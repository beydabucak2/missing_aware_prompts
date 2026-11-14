

<div align="center">
  <img src="fig/model.jpeg"/>
</div>

<div align="center">
  <img src="fig/model.jpeg"/>
</div>

# Masking Dummy Images in Missing-Aware Prompting (Hateful Memes)

This repo is a fork of the official code of Lee et al., *Multimodal Prompting with Missing Modalities for Visual Recognition* (CVPR 2023): https://github.com/YiLunLee/missing_aware_prompts

We reproduce three models from the paper and test one fix: masking the dummy image tokens in attention.

## Results

Hateful Memes, `test_seen`, AUROC (%), 70% missing rate, 10 epochs, seed 0.
One model per row is trained on the missing-both case and tested on three cases.

| Model | Flags | Both | Text | Image | Mean |
|---|---|---|---|---|---|
| baseline | `prompt_type=none` | 61.65 | 60.10 | 63.18 | 61.64 |
| input | `prompt_type=input` | 55.93 | 55.95 | 56.35 | 56.08 |
| attn | `prompt_type=attention` | 62.58 | 61.08 | 62.96 | 62.21 |
| input+mask | `prompt_type=input mask_missing_image=True` | 49.43 | 51.31 | 49.64 | 50.13 |
| baseline+mask | `prompt_type=none mask_missing_image=True` | 63.31 | 60.21 | 64.72 | 62.75 |
| attn+mask | `prompt_type=attention mask_missing_image=True` | 63.10 | 61.98 | 63.77 | 62.95 |

Only one seed was used, so differences of about one point can be noise.
Scores are in `results/*_scores.json`. Summary tables are in `EXPERIMENT_LOG.md`.

## What we changed

- `mask_missing_image=True` (`vilt/config.py`, `vilt/modules/vilt_missing_aware_prompt_module.py`): for samples with a missing image, the attention mask of all image tokens except the first is set to 0.
- `prompt_type=none`: no prompts. Only the pooler and the classifier are trained.
- `vilt/utils/write_hatememes.py`: removed a line that uses an undefined variable (`text_aug_dir`).
- `vilt/datasets/hatememes_dataset.py`: text column name changed from `plots` to `text`.
- `run_all.py`: prepares the data, trains the models, saves small checkpoints and evaluates in the three test cases.

## How to run (Kaggle)

1. Open a Kaggle notebook. Choose **GPU T4 x2** and turn **Internet on**.
2. Add two datasets as input:
   - `parthplc/facebook-hateful-meme-dataset` (images, `train.jsonl`, `dev.jsonl`)
   - a dataset with a **labelled** `test_seen.jsonl` (the script looks for it in `/kaggle/input`). If it finds none, it uses `dev.jsonl` as the test set and writes `dev` in `results/test_source.txt`.
3. Set up the environment and download the ViLT weights:

```bash
git clone [https://github.com/beydabucak2/missing_aware_prompts.git](https://github.com/beydabucak2/missing_aware_prompts.git)
cd missing_aware_prompts
wget -qO- [https://micro.mamba.pm/api/micromamba/linux-64/latest](https://micro.mamba.pm/api/micromamba/linux-64/latest) | tar -xj bin/micromamba
./bin/micromamba create -y -q -n mp -r /kaggle/working/mamba -c conda-forge python=3.7
PY=/kaggle/working/mamba/envs/mp/bin/python
$PY -m pip install -q torch==1.10.0+cu113 torchvision==0.11.0+cu113 -f [https://download.pytorch.org/whl/cu113/torch_stable.html](https://download.pytorch.org/whl/cu113/torch_stable.html)
grep -v pytorch-image-models requirements.txt > /kaggle/working/req.txt
$PY -m pip install -q -r /kaggle/working/req.txt timm==0.4.12
wget -q -O /kaggle/working/vilt_200k_mlm_itm.ckpt [https://github.com/dandelin/ViLT/releases/download/200k/vilt_200k_mlm_itm.ckpt](https://github.com/dandelin/ViLT/releases/download/200k/vilt_200k_mlm_itm.ckpt)
```

4. Prepare the data and the missing tables (seed 0). This takes about 15 minutes:

```bash
python run_all.py prepare
```

### Quick Demo (Evaluation Only — No Training Required)
Because pre-trained slim checkpoints (`results/*_slim.pt`) are already included in this repository, `run_all.py` automatically skips the multi-hour training stage if you only want to verify or demo a model's test evaluation.

To run a fast evaluation demo on a specific model (e.g., `attn_mask`) using the saved checkpoint:

```bash
rm results/attn_mask_scores.json
python run_all.py work 0 attn_mask
```

The script will merge the saved slim prompt/classifier weights (`results/attn_mask_slim.pt`) with the frozen ViLT backbone and evaluate `Both`, `Text`, and `Image` test splits in ~2 minutes.

### Full Training and Evaluation
5. To train and test all models from scratch (if `results/*_slim.pt` are deleted), run these two commands in separate processes, one for each GPU:

```bash
python run_all.py work 0 baseline,input,attn
python run_all.py work 1 input_mask,baseline_mask,attn_mask
python run_all.py summary
```

`run_all.py` skips a model if its files are already in `results/`.
To train a model again, delete `results/<name>_slim.pt` and `results/<name>_scores.json` first.
The script uses `/kaggle/...` paths, so it is written for Kaggle.

## Files

- `run_all.py`: data preparation, training, evaluation, summary
- `results/`: scores (`*_scores.json`), small checkpoints (`*_slim.pt`), `summary.md`
- `EXPERIMENT_LOG.md`: summary tables of the runs
- `DESIGN_NOTES.md`: experimental decisions and failure analysis
- `REUSED.md`: reused-component inventory and project separation note
- `logs/`: training and test logs (if present)
