# Experiment Log (Deney Günlüğü)

## 1. Initial Smoke Test (Hateful Memes — 1 Epoch)
- **Environment:** Kaggle 2xT4 (1 GPU used), Python 3.7.12, torch 1.10.0+cu113, timm 0.4.12
- **Dataset:** `parthplc/facebook-hateful-meme-dataset`, converted via `make_arrow.py`
- **Bug Fix 1:** `vilt/utils/write_hatememes.py` — removed undefined `text_aug_dir` line.
- **Bug Fix 2:** `vilt/datasets/hatememes_dataset.py` — changed `text_column_name` from `"plots"` to `"text"`.
- **Run Settings:** `task_finetune_hatememes`, `max_epoch=1`, `per_gpu_batchsize=8`, missing both 70%, input-level prompt (`seed=0`).
- **Outcome:** Completed in 8m 44s (`loss: 2.42 -> 0.75`, `val/the_metric: 0.544`), verified pipeline before full 10-epoch runs with labelled `test_seen.jsonl`.

## 2. Full Reproduction & Masking Runs (10 Epochs, Seed 0)
- **Backbone:** Frozen `vilt_200k_mlm_itm.ckpt`
- **Training Split:** Hateful Memes `train` (8,500 samples) with 70% missing-both (30% complete, 35% text-only, 35% image-only, fixed with `seed=0`)
- **Validation Split:** `dev` (500 samples), evaluated every 0.5 epoch (`val_check_interval=0.5`)
- **Test Split:** `test_seen` (1,000 labelled samples), evaluated across three 70% missing conditions (`Both`, `Text`, `Image`)
- **Batch1 (GPU 0 & GPU 1):** `baseline`, `input`, `attn`, `input_mask` (~1h 24m per model)
- **Batch 2 (Remaining Masked Variants):** `baseline_mask`, `attn_mask`

### Final Summary Table (AUROC %)

| Model | Both 70% | Text 70% | Image 70% | Mean |
|---|---|---|---|---|
| baseline | 61.65 | 60.10 | 63.18 | 61.64 |
| input | 55.93 | 55.95 | 56.35 | 56.08 |
| attn | 62.58 | 61.08 | 62.96 | 62.21 |
| input_mask | 49.43 | 51.31 | 49.64 | 50.13 |
| baseline_mask | 63.31 | 60.21 | 64.72 | 62.75 |
| attn_mask | 63.10 | 61.98 | 63.77 | 62.95 |

Test source: `test_seen` | Metric: AUROC (%) | 10 epochs | seed 0
