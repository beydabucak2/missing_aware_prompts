# Reused-Component Inventory

## 1. External Code & Pre-trained Weights (Reused)
- **Official Base Repository:** Forked from [YiLunLee/missing_aware_prompts](https://github.com/YiLunLee/missing_aware_prompts) (Lee et al., CVPR 2023).
  - Reused components: ViLT model definition (`vilt/modules/vilt_module.py`), base prompt attachment logic, PyArrow dataset conversion (`make_arrow.py`), and missing-modality table generation.
- **Pre-trained Backbone Weights:** `vilt_200k_mlm_itm.ckpt` from Kim et al. (ICML 2021) [dandelin/ViLT](https://github.com/dandelin/ViLT).
- **Dataset:** Hateful Memes dataset (Kiela et al., NeurIPS 2020) via Kaggle inputs.

## 2. Project-Specific Contributions (Authored / Modified by Us)
- **Targeted Improvement (`mask_missing_image`):** Implemented attention masking for dummy image tokens (`image_masks[miss, 1:] = 0`) in `vilt/modules/vilt_missing_aware_prompt_module.py` and added configuration support in `vilt/config.py`.
- **No-Prompt Baseline (`prompt_type=none`):** Modified `vilt_missing_aware_prompt_module.py` to bypass prompt concatenation and freeze unused prompt tensors so the baseline uses the exact same training pipeline.
- **Dataset Loader Bug Fixes:** Fixed undefined `text_aug_dir` bug in `vilt/utils/write_hatememes.py` and corrected the JSONL text column mapping (`plots` -> `text`) in `vilt/datasets/hatememes_dataset.py`.
- **Automated Training, Slimming & Evaluation Pipeline (`run_all.py`):** Wrote from scratch to automate data preparation, deterministic missing-table creation, training, checkpoint slimming/merging (`*_slim.pt`), and evaluation across test splits.

## 3. Course Project Separation Note (YZM5110)
This repository and all experiments herein are submitted exclusively for the **Research Project (Week 7)**. None of the code, trained weights, or experimental results from this repository will be reused for credit in the **Practical Project (Week 12)**.
