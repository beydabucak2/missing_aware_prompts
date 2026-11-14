# Design Notes & Experimental Decisions

## 1. Problem & Target Weakness Selection
In Lee et al. (CVPR 2023), missing images are replaced with an all-ones dummy image tensor. Although missing-aware prompts are attached to inform the model of the missing modality, the self-attention layers still attend to all 100+ patch tokens generated from this constant dummy image. Our hypothesis was that explicitly masking these uninformative dummy image tokens in the attention mask would reduce noise during multimodal fusion.

## 2. Design of the Targeted Fix (`+mask`)
- Instead of masking all visual tokens (which breaks the ViLT forward pass expecting at least one image token representation), we kept the first visual token visible and masked the remaining patch tokens: `image_masks[miss, 1:] = 0`.
- This design requires zero additional learnable parameters and isolates the effect of attention masking across all three architectures (`baseline`, `input`, `attn`).

## 3. Fair Control & Compute Constraints
- **Fixed Missingness:** We fixed the random seed (`seed=0`) during missing-table generation so that every model variant is trained and evaluated on the exact same missing-modality sample splits (70% missing rate).
- **Single-Training Evaluation:** Due to Kaggle T4 GPU compute limits, we trained each model once for 10 epochs on the `missing-both` (30% complete, 35% text-only, 35% image-only) split and evaluated the best validation checkpoint across all three test splits (`Both`, `Text`, `Image`).
- **Slim Checkpoints:** Since the ViLT backbone is frozen, we wrote `slim_and_merge()` in `run_all.py` to strip frozen backbone weights and store only the trainable parameters (`pooler`, `classifier`, `prompts`) in `results/*_slim.pt` (~3-5 MB per model instead of ~450 MB).

## 4. Analysis of the `input+mask` Failure
- In our reproduction, `input` prompts (length 16, layers 0–5) underperformed the no-prompt baseline (55.93 vs. 61.65 AUROC), and combining them with masking (`input+mask`) collapsed performance to ~50 AUROC (chance level).
- **Root Cause Hypothesis:** Input-level prompting prepends 16 tokens at each of the first 6 layers (adding 96 prompt tokens by layer 5). On Hateful Memes, where text inputs are short (max 128 tokens, often <20 tokens), masking out the image patches leaves the attention mechanism dominated almost entirely by newly initialized prompt tokens trained at a high learning rate (`1e-2`), leading to optimization divergence.
