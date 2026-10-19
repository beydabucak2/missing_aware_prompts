# Deney günlüğü
## 2026-10-18 ilk duman testi (Hateful Memes)
- Ortam: Kaggle 2xT4 (1 GPU kullanıldı), Python 3.7.12, torch 1.10.0+cu113, timm 0.4.12
- Veri: parthplc/facebook-hateful-meme-dataset, make_arrow ile dönüştürüldü
- Kod düzeltmesi 1: vilt/utils/write_hatememes.py, tanımsız text_aug_dir satırı silindi (sonradan kullanılmıyordu)
- Kod düzeltmesi 2: vilt/datasets/hatememes_dataset.py, text_column_name "plots" -> "text"
- Geçici çözüm: etiketsiz test.jsonl yüzünden hatememes_test.arrow = hatememes_dev.arrow kopyası. Test skoru gerçek test skoru DEĞİL.
- Koşu: task_finetune_hatememes, max_epoch=1, per_gpu_batchsize=8, missing both %70, input-level prompt
- Sonuç: 8 dk 44 sn, loss 2.42 -> 0.75, en iyi val/the_metric 0.544 (33 optimizer adımı, anlamlı sonuç değil)
## Baseline (10 Epochs, prompt_type="none")

## Prompt-based Model (10 Epochs, prompt_type="input")
- **Date:** October 19, 2026
- **Model:** ViLT with Missing-Aware Prompts (Fine-tuned on Hateful Memes)
- **Settings:** prompt_type="input", batch_size=8, epochs=10, gpus=1
- **Status:** Completed successfully (~1h 24m).
