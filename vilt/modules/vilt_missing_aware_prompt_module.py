import torch
import torch.nn as nn
import pytorch_lightning as pl
import vilt.modules.vision_transformer_prompts as vit
from transformers.models.bert.modeling_bert import BertConfig, BertEmbeddings
from vilt.modules import heads, objectives, vilt_utils

class ViLTransformerSS(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()

        bert_config = BertConfig(
            vocab_size=config["vocab_size"],
            hidden_size=config["hidden_size"],
            num_hidden_layers=config["num_layers"],
            num_attention_heads=config["num_heads"],
            intermediate_size=config["hidden_size"] * config["mlp_ratio"],
            max_position_embeddings=config["max_text_len"],
            hidden_dropout_prob=config["drop_rate"],
            attention_probs_dropout_prob=config["drop_rate"],
        )
        
        self.text_embeddings = BertEmbeddings(bert_config)
        self.text_embeddings.apply(objectives.init_weights)
        self.token_type_embeddings = nn.Embedding(2, config["hidden_size"])
        self.token_type_embeddings.apply(objectives.init_weights)

        self.transformer = getattr(vit, self.hparams.config["vit"])(
            pretrained=(self.hparams.config["load_path"] == ""), config=self.hparams.config
        )

        self.pooler = heads.Pooler(config["hidden_size"])
        self.pooler.apply(objectives.init_weights)

        if config["loss_names"]["mlm"] > 0:
            self.mlm_score = heads.MLMHead(bert_config)
            self.mlm_score.apply(objectives.init_weights)
        if config["loss_names"]["itm"] > 0:
            self.itm_score = heads.ITMHead(config["hidden_size"])
            self.itm_score.apply(objectives.init_weights)
        if config["loss_names"]["mpp"] > 0:
            self.mpp_score = heads.MPPHead(bert_config)
            self.mpp_score.apply(objectives.init_weights)

        if self.hparams.config["load_path"] != "" and not self.hparams.config["test_only"] and not self.hparams.config["finetune_first"]:
            ckpt = torch.load(self.hparams.config["load_path"], map_location="cpu")
            state_dict = ckpt["state_dict"]
            if config["max_text_len"] != 40:
                state_dict['text_embeddings.position_ids'] = torch.Tensor(range(config["max_text_len"])).long().view(1,-1)
                pos_emb = state_dict['text_embeddings.position_embeddings.weight']
                pos_emb = torch.nn.functional.interpolate(pos_emb.view(1,1,40,768), size=(config["max_text_len"],768), mode='bilinear').squeeze()
                state_dict['text_embeddings.position_embeddings.weight'] = pos_emb
            self.load_state_dict(state_dict, strict=False)

        hs = self.hparams.config["hidden_size"]
        if self.hparams.config["loss_names"]["hatememes"] > 0:
            cls_num = self.hparams.config["hatememes_class_num"]
            self.hatememes_classifier = nn.Sequential(
                nn.Linear(hs, hs * 2), nn.LayerNorm(hs * 2), nn.GELU(), nn.Linear(hs * 2, cls_num)
            )
            self.hatememes_classifier.apply(objectives.init_weights)

        if self.hparams.config["load_path"] != "" and self.hparams.config["finetune_first"]:
            ckpt = torch.load(self.hparams.config["load_path"], map_location="cpu")
            self.load_state_dict(ckpt["state_dict"], strict=False)            
  
        self.prompt_type = self.hparams.config["prompt_type"]
        prompt_length = self.hparams.config["prompt_length"]
        self.prompt_length = prompt_length
        embed_dim = self.hparams.config["hidden_size"]
        self.learnt_p = self.hparams.config["learnt_p"]
        self.prompt_layers = self.hparams.config["prompt_layers"]
        self.multi_layer_prompt = self.hparams.config["multi_layer_prompt"]
        prompt_num = len(self.prompt_layers) if self.multi_layer_prompt else 1

        complete_prompt = torch.zeros(prompt_num, prompt_length, embed_dim)
        complete_prompt[:,0:1,:].fill_(1)            
        if self.learnt_p and self.prompt_type == 'attention':
            complete_prompt[:,prompt_length//2:prompt_length//2+1,:].fill_(1)
        self.complete_prompt = nn.Parameter(complete_prompt)

        missing_text_prompt = torch.zeros(prompt_num, prompt_length, embed_dim)
        missing_text_prompt[:,2:3,:].fill_(1)            
        if self.learnt_p and self.prompt_type == 'attention':
            missing_text_prompt[:,prompt_length//2+2:prompt_length//2+3,:].fill_(1)
        self.missing_text_prompt = nn.Parameter(missing_text_prompt)

        missing_img_prompt = torch.zeros(prompt_num, prompt_length, embed_dim)
        missing_img_prompt[:,1:2,:].fill_(1)            
        if self.learnt_p and self.prompt_type == 'attention':
            missing_img_prompt[:,prompt_length//2+1:prompt_length//2+2,:].fill_(1)
        self.missing_img_prompt = nn.Parameter(missing_img_prompt)

        self.mask_missing_image = bool(self.hparams.config.get("mask_missing_image", False))
        if not self.learnt_p or self.prompt_type == 'none':
            self.complete_prompt.requires_grad = False
            self.missing_text_prompt.requires_grad = False           
            self.missing_img_prompt.requires_grad = False

        for param in self.transformer.parameters(): param.requires_grad = False
        for param in self.text_embeddings.parameters(): param.requires_grad = False
        for param in self.token_type_embeddings.parameters(): param.requires_grad = False

        vilt_utils.set_metrics(self)
        self.current_tasks = list()

        if self.hparams.config["load_path"] != "" and self.hparams.config["test_only"]:
            ckpt = torch.load(self.hparams.config["load_path"], map_location="cpu")
            self.load_state_dict(ckpt["state_dict"], strict=False)
        self.records = {}

    def infer(self, batch, mask_text=False, mask_image=False, image_token_type_idx=1, image_embeds=None, image_masks=None, is_train=None):
        imgkey = f"image_{image_token_type_idx - 1}" if f"image_{image_token_type_idx - 1}" in batch else "image"
        do_mlm = "_mlm" if mask_text else ""
        text_ids = batch[f"text_ids{do_mlm}"]
        text_labels = batch[f"text_labels{do_mlm}"]
        text_masks = batch[f"text_masks"]
        text_embeds = self.text_embeddings(text_ids)
        img = batch[imgkey][0]     
        
        if image_embeds is None and image_masks is None:
            image_embeds, image_masks, patch_index, image_labels = self.transformer.visual_embed(
                img, max_image_len=self.hparams.config["max_image_len"], mask_it=mask_image
            )
        else:
            patch_index, image_labels = None, None

        if self.mask_missing_image:
            _miss_img = torch.tensor([int(m) == 2 for m in batch["missing_type"]], device=image_masks.device)
            image_masks = image_masks.clone()
            image_masks[_miss_img, 1:] = 0

        text_embeds = text_embeds + self.token_type_embeddings(torch.zeros_like(text_masks))
        image_embeds = image_embeds + self.token_type_embeddings(torch.full_like(image_masks, image_token_type_idx))
        
        prompts = None
        for idx in range(len(img) if self.prompt_type != 'none' else 0):
            if batch["missing_type"][idx] == 0: prompt = self.complete_prompt        
            elif batch["missing_type"][idx] == 1: prompt = self.missing_text_prompt
            elif batch["missing_type"][idx] == 2: prompt = self.missing_img_prompt
            if prompt.size(0) != 1: prompt = prompt.unsqueeze(0)
            prompts = prompt if prompts is None else torch.cat([prompts, prompt], dim=0)

        if self.prompt_type == 'none':
            prompt_masks = text_masks.new_zeros(text_masks.size(0), 0)
        elif self.learnt_p:
            if self.prompt_type == 'attention':
                prompt_masks = torch.ones(prompts.shape[0], self.prompt_length//2, dtype=prompts.dtype, device=prompts.device).long()
            elif self.prompt_type == 'input':
                prompt_masks = torch.ones(prompts.shape[0], self.prompt_length*len(self.prompt_layers), dtype=prompts.dtype, device=prompts.device).long()
        else:
            prompt_masks = torch.ones(prompts.shape[0], self.prompt_length, dtype=prompts.dtype, device=prompts.device).long()   
        
        co_masks = torch.cat([prompt_masks, text_masks, image_masks], dim=1)
        x = torch.cat([text_embeds, image_embeds], dim=1).detach()

        for i, blk in enumerate(self.transformer.blocks):
            if self.prompt_type == 'none':
                x, _attn = blk(x, mask=co_masks)
            elif i in self.prompt_layers:
                if self.multi_layer_prompt:
                    x, _attn = blk(x, mask=co_masks, prompts=prompts[:,self.prompt_layers.index(i)], learnt_p=self.learnt_p, prompt_type=self.prompt_type)
                else:
                    x, _attn = blk(x, mask=co_masks, prompts=prompts, learnt_p=self.learnt_p)
            else:
                x, _attn = blk(x, mask=co_masks)

        x = self.transformer.norm(x)
        
        if self.prompt_type == 'none': total_prompt_len = 0
        elif self.prompt_type == 'input': total_prompt_len = len(self.prompt_layers) * prompts.shape[-2]
        elif self.prompt_type == 'attention': total_prompt_len = prompts.shape[-2]
        
        text_feats = x[:, total_prompt_len : total_prompt_len + text_embeds.shape[1]]
        image_feats = x[:, total_prompt_len + text_embeds.shape[1] :]
        cls_feats = self.pooler(x[:, total_prompt_len:total_prompt_len+1]) if self.prompt_type in ('input', 'none') else self.pooler(x)
            
        return {
            "text_feats": text_feats, "image_feats": image_feats, "cls_feats": cls_feats,
            "raw_cls_feats": x[:, 0], "image_labels": image_labels, "image_masks": image_masks,
            "text_labels": text_labels, "text_ids": text_ids, "text_masks": text_masks, "patch_index": patch_index,
        }

    def forward(self, batch):
        ret = dict()
        if len(self.current_tasks) == 0:
            ret.update(self.infer(batch))
            return ret
        if "hatememes" in self.current_tasks:
            ret.update(objectives.compute_hatememes(self, batch))
        return ret

    def training_step(self, batch, batch_idx):
        vilt_utils.set_task(self)
        output = self(batch)
        return sum([v for k, v in output.items() if "loss" in k])

    def training_epoch_end(self, outs): vilt_utils.epoch_wrapup(self)
    def validation_step(self, batch, batch_idx): vilt_utils.set_task(self); self(batch)
    def validation_epoch_end(self, outs): vilt_utils.epoch_wrapup(self)
    def test_step(self, batch, batch_idx): vilt_utils.set_task(self); self(batch); return dict()
    def test_epoch_end(self, outs): vilt_utils.epoch_wrapup(self)
    def configure_optimizers(self): return vilt_utils.set_schedule(self)
