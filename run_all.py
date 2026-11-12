import os, re, glob, json, shutil, random, subprocess

PY = "/kaggle/working/mamba/envs/mp/bin/python"
VILT = "/kaggle/working/vilt_200k_mlm_itm.ckpt"
DATA = "/kaggle/working/missing_aware_prompts/datasets/Hatefull_Memes"

def patch_and_data():
    for p, old, new in [
        ("vilt/utils/write_hatememes.py", "text_aug = text_aug_dir['{}.png'.format(data['id'])]", ""),
        ("vilt/datasets/hatememes_dataset.py", 'text_column_name="plots"', 'text_column_name="text"'),
        ("vilt/config.py", "multi_layer_prompt = True", "multi_layer_prompt = True\n    mask_missing_image = False")
    ]:
        s = open(p).read()
        if new == "" or new not in s:
            open(p, "w").write(s.replace(old, new))

    if os.path.exists(f"{DATA}/hatememes_test.arrow"):
        return
    base = os.path.dirname(glob.glob("/kaggle/input/**/train.jsonl", recursive=True)[0])
    raw = "/kaggle/working/raw/Hatefull_Memes/data"
    os.makedirs(raw, exist_ok=True)
    if not os.path.exists(f"{raw}/img"):
        os.symlink(f"{base}/img", f"{raw}/img")
    shutil.copy(f"{base}/train.jsonl", f"{raw}/train.jsonl")
    shutil.copy(f"{base}/dev.jsonl", f"{raw}/dev.jsonl")
    seen = glob.glob("/kaggle/input/**/test_seen.jsonl", recursive=True)
    shutil.copy(seen[0] if seen else f"{base}/dev.jsonl", f"{raw}/test.jsonl")
    subprocess.run([PY, "make_arrow.py", "--dataset", "hateful_memes", "--root", "/kaggle/working/raw"], check=True)

def make_tables():
    import torch
    from vilt.datasets import HateMemesDataset
    os.makedirs("./datasets/missing_tables", exist_ok=True)
    for split, typ, r in [("train","both",0.7), ("val","both",0.7), ("test","both",0.7), ("test","text",0.7), ("test","image",0.7)]:
        random.seed(0); torch.manual_seed(0)
        info = {"simulate_missing": False, "both_ratio": 0.5,
                "ratio": {"train": 0.7, "val": r, "test": r},
                "type": {"train": "both", "val": typ, "test": typ},
                "missing_table_root": "./datasets/missing_tables/"}
        HateMemesDataset(DATA, ["pixelbert"], split=split, image_size=384, max_text_len=128, missing_info=info)

def slim_and_merge(mode, ckpt_path, slim_path):
    import torch
    if mode == "slim":
        sd = torch.load(ckpt_path, map_location="cpu")["state_dict"]
        keep = {k: v for k, v in sd.items() if any(x in k for x in ["pooler.", "hatememes_classifier.", "prompt", "position_"])}
        torch.save(keep, slim_path)
    elif mode == "merge":
        sd = torch.load(VILT, map_location="cpu")["state_dict"]
        sd.update(torch.load(slim_path, map_location="cpu"))
        torch.save({"state_dict": sd}, ckpt_path)

def run_experiments():
    patch_and_data()
    subprocess.run([PY, "-c", "import run_all; run_all.make_tables()"], check=True)
    
    variants = {
        "baseline":   ["prompt_type=none"],
        "input":      ["prompt_type=input"],
        "attn":       ["prompt_type=attention"],
        "input_mask": ["prompt_type=input", "mask_missing_image=True"],
    }
    conds = [("both", 0.7), ("text", 0.7), ("image", 0.7)]
    os.makedirs("results", exist_ok=True)
    os.makedirs("records/hatememes", exist_ok=True)
    
    results = json.load(open("results/scores.json")) if os.path.exists("results/scores.json") else {}

    for v, flags in variants.items():
        slim_pt = f"results/{v}_slim.pt"
        ckpt = f"results/{v}_full.ckpt"
        base_cmd = [PY, "run.py", "with", f"data_root={DATA}", "num_gpus=1", "num_nodes=1", 
                    "per_gpu_batchsize=8", "num_workers=4", "task_finetune_hatememes"] + flags

        if not os.path.exists(slim_pt):
            print(f">>> Training {v}...")
            subprocess.run(base_cmd + [f"exp_name={v}", "seed=0", f"load_path={VILT}", "max_epoch=10"], check=True)
            cks = [c for c in glob.glob(f"result/{v}_seed0_*/version_*/checkpoints/*.ckpt") if "last.ckpt" not in c]
            best = max(cks, key=os.path.getmtime)
            subprocess.run([PY, "-c", f"import run_all; run_all.slim_and_merge('slim', '{best}', '{slim_pt}')"], check=True)
            shutil.rmtree(f"result/{v}_seed0_from_vilt_200k_mlm_itm", ignore_errors=True)

        subprocess.run([PY, "-c", f"import run_all; run_all.slim_and_merge('merge', '{ckpt}', '{slim_pt}')"], check=True)
        results.setdefault(v, {})

        for typ, r in conds:
            key = f"{typ}_{int(r*100)}"
            if key in results[v]:
                continue
            name = f"{v}_{key}"
            subprocess.run(base_cmd + [f"exp_name={name}", f"test_exp_name={name}", f"load_path={ckpt}",
                                       f"test_ratio={r}", f"test_type={typ}", "test_only=True"], check=True)
            line = open(f"records/hatememes/hatememes_{name}_on_missing_{typ}").read().strip().splitlines()[-1]
            m = re.search(r"AUROC: ([\d.]+), Accuracy: ([\d.]+)", line)
            results[v][key] = float(m[1])
            json.dump(results, open("results/scores.json", "w"), indent=2)

        if os.path.exists(ckpt):
            os.remove(ckpt)

    md = ["| Model | Missing Both (70%) | Missing Text (70%) | Missing Image (70%) |", "|---|---|---|---|"]
    for v in variants:
        r = results.get(v, {})
        md.append(f"| {v} | {r.get('both_70', '-')} | {r.get('text_70', '-')} | {r.get('image_70', '-')} |")
    open("results/summary.md", "w").write("\n".join(md))
    print("\n".join(md))

if __name__ == "__main__":
    run_experiments()
