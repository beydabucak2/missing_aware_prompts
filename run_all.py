import os, re, glob, json, shutil, random, subprocess, sys

PY = "/kaggle/working/mamba/envs/mp/bin/python"
VILT = "/kaggle/working/vilt_200k_mlm_itm.ckpt"
REPO = "/kaggle/working/missing_aware_prompts"
DATA = REPO + "/datasets/Hatefull_Memes"
TMP = "/kaggle/working/tmp"
RES = os.environ.get("RES", "results")
EPOCHS = int(os.environ.get("EPOCHS", 10))
VARIANTS = {
    "baseline": ["prompt_type=none"],
    "input": ["prompt_type=input"],
    "attn": ["prompt_type=attention"],
    "input_mask": ["prompt_type=input", "mask_missing_image=True"],
}
CONDS = [("both", 0.7), ("text", 0.7), ("image", 0.7)]


def prepare():
    base = os.path.dirname([p for p in glob.glob("/kaggle/input/**/train.jsonl", recursive=True) if "parthplc" in p][0])
    raw = "/kaggle/working/raw/Hatefull_Memes/data"
    os.makedirs(raw, exist_ok=True)
    if not os.path.exists(raw + "/img"):
        os.symlink(base + "/img", raw + "/img")
    for s in ("train", "dev"):
        shutil.copy(f"{base}/{s}.jsonl", f"{raw}/{s}.jsonl")
    seen = glob.glob("/kaggle/input/**/test_seen.jsonl", recursive=True)
    rows = [json.loads(l) for l in open(seen[0]) if l.strip()] if seen else []
    ok = bool(rows) and all("label" in r and os.path.exists(f"{base}/{r['img']}") for r in rows)
    shutil.copy(seen[0] if ok else f"{base}/dev.jsonl", f"{raw}/test.jsonl")
    os.makedirs(RES, exist_ok=True)
    open(f"{RES}/test_source.txt", "w").write("test_seen" if ok else "dev")
    print("test kaynagi:", "test_seen" if ok else "DEV (uyari)")
    subprocess.run([PY, "make_arrow.py", "--dataset", "hateful_memes", "--root", "/kaggle/working/raw"], check=True)
    subprocess.run([PY, "-c", "import run_all; run_all.make_tables()"], check=True)


def make_tables():
    import torch
    from vilt.datasets import HateMemesDataset
    os.makedirs("./datasets/missing_tables", exist_ok=True)
    jobs = [("train", "both", 0.7), ("val", "both", 0.7)] + [("test", t, r) for t, r in CONDS]
    for split, typ, r in jobs:
        random.seed(0)
        torch.manual_seed(0)
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
    else:
        sd = torch.load(VILT, map_location="cpu")["state_dict"]
        sd.update(torch.load(slim_path, map_location="cpu"))
        torch.save({"state_dict": sd}, ckpt_path)


def run(cmd, log, env):
    with open(log, "w") as f:
        rc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, env=env).returncode
    if rc != 0:
        print("HATA ->", log)
        print("".join(open(log, errors="ignore").readlines()[-40:]))
        sys.exit(1)


def py(code):
    subprocess.run([PY, "-c", "import run_all; run_all." + code], check=True)


def work(gpu, variants):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu, MASTER_PORT=str(29500 + int(gpu)), PYTHONUNBUFFERED="1")
    for d in (RES, "logs", "records/hatememes", TMP):
        os.makedirs(d, exist_ok=True)
    for v in variants:
        exp = f"{RES}_{v}"
        slim, sc_path, ckpt = f"{RES}/{v}_slim.pt", f"{RES}/{v}_scores.json", f"{TMP}/{exp}.ckpt"
        scores = json.load(open(sc_path)) if os.path.exists(sc_path) else {}
        cmd = [PY, "run.py", "with", f"data_root={DATA}", "num_gpus=1", "num_nodes=1", "per_gpu_batchsize=8",
               "num_workers=2", "task_finetune_hatememes"] + VARIANTS[v]
        if not os.path.exists(slim):
            print(f"[gpu{gpu}] egitim: {v}", flush=True)
            run(cmd + [f"exp_name={exp}", "seed=0", f"load_path={VILT}", f"max_epoch={EPOCHS}", "val_check_interval=0.5"],
                f"logs/{exp}_train.log", env)
            cks = [c for c in glob.glob(f"result/{exp}_seed0_*/version_*/checkpoints/*.ckpt") if "last.ckpt" not in c]
            py(f"slim_and_merge('slim', '{max(cks, key=os.path.getmtime)}', '{slim}')")
            shutil.rmtree(f"result/{exp}_seed0_from_vilt_200k_mlm_itm", ignore_errors=True)
        todo = [(t, r) for t, r in CONDS if f"{t}_{int(r * 100)}" not in scores]
        if todo:
            py(f"slim_and_merge('merge', '{ckpt}', '{slim}')")
        for typ, r in todo:
            key = f"{typ}_{int(r * 100)}"
            name = f"{exp}_{key}"
            rec = f"records/hatememes/hatememes_{name}_on_missing_{typ}"
            if os.path.exists(rec):
                os.remove(rec)
            run(cmd + [f"exp_name={name}", f"test_exp_name={name}", f"load_path={ckpt}", f"test_ratio={r}",
                       f"test_type={typ}", "test_only=True"], f"logs/{name}.log", env)
            scores[key] = float(re.search(r"AUROC: ([\d.]+)", open(rec).read().strip().splitlines()[-1]).group(1))
            json.dump(scores, open(sc_path, "w"), indent=2)
            print(f"[gpu{gpu}] {v} {key}: {scores[key]}", flush=True)
        if os.path.exists(ckpt):
            os.remove(ckpt)
    print(f"[gpu{gpu}] bitti", flush=True)


def summary():
    rows = ["| Model | Both 70% | Text 70% | Image 70% |", "|---|---|---|---|"]
    for v in VARIANTS:
        p = f"{RES}/{v}_scores.json"
        r = json.load(open(p)) if os.path.exists(p) else {}
        rows.append(f"| {v} | {r.get('both_70', '-')} | {r.get('text_70', '-')} | {r.get('image_70', '-')} |")
    rows.append(f"\nTest kaynagi: {open(RES + '/test_source.txt').read()} | AUROC | {EPOCHS} epoch | seed 0")
    open(f"{RES}/summary.md", "w").write("\n".join(rows))
    print("\n".join(rows))


if __name__ == "__main__":
    os.chdir(REPO)
    {"prepare": prepare, "work": lambda: work(sys.argv[2], sys.argv[3].split(",")), "summary": summary}[sys.argv[1]]()
