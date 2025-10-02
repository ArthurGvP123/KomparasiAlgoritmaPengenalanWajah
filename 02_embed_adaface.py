# 02_embed_adaface.py — AdaFace embedding (auto-fix Flatten.view -> reshape)
import os, sys, argparse, traceback, time, importlib, inspect
from pathlib import Path
import numpy as np
import cv2, torch
from torch import nn
from tqdm import tqdm

def add_repo_to_sys_path(repo_dir: str):
    repo = Path(repo_dir).resolve()
    if not repo.exists():
        raise FileNotFoundError(f"[!] AdaFace repo tidak ditemukan: {repo}")
    # pastikan repo di depan sys.path agar 'net' dari repo ini yang ter-load
    sys.path.insert(0, str(repo))

def import_net_and_models(adaface_repo: str):
    add_repo_to_sys_path(adaface_repo)
    net = importlib.import_module("net")
    print(f"[LOG] net module loaded from: {net.__file__}")
    # ambil constructor IR_50/IR_101
    IR_50 = getattr(net, "IR_50")
    IR_101 = getattr(net, "IR_101")
    return net, IR_50, IR_101

def build_model(adaface_repo: str, arch: str = "ir101"):
    net, IR_50, IR_101 = import_net_and_models(adaface_repo)
    arch = arch.lower()
    print(f"[LOG] build_model: arch={arch}")
    if arch in ["ir101", "ir_101"]:
        model = IR_101(input_size=(112, 112))
    elif arch in ["ir50", "ir_50"]:
        model = IR_50(input_size=(112, 112))
    else:
        raise ValueError(f"Arch tidak dikenal: {arch} (pakai ir50/ir101)")
    return model, net

class SafeFlatten(nn.Module):
    def forward(self, x):
        return x.reshape(x.size(0), -1)

def replace_flatten_with_safe(model: nn.Module):
    """
    Ganti semua submodule bernama/bertipe Flatten menjadi SafeFlatten
    agar tidak crash saat tensor non-contiguous.
    """
    replaced = 0
    for name, child in list(model.named_children()):
        # deteksi via nama kelas (agar tetap jalan walau net.Flatten berbeda tipe)
        if child.__class__.__name__ == "Flatten":
            setattr(model, name, SafeFlatten())
            replaced += 1
        else:
            replaced += replace_flatten_with_safe(child)
    return replaced

def load_backbone(weights: str, adaface_repo: str, arch: str, device: str):
    print(f"[LOG] load_backbone: weights={weights}")
    model, net = build_model(adaface_repo, arch)

    # Auto-fix: replace Flatten modules
    nrep = replace_flatten_with_safe(model)
    if nrep > 0:
        print(f"[LOG] Replaced {nrep} Flatten -> SafeFlatten")

    # Muat checkpoint
    try:
        ckpt = torch.load(weights, map_location="cpu", weights_only=True)  # type: ignore[arg-type]
    except TypeError:
        ckpt = torch.load(weights, map_location="cpu")

    # Ambil state_dict
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        sd = ckpt["state_dict"]
    elif isinstance(ckpt, dict):
        sd = ckpt
    else:
        sd = {}

    # Bersihkan prefix umum
    PREFIXES = (
        "features.module.", "module.features.", "features.",
        "module.", "model.", "backbone.", "net.", "encoder."
    )
    cleaned = {}
    for k, v in sd.items():
        nk = k
        for pref in PREFIXES:
            if nk.startswith(pref):
                nk = nk[len(pref):]
        cleaned[nk] = v

    # Partial load: hanya key yang ada & shape cocok
    model_sd = model.state_dict()
    to_load, skipped = {}, []
    for k, v in cleaned.items():
        if k not in model_sd:
            skipped.append((k, tuple(v.shape), "missing_in_model")); continue
        if tuple(model_sd[k].shape) != tuple(v.shape):
            skipped.append((k, tuple(v.shape), tuple(model_sd[k].shape))); continue
        to_load[k] = v

    missing, unexpected = model.load_state_dict(to_load, strict=False)
    print(f"[Info] Loaded params: {len(to_load)} | missing={len(missing)} | unexpected={len(unexpected)}")
    if skipped:
        print("[Info] Skipped (shape mismatch / not in model), contoh <=10:")
        for i, (k, shp_v, shp_m) in enumerate(skipped[:10], 1):
            print(f"  {i:02d}. {k}: ckpt{shp_v} vs model{shp_m}")

    model.eval().to(device).float()
    return model

def preprocess_bgr_adaface(img_bgr, size=112):
    # BGR, 112x112, normalize [-1,1]
    if img_bgr.shape[:2] != (size, size):
        img_bgr = cv2.resize(img_bgr, (size, size), interpolation=cv2.INTER_LINEAR)
    img = img_bgr.astype(np.float32) / 255.0
    img = (img - 0.5) / 0.5
    img = np.transpose(img, (2, 0, 1))  # CHW
    return img

@torch.no_grad()
def embed_batch(model, batch_imgs, device):
    x = torch.from_numpy(np.stack(batch_imgs)).to(device).float()
    y = model(x)
    if isinstance(y, (tuple, list)):  # beberapa model mengembalikan (feat, norm)
        y = y[0]
    feat = y.detach().cpu().numpy()
    feat = feat / np.clip(np.linalg.norm(feat, axis=1, keepdims=True), 1e-12, None)
    return feat

def collect_images(root: Path):
    exts = (".jpg", ".jpeg", ".png")
    return sorted([str(p) for p in root.rglob("*") if p.suffix.lower() in exts])

def main():
    t0 = time.time()
    print("== AdaFace Embedding v3 ==")

    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="crops_112", help="root gambar aligned 112x112")
    ap.add_argument("--weights", required=True, help="path .ckpt/.pth AdaFace")
    ap.add_argument("--adaface-repo", required=True, help="folder repo AdaFace (berisi net.py IR_50/IR_101)")
    ap.add_argument("--arch", default="ir101", choices=["ir50","ir101"])
    ap.add_argument("--out", default="embeds_adaface_ir101.npz")
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--limit", type=int, default=0, help="uji: batasi jumlah gambar (0=semua)")
    ap.add_argument("--save-magnitude-csv", default="adaface_magnitude.csv",
                    help="opsional: simpan norma fitur per gambar (None untuk skip)")
    args = ap.parse_args()

    print(f"[LOG] args: {args}")
    root = Path(args.root)
    if not root.exists():
        print(f"[!] Folder root tidak ada: {root}")
        return

    paths = collect_images(root)
    print(f"[LOG] ditemukan {len(paths)} gambar di {root}")
    if len(paths) == 0:
        print("[!] Tidak ada gambar .jpg/.jpeg/.png. Stop.")
        return
    if args.limit and args.limit > 0:
        paths = paths[:args.limit]
        print(f"[LOG] MODE UJI: {len(paths)} gambar pertama.")

    model = load_backbone(args.weights, args.adaface_repo, args.arch, args.device)
    print(f"[LOG] device: {args.device}")

    rels = [str(Path(p).relative_to(root)).replace("\\","/") for p in paths]
    feats, mags = {}, {}
    B = max(1, int(args.batch))
    buf_imgs, buf_idx = [], []

    # sanity-check 1 sampel
    try:
        sample = cv2.imread(paths[0]); assert sample is not None
        s_pre = preprocess_bgr_adaface(sample, 112)
        s_feat = embed_batch(model, [s_pre], args.device)[0]
        print(f"[LOG] sanity-check: 1 sample embed shape={s_feat.shape}, norm={np.linalg.norm(s_feat):.6f}")
    except Exception as e:
        print("[ERR] sanity-check gagal:")
        print("".join(traceback.format_exception(type(e), e, e.__traceback__)))
        return

    # proses batch
    proc = 0
    for i, p in enumerate(tqdm(paths, desc=f"Embedding[AdaFace-{args.arch}]")):
        img = cv2.imread(p)
        if img is None: continue
        buf_imgs.append(preprocess_bgr_adaface(img, 112))
        buf_idx.append(i)
        if len(buf_imgs) == B:
            F = embed_batch(model, buf_imgs, args.device)
            for j, ii in enumerate(buf_idx):
                feats[rels[ii]] = F[j]
                mags[rels[ii]]  = float(np.linalg.norm(F[j]))
            proc += len(buf_imgs)
            buf_imgs, buf_idx = [], []

    if buf_imgs:
        F = embed_batch(model, buf_imgs, args.device)
        for j, ii in enumerate(buf_idx):
            feats[rels[ii]] = F[j]
            mags[rels[ii]]  = float(np.linalg.norm(F[j]))
        proc += len(buf_imgs)

    # simpan
    out_npz = Path(args.out)
    np.savez_compressed(out_npz, **feats)
    print(f"[OK] Saved {len(feats)} embeddings -> {out_npz}")

    if args.save_magnitude_csv and len(mags):
        out_csv = Path(args.save_magnitude_csv)
        with out_csv.open("w", encoding="utf-8") as f:
            f.write("path,mag\n")
            for k,v in mags.items():
                f.write(f"{k},{v:.8f}\n")
        print(f"[OK] Saved magnitudes -> {out_csv}")

    dt = time.time() - t0
    print(f"[DONE] processed={proc}/{len(paths)} images in {dt:.2f}s")

if __name__ == "__main__":
    torch.backends.cudnn.benchmark = False
    try:
        main()
    except Exception as e:
        print("[FATAL] Uncaught exception:")
        print("".join(traceback.format_exception(type(e), e, e.__traceback__)))
        raise
