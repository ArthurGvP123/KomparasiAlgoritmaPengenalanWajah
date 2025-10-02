# 02_embed_curricular_fixed.py
import os, sys, argparse
from pathlib import Path
import cv2, numpy as np, torch
from tqdm import tqdm

HERE = Path(__file__).resolve().parent
sys.path.append(str(HERE))  # pastikan bisa import "backbone.*"

def build_model(arch: str):
    """
    Bangun backbone sesuai argumen:
      - 'irse101'  -> backbone.model_irse.IR_101(input_size=(112,112))
      - 'ir100'    -> backbone.iresnet.iresnet100()
    """
    arch = arch.lower()
    if arch in ["irse101", "ir101"]:
        from backbone.model_irse import IR_101
        model = IR_101(input_size=(112, 112))   # tuple!
        return model, 112
    elif arch == "ir100":
        from backbone.iresnet import iresnet100
        model = iresnet100()                    # juga 112x112
        return model, 112
    else:
        raise ValueError(f"Arch tidak dikenal: {arch}. Pilih 'irse101' atau 'ir100'.")

def load_backbone(weight_path: str, arch: str, device: str = "cuda"):
    model, input_size = build_model(arch)
    # Coba pakai weights_only=True (PyTorch >=2.4); fallback jika tidak didukung
    try:
        ckpt = torch.load(weight_path, map_location="cpu", weights_only=True)  # type: ignore[arg-type]
    except TypeError:
        ckpt = torch.load(weight_path, map_location="cpu")

    # Jika ckpt adalah dict dengan 'state_dict', ambil isinya
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        ckpt = ckpt["state_dict"]

    # Bersihkan prefix umum di state_dict (module./model./backbone.)
    cleaned = {}
    for k, v in (ckpt.items() if isinstance(ckpt, dict) else []):
        nk = k
        for pref in ("module.", "model.", "backbone."):
            if nk.startswith(pref):
                nk = nk[len(pref):]
        cleaned[nk] = v

    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    if missing or unexpected:
        print("[Warn] load_state_dict non-strict:",
              f"missing={len(missing)} unexpected={len(unexpected)}")

    model.eval().to(device).float()
    return model, input_size

def preprocess_arcface(img_bgr, size=112):
    # ArcFace/CurricularFace style: RGB, (x-127.5)/128, size x size, CHW
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    if img.shape[:2] != (size, size):
        img = cv2.resize(img, (size, size), interpolation=cv2.INTER_LINEAR)
    img = img.astype(np.float32)
    img = (img - 127.5) / 128.0
    img = np.transpose(img, (2, 0, 1))  # CHW
    return img

@torch.no_grad()
def embed_batch(model, batch_imgs, device):
    x = torch.from_numpy(np.stack(batch_imgs)).to(device).float()
    y = model(x)
    # Beberapa backbone (IRSE) mengembalikan tuple (feat, conv_out)
    if isinstance(y, tuple):
        y = y[0]
    feat = y.detach().cpu().numpy()
    # L2-normalize per baris
    feat = feat / np.linalg.norm(feat, axis=1, keepdims=True)
    return feat

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="crops_112", help="root gambar aligned")
    ap.add_argument("--weights", default="weights/CurricularFace_Backbone.pth")
    ap.add_argument("--out", default="embeds_curricularface.npz")
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--arch", default="irse101", choices=["irse101","ir101","ir100"],
                    help="backbone yang cocok dengan checkpoint")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    # 1) Muat model
    model, in_size = load_backbone(args.weights, args.arch, args.device)

    # 2) Kumpulkan path gambar
    root = Path(args.root)
    exts = (".jpg", ".jpeg", ".png")
    paths = sorted([str(p) for p in root.rglob("*") if p.suffix.lower() in exts])
    if not paths:
        print(f"[!] Tidak ada gambar di {args.root}. Pastikan langkah align sudah benar.")
        return

    rels = [str(Path(p).relative_to(root)) for p in paths]
    feats = {}
    B = max(1, int(args.batch))
    buf_imgs, buf_idx = [], []

    # 3) Proses batch
    for i, p in enumerate(tqdm(paths, desc=f"Embedding[{args.arch}]")):
        img = cv2.imread(p)
        if img is None:
            continue
        buf_imgs.append(preprocess_arcface(img, size=in_size))
        buf_idx.append(i)
        if len(buf_imgs) == B:
            F = embed_batch(model, buf_imgs, args.device)
            for j, ii in enumerate(buf_idx):
                feats[rels[ii]] = F[j]
            buf_imgs, buf_idx = [], []

    if buf_imgs:
        F = embed_batch(model, buf_imgs, args.device)
        for j, ii in enumerate(buf_idx):
            feats[rels[ii]] = F[j]

    # 4) Simpan
    np.savez_compressed(args.out, **feats)
    print(f"[OK] Saved {len(feats)} embeddings -> {args.out}")

if __name__ == "__main__":
    torch.backends.cudnn.benchmark = False
    main()
