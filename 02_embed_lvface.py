# 02_embed_lvface.py
import argparse
from pathlib import Path
import os
import numpy as np
from PIL import Image
from tqdm import tqdm

# onnx / torch optional imports
try:
    import onnxruntime as ort
except Exception:
    ort = None
try:
    import torch
    import torch.nn as nn
except Exception:
    torch = None

# (opsional) unduh dari HF kalau kamu pilih --hf-id
from huggingface_hub import hf_hub_download

def list_images(root: Path):
    exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    return [p for p in root.rglob("*") if p.suffix.lower() in exts]

def preprocess(img_path: Path, img_size: int = 112):
    im = Image.open(str(img_path)).convert("RGB")
    if im.size != (img_size, img_size):
        im = im.resize((img_size, img_size), Image.BILINEAR)
    x = np.asarray(im).astype(np.float32) / 255.0  # HWC
    x = (x - 0.5) / 0.5                           # [-1,1]
    x = np.transpose(x, (2, 0, 1))                # CHW
    return x

def l2norm(x: np.ndarray, eps=1e-12):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(n, eps)

# ---------------- ONNX path ----------------
class LVFaceONNX:
    def __init__(self, onnx_path: str):
        assert ort is not None, "onnxruntime belum terpasang. `pip install onnxruntime`"
        so = ort.SessionOptions()
        so.log_severity_level = 3
        self.sess = ort.InferenceSession(onnx_path, sess_options=so, providers=["CPUExecutionProvider"])
        # deteksi nama input & output
        self.inp = self.sess.get_inputs()[0].name
        self.out = self.sess.get_outputs()[0].name
        # Banyak model face expect NCHW float32
    def __call__(self, batch: np.ndarray) -> np.ndarray:
        # batch: (N,3,H,W) float32
        out = self.sess.run([self.out], {self.inp: batch})[0]
        if out.ndim == 4:  # kadang (N,1,1,512)
            out = out.reshape(out.shape[0], -1)
        return out.astype(np.float32)

# ---------------- Torch path ----------------
class LVFaceTorch:
    def __init__(self, pt_path: str, device: str = "cpu"):
        assert torch is not None, "PyTorch belum terpasang. `pip install torch torchvision`"
        self.device = torch.device(device)
        # 1) coba TorchScript
        try:
            model = torch.jit.load(pt_path, map_location="cpu")
            model.eval()
            self.model = model.to(self.device)
            self.torchscript = True
            return
        except Exception:
            self.torchscript = False

        # 2) coba nn.Module pickled lengkap
        try:
            obj = torch.load(pt_path, map_location="cpu", weights_only=False)
            if isinstance(obj, nn.Module):
                self.model = obj.eval().to(self.device)
                return
        except Exception:
            pass

        raise RuntimeError(
            "Gagal memuat .pt sebagai TorchScript/nn.Module lengkap. "
            "Kemungkinan file ini hanya berisi state_dict.\n"
            "Saran: gunakan file ONNX (LVFace-L_Glint360K.onnx) untuk inference CPU yang stabil."
        )

    def __call__(self, batch: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            x = torch.from_numpy(batch).to(self.device)  # (N,3,H,W)
            y = self.model(x)
            if isinstance(y, (tuple, list)):
                y = y[0]
            y = y.detach().cpu().float().numpy()
            if y.ndim == 4:
                y = y.reshape(y.shape[0], -1)
            return y.astype(np.float32)

# ---------------- Loader util ----------------
def get_model(weights: str, fmt: str, device: str, hf_id: str, model_file: str):
    """
    fmt: 'auto' | 'onnx' | 'pt'
    weights: path lokal ke .onnx atau .pt
    hf_id + model_file: kalau weights kosong, unduh dari HF
    """
    if not weights and not hf_id:
        raise ValueError("Berikan --weights (lokal) atau --hf-id + --model-file (nama file di repo HF).")

    model_path = weights
    if not model_path:
        # unduh dari HF
        model_path = hf_hub_download(repo_id=hf_id, filename=model_file, local_dir="./weights", local_dir_use_symlinks=False)

    ext = Path(model_path).suffix.lower()
    use_onnx = (fmt == "onnx") or (fmt == "auto" and ext == ".onnx")
    use_pt   = (fmt == "pt")   or (fmt == "auto" and ext == ".pt")

    if use_onnx:
        return LVFaceONNX(model_path), "onnx", model_path
    if use_pt:
        return LVFaceTorch(model_path, device=device), "pt", model_path

    raise ValueError(f"Tidak bisa menentukan format model dari: {model_path}. Gunakan --format onnx|pt.")

# ---------------- Main ----------------
def main():
    ap = argparse.ArgumentParser(description="Embed LVFace (ONNX/PyTorch) – konsisten dengan pipeline sebelumnya")
    ap.add_argument("--root", default="./crops_112", help="Folder gambar yang sudah aligned (default: ./crops_112)")
    ap.add_argument("--weights", default="", help="Path lokal model (.onnx atau .pt). Boleh kosong jika pakai --hf-id")
    ap.add_argument("--hf-id", default="", help="Repo HuggingFace, contoh: bytedance-research/LVFace")
    ap.add_argument("--model-file", default="LVFace-L_Glint360K.onnx", help="Nama file di repo HF (default: LVFace-L_Glint360K.onnx)")
    ap.add_argument("--format", choices=["auto","onnx","pt"], default="auto", help="Paksa format model (default: auto)")
    ap.add_argument("--out", default="./embeds_lvface_l.npz", help="File output embeddings .npz")
    ap.add_argument("--batch", type=int, default=128, help="Batch size (default: 128)")
    ap.add_argument("--device", choices=["cpu","cuda"], default="cpu", help="(Hanya untuk .pt) device target (default: cpu)")
    ap.add_argument("--img-size", type=int, default=112, help="Ukuran input (default: 112)")
    ap.add_argument("--limit", type=int, default=0, help="Batasi jumlah gambar (0=semua)")
    args = ap.parse_args()

    root = Path(args.root)
    if not root.exists():
        raise FileNotFoundError(f"Folder tidak ditemukan: {root}")

    print("== LVFace Embedding ==")
    print("[LOG] args:", args)

    model, mode, model_path = get_model(args.weights, args.format, args.device, args.hf_id, args.model_file)
    print(f"[LOG] backend: {mode} | model: {model_path}")

    imgs = list_images(root)
    print(f"[LOG] ditemukan {len(imgs)} gambar di {root}")
    if args.limit and args.limit > 0:
        imgs = imgs[:args.limit]
        print(f"[LOG] MODE UJI: membatasi ke {len(imgs)} gambar pertama.")

    # siapkan buffer embedding
    keys, feats = [], []

    # batching
    buf = []
    buf_keys = []
    for p in tqdm(imgs, desc=f"Embedding[LVFace-{mode}]", unit_scale=True):
        x = preprocess(p, img_size=args.img_size)
        buf.append(x); buf_keys.append(p)
        if len(buf) >= args.batch:
            X = np.stack(buf, axis=0)  # (N,3,H,W)
            F = model(X)               # (N, D)
            F = l2norm(F)
            feats.append(F); keys.extend(buf_keys)
            buf, buf_keys = [], []

    if buf:
        X = np.stack(buf, axis=0)
        F = model(X)
        F = l2norm(F)
        feats.append(F); keys.extend(buf_keys)

    if not feats:
        raise RuntimeError("Tidak ada embedding yang dihasilkan.")

    FEAT = np.concatenate(feats, axis=0)
    # kunci: path relatif terhadap root, forward-slash
    key_strs = []
    for k in keys:
        rel = k.relative_to(root).as_posix()
        # seragamkan awalan (ikuti pola kamu sebelumnya)
        key_strs.append(rel)

    # simpan ke npz (key->vec)
    # np.savez tidak suka dict langsung; simpan satu per satu
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    npz_dict = {k: FEAT[i] for i, k in enumerate(key_strs)}
    np.savez(out_path, **npz_dict)
    print(f"[OK] embeddings tersimpan: {out_path} | total: {FEAT.shape[0]} | dim: {FEAT.shape[1]}")

if __name__ == "__main__":
    main()
