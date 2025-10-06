# 02_embed_lvface.py
import argparse
from pathlib import Path
import os
import sys
import numpy as np
from PIL import Image
from tqdm import tqdm

# onnx / torch (opsional)
try:
    import onnxruntime as ort
except Exception:
    ort = None

try:
    import torch
    import torch.nn as nn
except Exception:
    torch = None

# (opsional) unduh dari HF kalau --weights kosong
from huggingface_hub import hf_hub_download


def list_images(root: Path):
    exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    return [p for p in root.rglob("*") if p.suffix.lower() in exts]


def preprocess(img_path: Path, img_size: int = 112):
    im = Image.open(str(img_path)).convert("RGB")
    if im.size != (img_size, img_size):
        im = im.resize((img_size, img_size), Image.BILINEAR)
    x = np.asarray(im).astype(np.float32) / 255.0   # HWC
    x = (x - 0.5) / 0.5                             # [-1, 1]
    x = np.transpose(x, (2, 0, 1))                  # CHW
    return x


def l2norm(x: np.ndarray, eps=1e-12):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(n, eps)


# ---------------- ONNX path ----------------
class LVFaceONNX:
    def __init__(self, onnx_path: str, device: str = "cpu"):
        assert ort is not None, "onnxruntime belum terpasang. `pip install onnxruntime`"
        so = ort.SessionOptions()
        so.log_severity_level = 3

        # pilih provider sesuai device
        providers = ["CPUExecutionProvider"]
        if device == "cuda":
            try:
                av = ort.get_available_providers()
                if "CUDAExecutionProvider" in av:
                    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
                else:
                    print("[WARN] CUDAExecutionProvider tidak tersedia di onnxruntime; jatuh ke CPU.")
            except Exception:
                print("[WARN] Gagal memeriksa provider onnxruntime; jatuh ke CPU.")

        self.sess = ort.InferenceSession(onnx_path, sess_options=so, providers=providers)
        self.inp = self.sess.get_inputs()[0].name
        self.out = self.sess.get_outputs()[0].name

    def __call__(self, batch: np.ndarray) -> np.ndarray:
        # expect (N,3,H,W) float32
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
            "Kemungkinan file hanya berisi state_dict.\n"
            "Saran: gunakan file ONNX (mis. LVFace-L_Glint360K.onnx) untuk inference yang stabil."
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
    hf_id + model_file: jika weights kosong, unduh dari HF
    """
    if not weights and not hf_id:
        raise ValueError("Berikan --weights (lokal) atau --hf-id + --model-file (nama file di repo HF).")

    model_path = weights
    if not model_path:
        model_path = hf_hub_download(
            repo_id=hf_id,
            filename=model_file,
            local_dir="./algoritma/weights",
            local_dir_use_symlinks=False
        )

    ext = Path(model_path).suffix.lower()
    use_onnx = (fmt == "onnx") or (fmt == "auto" and ext == ".onnx")
    use_pt   = (fmt == "pt")   or (fmt == "auto" and ext == ".pt")

    if use_onnx:
        return LVFaceONNX(model_path, device=device), "onnx", model_path
    if use_pt:
        return LVFaceTorch(model_path, device=device), "pt", model_path

    raise ValueError(f"Tidak bisa menentukan format model dari: {model_path}. Pakai --format onnx|pt atau berikan ekstensi yang benar.")


# ---------------- Main ----------------
def main():
    ap = argparse.ArgumentParser(description="Embed LVFace (ONNX/PyTorch) – seragam dengan pipeline lain")

    # Argumen diseragamkan:
    ap.add_argument("--repo-name", default="", help="Folder repo LVFace (opsional, untuk konsistensi antarmuka)")
    ap.add_argument("--dataset-name", required=True, help="Folder dataset aligned (mis. .\\dataset\\Dosen_112)")
    ap.add_argument("--weights", default="", help="Path lokal model (.onnx/.pt). Boleh kosong jika pakai --hf-id")
    ap.add_argument("--out", required=True, help="File output embeddings .npz (mis. .\\embeds\\embeds_lvface_l.npz)")

    # Opsi khusus LVFace
    ap.add_argument("--hf-id", default="", help="Repo HuggingFace, contoh: bytedance-research/LVFace")
    ap.add_argument("--model-file", default="LVFace-L_Glint360K.onnx", help="Nama file di repo HF (default: LVFace-L_Glint360K.onnx)")
    ap.add_argument("--format", choices=["auto","onnx","pt"], default="auto", help="Pilih backend model (default: auto)")

    # Lainnya
    ap.add_argument("--batch", type=int, default=128, help="Batch size (default: 128)")
    ap.add_argument("--device", choices=["cpu","cuda"], default="cpu", help="Device untuk inference (ONNX/Torch)")
    ap.add_argument("--img-size", type=int, default=112, help="Ukuran input (default: 112)")
    ap.add_argument("--limit", type=int, default=0, help="Batasi jumlah gambar (0=semua)")

    args = ap.parse_args()

    dataset_root = Path(args.dataset_name).resolve()
    if not dataset_root.exists():
        raise FileNotFoundError(f"Folder dataset tidak ditemukan: {dataset_root}")

    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print("== LVFace Embedding ==")
    print("[LOG] repo-name   :", args.repo_name or "(tidak dipakai)")
    print("[LOG] dataset_root:", dataset_root)
    print("[LOG] out_path    :", out_path)
    print("[LOG] device      :", args.device)

    # Muat model
    model, mode, model_path = get_model(args.weights, args.format, args.device, args.hf_id, args.model_file)
    print(f"[LOG] backend: {mode} | model: {model_path}")

    # Kumpulkan gambar
    imgs = list_images(dataset_root)
    print(f"[LOG] ditemukan {len(imgs)} gambar di {dataset_root}")
    if not imgs:
        print("[!] Tidak ada gambar ditemukan. Pastikan 01_merapikan_dataset sudah menghasilkan folder aligned.")
        return
    if args.limit and args.limit > 0:
        imgs = imgs[:args.limit]
        print(f"[LOG] MODE UJI: membatasi ke {len(imgs)} gambar pertama.")

    # Embedding (batch)
    keys, feats = [], []
    buf, buf_keys = [], []
    for p in tqdm(imgs, desc=f"Embedding[LVFace-{mode}]"):
        x = preprocess(p, img_size=args.img_size)
        buf.append(x); buf_keys.append(p)
        if len(buf) >= args.batch:
            X = np.stack(buf, axis=0)           # (N,3,H,W)
            F = model(X)                        # (N,D)
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
    key_strs = [k.relative_to(dataset_root).as_posix() for k in keys]

    # simpan ke .npz (key->vec)
    np.savez(out_path, **{k: FEAT[i] for i, k in enumerate(key_strs)})
    print(f"[OK] embeddings tersimpan: {out_path} | total: {FEAT.shape[0]} | dim: {FEAT.shape[1]}")


if __name__ == "__main__":
    main()
