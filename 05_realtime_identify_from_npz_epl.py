import argparse
import sys
import time
import types
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from torch import nn

# =========================
# Util umum
# =========================
def l2norm(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.maximum(n, eps)

def cosine_sim_matrix(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    # A dan b diasumsikan L2-normalized: dot = cosine
    return A @ b

def parse_npz_entry(val) -> Tuple[np.ndarray, Optional[str]]:
    """
    Mendukung:
      - Lama: value = vektor float (D,)
      - Baru: structured dtype dengan fields {'feat','label'}
    Return: (feat: float[D], label: Optional[str])
    """
    arr = np.asarray(val)
    if hasattr(arr, "dtype") and arr.dtype.fields:
        feat = np.asarray(arr["feat"]).reshape(-1).astype(np.float32, copy=False)
        lab = None
        if "label" in arr.dtype.fields:
            lab_arr = np.asarray(arr["label"]).reshape(-1)
            if lab_arr.size > 0:
                lab = str(lab_arr[0])
        return feat, lab
    return arr.astype(np.float32, copy=False).reshape(-1), None

def label_from_key_path(key: str) -> Optional[str]:
    parts = key.replace("\\", "/").split("/")
    if len(parts) >= 2:
        return parts[-2]
    return None

def load_gallery_templates_from_npz(npz_path: Path) -> Tuple[np.ndarray, List[str]]:
    """
    Baca .npz -> kumpulkan fitur per label -> template = mean(L2norm(feat)).
    Return: (G [N_labels, D], labels: list[str])
    """
    data = np.load(str(npz_path))
    buckets: Dict[str, List[np.ndarray]] = {}
    for k in data.keys():
        feat, lab = parse_npz_entry(data[k])
        lab = lab or label_from_key_path(k) or "UNKNOWN"
        feat = l2norm(feat.reshape(1, -1)).reshape(-1)
        buckets.setdefault(lab, []).append(feat)

    labels = sorted(buckets.keys())
    G = []
    for lab in labels:
        M = np.stack(buckets[lab], axis=0)   # [n_i, D]
        m = l2norm(M.mean(axis=0, keepdims=True)).reshape(-1)
        G.append(m)
    G = np.stack(G, axis=0).astype(np.float32)
    return G, labels

# =========================
# Loader EPL (ckpt objek model)
# =========================
def _get_attr_chain(obj, chain: str):
    cur = obj
    for part in chain.split("."):
        if not hasattr(cur, part):
            return None
        cur = getattr(cur, part)
    return cur

def _find_backbone_module(obj):
    if isinstance(obj, nn.Module):
        # Jika objek ini langsung usable (punya forward), pakai
        if hasattr(obj, "forward") and obj.__class__.__name__ not in ("_Dummy","Module"):
            return obj
        # Coba beberapa jalur umum
        for ch in (
            "backbone",
            "module.backbone",
            "model.backbone",
            "module.model.backbone",
            "net.backbone",
            "module.net.backbone",
        ):
            sub = _get_attr_chain(obj, ch)
            if isinstance(sub, nn.Module):
                return sub
    return None

def _install_stub_head_vpcl():
    """
    Pasang modul 'head_VPCL' dummy agar unpickle ckpt yang mendefinisikan head custom tidak gagal.
    Stub ini tidak dipakai saat inference backbone.
    """
    if "head_VPCL" in sys.modules:
        return

    mod = types.ModuleType("head_VPCL")

    class _HeadStub(nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()
        def forward(self, *args, **kwargs):
            raise RuntimeError("Stub head_VPCL dipanggil; untuk embedding, head tidak diperlukan.")

    def _make_stub(name: str):
        cls = type(name, (_HeadStub,), {})
        setattr(mod, name, cls)
        return cls

    # beberapa nama umum
    for name in [
        "UnifiedContrastive", "Cosface_uni_VPCL_k", "Cosface_uni_VPCL",
        "Cosface_VPCL", "Arcface_VPCL", "Arcface_uni_VPCL_k", "MV_Softmax_VPCL",
    ]:
        _make_stub(name)

    def __getattr__(name):
        return _make_stub(name)
    mod.__getattr__ = __getattr__
    sys.modules["head_VPCL"] = mod

def _safe_import_epl_modules(epl_repo: Optional[Path]):
    if epl_repo:
        repo = str(epl_repo.resolve())
        if repo not in sys.path:
            sys.path.insert(0, repo)
    # Pastikan head_VPCL ada
    try:
        __import__("head_VPCL")
    except Exception:
        _install_stub_head_vpcl()
    # Import lain jika perlu
    try:
        __import__("backbone")
    except Exception:
        pass

def load_epl_backbone(weights_path: Path, repo_dir: Optional[Path], device: str = "cpu") -> nn.Module:
    _safe_import_epl_modules(repo_dir)
    # Gunakan unpickle penuh (pastikan ckpt tepercaya)
    ckpt = torch.load(str(weights_path), map_location="cpu", weights_only=False)

    bb = None
    if isinstance(ckpt, nn.Module):
        bb = _find_backbone_module(ckpt)
        if bb is None and hasattr(ckpt, "module") and isinstance(ckpt.module, nn.Module):
            bb = _find_backbone_module(ckpt.module)

    if bb is None:
        if isinstance(ckpt, dict) and "state_dict" in ckpt:
            raise RuntimeError(
                "Checkpoint EPL ini hanya berisi state_dict. "
                "Gunakan ckpt yang menyimpan objek model utuh, atau jalankan dengan repo EPL lengkap "
                "dan rakit backbone dari definisi kelas aslinya."
            )
        raise RuntimeError(
            "Tidak menemukan submodule backbone di checkpoint. "
            "Pastikan --repo-name menunjuk ke repo EPL yang benar atau gunakan ckpt model utuh."
        )

    bb.eval().to(device)
    for p in bb.parameters():
        p.requires_grad_(False)
    return bb

# =========================
# Preprocess & embed
# =========================
def preprocess_arcface(img_bgr, size=112):
    # ArcFace-style: RGB, (x-127.5)/128, CHW
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    if img.shape[:2] != (size, size):
        img = cv2.resize(img, (size, size), interpolation=cv2.INTER_LINEAR)
    img = img.astype(np.float32)
    img = (img - 127.5) / 128.0
    img = np.transpose(img, (2, 0, 1))
    return img

@torch.no_grad()
def embed_batch_epl(model: nn.Module, batch_imgs_chw: List[np.ndarray], device: str) -> np.ndarray:
    x = torch.from_numpy(np.stack(batch_imgs_chw)).to(device).float()
    y = model(x)
    if isinstance(y, (tuple, list)):
        y = y[0]
    if isinstance(y, dict):
        for k in ("embeddings","feat","features","x","out"):
            if k in y:
                y = y[k]; break
    if not torch.is_tensor(y):
        raise RuntimeError(f"Output model bukan tensor: {type(y)}")
    feat = y.detach().cpu().numpy().astype(np.float32)
    feat = l2norm(feat)
    return feat

# =========================
# Face detector (dlib/haar)
# =========================
def detect_faces_bgr(img_bgr, detector="auto", upsample=0):
    """
    Return list of rect (x1,y1,x2,y2)
    """
    det = detector.lower()
    rects = []
    if det in ("auto", "dlib"):
        try:
            import dlib  # noqa
            d = dlib.get_frontal_face_detector()
            rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            rs = d(rgb, upsample)
            for r in rs:
                rects.append((r.left(), r.top(), r.right(), r.bottom()))
            if rects or det == "dlib":
                return rects
        except Exception:
            if det == "dlib":
                return rects  # kosong jika gagal
    # Haar cascade fallback
    try:
        haar = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        faces = haar.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
        for (x, y, w, h) in faces:
            rects.append((int(x), int(y), int(x+w), int(y+h)))
    except Exception:
        pass
    return rects

# =========================
# Main
# =========================
def main():
    ap = argparse.ArgumentParser(description="Realtime Identify from NPZ (EPL backbone)")
    # Gallery NPZ
    ap.add_argument("--npz", required=True, help="Path file NPZ gallery (mis. ./embeds/embeds_epl_ir100.npz)")
    # EPL backbone
    ap.add_argument("--repo-name", default="", help="Folder repo EPL (berisi backbone.py, dll). Disarankan diisi.")
    ap.add_argument("--weights", required=True, help="Path checkpoint EPL (biasanya model utuh .pth)")
    # Runtime
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    ap.add_argument("--camera", type=int, default=0, help="Index kamera (default: 0)")
    ap.add_argument("--thr", type=float, default=0.5, help="Threshold cosine")
    ap.add_argument("--width", type=int, default=960, help="Lebar tampilan video")
    ap.add_argument("--detector", choices=["auto", "dlib", "haar"], default="auto")
    ap.add_argument("--upsample", type=int, default=0, help="Detektor dlib upsample (0/1)")

    args = ap.parse_args()
    device = "cuda" if (args.device == "cuda" and torch.cuda.is_available()) else "cpu"
    if args.device == "cuda" and device != "cuda":
        print("[WARN] CUDA diminta tetapi tidak tersedia. Pakai CPU.")

    npz_path = Path(args.npz).resolve()
    repo_dir = Path(args.repo_name).resolve() if args.repo_name else None
    w_path   = Path(args.weights).resolve()
    if not npz_path.exists():
        raise FileNotFoundError(f"NPZ tidak ditemukan: {npz_path}")
    if not w_path.exists():
        raise FileNotFoundError(f"Weights EPL tidak ditemukan: {w_path}")
    if repo_dir is not None and not repo_dir.exists():
        raise FileNotFoundError(f"Repo EPL tidak ditemukan: {repo_dir}")

    # 1) Gallery templates
    print("[LOG] Memuat gallery templates:", npz_path)
    G, labels = load_gallery_templates_from_npz(npz_path)
    print(f"[LOG] Label: {len(labels)} | dim(gallery): {G.shape[1]}")

    # 2) Backbone EPL
    print("[LOG] Memuat backbone EPL...")
    model = load_epl_backbone(w_path, repo_dir, device=device)
    print(f"[OK] Model EPL siap di {device}")

    # Sanity check dimensi
    try:
        dummy = np.zeros((1, 3, 112, 112), dtype=np.float32)
        with torch.no_grad():
            y = model(torch.from_numpy(dummy).to(device))
            if isinstance(y, (tuple, list)):
                y = y[0]
            if isinstance(y, dict):
                for k in ("embeddings","feat","features","x","out"):
                    if k in y:
                        y = y[k]; break
            D_model = int(y.shape[-1])
        if G.shape[1] != D_model:
            raise RuntimeError(f"Dimensi gallery ({G.shape[1]}) ≠ dimensi model EPL ({D_model}). "
                               "Pastikan NPZ dan backbone menggunakan dimensi embedding yang sama.")
    except Exception as e:
        print(f"[WARN] Sanity-check dim dilewati: {e}")

    # 3) Kamera
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError("Tidak bisa membuka kamera. Coba ganti --camera 1 atau pastikan tidak dipakai aplikasi lain.")

    print("[INFO] Tekan 'q' untuk keluar.")
    last_fps, t_prev = 0.0, time.time()

    while True:
        ok, frame = cap.read()
        if not ok:
            print("[WARN] Frame kosong."); break

        # Resize display
        h, w = frame.shape[:2]
        if args.width > 0 and w != args.width:
            scale = args.width / w
            frame = cv2.resize(frame, (args.width, int(h * scale)))
        vis = frame.copy()

        # Deteksi wajah
        rects = detect_faces_bgr(frame, detector=args.detector, upsample=args.upsample)

        # Batch embed
        crops, boxes = [], []
        for (x1, y1, x2, y2) in rects:
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(frame.shape[1]-1, x2), min(frame.shape[0]-1, y2)
            if x2 <= x1 or y2 <= y1: continue
            crop = frame[y1:y2, x1:x2, :]
            crops.append(preprocess_arcface(crop, 112))
            boxes.append((x1, y1, x2, y2))

        if crops:
            F = embed_batch_epl(model, crops, device)  # [N, D]
            for i, (x1, y1, x2, y2) in enumerate(boxes):
                v = F[i]
                sims = cosine_sim_matrix(G, v)
                j = int(np.argmax(sims))
                score = float(sims[j])
                name = labels[j] if score >= args.thr else "Unknown"

                color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
                txt = f"{name} ({score:.2f})"
                tw = 8 * len(txt)
                cv2.rectangle(vis, (x1, y1 - 22), (x1 + tw + 6, y1), (0, 0, 0), -1)
                cv2.putText(vis, txt, (x1 + 4, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

        # FPS overlay
        now = time.time()
        dt = now - t_prev
        if dt > 0:
            last_fps = 0.9 * last_fps + 0.1 * (1.0 / dt)
        t_prev = now
        cv2.putText(vis, f"FPS: {last_fps:.1f} | thr={args.thr}", (8, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow("Realtime Identify (EPL + NPZ)", vis)
        if (cv2.waitKey(1) & 0xFF) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    torch.backends.cudnn.benchmark = False
    main()
