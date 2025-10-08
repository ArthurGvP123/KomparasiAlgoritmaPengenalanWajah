import argparse
import sys
import time
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
    Mendukung 2 format:
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
# AdaFace loader
# =========================
def add_repo_to_sys_path(repo_dir: Path):
    repo = repo_dir.resolve()
    if not repo.exists():
        raise FileNotFoundError(f"[!] AdaFace repo tidak ditemukan: {repo}")
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))

def import_net_and_models(adaface_repo: Path):
    add_repo_to_sys_path(adaface_repo)
    net = __import__("net")
    IR_50  = getattr(net, "IR_50")
    IR_101 = getattr(net, "IR_101")
    return net, IR_50, IR_101

def normalize_arch_name(arch_raw: str) -> str:
    a = arch_raw.strip().lower().replace("_", "")
    if a in {"ir50", "irse50"}:
        return "ir50"
    if a in {"ir101", "ir100"}:  # izinkan alias ir100 -> ir101 (AdaFace tidak punya ir100)
        return "ir101"
    raise ValueError(f"Arch tidak dikenal: {arch_raw}. Gunakan ir50/ir101.")

class SafeFlatten(nn.Module):
    def forward(self, x):
        return x.reshape(x.size(0), -1)

def replace_flatten_with_safe(model: nn.Module) -> int:
    replaced = 0
    for name, child in list(model.named_children()):
        if child.__class__.__name__ == "Flatten":
            setattr(model, name, SafeFlatten()); replaced += 1
        else:
            replaced += replace_flatten_with_safe(child)
    return replaced

def build_adaface_model(adaface_repo: Path, arch: str = "ir101"):
    net, IR_50, IR_101 = import_net_and_models(adaface_repo)
    arch_norm = normalize_arch_name(arch)
    if arch_norm == "ir101":
        model = IR_101(input_size=(112, 112))
    else:
        model = IR_50(input_size=(112, 112))
    nrep = replace_flatten_with_safe(model)
    if nrep > 0:
        print(f"[LOG] Replaced {nrep} Flatten -> SafeFlatten")
    return model, arch_norm

def load_weights_into_model(model: nn.Module, weights_path: Path):
    try:
        ckpt = torch.load(str(weights_path), map_location="cpu", weights_only=True)  # PyTorch >=2.4
    except TypeError:
        ckpt = torch.load(str(weights_path), map_location="cpu")

    # Ambil state_dict dari berbagai format
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        sd = ckpt["state_dict"]
    elif isinstance(ckpt, dict):
        sd = ckpt
    else:
        sd = {}

    PREFIXES = (
        "features.module.", "module.features.", "features.",
        "module.", "model.", "backbone.", "net.", "encoder."
    )
    cleaned = {}
    for k, v in (sd.items() if isinstance(sd, dict) else []):
        nk = k
        for pref in PREFIXES:
            if nk.startswith(pref):
                nk = nk[len(pref):]
        cleaned[nk] = v

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

def preprocess_bgr_adaface(img_bgr, size=112):
    # BGR -> (x/255-0.5)/0.5, CHW float32; konsisten dg 02_embed_adaface.py
    if img_bgr.shape[:2] != (size, size):
        img_bgr = cv2.resize(img_bgr, (size, size), interpolation=cv2.INTER_LINEAR)
    img = img_bgr.astype(np.float32) / 255.0
    img = (img - 0.5) / 0.5
    img = np.transpose(img, (2, 0, 1))  # CHW
    return img

@torch.no_grad()
def embed_batch_adaface(model, batch_imgs_chw, device: str) -> np.ndarray:
    x = torch.from_numpy(np.stack(batch_imgs_chw)).to(device).float()
    y = model(x)
    if isinstance(y, (tuple, list)):
        y = y[0]
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
    # Haar cascade
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
    ap = argparse.ArgumentParser(description="Realtime Identify from NPZ (AdaFace IR50/IR101)")
    # Gallery NPZ
    ap.add_argument("--npz", required=True, help="Path file NPZ gallery (mis. ./embeds/embeds_adaface_ir101.npz)")
    # AdaFace backbone
    ap.add_argument("--repo-name", required=True, help="Folder repo AdaFace (berisi net.py)")
    ap.add_argument("--weights", required=True, help="Path weights AdaFace (mis. ./algoritma/weights/adaface_ir101_ms1mv2.ckpt)")
    ap.add_argument("--arch", default="ir101", help="ir50 atau ir101 (alias 'ir100' akan dipetakan ke ir101)")
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
    repo_dir = Path(args.repo_name).resolve()
    w_path   = Path(args.weights).resolve()
    if not npz_path.exists():
        raise FileNotFoundError(f"NPZ tidak ditemukan: {npz_path}")
    if not repo_dir.exists():
        raise FileNotFoundError(f"Repo AdaFace tidak ditemukan: {repo_dir}")
    if not w_path.exists():
        raise FileNotFoundError(f"Weights AdaFace tidak ditemukan: {w_path}")

    # 1) Gallery templates
    print("[LOG] Memuat gallery templates:", npz_path)
    G, labels = load_gallery_templates_from_npz(npz_path)
    print(f"[LOG] Label: {len(labels)} | dim: {G.shape[1]}")

    # 2) Backbone AdaFace
    print("[LOG] Memuat backbone AdaFace...")
    model, arch_norm = build_adaface_model(repo_dir, args.arch)
    load_weights_into_model(model, w_path)
    model.eval().to(device).float()
    print(f"[OK] Model: AdaFace-{arch_norm} on {device}")

    # Sanity check dim
    try:
        dummy = np.zeros((1, 3, 112, 112), dtype=np.float32)
        with torch.no_grad():
            y = model(torch.from_numpy(dummy).to(device))
            if isinstance(y, (tuple, list)):
                y = y[0]
            D_model = int(y.shape[-1])
        if G.shape[1] != D_model:
            raise RuntimeError(f"Dimensi gallery ({G.shape[1]}) ≠ dimensi model ({D_model}). Pastikan NPZ dari AdaFace yang sama.")
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
            print("[WARN] Frame kosong.")
            break

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
            crops.append(preprocess_bgr_adaface(crop, 112))
            boxes.append((x1, y1, x2, y2))

        if crops:
            F = embed_batch_adaface(model, crops, device)  # [N, D]
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
                cv2.rectangle(vis, (x1, y1 - 22), (x1 + tw, y1), (0, 0, 0), -1)
                cv2.putText(vis, txt, (x1 + 4, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

        # FPS overlay
        now = time.time()
        dt = now - t_prev
        if dt > 0:
            last_fps = 0.9 * last_fps + 0.1 * (1.0 / dt)
        t_prev = now
        cv2.putText(vis, f"FPS: {last_fps:.1f} | thr={args.thr}", (8, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow("Realtime Identify (AdaFace + NPZ)", vis)
        if (cv2.waitKey(1) & 0xFF) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    torch.backends.cudnn.benchmark = False
    main()
