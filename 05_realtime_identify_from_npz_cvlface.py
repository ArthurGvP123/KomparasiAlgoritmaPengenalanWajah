import argparse
import os
import sys
import time
import shutil
import warnings
from pathlib import Path
from typing import List, Tuple, Optional, Dict

import numpy as np
import cv2
import torch
from huggingface_hub import snapshot_download
from transformers import AutoModel

warnings.filterwarnings("ignore", category=UserWarning)

# ================== Util umum ==================
def l2norm(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.maximum(n, eps)

def cosine_sim_matrix(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    # A dan b diasumsikan sudah L2-norm; dot product = cosine
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
    # Ambil nama folder induk: ".../<LABEL>/<file>"
    parts = key.replace("\\", "/").split("/")
    if len(parts) >= 2:
        return parts[-2]
    return None

def load_gallery_templates_from_npz(npz_path: Path) -> Tuple[np.ndarray, List[str]]:
    """
    Baca .npz -> kumpulkan fitur per label -> template = mean(L2norm(feat)).
    Return:
      G: [N_labels, D]
      labels: list[str]
    """
    npz = np.load(str(npz_path))
    buckets: Dict[str, List[np.ndarray]] = {}
    for k in npz.keys():
        feat, lab = parse_npz_entry(npz[k])
        lab = lab or label_from_key_path(k) or "UNKNOWN"
        feat = l2norm(feat.reshape(1, -1)).reshape(-1)
        buckets.setdefault(lab, []).append(feat)

    labels = sorted(buckets.keys())
    G = []
    for lab in labels:
        M = np.stack(buckets[lab], axis=0)     # [n_i, D]
        m = l2norm(M.mean(axis=0, keepdims=True)).reshape(-1)
        G.append(m)
    G = np.stack(G, axis=0).astype(np.float32)
    return G, labels

# ================== CVLface loader (HF) ==================
def prepare_pretrained_subdir(model_dir: Path) -> Path:
    pm = model_dir / "pretrained_model"
    pm.mkdir(exist_ok=True)
    for name in ["model.yaml", "config.yaml", "model.pt", "model.safetensors"]:
        src, dst = model_dir / name, pm / name
        if src.exists() and not dst.exists():
            try: shutil.copy2(src, dst)
            except Exception: pass
    return pm

def ensure_models_package(model_dir: Path) -> Path:
    mdir = model_dir / "models"
    mdir.mkdir(parents=True, exist_ok=True)
    for fn in [
        "vit.py","kprpe_shared.py","relative_keypoints.py","rpe_index.py",
        "rpe_options.py","utils.py","dist.py","__init__.py"
    ]:
        src, dst = model_dir / fn, mdir / fn
        if src.exists() and not dst.exists():
            try: shutil.copy2(src, dst)
            except Exception: pass
    init = mdir / "__init__.py"
    if not init.exists():
        init.write_text(
            "# auto-generated\n"
            "from .vit import *\nfrom .kprpe_shared import *\nfrom .relative_keypoints import *\n"
            "from .rpe_index import *\nfrom .rpe_options import *\nfrom .utils import *\nfrom .dist import *\n",
            encoding="utf-8",
        )
    return mdir

def download_snapshot(repo_id: str, local_dir: Path) -> Path:
    local_dir.mkdir(parents=True, exist_ok=True)
    path = snapshot_download(
        repo_id=repo_id,
        local_dir=str(local_dir),
        local_dir_use_symlinks=False,
        ignore_patterns=["*.bin"],
    )
    return Path(path)

def _try_import_rpe_now(model_dir: Path) -> bool:
    try:
        import rpe_index_cpp  # noqa: F401
        return True
    except Exception:
        # cari build .pyd di snapshot
        cand = list(model_dir.rglob("rpe_ops/build/**/rpe_index_cpp*.pyd"))
        for p in cand:
            pdir = str(Path(p).parent)
            if pdir not in sys.path:
                sys.path.insert(0, pdir)
                try:
                    import rpe_index_cpp  # noqa: F401
                    return True
                except Exception:
                    continue
    return False

def _install_rpe_ops(model_dir: Path) -> None:
    rpe_dir = model_dir / "models" / "vit_kprpe" / "RPE" / "rpe_ops"
    if not rpe_dir.exists():
        cand = list(model_dir.rglob("rpe_ops/setup.py"))
        if cand:
            rpe_dir = cand[0].parent
        else:
            raise RuntimeError("Tidak menemukan folder rpe_ops untuk build.")
    import subprocess
    print("[INFO] Membangun rpe_ops (sekali saja)...")
    subprocess.check_call([sys.executable, "setup.py", "install", "--user"], cwd=str(rpe_dir))
    print("[INFO] rpe_ops terpasang.")

def ensure_rpe_ops_available(model_dir: Path):
    if _try_import_rpe_now(model_dir):
        return
    _install_rpe_ops(model_dir)
    if not _try_import_rpe_now(model_dir):
        raise RuntimeError("`rpe_ops` sudah di-install tapi belum bisa di-import. Coba jalankan ulang skrip.")

def load_cvlface_and_aligner(
    hf_id: str,
    hf_local: str,
    device: str = "cpu",
):
    """
    Muat backbone CVLface (AutoModel trust_remote_code) + DFA-mobilenet aligner.
    """
    if hf_local:
        model_dir = Path(hf_local).resolve()
        if not model_dir.exists():
            raise FileNotFoundError(f"hf-local tidak ditemukan: {model_dir}")
    else:
        cache_root = Path.home() / ".cvlface_cache" / hf_id.replace("/", "__")
        model_dir = download_snapshot(hf_id, cache_root)

    prepare_pretrained_subdir(model_dir)
    ensure_models_package(model_dir)
    ensure_rpe_ops_available(model_dir)

    # Import model HF dengan trust_remote_code
    cwd0 = Path.cwd()
    added = False
    try:
        os.chdir(model_dir)
        if str(model_dir) not in sys.path:
            sys.path.insert(0, str(model_dir)); added = True
        model = AutoModel.from_pretrained(str(model_dir), trust_remote_code=True)
    finally:
        os.chdir(cwd0)
        if added:
            try: sys.path.remove(str(model_dir))
            except ValueError: pass

    # Aligner (DFA-mobilenet)
    repo_aligner = "minchul/cvlface_DFA_mobilenet"
    align_dir = download_snapshot(repo_aligner, Path.home() / ".cvlface_cache" / repo_aligner.replace("/", "__"))
    added = False
    try:
        os.chdir(align_dir)
        if str(align_dir) not in sys.path:
            sys.path.insert(0, str(align_dir)); added = True
        aligner = AutoModel.from_pretrained(str(align_dir), trust_remote_code=True)
    finally:
        os.chdir(cwd0)
        if added:
            try: sys.path.remove(str(align_dir))
            except ValueError: pass

    model.eval().to(device)
    aligner.eval().to(device)
    return model, aligner, model_dir

# ================== Preprocess & embed ==================
def preprocess_arcface(img_bgr, size=112):
    # RGB, (x-127.5)/128, CHW
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    if img.shape[:2] != (size, size):
        img = cv2.resize(img, (size, size), interpolation=cv2.INTER_LINEAR)
    img = img.astype(np.float32)
    img = (img - 127.5) / 128.0
    img = np.transpose(img, (2, 0, 1))
    return img

@torch.no_grad()
def embed_batch_cvlface(model, aligner, batch_imgs_chw, device: str):
    """
    batch_imgs_chw: List[np.ndarray CHW float32, nilai sekitar [-1,1]]
    """
    x = torch.from_numpy(np.stack(batch_imgs_chw)).to(device).float()  # [N,3,112,112]
    # 1) dapatkan keypoints dari aligner
    ax, orig_ldmks, *_ = aligner(x)
    keypoints = orig_ldmks  # [N,5,2]
    # 2) forward backbone
    y = model(x, keypoints=keypoints)
    if isinstance(y, (list, tuple)):
        y = y[0]
    elif isinstance(y, dict):
        for k in ["embeddings", "feat", "features", "last_hidden_state", "pooler_output", "output"]:
            if k in y:
                y = y[k]; break
        else:
            y = next(iter(y.values()))
    if not torch.is_tensor(y):
        raise RuntimeError(f"Unexpected model output: {type(y)}")
    feat = y.detach().cpu().numpy().astype(np.float32)
    feat = l2norm(feat)
    return feat

# ================== Face detector (dlib atau Haar) ==================
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
            # else: fallback ke haar
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

# ================== Main loop ==================
def main():
    ap = argparse.ArgumentParser(description="Realtime Identify from NPZ (CVLface ViT-KPRPE + DFA-mobilenet)")
    ap.add_argument("--npz", required=True, help="Path file NPZ gallery (mis. ./embeds/embeds_cvlface_vitb_kprpe.npz)")
    ap.add_argument("--hf-id", default="minchul/cvlface_adaface_vit_base_kprpe_webface4m",
                    help="HuggingFace repo id untuk backbone CVLface")
    ap.add_argument("--hf-local", default="", help="(Opsional) path snapshot lokal model CVLface")

    ap.add_argument("--device", choices=["cpu", "cuda"], default="cpu", help="Device untuk inference")
    ap.add_argument("--camera", type=int, default=0, help="Index kamera (default: 0)")
    ap.add_argument("--thr", type=float, default=0.5, help="Threshold cosine untuk identifikasi")
    ap.add_argument("--width", type=int, default=960, help="Lebar tampilan video")
    ap.add_argument("--detector", choices=["auto", "dlib", "haar"], default="auto", help="Detektor wajah")
    ap.add_argument("--upsample", type=int, default=0, help="Detektor dlib upsample (0/1)")

    args = ap.parse_args()
    device = "cuda" if (args.device == "cuda" and torch.cuda.is_available()) else "cpu"
    if args.device == "cuda" and device != "cuda":
        print("[WARN] CUDA diminta tetapi tidak tersedia. Pakai CPU.")

    npz_path = Path(args.npz).resolve()
    if not npz_path.exists():
        raise FileNotFoundError(f"NPZ tidak ditemukan: {npz_path}")

    print("[LOG] Memuat gallery templates:", npz_path)
    G, labels = load_gallery_templates_from_npz(npz_path)
    print(f"[LOG] Label: {len(labels)} | dim: {G.shape[1]}")

    # Muat backbone + aligner
    print("[LOG] Memuat CVLface backbone + DFA-mobilenet aligner...")
    model, aligner, _ = load_cvlface_and_aligner(args.hf_id, args.hf_local, device)
    print("[OK] Model siap.")

    # Sanity-check dim
    D_model = None
    try:
        dummy = np.zeros((1, 3, 112, 112), dtype=np.float32)
        with torch.no_grad():
            X = torch.from_numpy(dummy).to(device)
            ax, orig_ldmks, *_ = aligner(X)
            y = model(X, keypoints=orig_ldmks)
            if isinstance(y, (list, tuple)):
                y = y[0]
            elif isinstance(y, dict):
                y = next(iter(y.values()))
            if torch.is_tensor(y):
                D_model = int(y.shape[-1])
    except Exception:
        pass
    if D_model is not None and G.shape[1] != D_model:
        raise RuntimeError(f"Dimensi gallery ({G.shape[1]}) ≠ dimensi model ({D_model}). Pastikan NPZ dari CVLface yang sama.")

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError("Tidak bisa membuka kamera. Coba ganti --camera 1 atau pastikan kamera tidak dipakai aplikasi lain.")

    print("[INFO] Tekan 'q' untuk keluar.")
    last_fps = 0.0
    t_prev = time.time()

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

        # Siapkan embedding untuk semua wajah di frame (batch)
        crops, boxes = [], []
        for (x1, y1, x2, y2) in rects:
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(frame.shape[1] - 1, x2), min(frame.shape[0] - 1, y2)
            if x2 <= x1 or y2 <= y1:
                continue
            crop = frame[y1:y2, x1:x2, :]
            crops.append(preprocess_arcface(crop, 112))  # CHW float32
            boxes.append((x1, y1, x2, y2))

        if crops:
            F = embed_batch_cvlface(model, aligner, crops, device)  # [N, D]
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

        cv2.imshow("Realtime Identify (CVLface + NPZ)", vis)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    torch.backends.cudnn.benchmark = False
    main()
