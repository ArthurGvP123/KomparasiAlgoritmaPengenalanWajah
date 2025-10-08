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
# ElasticFace loader
# =========================
def add_repo_to_sys_path(repo_dir: Path):
    repo = repo_dir.resolve()
    if not repo.exists():
        raise FileNotFoundError(f"[!] ElasticFace repo tidak ditemukan: {repo}")
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))

# ---- Fallback IResNet (kompatibel 512-D) ----
class _IBasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(inplanes, eps=2e-5, momentum=0.9)
        self.conv1 = nn.Conv2d(inplanes, planes, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, eps=2e-5, momentum=0.9)
        self.prelu = nn.PReLU(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes, eps=2e-5, momentum=0.9)
        self.downsample = downsample
    def forward(self, x):
        identity = x
        out = self.bn1(x); out = self.conv1(out)
        out = self.bn2(out); out = self.prelu(out)
        out = self.conv2(out); out = self.bn3(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        return out + identity

def _make_layer(block, inplanes_ref, planes, blocks, stride=1):
    inplanes, modules = inplanes_ref[0], []
    downsample = None
    if stride != 1 or inplanes != planes:
        downsample = nn.Sequential(
            nn.Conv2d(inplanes, planes, 1, stride, bias=False),
            nn.BatchNorm2d(planes, eps=2e-5, momentum=0.9),
        )
    modules.append(block(inplanes, planes, stride, downsample)); inplanes = planes
    for _ in range(1, blocks):
        modules.append(block(inplanes, planes))
    inplanes_ref[0] = inplanes
    return nn.Sequential(*modules)

class IResNet_Fallback(nn.Module):
    # 112x112 -> 7x7 spatial at the end, embedding 512-D
    def __init__(self, layers=(3,13,30,3), embedding_size=512):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(64, eps=2e-5, momentum=0.9)
        self.prelu = nn.PReLU(64)
        inref = [64]
        self.layer1 = _make_layer(_IBasicBlock, inref, 64,  layers[0], stride=2)
        self.layer2 = _make_layer(_IBasicBlock, inref, 128, layers[1], stride=2)
        self.layer3 = _make_layer(_IBasicBlock, inref, 256, layers[2], stride=2)
        self.layer4 = _make_layer(_IBasicBlock, inref, 512, layers[3], stride=2)
        self.bn2 = nn.BatchNorm2d(512, eps=2e-5, momentum=0.9)
        self.dropout = nn.Dropout2d(p=0.4, inplace=True)
        self.fc = nn.Linear(512 * 7 * 7, embedding_size)
        self.features = nn.BatchNorm1d(embedding_size, eps=2e-5, momentum=0.9)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1); nn.init.constant_(m.bias, 0)
    def forward(self, x):
        x = self.conv1(x); x = self.bn1(x); x = self.prelu(x)
        x = self.layer1(x); x = self.layer2(x); x = self.layer3(x); x = self.layer4(x)
        x = self.bn2(x); x = self.dropout(x)
        x = x.reshape(x.size(0), -1)  # reshape aman utk non-contiguous
        x = self.fc(x); x = self.features(x)
        return x

def build_elastic_model(repo_dir: Optional[Path], arch: str = "ir100"):
    """
    Coba import iresnet100 dari repo ElasticFace/backbones.
    Jika gagal, gunakan fallback IResNet_Fallback (512-D).
    """
    model = None
    if repo_dir:
        add_repo_to_sys_path(repo_dir)
        try:
            from backbones.iresnet import iresnet100  # dari repo resmi
            if arch.lower() not in ["ir100","iresnet100","ires100"]:
                raise ValueError("Untuk ElasticFace gunakan arch 'ir100'.")
            model = iresnet100(num_classes=512)
            print("[LOG] Loaded iresnet100 from ElasticFace repo.")
        except Exception as e:
            print(f"[WARN] Gagal impor iresnet100 dari repo: {e}")
    if model is None:
        print("[LOG] Menggunakan fallback IResNet_Fallback (512-D).")
        model = IResNet_Fallback()
    return model

def load_weights_into_model(model: nn.Module, weights_path: Path):
    # torch.load kompatibel PyTorch baru & lama
    try:
        ckpt = torch.load(str(weights_path), map_location="cpu", weights_only=True)  # PyTorch >=2.4
    except TypeError:
        ckpt = torch.load(str(weights_path), map_location="cpu")

    # Ambil state_dict jika dibungkus
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        ckpt = ckpt["state_dict"]

    cleaned = {}
    if isinstance(ckpt, dict):
        for k, v in ckpt.items():
            nk = k
            for pref in ("features.module.", "module.features.", "features.",
                          "module.", "model.", "backbone.", "encoder."):
                if nk.startswith(pref):
                    nk = nk[len(pref):]
            cleaned[nk] = v

    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    print(f"[Info] Loaded params: {len(cleaned)} | missing={len(missing)} | unexpected={len(unexpected)}")
    if missing or unexpected:
        print("[Info] load_state_dict non-strict. Contoh missing/unexpected (<=5):")
        for i, k in enumerate(list(missing)[:5], 1):   print(f"  missing {i:02d}: {k}")
        for i, k in enumerate(list(unexpected)[:5], 1): print(f"  unexpected {i:02d}: {k}")

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
def embed_batch_elastic(model, batch_imgs_chw, device: str) -> np.ndarray:
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
    ap = argparse.ArgumentParser(description="Realtime Identify from NPZ (ElasticFace IR-100)")
    # Gallery NPZ
    ap.add_argument("--npz", required=True, help="Path file NPZ gallery (mis. ./embeds/embeds_elasticface_ir100.npz)")
    # ElasticFace backbone
    ap.add_argument("--repo-name", default="", help="Folder repo ElasticFace (berisi backbones/). Boleh kosong (fallback).")
    ap.add_argument("--weights", required=True, help="Path weights ElasticFace (.pth), mis. ./algoritma/weights/elasticface_ir100_backbone.pth")
    ap.add_argument("--arch", default="ir100", choices=["ir100"], help="Backbone (ir100)")
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
        raise FileNotFoundError(f"Weights ElasticFace tidak ditemukan: {w_path}")
    if repo_dir is not None and not repo_dir.exists():
        raise FileNotFoundError(f"Repo ElasticFace tidak ditemukan: {repo_dir}")

    # 1) Gallery templates
    print("[LOG] Memuat gallery templates:", npz_path)
    G, labels = load_gallery_templates_from_npz(npz_path)
    print(f"[LOG] Label: {len(labels)} | dim: {G.shape[1]}")

    # 2) Backbone ElasticFace
    print("[LOG] Memuat backbone ElasticFace...")
    model = build_elastic_model(repo_dir, args.arch)
    load_weights_into_model(model, w_path)
    model.eval().to(device).float()
    print(f"[OK] Model: ElasticFace-{args.arch} on {device}")

    # Sanity check dimensi
    try:
        dummy = np.zeros((1, 3, 112, 112), dtype=np.float32)
        with torch.no_grad():
            y = model(torch.from_numpy(dummy).to(device))
            if isinstance(y, (tuple, list)):
                y = y[0]
            D_model = int(y.shape[-1])
        if G.shape[1] != D_model:
            raise RuntimeError(f"Dimensi gallery ({G.shape[1]}) ≠ dimensi model ({D_model}). Pastikan NPZ berasal dari ElasticFace dengan dimensi sama.")
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
            crops.append(preprocess_arcface(crop, 112))
            boxes.append((x1, y1, x2, y2))

        if crops:
            F = embed_batch_elastic(model, crops, device)  # [N, D]
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

        cv2.imshow("Realtime Identify (ElasticFace + NPZ)", vis)
        if (cv2.waitKey(1) & 0xFF) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    torch.backends.cudnn.benchmark = False
    main()
