# pyright: reportMissingImports=false
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import sys
import time
import math
import urllib.request
import importlib
from typing import Tuple, List, Dict, Optional

import cv2
import numpy as np
import torch


# ============================ Utils ============================

def l2_normalize(x: np.ndarray, axis: int = -1, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(x, axis=axis, keepdims=True)
    n = np.maximum(n, eps)
    return x / n


def cosine_similarity(q: np.ndarray, gallery: np.ndarray) -> np.ndarray:
    """q: (D,), gallery: (N,D) assumed L2-normalized."""
    return np.dot(gallery, q)


def euclidean_distance(q: np.ndarray, gallery: np.ndarray) -> np.ndarray:
    """q: (D,), gallery: (N,D)."""
    diff = gallery - q[None, :]
    return np.sqrt(np.sum(diff * diff, axis=1))


def threshold_pass(score: float, metric: str, thr: float) -> bool:
    """Accept if score passes threshold. For cosine: score >= thr, for euclidean: distance <= thr."""
    if metric == "cosine":
        return score >= thr
    return score <= thr


def cosine_to_l2_equiv(cos_thr: float) -> float:
    """
    Untuk vektor yang sudah L2-normalized, relasi:
    L2^2 = 2 - 2*cos  ->  L2 = sqrt(2*(1 - cos))
    """
    cos_thr = np.clip(cos_thr, -1.0, 1.0)
    return float(np.sqrt(2.0 * (1.0 - cos_thr)))


def l2_to_cosine_equiv(l2_thr: float) -> float:
    """cos = 1 - (L2^2)/2"""
    return float(1.0 - (l2_thr * l2_thr) / 2.0)


def _choose_first_available(d: Dict, keys: List[str]):
    for k in keys:
        if k in d:
            return d[k]
    return None


def _derive_identity_from_string(s: str, mode: str = "parentdir") -> str:
    """
    mode:
      - parentdir: ambil nama folder induk (…/IDENTITY/filename.jpg)
      - stem     : ambil nama file tanpa ekstensi
      - filename : ambil nama file dengan ekstensi
      - raw      : pakai string apa adanya
    """
    try:
        if mode == "parentdir":
            return os.path.basename(os.path.dirname(s)) or s
        elif mode == "stem":
            return os.path.splitext(os.path.basename(s))[0]
        elif mode == "filename":
            return os.path.basename(s)
        else:
            return s
    except Exception:
        return s


def load_gallery(
    npz_path: str,
    label_from: str = "parentdir",
    collapse: str = "mean",
) -> Tuple[np.ndarray, List[str], Dict]:
    """
    Memuat embedding dari .npz dan mengubah label menjadi identity yang rapi.
    Bisa kompresi per-identity (mean) agar N kecil & cepat.
    """
    if not os.path.isfile(npz_path):
        raise FileNotFoundError(f"File tidak ditemukan: {npz_path}")

    data = np.load(npz_path, allow_pickle=True)
    # Cari kunci umum untuk embedding & label
    embeds = _choose_first_available(data, ["embeds", "embeddings", "features", "X", "arr_0"])
    raw_labels = _choose_first_available(data, ["labels", "paths", "files", "filenames", "names", "y", "arr_1"])

    if embeds is None:
        raise KeyError(
            f"Tidak menemukan array embedding di {npz_path}. "
            f"Harap simpan dengan key 'embeds' atau 'embeddings' dsb."
        )
    embeds = np.asarray(embeds)

    # Normalisasi embedding (umumnya model face-rec sudah unit-norm; normalize lagi biar aman)
    embeds = l2_normalize(embeds.astype(np.float32), axis=1)

    if raw_labels is None:
        # Kalau tidak ada label sama sekali, buat dummy label bernomor
        labels_str = [f"id_{i}" for i in range(len(embeds))]
    else:
        raw = list(raw_labels.tolist() if hasattr(raw_labels, "tolist") else list(raw_labels))
        # Convert ke string (kalau integer class id, jadi "0","1",…)
        raw = [str(x) for x in raw]
        labels_str = [_derive_identity_from_string(s, mode=label_from) for s in raw]

    meta = {
        "total_items": int(embeds.shape[0]),
        "label_from": label_from,
        "collapse": collapse,
    }

    if collapse.lower() == "none":
        # Tidak dikompresi: 1 embedding per file
        # label identitas = hasil parsing (parentdir/stem/filename/raw)
        return embeds, labels_str, meta

    # Kompresi per identitas (default: mean)
    by_id: Dict[str, List[int]] = {}
    for i, name in enumerate(labels_str):
        by_id.setdefault(name, []).append(i)

    ids = sorted(by_id.keys())
    out_embeds = []
    out_labels = []
    for ident in ids:
        idxs = by_id[ident]
        vecs = embeds[idxs]
        if collapse.lower() == "mean":
            proto = np.mean(vecs, axis=0)
        elif collapse.lower() == "median":
            proto = np.median(vecs, axis=0)
        else:
            # fallback ke mean jika opsi tak dikenal
            proto = np.mean(vecs, axis=0)
        proto = l2_normalize(proto)
        out_embeds.append(proto)
        out_labels.append(ident)

    out_embeds = np.stack(out_embeds, axis=0).astype(np.float32)
    meta["unique_identities"] = len(out_labels)
    return out_embeds, out_labels, meta


# ====================== Detectors (MediaPipe & DNN) ======================

_MEDIAPIPE_IMPORT_ERROR = None
mp = None
try:
    import mediapipe as mp  # type: ignore
except Exception as e:
    _MEDIAPIPE_IMPORT_ERROR = e
    mp = None


class MediapipeFaceDetector:
    def __init__(self, min_conf=0.5, model_selection=0):
        self.mp_fd = mp.solutions.face_detection
        self.det = self.mp_fd.FaceDetection(
            model_selection=model_selection,
            min_detection_confidence=min_conf,
        )

    def detect(self, frame_bgr):
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        res = self.det.process(rgb)
        out = []
        if res.detections:
            H, W = frame_bgr.shape[:2]
            for det in res.detections:
                rb = det.location_data.relative_bounding_box
                x1, y1 = max(0, int(rb.xmin * W)), max(0, int(rb.ymin * H))
                x2 = min(W - 1, int((rb.xmin + rb.width) * W))
                y2 = min(H - 1, int((rb.ymin + rb.height) * H))
                kps = [(int(kp.x * W), int(kp.y * H)) for kp in det.location_data.relative_keypoints]
                out.append({"bbox": (x1, y1, x2, y2), "kps": kps, "score": float(det.score[0])})
        return out


DNN_PROTO_URL = "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt"
DNN_MODEL_URL = "https://raw.githubusercontent.com/opencv/opencv_3rdparty/master/dnn/face_detector/res10_300x300_ssd_iter_140000_fp16.caffemodel"


def ensure_dnn_files(model_dir: str) -> Tuple[str, str]:
    os.makedirs(model_dir, exist_ok=True)
    proto = os.path.join(model_dir, "deploy.prototxt")
    caffemodel = os.path.join(model_dir, "res10_300x300_ssd_iter_140000_fp16.caffemodel")
    if not os.path.isfile(proto):
        urllib.request.urlretrieve(DNN_PROTO_URL, proto)
    if not os.path.isfile(caffemodel):
        urllib.request.urlretrieve(DNN_MODEL_URL, caffemodel)
    return proto, caffemodel


class DnnFaceDetector:
    def __init__(self, min_conf=0.5, model_dir="algoritma/weights/face_detector"):
        proto, caffemodel = ensure_dnn_files(model_dir)
        self.net = cv2.dnn.readNetFromCaffe(proto, caffemodel)
        self.min_conf = float(min_conf)

    def detect(self, frame_bgr):
        (H, W) = frame_bgr.shape[:2]
        blob = cv2.dnn.blobFromImage(frame_bgr, 1.0, (300, 300), (104, 177, 123), swapRB=False, crop=False)
        self.net.setInput(blob)
        detections = self.net.forward()
        out = []
        for i in range(detections.shape[2]):
            conf = float(detections[0, 0, i, 2])
            if conf < self.min_conf:
                continue
            x1, y1, x2, y2 = (detections[0, 0, i, 3:7] * np.array([W, H, W, H])).astype("int")
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(W - 1, x2), min(H - 1, y2)
            out.append({"bbox": (x1, y1, x2, y2), "kps": [], "score": conf})
        return out


# ====================== Crop / Align ======================

def _expand_bbox(x1, y1, x2, y2, w, h, margin=0.3):
    bw = x2 - x1
    bh = y2 - y1
    cx = x1 + bw / 2.0
    cy = y1 + bh / 2.0
    bw *= (1.0 + margin)
    bh *= (1.0 + margin)
    nx1 = int(max(0, cx - bw / 2.0))
    ny1 = int(max(0, cy - bh / 2.0))
    nx2 = int(min(w - 1, cx + bw / 2.0))
    ny2 = int(min(h - 1, cy + bh / 2.0))
    return nx1, ny1, nx2, ny2


def crop_square(img, bbox, size=112, margin=0.3):
    h, w = img.shape[:2]
    x1, y1, x2, y2 = bbox
    x1, y1, x2, y2 = _expand_bbox(x1, y1, x2, y2, w, h, margin)
    face = img[y1:y2, x1:x2]
    if face.size == 0:
        # fallback ke center crop kecil agar tidak crash
        cx, cy = w // 2, h // 2
        half = min(w, h) // 4
        face = img[max(0, cy - half):min(h, cy + half), max(0, cx - half):min(w, cx + half)]
    face = cv2.resize(face, (size, size))
    return face


def rotate_and_crop_by_eyes(img, bbox, kps, size=112, margin=0.3):
    # Butuh minimal 2 keypoints (eye_left, eye_right)
    if len(kps) < 2:
        return crop_square(img, bbox, size=size, margin=margin)
    (ex1, ey1), (ex2, ey2) = kps[0], kps[1]
    dx, dy = ex2 - ex1, ey2 - ey1
    angle = math.degrees(math.atan2(dy, dx))
    # rotasi sekitar tengah wajah
    x1, y1, x2, y2 = bbox
    cx = int((x1 + x2) / 2)
    cy = int((y1 + y2) / 2)
    M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
    rotated = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
    # recalc bbox pada gambar ter-rotasi: pakai bbox yg sama (approx) lalu expand + crop
    return crop_square(rotated, bbox, size=size, margin=margin)


# ====================== AdaFace extractor ======================

class AdaFaceTorchExtractor:
    def __init__(self, repo_dir: str, arch: str, ckpt_path: str, device: Optional[str] = None):
        self.repo_dir = os.path.abspath(repo_dir)
        if self.repo_dir not in sys.path:
            sys.path.insert(0, self.repo_dir)
        try:
            net = importlib.import_module("net")  # type: ignore
        except Exception as e:
            raise ImportError(
                f"Tidak bisa import modul 'net' dari {self.repo_dir}. "
                f"Pastikan repo AdaFace sudah di-clone. Error: {e}"
            )

        try:
            self.model = net.build_model(arch)
        except Exception as e:
            raise RuntimeError(f"build_model('{arch}') gagal: {e}")

        if not os.path.isfile(ckpt_path):
            raise FileNotFoundError(f"Checkpoint tidak ditemukan: {ckpt_path}")

        ckpt = torch.load(ckpt_path, map_location="cpu")
        sd = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
        sd = {(k[6:] if k.startswith("model.") else k): v for k, v in sd.items()}
        missing, unexpected = self.model.load_state_dict(sd, strict=False)
        if missing:
            print(f"[WARN] missing keys: {len(missing)} (contoh: {missing[:5]})")
        if unexpected:
            print(f"[WARN] unexpected keys: {len(unexpected)} (contoh: {unexpected[:5]})")

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device).eval()

    @staticmethod
    def preprocess_bgr_112(face_bgr: np.ndarray) -> torch.Tensor:
        if face_bgr.shape[:2] != (112, 112):
            face_bgr = cv2.resize(face_bgr, (112, 112))
        img = (face_bgr.astype("float32") / 255.0 - 0.5) / 0.5
        return torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0).float()

    @torch.no_grad()
    def embed(self, face_bgr: np.ndarray) -> np.ndarray:
        x = self.preprocess_bgr_112(face_bgr).to(self.device)
        feat, _ = self.model(x)
        feat = feat[0].detach().cpu().numpy().astype("float32")
        return l2_normalize(feat)


# ============================ Index ============================

class EmbeddingIndex:
    def __init__(self, gallery: np.ndarray, labels: List[str], metric: str = 'cosine'):
        self.gallery = gallery
        self.labels = labels
        self.metric = metric

    def query(self, q: np.ndarray, topk: int = 1):
        if self.metric == 'cosine':
            scores = cosine_similarity(q, self.gallery)
            idx = np.argsort(-scores)[:topk]
            vals = scores[idx]
        else:
            dists = euclidean_distance(q, self.gallery)
            idx = np.argsort(dists)[:topk]
            vals = dists[idx]
        return [self.labels[i] for i in idx], vals, idx


# ============================== Main ==============================

def main():
    ap = argparse.ArgumentParser("Realtime AdaFace Recognition (auto-fallback detector, clean labels)")
    ap.add_argument("--embeds_path", required=True, help="Path ke .npz, misal embeds/embeds_adaface_ir100.npz")
    ap.add_argument("--repo_dir", default="algoritma/AdaFace", help="Folder repo resmi AdaFace")
    ap.add_argument("--arch", default="ir_101", help="Arsitektur AdaFace, contoh: ir_50, ir_101")
    ap.add_argument("--ckpt_path", default="algoritma/weights/adaface_ir101_ms1mv2.ckpt", help="Path checkpoint .ckpt")
    ap.add_argument("--similarity", choices=["cosine", "euclidean"], default="cosine")
    ap.add_argument("--threshold", type=float, default=0.5, help="Ambang penerimaan match")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--min_det_conf", type=float, default=0.6)
    ap.add_argument("--align", action="store_true", help="Rotate-by-eyes jika keypoint tersedia (MediaPipe)")
    ap.add_argument("--topk", type=int, default=1)
    ap.add_argument("--stride", type=int, default=1, help="Hitung embedding setiap N frame (hemat CPU/GPU)")
    ap.add_argument("--detector", choices=["auto", "mediapipe", "dnn"], default="auto")

    # --- arg baru untuk bersihin label & kompresi galeri ---
    ap.add_argument("--label_from", choices=["parentdir", "stem", "filename", "raw"], default="parentdir",
                    help="Cara ekstrak identitas dari label/path di .npz")
    ap.add_argument("--collapse", choices=["mean", "median", "none"], default="mean",
                    help="Kompresi galeri per identitas (mean/median) atau tidak (none)")

    args = ap.parse_args()

    # Load galeri (dengan pembersihan label & kompresi)
    gallery, labels, meta = load_gallery(args.embeds_path, label_from=args.label_from, collapse=args.collapse)
    dim = int(gallery.shape[1])
    print(f"[INFO] Gallery loaded: items={gallery.shape[0]} dim={dim} "
          f"(from {meta.get('total_items', gallery.shape[0])} raw, mode={args.collapse}, label_from={args.label_from})")

    if args.similarity == "cosine":
        print(f"[INFO] Thr cosine = {args.threshold:.3f} (≈ L2 {cosine_to_l2_equiv(args.threshold):.3f})")
    else:
        print(f"[INFO] Thr L2 = {args.threshold:.3f} (≈ cosine {l2_to_cosine_equiv(args.threshold):.3f})")

    index = EmbeddingIndex(gallery, labels, metric=args.similarity)

    print(f"[INFO] Muat AdaFace arch={args.arch} ckpt={args.ckpt_path}")
    extractor = AdaFaceTorchExtractor(args.repo_dir, args.arch, args.ckpt_path)

    # Pilih detector + robust fallback
    detector_impl = None
    choice = args.detector
    if choice in ("auto", "mediapipe"):
        try:
            if mp is None:
                raise ImportError(_MEDIAPIPE_IMPORT_ERROR or "mediapipe not available")
            detector_impl = MediapipeFaceDetector(min_conf=args.min_det_conf, model_selection=0)
            # trigger awal utk cepat mendeteksi problem protobuf
            _ = detector_impl.detect(np.zeros((240, 320, 3), dtype=np.uint8))
            print("[INFO] Detector: MediaPipe")
        except Exception as e:
            print(f"[WARN] MediaPipe tidak tersedia/bermasalah ({e}). Fallback ke OpenCV DNN.")
            detector_impl = DnnFaceDetector(min_conf=args.min_det_conf)
    if choice == "dnn":
        detector_impl = DnnFaceDetector(min_conf=args.min_det_conf)
        print("[INFO] Detector: OpenCV DNN")

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print("[ERROR] Kamera tidak terbuka."); return

    print("[INFO] Tekan Q untuk keluar, S untuk snapshot.")
    prev_t, frame_id = time.time(), 0

    while True:
        ok, frame = cap.read()
        if not ok:
            print("[WARN] Gagal baca frame."); break
        frame_id += 1
        canvas = frame.copy()

        now = time.time()
        dt = now - prev_t
        prev_t = now
        fps = (1.0 / dt) if dt > 0 else 0.0

        dets = detector_impl.detect(frame)
        do_infer = (frame_id % max(1, args.stride) == 0)

        for det in dets:
            x1, y1, x2, y2 = det["bbox"]
            score = det["score"]
            kps = det.get("kps", [])
            cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 220, 0), 2)
            cv2.putText(canvas, f"{score:.2f}", (x1, max(0, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 0), 1)

            if args.align and len(kps) >= 2:
                face = rotate_and_crop_by_eyes(frame, (x1, y1, x2, y2), kps, 112, 0.3)
            else:
                face = crop_square(frame, (x1, y1, x2, y2), 112, 0.3)

            name_show, info_show = "...", ""
            if do_infer:
                try:
                    feat = extractor.embed(face)  # (D,)
                    if feat.shape[0] != dim:
                        name_show = "DIM_MISMATCH"
                        info_show = f"probe:{feat.shape[0]} vs gallery:{dim}"
                    else:
                        names, vals, _ = index.query(feat, topk=max(1, args.topk))
                        primary = float(vals[0])
                        accept = threshold_pass(primary, args.similarity, args.threshold)
                        name_show = names[0] if accept else "UNKNOWN"
                        info_show = (f"cos={primary:.3f}" if args.similarity == "cosine" else f"L2={primary:.3f}")

                        # tampilkan top-k ringkas (1 baris/entry)
                        for i, (nm, sc) in enumerate(zip(names, vals)):
                            ytxt = y2 + 20 + 18 * i
                            if ytxt < canvas.shape[0] - 5:
                                cv2.putText(canvas, f"{i+1}. {nm}: {float(sc):.3f}",
                                            (x1, ytxt), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                except Exception as e:
                    name_show, info_show = "INFER_ERROR", str(e)

            cv2.putText(canvas, name_show, (x1, y2 + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            if info_show:
                cv2.putText(canvas, info_show, (x1, y2 + 36), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        cv2.putText(canvas, f"FPS: {fps:.1f}", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        cv2.imshow("Realtime AdaFace Recognition", canvas)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), ord('Q')):
            break
        if key in (ord('s'), ord('S')):
            out = f"snapshot_{int(time.time())}.jpg"
            cv2.imwrite(out, canvas)
            print(f"[INFO] Snapshot: {out}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
