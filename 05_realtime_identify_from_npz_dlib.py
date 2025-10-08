import argparse
from pathlib import Path
import time
import numpy as np
import cv2
import dlib

# ---------------- utils ----------------
def l2norm(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    n = np.maximum(n, eps)
    return x / n

def cosine_sim_matrix(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    # A: [N,D], b: [D]
    return A @ b  # asumsikan A dan b sudah L2-normalized

def parse_npz_entry(val) -> tuple[np.ndarray, str | None]:
    """
    Mendukung 2 format:
      - Lama: value = vektor float (D,)
      - Baru: value = structured array dgn fields {'feat','label'}
    Return: (feat: float[D], label: Optional[str])
    """
    arr = np.asarray(val)
    if hasattr(arr, "dtype") and arr.dtype.fields:
        # structured
        feat = np.asarray(arr["feat"]).reshape(-1).astype(np.float32, copy=False)
        label = None
        if "label" in arr.dtype.fields:
            lab = np.asarray(arr["label"]).reshape(-1)
            if lab.size > 0:
                label = str(lab[0])
        return feat, label
    # vektor biasa
    return arr.astype(np.float32, copy=False).reshape(-1), None

def label_from_key_path(key: str) -> str | None:
    # Ambil nama folder induk (ID) dari path key: ".../<LABEL>/<file>"
    parts = key.replace("\\", "/").split("/")
    if len(parts) >= 2:
        return parts[-2]
    return None

def load_gallery_templates_from_npz(npz_path: Path) -> tuple[np.ndarray, list[str]]:
    """
    Baca .npz -> kumpulkan fitur per label -> template = mean(L2norm(feat)).
    Return:
      G: [N_labels, D]
      labels: list[str] (panjang N_labels)
    """
    npz = np.load(str(npz_path))
    buckets: dict[str, list[np.ndarray]] = {}

    for k in npz.keys():
        feat, lab = parse_npz_entry(npz[k])
        # fallback label dari path jika label kosong
        lab = lab or label_from_key_path(k) or "UNKNOWN"
        feat = l2norm(feat.reshape(1, -1)).reshape(-1)
        buckets.setdefault(lab, []).append(feat)

    labels = sorted(buckets.keys())
    mats = []
    for lab in labels:
        M = np.stack(buckets[lab], axis=0)      # [n_i, D]
        m = l2norm(M.mean(axis=0, keepdims=True)).reshape(-1)
        mats.append(m)
    G = np.stack(mats, axis=0).astype(np.float32)  # [N_labels, D]
    return G, labels

# --------------- dlib embedding ---------------
def embed_one_face_rgb(rgb_img: np.ndarray, rect: dlib.rectangle,
                       sp5: dlib.shape_predictor,
                       facerec: dlib.face_recognition_model_v1) -> np.ndarray:
    shape = sp5(rgb_img, rect)
    # Anda bisa langsung pakai image+shape (tanpa chip) — ini cepat & akurat
    desc = facerec.compute_face_descriptor(rgb_img, shape)
    v = np.array(desc, dtype=np.float32)  # (128,)
    return l2norm(v.reshape(1, -1)).reshape(-1)

# --------------- main loop ---------------
def main():
    ap = argparse.ArgumentParser(description="Realtime Identify from NPZ (dlib)")
    ap.add_argument("--npz", required=True, help="Path file NPZ (contoh: ./embeds/embeds_dlib.npz)")
    ap.add_argument("--weights-recog", default="./algoritma/weights/dlib_face_recognition_resnet_model_v1.dat",
                    help="Bobot dlib face recognition resnet v1")
    ap.add_argument("--weights-sp5",   default="./algoritma/weights/shape_predictor_5_face_landmarks.dat",
                    help="Bobot shape predictor 5 landmarks")
    ap.add_argument("--camera", type=int, default=0, help="Index kamera (default: 0)")
    ap.add_argument("--thr", type=float, default=0.5, help="Threshold cosine untuk identifikasi (default: 0.5)")
    ap.add_argument("--width", type=int, default=960, help="Lebar tampilan video (auto scale tinggi)")
    ap.add_argument("--upsample", type=int, default=0, help="Dlib detector upsample (0/1) — nilai >0 memperbanyak deteksi tapi lebih lambat")
    args = ap.parse_args()

    npz_path = Path(args.npz).resolve()
    if not npz_path.exists():
        raise FileNotFoundError(f"NPZ tidak ditemukan: {npz_path}")

    print("[LOG] Memuat gallery templates dari:", npz_path)
    G, labels = load_gallery_templates_from_npz(npz_path)
    print(f"[LOG] Jumlah label: {len(labels)} | dim: {G.shape[1]}")

    # Muat dlib models
    wr = Path(args.weights_recog).resolve()
    ws = Path(args.weights_sp5).resolve()
    if not wr.exists() or not ws.exists():
        raise FileNotFoundError(
            f"Bobot dlib tidak ditemukan.\n  - recogn: {wr}\n  - sp5   : {ws}"
        )
    facerec = dlib.face_recognition_model_v1(str(wr))
    sp5     = dlib.shape_predictor(str(ws))
    detector = dlib.get_frontal_face_detector()

    # Cek dimensi kompatibel (dlib 128D)
    d_probe = 128
    if G.shape[1] != d_probe:
        raise RuntimeError(
            f"Dimensi embedding NPZ ({G.shape[1]}) tidak cocok dengan dlib probe (128).\n"
            f"Gunakan NPZ hasil algoritma dlib (contoh: embeds_dlib.npz)."
        )

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError("Tidak bisa membuka kamera. Coba ganti --camera 1 atau pastikan kameranya tidak dipakai aplikasi lain.")

    print("[INFO] Tekan 'q' untuk keluar.")
    last_fps = 0.0
    t_prev = time.time()

    while True:
        ok, frame = cap.read()
        if not ok:
            print("[WARN] Frame kosong dari kamera.")
            break

        # Resize agar ringan ditampilkan
        h, w = frame.shape[:2]
        if w != args.width and args.width > 0:
            scale = args.width / w
            frame = cv2.resize(frame, (args.width, int(h*scale)))
        vis = frame.copy()

        # dlib expect RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # deteksi wajah
        rects = detector(rgb, args.upsample)

        for r in rects:
            # embedding satu wajah
            v = embed_one_face_rgb(rgb, r, sp5, facerec)  # (128,)
            # cosine ke gallery
            sims = cosine_sim_matrix(G, v)  # [N_labels]
            j = int(np.argmax(sims))
            score = float(sims[j])
            name = labels[j] if score >= args.thr else "Unknown"

            # gambar kotak + label
            x1, y1, x2, y2 = r.left(), r.top(), r.right(), r.bottom()
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0,255,0) if name!="Unknown" else (0,0,255), 2)
            txt = f"{name}  ({score:.2f})"
            cv2.rectangle(vis, (x1, y1-22), (x1 + 8*len(txt), y1), (0,0,0), -1)
            cv2.putText(vis, txt, (x1+4, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 1, cv2.LINE_AA)

        # FPS
        t_now = time.time()
        dt = t_now - t_prev
        if dt > 0:
            last_fps = 0.9*last_fps + 0.1*(1.0/dt)
        t_prev = t_now
        cv2.putText(vis, f"FPS: {last_fps:.1f} | thr={args.thr}", (8, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2, cv2.LINE_AA)

        cv2.imshow("Realtime Identify (dlib + NPZ)", vis)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
