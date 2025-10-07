# 03_evaluasi_performa_cosine.py
# IDENTIFIKASI (gallery -> probe) dgn Cosine Similarity + threshold
#
# Output per algoritma: <outdir>/<algo>/
#   - <algo>_results_identification_thrXX.json
#   - <algo>_results_metrics_thrXX.json
# CSV agregat semua algoritma: <outdir>/metrics_all_cosine.csv
#
# Kompatibel FORMAT EMBED:
# 1) Baru (tuple/object): value = np.array([feat, label], dtype=object) ATAU (feat,label) dalam array 1-elemen
# 2) Baru (structured):   value.dtype.fields ada & punya field "feat","label"
# 3) Lama (vektor saja):  value = np.ndarray 1D vektor fitur (label diambil dari path)
#
# Contoh:
#   python 03_evaluasi_performa_cosine.py --embeds .\embeds\embeds_adaface_ir100.npz --fixed-thr 0.5 --outdir eval_verif_cosine
#   python 03_evaluasi_performa_cosine.py --embeds-dir .\embeds --fixed-thr 0.5 --outdir eval_verif_cosine

import argparse, csv, json
from pathlib import Path
from typing import Optional, List, Dict, Tuple
import numpy as np
from collections import defaultdict

def algo_name_from_embeds(path: Path) -> str:
    return path.stem

def thr_to_tag(val: float) -> str:
    return f"thr{val:.2f}"

def name_with_thr(base: str, thr_tag: str) -> str:
    p = Path(base)
    return f"{p.stem}_{thr_tag}{p.suffix}"

# ---- path normalizer ----
def _norm_once(p: str) -> str:
    p = p.strip().replace("\\", "/")
    while p.startswith("./"):
        p = p[2:]
    if "." in p:
        head, _, ext = p.rpartition(".")
        if ext:
            p = head + "." + ext.lower()
    return p

def l2norm(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(v)
    if n < eps:
        return v
    return v / n

def extract_role_and_id_from_key(k: str) -> Tuple[Optional[str], Optional[str]]:
    """
    role: 'gallery' | 'probe' | None
    pid : ID jika dapat diekstrak dari path.
          - gallery/.../<ID>/file -> pid=<ID>
          - probe/.../<ID>/file   -> pid=<ID>
          - probe/file (flat)     -> pid=None
    """
    kn = _norm_once(k)
    parts = kn.split("/")
    lowers = [s.lower() for s in parts]

    role, pid = None, None
    if "gallery" in lowers:
        role = "gallery"
        idx = lowers.index("gallery")
        if idx + 1 < len(parts):
            nxt = parts[idx + 1]
            if "." not in nxt.lower():
                pid = nxt
    elif "probe" in lowers:
        role = "probe"
        idx = lowers.index("probe")
        if idx + 1 < len(parts):
            nxt = parts[idx + 1]
            if "." not in nxt.lower():
                pid = nxt
    return role, pid

def fallback_label_from_path(k: str) -> Optional[str]:
    """Jika tidak ada label di value, ambil dari path:
       - prefer segmen setelah 'gallery'/'probe'
       - jika tidak ada, pakai nama folder induk
       - fallback terakhir: nama file tanpa ekstensi
    """
    kn = _norm_once(k)
    role, pid = extract_role_and_id_from_key(kn)
    if pid:
        return pid
    parts = kn.split("/")
    if len(parts) >= 2:
        return parts[-2]
    return Path(kn).stem

# ---- parser value feat+label (robust untuk 3 format) ----
def parse_feat_and_label(val, key_for_fallback: str) -> Tuple[np.ndarray, Optional[str]]:
    """
    Return: (feat_vector_1D_float64, label_str_or_None)
    Mendukung:
      - structured dtype dengan fields {'feat','label'}
      - array/tuple object berisi [feat, label] (urutan 0=feat, 1=label)
      - legacy: vektor saja (tanpa label)
    """
    # 1) Structured dtype
    if isinstance(val, np.ndarray) and getattr(val.dtype, "fields", None):
        names = set(val.dtype.names or [])
        if "feat" in names and "label" in names:
            # Bisa shape (1,) atau scalar structured
            try:
                feat_field = val["feat"]
                label_field = val["label"]
                if feat_field.shape == ():
                    feat = np.asarray(feat_field, dtype=np.float64).reshape(-1)
                else:
                    feat = np.asarray(feat_field[0], dtype=np.float64).reshape(-1)
                if label_field.shape == ():
                    label = str(label_field)
                else:
                    label = str(label_field[0])
                return feat, label
            except Exception:
                pass  # lanjut ke mode lain bila gagal parsing

    # 2) Array/tuple object: [feat, label] atau ((feat,label),) dst.
    if isinstance(val, np.ndarray) and val.dtype == object:
        # kemungkinan langsung [feat, label]
        if val.ndim == 1 and val.size == 2:
            feat = np.asarray(val[0], dtype=np.float64).reshape(-1)
            label = None if val[1] is None else str(val[1])
            return feat, label
        # kemungkinan (1,) lalu elemennya tuple/list
        if val.ndim == 1 and val.size == 1:
            inner = val[0]
            if isinstance(inner, (tuple, list)) and len(inner) >= 2:
                feat = np.asarray(inner[0], dtype=np.float64).reshape(-1)
                label = None if inner[1] is None else str(inner[1])
                return feat, label

    # 3) Legacy: vektor saja
    if isinstance(val, np.ndarray) and val.ndim >= 1 and val.dtype != object:
        feat = np.asarray(val, dtype=np.float64).reshape(-1)
        label = None  # ambil dari path nanti bila perlu
        return feat, label

    # 4) Terakhir, coba treat sebagai (feat,label) tuple langsung
    if isinstance(val, (tuple, list)) and len(val) >= 2:
        feat = np.asarray(val[0], dtype=np.float64).reshape(-1)
        label = None if val[1] is None else str(val[1])
        return feat, label

    # Gagal total
    raise ValueError(f"Format value embed tidak dikenali untuk key='{key_for_fallback}' (type={type(val)})")

# ---- metrik helper ----
def safe_div(num: float, den: float) -> Optional[float]:
    return (num / den) if den != 0 else None

def fmt(x: Optional[float]) -> Optional[float]:
    if x is None:
        return None
    return float(f"{x:.8f}")

# =============== IDENTIFICATION PIPELINE ===============
def eval_identification_pipeline(
    embeds_path: Path,
    outdir_root: Path,
    fixed_thr: float = 0.6,
):
    """
    Evaluasi IDENTIFIKASI (Top-1) menggunakan Cosine Similarity.
    - Template per-ID dari GALLERY = mean embedding (L2-norm)
    - Untuk tiap PROBE: cosine ke semua template -> top-1 skor & ID
      Threshold:
        skor >= thr & pred==true -> TP
        skor >= thr & pred!=true -> FP
        skor <  thr:
           true di gallery -> FN
           true tdk di gallery -> TN
    - Probe tanpa label (tak bisa ambil ID) -> diskip
    - Membaca embed format baru: per-key menyimpan (feat,label)
    """
    outdir_root.mkdir(parents=True, exist_ok=True)

    algo = algo_name_from_embeds(embeds_path)
    thr_tag = thr_to_tag(fixed_thr)

    algo_dir = outdir_root / algo
    algo_dir.mkdir(parents=True, exist_ok=True)

    metrics_all_csv = outdir_root / "metrics_all_cosine.csv"

    # === Load embeddings ===
    # allow_pickle=True diperlukan untuk dtype=object (tuple [feat,label])
    EMB = np.load(str(embeds_path), allow_pickle=True)
    all_keys = list(EMB.keys())

    # Pisahkan GALLERY & PROBE + ambil ID dari field "label" (fallback: path)
    gal_id2vecs: Dict[str, List[np.ndarray]] = defaultdict(list)
    probe_items: List[Tuple[str, Optional[str], np.ndarray]] = []  # (key, pid_true, vec)

    for k in all_keys:
        role, pid_from_path = extract_role_and_id_from_key(k)

        feat, lbl = parse_feat_and_label(EMB[k], k)
        feat = l2norm(feat)

        # jika lbl tak ada, fallback dari path
        pid = lbl if (lbl is not None and str(lbl).strip() != "") else pid_from_path

        if role == "gallery":
            if pid is not None:
                gal_id2vecs[pid].append(feat)
        elif role == "probe":
            probe_items.append((k, pid, feat))
        else:
            # abaikan selain gallery/probe
            pass

    if not gal_id2vecs:
        raise RuntimeError(f"Tidak ada data GALLERY pada embeddings: {embeds_path}")

    # Template per-ID
    gal_ids = sorted(gal_id2vecs.keys())
    gal_mat = []
    for pid in gal_ids:
        m = np.mean(np.stack(gal_id2vecs[pid], axis=0), axis=0)
        gal_mat.append(l2norm(m))
    G = np.stack(gal_mat, axis=0)  # [n_id, d]
    d = G.shape[1]

    n_gallery_ids = len(gal_ids)
    n_gallery_imgs = sum(len(v) for v in gal_id2vecs.values())
    n_probe_imgs = len(probe_items)

    # === Evaluasi probe ===
    TP = TN = FP = FN = 0
    skipped_unlabeled = 0
    used = 0

    for k, pid_true, v in probe_items:
        if pid_true is None:
            skipped_unlabeled += 1
            continue
        if v.shape[0] != d:
            skipped_unlabeled += 1
            continue

        sims = G @ v  # cosine (karena G & v sudah L2-norm)
        j = int(np.argmax(sims))
        score = float(sims[j])
        pred_id = gal_ids[j]

        if score >= fixed_thr:
            if pred_id == pid_true:
                TP += 1
            else:
                FP += 1
        else:
            if pid_true in gal_id2vecs:
                FN += 1
            else:
                TN += 1
        used += 1

    total = TP + TN + FP + FN
    accuracy  = safe_div(TP + TN, total)
    precision = safe_div(TP, TP + FP)
    recall    = safe_div(TP, TP + FN)
    f1 = None if (precision is None or recall is None or (precision + recall) == 0) \
        else 2.0 * (precision * recall) / (precision + recall)

    # JSON ringkasan IDENTIFIKASI
    summary_id = {
        "algo": algo,
        "task": "identification",
        "threshold_used": float(fixed_thr),
        "N_gallery_ids": int(n_gallery_ids),
        "N_gallery_images": int(n_gallery_imgs),
        "N_probe_images": int(n_probe_imgs),
        "N_probe_used": int(used),
        "N_probe_skipped_unlabeled": int(skipped_unlabeled),
        "TP": int(TP), "TN": int(TN), "FP": int(FP), "FN": int(FN),
        "accuracy": fmt(accuracy),
        "precision": fmt(precision),
        "recall": fmt(recall),
        "f1": fmt(f1),
        "embeds": str(embeds_path),
    }
    id_json_path = algo_dir / name_with_thr(f"{algo}_results_identification.json", thr_tag)
    id_json_path.write_text(json.dumps(summary_id, indent=2), encoding="utf-8")

    # JSON metrics (identik untuk konsistensi)
    summary_met = dict(summary_id)
    met_json_path = algo_dir / name_with_thr(f"{algo}_results_metrics.json", thr_tag)
    met_json_path.write_text(json.dumps(summary_met, indent=2), encoding="utf-8")

    # CSV agregat
    fieldnames = [
        "algo",
        "accuracy", "precision", "recall", "f1",
        "TP", "TN", "FP", "FN",
        "total_pairs",  # kompatibel (isi = N_probe_used)
        "threshold_used",
        "N_gallery_ids", "N_gallery_images", "N_probe_images", "N_probe_used", "N_probe_skipped_unlabeled",
    ]
    write_header = not metrics_all_csv.exists()
    with metrics_all_csv.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            w.writeheader()
        row = {
            "algo": algo,
            "accuracy": summary_met["accuracy"],
            "precision": summary_met["precision"],
            "recall": summary_met["recall"],
            "f1": summary_met["f1"],
            "TP": TP, "TN": TN, "FP": FP, "FN": FN,
            "total_pairs": used,
            "threshold_used": float(fixed_thr),
            "N_gallery_ids": n_gallery_ids,
            "N_gallery_images": n_gallery_imgs,
            "N_probe_images": n_probe_imgs,
            "N_probe_used": used,
            "N_probe_skipped_unlabeled": skipped_unlabeled,
        }
        w.writerow(row)

    print("\n== IDENTIFICATION SUMMARY ==")
    print(json.dumps(summary_id, indent=2))

    return summary_id, summary_met

def main():
    ap = argparse.ArgumentParser(
        description="Evaluasi IDENTIFIKASI (gallery -> probe) berbasis Cosine Similarity + threshold."
    )
    g = ap.add_mutually_exclusive_group(required=False)
    g.add_argument("--embeds", help="Path ke file .npz hasil embedding (satu file).")
    g.add_argument("--embeds-dir", help="Folder berisi banyak .npz untuk dievaluasi semuanya.")

    ap.add_argument("--outdir", default="eval_verif_cosine",
                    help="Folder output hasil program. Default: eval_verif_cosine")
    ap.add_argument("--fixed-thr", type=float, default=0.6, help="Threshold Cosine (mis. 0.3/0.5/0.7).")

    args = ap.parse_args()

    outdir_root = Path(args.outdir)
    outdir_root.mkdir(parents=True, exist_ok=True)

    targets: List[Path] = []
    if args.embeds_dir:
        d = Path(args.embeds_dir)
        if not d.exists():
            raise FileNotFoundError(f"Folder embeds tidak ditemukan: {d}")
        targets = sorted([p for p in d.glob("*.npz") if p.is_file()])
        if not targets:
            raise RuntimeError(f"Tidak ada file .npz di: {d}")
    elif args.embeds:
        p = Path(args.embeds)
        if not p.exists():
            raise FileNotFoundError(f"Embeddings tidak ditemukan: {p}")
        targets = [p]
    else:
        # default: cari di ./embeds
        d = Path("embeds")
        if not d.exists():
            raise FileNotFoundError("Tidak ada --embeds atau --embeds-dir, dan folder default './embeds' tidak ada.")
        targets = sorted([p for p in d.glob("*.npz") if p.is_file()])
        if not targets:
            raise RuntimeError("Folder './embeds' ada tapi tidak berisi file .npz.")

    for emb in targets:
        try:
            eval_identification_pipeline(
                embeds_path=emb,
                outdir_root=outdir_root,
                fixed_thr=args.fixed_thr,
            )
        except Exception as e:
            print(f"[ERROR] Gagal evaluasi {emb.name}: {type(e).__name__}: {e}")

if __name__ == "__main__":
    main()
