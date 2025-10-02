# 03_evaluasi_performa_euclidean.py
# Pipeline terpadu (varian Euclidean/L2):
#  - SELALU membuat pairs otomatis dari embeddings
#  - Evaluasi verifikasi (ROC/AUC/EER/TAR) berbasis skor = -L2, + confusion matrix di threshold L2
#  - Hitung Accuracy / Precision / Recall / F1
#  - Simpan (per algoritma di eval_verif_euclidean/<algo>/):
#      * <algo>_pairs_auto_thrXX.csv
#      * <algo>_pairs_fixed_thrXX.csv
#      * <algo>_results_verification_thrXX.json
#      * <algo>_results_metrics_thrXX.json
#      * roc_<algo>_thrXX.png
#  - CSV agregat semua algoritma: eval_verif_euclidean/metrics_all.csv (TANPA kolom 'metrik')
#
# Jalankan contoh:
#   python 03_evaluasi_performa_euclidean.py --embeds .\embeds\embeds_adaface_ir100.npz --outdir eval_verif_euclidean --fixed-thr 0.6

import argparse, csv, json, random
from pathlib import Path
from typing import Optional, List, Dict
from collections import defaultdict

import numpy as np
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt


# =============== Util path & naming ===============
EVAL_ROOT_NAME = "eval_verif_euclidean"

def find_eval_root(outdir: Path) -> Path:
    """Cari folder 'eval_verif_euclidean' terdekat di jalur outdir; fallback ke outdir."""
    if outdir.name == EVAL_ROOT_NAME:
        return outdir
    for p in [outdir] + list(outdir.parents):
        if p.name == EVAL_ROOT_NAME:
            return p
    return outdir

def algo_name_from_embeds(path: Path) -> str:
    """Nama algoritma dari nama file .npz, tanpa ekstensi."""
    return path.stem

def thr_to_tag(val: float) -> str:
    """Buat tag threshold untuk nama file, mis. 0.6 -> 'thr0.60'."""
    return f"thr{val:.2f}"

def name_with_thr(base: str, thr_tag: str) -> str:
    """
    Sisipkan tag threshold sebelum ekstensi.
    base='roc_algo.png' -> 'roc_algo_thr0.60.png'
    base='algo_results.json' -> 'algo_results_thr0.60.json'
    """
    p = Path(base)
    stem = p.stem
    ext = p.suffix
    return f"{stem}_{thr_tag}{ext}"

def cosine_to_l2_thr(tau_cos: float) -> float:
    """
    Konversi ambang cosine -> ambang L2 (asumsi embedding sudah dinormalisasi L2):
        ||x - y||_2 = sqrt(2 - 2 * cos)
    """
    tc = max(-1.0, min(1.0, float(tau_cos)))     # clamp ke [-1, 1]
    val = 2.0 - 2.0 * tc
    if val < 0.0:
        val = 0.0
    return float(np.sqrt(val))


# =============== 03a - Generate Pairs ===============
def guess_id_from_key(k: str) -> str:
    parts = k.split("/")
    lowers = [p.lower() for p in parts]
    for anchor in ("persons", "gallery", "probe"):
        if anchor in lowers:
            idx = lowers.index(anchor)
            if idx + 1 < len(parts):
                return parts[idx + 1]
    if len(parts) >= 2:
        return parts[-2]
    return "UNKNOWN"

def build_pairs_from_embeds(npz_path: str, pos_per_id: int = 50, seed: int = 42):
    E = np.load(npz_path)
    keys = list(E.keys())
    random.seed(seed)

    id2keys = defaultdict(list)
    for k in keys:
        pid = guess_id_from_key(k)
        id2keys[pid].append(k)

    # positive pairs
    pos_pairs = []
    for pid, lst in id2keys.items():
        if len(lst) < 2:
            continue
        seen = set()
        for _ in range(pos_per_id * 2):  # sampling beberapa kali
            a, b = random.sample(lst, 2)
            key = tuple(sorted([a, b]))
            if key in seen:
                continue
            seen.add(key)
            pos_pairs.append((a, b, 1))
            if len(seen) >= pos_per_id:
                break

    # negative pairs (≈ jumlah positive)
    neg_pairs = []
    ids = list(id2keys.keys())
    target = len(pos_pairs)
    if len(ids) >= 2:
        while len(neg_pairs) < target:
            id1, id2 = random.sample(ids, 2)
            a = random.choice(id2keys[id1])
            b = random.choice(id2keys[id2])
            neg_pairs.append((a, b, 0))

    pairs = pos_pairs + neg_pairs
    random.shuffle(pairs)
    return pairs, id2keys


# =============== 04 - Evaluasi Verifikasi ===============
def tar_at_far(fpr: np.ndarray, tpr: np.ndarray, target: float) -> float:
    target = float(target)
    if target <= fpr[0]:
        return float(tpr[0])
    if target >= fpr[-1]:
        return float(tpr[-1])
    return float(np.interp(target, fpr, tpr))

def coerce_label(s: str) -> Optional[int]:
    t = s.strip().lower()
    if t.isdigit():
        return int(t)
    try:
        fv = float(t)
        if fv in (0.0, 1.0):
            return int(fv)
    except Exception:
        pass
    truthy = {"1", "true", "yes", "y", "same", "match", "positive"}
    falsy  = {"0", "false", "no", "n", "diff", "different", "mismatch", "negative", "not_same"}
    if t in truthy:
        return 1
    if t in falsy:
        return 0
    return None

def read_pairs_raw(pairs_file: Path):
    rows, labels, skipped = [], [], 0
    with pairs_file.open(newline="", encoding="utf-8") as f:
        r = csv.reader(f)
        for row in r:
            if not row or len(row) < 3:
                skipped += 1
                continue
            p1, p2, labraw = row[0].strip(), row[1].strip(), row[2]
            lab = coerce_label(labraw)
            if lab is None:
                skipped += 1
                continue
            rows.append((p1, p2))
            labels.append(lab)
    return rows, np.array(labels, dtype=int), skipped

# ---- normalisasi path (case-insensitive) ----
def _norm_once(p: str) -> str:
    p = p.strip().replace("\\", "/")
    while p.startswith("./"):
        p = p[2:]
    # normalisasi ekstensi
    if "." in p:
        head, _, ext = p.rpartition(".")
        if ext:
            p = head + "." + ext.lower()
    # hilangkan prefix umum
    for pref in ("dataset_raw/", "crops_112/", "crops_224/"):
        if p.lower().startswith(pref):
            p = p[len(pref):]
    return p

def norm_path_casefold(p: str) -> str:
    return _norm_once(p).casefold()

def build_indices_casefold(emb_keys: List[str]):
    lower_to_orig: Dict[str, str] = {}
    fname_to_keys: Dict[str, List[str]] = defaultdict(list)
    suffix_maps = {2: {}, 3: {}, 4: {}, 5: {}}

    for k in emb_keys:
        k_norm = _norm_once(k)
        kl = k_norm.casefold()
        lower_to_orig[kl] = k

        parts = k_norm.split("/")
        if parts:
            fname_to_keys[parts[-1].casefold()].append(k)
        for L in (2, 3, 4, 5):
            if len(parts) >= L:
                suf = "/".join(parts[-L:]).casefold()
                suffix_maps[L][suf] = k

    return lower_to_orig, fname_to_keys, suffix_maps

def resolve_to_key(raw: str, lower_to_orig, fname_to_keys, suffix_maps) -> Optional[str]:
    pl = norm_path_casefold(raw)

    # 1) full match
    if pl in lower_to_orig:
        return lower_to_orig[pl]

    # 2) suffix match 5..2 segmen
    parts = pl.split("/")
    for L in (5, 4, 3, 2):
        if len(parts) >= L:
            suf = "/".join(parts[-L:])
            if suf in suffix_maps[L]:
                return suffix_maps[L][suf]

    # 3) filename unik
    fname = parts[-1] if parts else pl
    cands = fname_to_keys.get(fname, [])
    if len(cands) == 1:
        return cands[0]

    # 4) filename + parent dir hint
    if len(cands) > 1 and len(parts) >= 2:
        parent_hint = parts[-2]
        filtered = []
        for k in cands:
            k_parts = _norm_once(k).split("/")
            if len(k_parts) >= 2 and k_parts[-2].casefold() == parent_hint:
                filtered.append(k)
        if len(filtered) == 1:
            return filtered[0]

    return None


# =============== 05 - Metrics helper ===============
def safe_div(num: float, den: float) -> Optional[float]:
    return (num / den) if den != 0 else None

def fmt(x: Optional[float]) -> Optional[float]:
    if x is None:
        return None
    return float(f"{x:.8f}")


# =============== Pipeline utama (Euclidean) ===============
def eval_verification_pipeline(
    embeds_path: Path,
    outdir: Path,
    pos_per_id: int = 50,
    seed: int = 42,
    use_pairs: str = "auto",           # "auto" atau "given"
    pairs_given: Optional[Path] = None,
    fixed_thr: float = 0.6,            # threshold COSINE (akan dikonversi ke L2)
    roc_filename: Optional[str] = None,
):
    """
    Jalankan end-to-end untuk satu file .npz (Euclidean).
    - Semua output disimpan di eval_verif_euclidean/<algo>/ dengan nama file bertag threshold cosine.
    - CSV agregat semua algoritma: eval_verif_euclidean/metrics_all.csv
    """
    eval_root = find_eval_root(outdir)
    algo = algo_name_from_embeds(embeds_path)
    thr_tag = thr_to_tag(fixed_thr)               # label file pakai nilai COSINE yang kamu input
    thr_used = cosine_to_l2_thr(fixed_thr)        # AMBANG L2 sesungguhnya untuk keputusan

    # Folder algoritma
    algo_dir = eval_root / algo
    algo_dir.mkdir(parents=True, exist_ok=True)

    metrics_all_csv = eval_root / "metrics_all_euclidean.csv"

    # --- SELALU: generate pairs otomatis dari embeddings
    pairs_auto, id2keys = build_pairs_from_embeds(str(embeds_path), pos_per_id=pos_per_id, seed=seed)
    pairs_auto_path = algo_dir / f"{algo}_pairs_auto_{thr_tag}.csv"
    with pairs_auto_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["path_img1", "path_img2", "is_same"])
        for a, b, y in pairs_auto:
            w.writerow([a, b, y])

    n_ids = sum(1 for _, v in id2keys.items() if len(v) >= 1)
    n_ids_pos = sum(1 for _, v in id2keys.items() if len(v) >= 2)
    print({
        "pairs_auto": str(pairs_auto_path),
        "N_pairs_auto": len(pairs_auto),
        "N_ids": n_ids,
        "N_ids_pos>=2": n_ids_pos
    })

    # --- pilih sumber pairs untuk evaluasi
    if use_pairs == "given":
        if pairs_given is None or (not pairs_given.exists()):
            raise FileNotFoundError("use_pairs=given dipilih, tetapi --pairs-in tidak valid/ada.")
        pairs_eval = pairs_given
    else:
        pairs_eval = pairs_auto_path  # default: pakai yang otomatis

    # --- Evaluasi verifikasi
    EMB = np.load(str(embeds_path))
    emb_keys = list(EMB.keys())
    lower_to_orig, fname_to_keys, suffix_maps = build_indices_casefold(emb_keys)

    raw_pairs, y_true_all, skipped_header = read_pairs_raw(pairs_eval)

    fixed_pairs = []
    missing_samples = []
    skipped_not_found = 0
    for (p1_raw, p2_raw), lab in zip(raw_pairs, y_true_all):
        k1 = resolve_to_key(p1_raw, lower_to_orig, fname_to_keys, suffix_maps)
        k2 = resolve_to_key(p2_raw, lower_to_orig, fname_to_keys, suffix_maps)
        if (k1 is None) or (k2 is None):
            skipped_not_found += 1
            if len(missing_samples) < 30:
                missing_samples.append((p1_raw, k1, p2_raw, k2))
            continue
        fixed_pairs.append((k1, k2, lab))

    if len(fixed_pairs) == 0:
        report = algo_dir / f"{algo}_missing_paths_report_{thr_tag}.txt"
        with report.open("w", encoding="utf-8") as f:
            f.write("== DEBUG PATH REPORT ==\n")
            f.write(f"Total pairs (valid rows): {len(raw_pairs)}\n")
            f.write("All skipped because keys not found in embeddings.\n\n")
            f.write("Example embedding keys (<=30):\n")
            for i, k in enumerate(emb_keys[:30], 1):
                f.write(f"  {i:02d}. {k}\n")
            f.write("\nExample failing pairs (<=30):\n")
            for i, (p1, k1, p2, k2) in enumerate(missing_samples, 1):
                f.write(f"  {i:02d}. p1='{p1}' -> '{k1}', p2='{p2}' -> '{k2}'\n")
        raise RuntimeError(
            "Semua pasangan ter-skip setelah normalisasi (case-insensitive). "
            f"Detail ditulis ke: {report}"
        )

    pairs_fixed_path = algo_dir / f"{algo}_pairs_fixed_{thr_tag}.csv"
    with pairs_fixed_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["path_img1", "path_img2", "is_same"])
        for k1, k2, lab in fixed_pairs:
            w.writerow([k1, k2, int(lab)])

    # --- Hitung jarak L2 & skor untuk ROC (score = -distance agar "semakin besar semakin mirip")
    y_true = np.array([lab for _, _, lab in fixed_pairs], dtype=int)
    dists = np.array([float(np.linalg.norm(EMB[k1] - EMB[k2])) for k1, k2, _ in fixed_pairs], dtype=float)
    scores = -dists

    # ROC & metrik
    fpr, tpr, thr_scores = roc_curve(y_true, scores)
    roc_auc = float(auc(fpr, tpr))
    tar1e2 = tar_at_far(fpr, tpr, 1e-2)
    tar1e3 = tar_at_far(fpr, tpr, 1e-3)
    tar1e4 = tar_at_far(fpr, tpr, 1e-4)
    eer = float(1.0 - np.max(tpr - fpr))

    # Ambang (distance) yang ekuivalen di FAR=1e-4 (konversi balik dari skor ke jarak)
    if len(thr_scores) > 0:
        idx_1e4 = np.searchsorted(fpr, 1e-4, side="right") - 1
        idx_1e4 = int(np.clip(idx_1e4, 0, len(thr_scores) - 1))
        thr_1e4_dist = float(-thr_scores[idx_1e4])  # balik ke jarak (karena score=-dist)
    else:
        thr_1e4_dist = 0.0

    # Confusion matrix dengan threshold L2 (semakin kecil semakin mirip)
    pred = (dists <= thr_used).astype(int)
    TP = int(((pred == 1) & (y_true == 1)).sum())
    TN = int(((pred == 0) & (y_true == 0)).sum())
    FP = int(((pred == 1) & (y_true == 0)).sum())
    FN = int(((pred == 0) & (y_true == 1)).sum())

    # Simpan ROC ke folder algoritma
    default_roc_name = f"roc_{algo}.png"
    roc_base = roc_filename if roc_filename else default_roc_name
    roc_name = name_with_thr(roc_base, thr_tag)
    plt.figure()
    plt.plot(fpr, tpr, label=f"ROC (AUC={roc_auc:.5f})")
    plt.plot([0, 1], [0, 1], "--", linewidth=1)
    plt.xlabel("FAR (FPR)")
    plt.ylabel("TAR (TPR)")
    plt.title(f"ROC – {algo} (Euclidean)")
    plt.grid(True, linewidth=0.3)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(algo_dir / roc_name, dpi=150)
    plt.close()

    # Ringkasan verifikasi (JSON per algoritma)
    summary_ver = {
        "algo": algo,
        "metric": "euclidean_l2",
        "N_pairs_used": int(len(y_true)),
        "N_pairs_skipped_header": int(skipped_header),
        "N_pairs_skipped_not_found": int(skipped_not_found),
        "AUC": roc_auc,
        "EER": eer,
        "TAR@1e-2": tar1e2,
        "TAR@1e-3": tar1e3,
        "TAR@1e-4": tar1e4,
        "thr@1e-4(dist)": thr_1e4_dist,
        "thr_input_cosine": float(fixed_thr),
        "thr_used_l2": float(thr_used),
        "thr_source": "cosine->l2",
        "TP@thr": TP, "TN@thr": TN, "FP@thr": FP, "FN@thr": FN,
        "embeds": str(embeds_path),
        "pairs_input": str(pairs_eval),
        "pairs_fixed": str(pairs_fixed_path),
        "pairs_auto": str(pairs_auto_path),
        "roc_png": str(algo_dir / roc_name),
        "use_pairs": ("given" if pairs_eval == pairs_given else "auto"),
    }
    ver_json_path = algo_dir / name_with_thr(f"{algo}_results_verification.json", thr_tag)
    ver_json_path.write_text(json.dumps(summary_ver, indent=2), encoding="utf-8")
    print("\n== VERIFICATION SUMMARY ==")
    print(json.dumps(summary_ver, indent=2))

    # Metrik akhir + JSON per algoritma
    total = TP + TN + FP + FN
    accuracy = safe_div(TP + TN, total)
    precision = safe_div(TP, TP + FP)
    recall = safe_div(TP, TP + FN)
    f1 = None if (precision is None or recall is None or (precision + recall) == 0) else \
        2.0 * (precision * recall) / (precision + recall)

    summary_met = {
        "algo": algo,
        "metric": "euclidean_l2",
        "TP": TP, "TN": TN, "FP": FP, "FN": FN,
        "total_pairs": total,
        "threshold_used": float(thr_used),
        "threshold_source": "cosine->l2",
        "accuracy": fmt(accuracy),
        "precision": fmt(precision),
        "recall": fmt(recall),
        "f1": fmt(f1),
        # metadata (tetap di JSON; TIDAK dimasukkan ke CSV agregat)
        "embeds": summary_ver.get("embeds"),
        "pairs_fixed": summary_ver.get("pairs_fixed"),
        "pairs_auto": summary_ver.get("pairs_auto"),
        "roc_png": summary_ver.get("roc_png"),
        "AUC": summary_ver.get("AUC"),
        "EER": summary_ver.get("EER"),
        "TAR@1e-4": summary_ver.get("TAR@1e-4"),
        "TAR@1e-3": summary_ver.get("TAR@1e-3"),
        "TAR@1e-2": summary_ver.get("TAR@1e-2"),
    }
    met_json_path = algo_dir / name_with_thr(f"{algo}_results_metrics.json", thr_tag)
    met_json_path.write_text(json.dumps(summary_met, indent=2), encoding="utf-8")
    print("\n== METRICS SUMMARY ==")
    print(json.dumps(summary_met, indent=2))

    # === SATU CSV agregat untuk SEMUA algoritma ===
    # Kolom path & metadata tidak dimasukkan; dan TIDAK ADA kolom 'metrik'
    metrics_all_csv_fieldnames = [
        "algo",
        "accuracy", "precision", "recall", "f1",
        "TP", "TN", "FP", "FN", "total_pairs",
        "threshold_used",
        "AUC", "EER", "TAR@1e-4", "TAR@1e-3", "TAR@1e-2",
    ]
    write_header = not metrics_all_csv.exists()
    with metrics_all_csv.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=metrics_all_csv_fieldnames)
        if write_header:
            w.writeheader()
        row = {k: (summary_met.get(k) if k in summary_met else summary_ver.get(k)) for k in metrics_all_csv_fieldnames}
        w.writerow(row)

    return summary_ver, summary_met


def main():
    ap = argparse.ArgumentParser(
        description="Evaluasi verifikasi + metrik (Euclidean/L2) dengan output per-algoritma & CSV agregat."
    )
    ap.add_argument("--embeds", required=True, help="Path ke file .npz hasil embedding.")
    ap.add_argument("--outdir", default=EVAL_ROOT_NAME, help=f"Folder basis output (default: {EVAL_ROOT_NAME}).")

    # generate pairs otomatis (SELALU dibuat)
    ap.add_argument("--pos-per-id", type=int, default=50, help="Jumlah pairs positif/ID saat generate (default: 50).")
    ap.add_argument("--seed", type=int, default=42, help="Seed random (default: 42).")

    # pilih sumber evaluasi
    ap.add_argument("--use-pairs", choices=["auto", "given"], default="auto",
                    help="Pilih sumber pairs untuk evaluasi: 'auto' (default) atau 'given' (--pairs-in).")
    ap.add_argument("--pairs-in", default="", help="CSV pairs yang sudah ada (dipakai jika --use-pairs given).")

    # evaluasi
    ap.add_argument("--fixed-thr", type=float, default=0.6,
                    help="Threshold COSINE (akan dikonversi otomatis ke L2).")
    ap.add_argument("--roc", default="", help="Nama file ROC PNG (default: otomatis: roc_<algo>.png).")

    args = ap.parse_args()

    embeds_path = Path(args.embeds)
    if not embeds_path.exists():
        raise FileNotFoundError(f"Embeddings tidak ditemukan: {embeds_path}")

    outdir = Path(args.outdir)
    pairs_given = Path(args.pairs_in) if args.pairs_in else None
    roc_file = args.roc if args.roc else None

    eval_verification_pipeline(
        embeds_path=embeds_path,
        outdir=outdir,
        pos_per_id=args.pos_per_id,
        seed=args.seed,
        use_pairs=args.use_pairs,
        pairs_given=pairs_given,
        fixed_thr=args.fixed_thr,
        roc_filename=roc_file,
    )


if __name__ == "__main__":
    main()
