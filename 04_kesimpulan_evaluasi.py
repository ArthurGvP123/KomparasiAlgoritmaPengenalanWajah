# 04_kesimpulan_evaluasi.py
# Rekap "pemenang akurasi" per kategori threshold untuk Cosine & Euclidean
# Sumber:
#   - eval_verif_cosine/metrics_all.csv
#   - eval_verif_euclidean/metrics_all.csv
# Hasil:
#   - kesimpulan_evaluasi.csv (di folder kerja saat dijalankan)

import argparse
import csv
import math
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# --------- Helper parsing ---------
def _to_float(v) -> Optional[float]:
    try:
        if v is None or v == "":
            return None
        return float(v)
    except Exception:
        return None

def _fmt_thr(v: float, ndigits: int) -> str:
    # Format kategori threshold, mis. 0.6 -> "0.60" bila ndigits=2
    return f"{v:.{ndigits}f}"

# --------- I/O ---------
def read_metrics_csv(path: Path, similarity_label: str) -> List[Dict]:
    """
    Baca metrics_all.csv dan kembalikan list of dict dengan kolom-kolom penting.
    'similarity' diisi dengan similarity_label ("Cosine" / "Euclidean").
    Baris tanpa 'accuracy' atau 'threshold_used' akan diskip.
    """
    rows: List[Dict] = []
    if not path.exists():
        return rows

    with path.open("r", newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            acc = _to_float(row.get("accuracy"))
            thr = _to_float(row.get("threshold_used"))
            if acc is None or thr is None:
                # baris tidak valid untuk perbandingan
                continue

            # Ambil kolom yang mungkin ada; default ke None kalau tidak tersedia
            item = {
                "similarity": similarity_label,
                "algo": row.get("algo", ""),
                "accuracy": acc,
                "precision": _to_float(row.get("precision")),
                "recall": _to_float(row.get("recall")),
                "f1": _to_float(row.get("f1")),
                "TP": int(_to_float(row.get("TP")) or 0),
                "TN": int(_to_float(row.get("TN")) or 0),
                "FP": int(_to_float(row.get("FP")) or 0),
                "FN": int(_to_float(row.get("FN")) or 0),
                "total_pairs": int(_to_float(row.get("total_pairs")) or 0),
                "threshold_used": thr,
                "AUC": _to_float(row.get("AUC")),
                "EER": _to_float(row.get("EER")),
                "TAR@1e-4": _to_float(row.get("TAR@1e-4")),
                "TAR@1e-3": _to_float(row.get("TAR@1e-3")),
                "TAR@1e-2": _to_float(row.get("TAR@1e-2")),
            }
            rows.append(item)
    return rows

# --------- Core logic ---------
def best_by_threshold(rows: List[Dict], ndigits: int) -> List[Dict]:
    """
    Pilih satu "pemenang" (akurasi tertinggi) untuk setiap threshold (dibulatkan ke ndigits).
    Grouping & pemilihan dilakukan PER similarity (Cosine/EU) secara terpisah.
    Input rows diharapkan sudah untuk 1 similarity; kalau mixed, hasil tetap benar karena grouping pakai (similarity, thr_group).
    """
    groups: Dict[Tuple[str, str], List[Dict]] = {}
    for r in rows:
        sim = r["similarity"]
        thr_group = _fmt_thr(r["threshold_used"], ndigits)
        key = (sim, thr_group)
        groups.setdefault(key, []).append(r)

    winners: List[Dict] = []
    for key, items in groups.items():
        # pilih row dengan accuracy terbesar; tie-break: AUC lebih besar, lalu algo nama asc
        items_sorted = sorted(
            items,
            key=lambda x: (
                -(x["accuracy"] if x["accuracy"] is not None else -1.0),
                -(x["AUC"] if x["AUC"] is not None else -1.0),
                x.get("algo", "")
            ),
        )
        best = items_sorted[0]
        # letakkan kolom ringkas + kategori
        winners.append({
            "similarity": key[0],
            "threshold": key[1],
            "algo": best.get("algo", ""),
            "accuracy": best.get("accuracy"),
            "precision": best.get("precision"),
            "recall": best.get("recall"),
            "f1": best.get("f1"),
            "TP": best.get("TP"),
            "TN": best.get("TN"),
            "FP": best.get("FP"),
            "FN": best.get("FN"),
            "total_pairs": best.get("total_pairs"),
            "AUC": best.get("AUC"),
            "EER": best.get("EER"),
            "TAR@1e-4": best.get("TAR@1e-4"),
            "TAR@1e-3": best.get("TAR@1e-3"),
            "TAR@1e-2": best.get("TAR@1e-2"),
        })

    # Urutkan output: similarity (Cosine dulu), lalu threshold naik
    order_sim = {"Cosine": 0, "Euclidean": 1}
    winners_sorted = sorted(
        winners,
        key=lambda x: (order_sim.get(x["similarity"], 99), float(x["threshold"]))
    )
    return winners_sorted

def main():
    ap = argparse.ArgumentParser(
        description="Rekap algoritma dengan akurasi tertinggi per kategori threshold untuk Cosine & Euclidean."
    )
    ap.add_argument("--cosine-csv", default=str(Path("eval_verif_cosine") / "metrics_all.csv"),
                    help="Path ke metrics_all.csv (Cosine). Default: eval_verif_cosine/metrics_all.csv")
    ap.add_argument("--euclidean-csv", default=str(Path("eval_verif_euclidean") / "metrics_all.csv"),
                    help="Path ke metrics_all.csv (Euclidean). Default: eval_verif_euclidean/metrics_all.csv")
    ap.add_argument("--threshold-round", type=int, default=2,
                    help="Pembulatan threshold untuk kategori (mis. 2 -> 0.60). Default: 2")
    ap.add_argument("--out", default="kesimpulan_evaluasi.csv",
                    help="Nama file keluaran CSV di folder kerja. Default: kesimpulan_evaluasi.csv")
    args = ap.parse_args()

    cosine_path = Path(args.cosine_csv)
    euclid_path = Path(args.euclidean_csv)

    # Baca keduanya (jika ada)
    rows_cos = read_metrics_csv(cosine_path, similarity_label="Cosine")
    rows_euc = read_metrics_csv(euclid_path, similarity_label="Euclidean")

    if not rows_cos and not rows_euc:
        sys.exit("[ERROR] Tidak menemukan data pada kedua CSV. Periksa path atau jalankan evaluasi terlebih dulu.")

    winners: List[Dict] = []
    if rows_cos:
        winners += best_by_threshold(rows_cos, ndigits=args.threshold_round)
    if rows_euc:
        winners += best_by_threshold(rows_euc, ndigits=args.threshold_round)

    if not winners:
        sys.exit("[ERROR] Tidak ada pemenang terdeteksi (mungkin kolom accuracy/threshold_used kosong).")

    out_path = Path(args.out)
    fieldnames = [
        "similarity",   # "Cosine" / "Euclidean"
        "threshold",    # kategori, sudah dibulatkan (string)
        "algo",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "TP", "TN", "FP", "FN",
        "total_pairs",
        "AUC", "EER", "TAR@1e-4", "TAR@1e-3", "TAR@1e-2",
    ]
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in winners:
            w.writerow(row)

    # Tampilkan ringkas
    print(f"[OK] Tersimpan: {out_path}")
    print("Ringkas pemenang per kategori (top 10 tampilan):")
    for row in winners[:10]:
        print(f" - {row['similarity']} @ {row['threshold']}: {row['algo']} (acc={row['accuracy']:.6f})")

if __name__ == "__main__":
    main()
