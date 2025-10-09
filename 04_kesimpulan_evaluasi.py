# 04_kesimpulan_evaluasi.py
# Rekap "pemenang akurasi" per kategori threshold untuk Cosine & Euclidean
# Sumber:
#   - eval_verif_cosine/metrics_all.csv
#   - eval_verif_euclidean/metrics_all.csv
# Hasil:
#   - kesimpulan_evaluasi.csv (di folder kerja saat dijalankan)
#
# Perubahan:
# - Tidak ada lagi argumen --threshold-round. Kategori threshold diambil dari nilai string di CSV.
# - Hapus kolom 13–17 (AUC, EER, TAR@1e-4, TAR@1e-3, TAR@1e-2) dari output.
# - Jika ada seri akurasi tertinggi di satu kategori, tulis semua pemenang (tanpa tie-break AUC).

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

def _to_int(v) -> int:
    try:
        if v is None or v == "":
            return 0
        return int(float(v))
    except Exception:
        return 0

# --------- I/O ---------
def read_metrics_csv(path: Path, similarity_label: str) -> List[Dict]:
    """
    Baca metrics_all.csv dan kembalikan list of dict dengan kolom-kolom penting.
    'similarity' diisi dengan similarity_label ("Cosine" / "Euclidean").
    Baris tanpa 'accuracy' atau 'threshold_used' akan diskip.
    Kategori threshold akan memakai string asli dari kolom 'threshold_used' di CSV.
    """
    rows: List[Dict] = []
    if not path.exists():
        return rows

    with path.open("r", newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            acc = _to_float(row.get("accuracy"))
            thr_float = _to_float(row.get("threshold_used"))
            thr_raw = (row.get("threshold_used") or "").strip()

            if acc is None or thr_float is None or thr_raw == "":
                # baris tidak valid untuk perbandingan
                continue

            item = {
                "similarity": similarity_label,
                "algo": row.get("algo", "").strip(),
                "accuracy": acc,
                "precision": _to_float(row.get("precision")),
                "recall": _to_float(row.get("recall")),
                "f1": _to_float(row.get("f1")),
                "TP": _to_int(row.get("TP")),
                "TN": _to_int(row.get("TN")),
                "FP": _to_int(row.get("FP")),
                "FN": _to_int(row.get("FN")),
                "total_pairs": _to_int(row.get("total_pairs")),
                # Simpan dua versi threshold: string asli (untuk kategori/output) & float (untuk sorting)
                "threshold_used": thr_float,
                "threshold_str": thr_raw,
                # Kolom metrik lain boleh ada di CSV, tapi tidak digunakan lagi untuk output/tie-break
                "AUC": _to_float(row.get("AUC")),
                "EER": _to_float(row.get("EER")),
                "TAR@1e-4": _to_float(row.get("TAR@1e-4")),
                "TAR@1e-3": _to_float(row.get("TAR@1e-3")),
                "TAR@1e-2": _to_float(row.get("TAR@1e-2")),
            }
            rows.append(item)
    return rows

# --------- Core logic ---------
def winners_by_exact_threshold(rows: List[Dict]) -> List[Dict]:
    """
    Pilih semua "pemenang" (akurasi tertinggi) untuk setiap threshold persis seperti di CSV (pakai string).
    Grouping & seleksi dilakukan PER similarity (Cosine/Euclidean) terpisah.
    Seri akurasi: semua ditulis (tanpa tie-break AUC).
    """
    # group key: (similarity, threshold_str)
    groups: Dict[Tuple[str, str], List[Dict]] = {}
    for r in rows:
        sim = r["similarity"]
        thr_str = r.get("threshold_str") or (str(r.get("threshold_used")) if r.get("threshold_used") is not None else "")
        key = (sim, thr_str)
        groups.setdefault(key, []).append(r)

    winners: List[Dict] = []
    for key, items in groups.items():
        # Cari akurasi maksimum di grup
        max_acc = None
        for it in items:
            a = it.get("accuracy")
            if a is None:
                continue
            if (max_acc is None) or (a > max_acc):
                max_acc = a

        if max_acc is None:
            continue

        # Ambil semua yang akurasinya "setara" (pakai toleransi floating)
        chosen = [it for it in items if it.get("accuracy") is not None and math.isclose(it["accuracy"], max_acc, rel_tol=0.0, abs_tol=1e-12)]

        # Untuk urutan yang rapi, sort pemenang berdasarkan nama algo
        chosen.sort(key=lambda x: (x.get("algo", "")))

        # Masukkan ke winners dengan kolom yang dipertahankan
        for best in chosen:
            winners.append({
                "similarity": key[0],
                "threshold": key[1],  # string asli dari CSV
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
                # simpan juga angka float threshold sebagai bantuan sorting akhir (tidak ditulis ke CSV)
                "_thr_float": best.get("threshold_used"),
            })

    # Urutkan output: similarity (Cosine dulu), lalu threshold naik (pakai float jika bisa), lalu nama algo
    order_sim = {"Cosine": 0, "Euclidean": 1}

    def _thr_sort_key(x: Dict):
        thr_f = x.get("_thr_float")
        if isinstance(thr_f, float):
            return thr_f
        # fallback ke parsing dari string; kalau gagal, pakai 1e9 biar di akhir
        try:
            return float(x.get("threshold", "inf"))
        except Exception:
            return float("inf")

    winners_sorted = sorted(
        winners,
        key=lambda x: (order_sim.get(x["similarity"], 99), _thr_sort_key(x), x.get("algo", ""))
    )

    # Buang key bantu _thr_float sebelum ditulis
    for it in winners_sorted:
        if "_thr_float" in it:
            del it["_thr_float"]

    return winners_sorted

def main():
    ap = argparse.ArgumentParser(
        description="Rekap algoritma dengan akurasi tertinggi per kategori threshold (berdasarkan nilai asli di CSV) untuk Cosine & Euclidean."
    )
    ap.add_argument("--cosine-csv", default=str(Path("eval_verif_cosine") / "metrics_all.csv"),
                    help="Path ke metrics_all.csv (Cosine). Default: eval_verif_cosine/metrics_all.csv")
    ap.add_argument("--euclidean-csv", default=str(Path("eval_verif_euclidean") / "metrics_all.csv"),
                    help="Path ke metrics_all.csv (Euclidean). Default: eval_verif_euclidean/metrics_all.csv")
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
        winners += winners_by_exact_threshold(rows_cos)
    if rows_euc:
        winners += winners_by_exact_threshold(rows_euc)

    if not winners:
        sys.exit("[ERROR] Tidak ada pemenang terdeteksi (mungkin kolom accuracy/threshold_used kosong).")

    out_path = Path(args.out)
    # Kolom 13–17 (AUC, EER, TAR@...) DIHAPUS
    fieldnames = [
        "similarity",   # "Cosine" / "Euclidean"
        "threshold",    # kategori persis dari CSV (string)
        "algo",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "TP", "TN", "FP", "FN",
        "total_pairs",
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
        try:
            acc_str = f"{row['accuracy']:.6f}" if row.get("accuracy") is not None else "NA"
        except Exception:
            acc_str = "NA"
        print(f" - {row['similarity']} @ {row['threshold']}: {row['algo']} (acc={acc_str})")

if __name__ == "__main__":
    main()
