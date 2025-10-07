# 01_merapikan_dataset.py
import os
import sys
import cv2
import json
import shutil
import random
import argparse
import re
from pathlib import Path
from typing import Dict, List
from tqdm import tqdm

from insightface.app import FaceAnalysis
from insightface.utils.face_align import norm_crop

# Root kumpulan dataset
DATASETS_DIR = Path("dataset")

# Ekstensi gambar yang didukung
SUPPORTED_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff")

# Ukuran asli align yang didukung norm_crop; selain ini akan di-resize
_SUPPORTED_ALIGN_SIZES = {112, 224}


# ---------- Utils ----------
def collect_images(root: Path) -> List[Path]:
    files = []
    for dp, _, fs in os.walk(root):
        for f in fs:
            if f.lower().endswith(SUPPORTED_EXT):
                files.append(Path(dp) / f)
    return files


def largest_face(faces):
    if not faces:
        return None
    return max(faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]))


def align_to_any_size(img, kps, target_size: int):
    """Align pakai norm_crop 112/224, lalu resize ke target_size jika perlu."""
    if target_size in _SUPPORTED_ALIGN_SIZES:
        return norm_crop(img, kps, image_size=target_size)
    base = 112 if target_size <= 168 else 224
    aligned = norm_crop(img, kps, image_size=base)
    interp = cv2.INTER_AREA if target_size < base else cv2.INTER_CUBIC
    return cv2.resize(aligned, (target_size, target_size), interpolation=interp)


def copy_file(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def safe_filename(name: str) -> str:
    """Ganti karakter aneh agar aman di semua OS (biarkan . _ -)."""
    return re.sub(r"[^A-Za-z0-9._\-]+", "_", name)


def ensure_unique_path(p: Path) -> Path:
    """Jika path sudah ada, tambahkan akhiran _1, _2, ... hingga unik."""
    if not p.exists():
        return p
    stem, suf = p.stem, p.suffix
    i = 1
    while True:
        cand = p.with_name(f"{stem}_{i}{suf}")
        if not cand.exists():
            return cand
        i += 1


def write_json(p: Path, obj: dict):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2), encoding="utf-8")


# ---------- Main ----------
def main():
    ap = argparse.ArgumentParser(
        description="Merapikan dataset: align (crop) dahulu, lalu opsional mengorganisir ke gallery/probe."
    )
    ap.add_argument(
        "--dataset-name", required=True,
        help="Nama folder dataset sumber di dalam folder 'dataset'. Contoh: 'Sewa' -> dataset/Sewa"
    )
    ap.add_argument(
        "--size", type=int, required=True,
        help="Ukuran sisi output (tanpa 'x'). Contoh: 112, 150, 160."
    )
    ap.add_argument(
        "--sort", choices=["on", "off", "of"], default="on",
        help="on  = buat folder gallery (per-ID) & probe (kini juga per-ID). "
             "off/of = hanya align, tanpa struktur gallery/probe."
    )

    # === Mutual exclusive: --probe-per-id vs --gallery-per-id ===
    # Keduanya opsional. Jika dua-duanya None -> default: probe-per-id = 1 (perilaku lama).
    ap.add_argument(
        "--probe-per-id", type=int, default=None,
        help="Maksimal JUMLAH file per-ID ke 'probe/<ID>/' saat --sort on. "
             "Sisa file per-ID masuk ke 'gallery/<ID>/'. "
             "Mutually exclusive dgn --gallery-per-id. Gunakan 0 untuk semua ke gallery."
    )
    ap.add_argument(
        "--gallery-per-id", type=int, default=None,
        help="Maksimal JUMLAH file per-ID ke 'gallery/<ID>/' saat --sort on. "
             "Sisa file per-ID masuk ke 'probe/<ID>/'. "
             "Mutually exclusive dgn --probe-per-id. Gunakan 0 untuk semua ke probe."
    )

    ap.add_argument(
        "--seed", type=int, default=42,
        help="Seed random untuk pemilihan file. Default: 42"
    )
    ap.add_argument(
        "--det-size", type=int, default=640,
        help="Ukuran deteksi wajah insightface (per sisi). Default: 640"
    )

    args = ap.parse_args()
    random.seed(args.seed)

    # Validasi eksklusivitas
    if args.probe_per_id is not None and args.gallery_per_id is not None:
        sys.exit("[ERROR] --probe-per-id dan --gallery-per-id tidak boleh digunakan bersamaan.")

    # Tetapkan default bila dua-duanya None (pertahankan perilaku lama)
    if args.probe_per_id is None and args.gallery_per_id is None:
        args.probe_per_id = 1  # default lama

    # Normalisasi nilai sort (terima 'of' sebagai alias 'off')
    sort_mode = "off" if args.sort in ("off", "of") else "on"

    # Validasi dataset sumber
    dataset_src = DATASETS_DIR / args.dataset_name
    if not dataset_src.exists() or not dataset_src.is_dir():
        sys.exit(f"[ERROR] Dataset sumber tidak ditemukan: {dataset_src}")

    # Folder output utama untuk dataset ini
    out_root = DATASETS_DIR / f"{args.dataset_name}_{args.size}"

    # Temp folder untuk hasil aligned (mirror struktur sumber)
    tmp_aligned = out_root / "_aligned"

    # Siapkan insightface (GPU jika ada, fallback CPU)
    try:
        app = FaceAnalysis(name="buffalo_l", providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        app.prepare(ctx_id=0, det_size=(args.det_size, args.det_size))
    except Exception as e:
        print("[Info] CUDA tidak tersedia / gagal inisialisasi, fallback ke CPU:", e)
        app = FaceAnalysis(name="buffalo_l", providers=['CPUExecutionProvider'])
        app.prepare(ctx_id=-1, det_size=(args.det_size, args.det_size))

    # Kumpulkan semua gambar dari dataset sumber
    all_imgs = collect_images(dataset_src)

    # Statistik proses align
    align_stat = {"total": 0, "aligned": 0, "no_face": 0, "failed": 0}
    debug_errs = []

    # --- Tahap 1: Align/crop ke tmp_aligned ---
    for src_img in tqdm(all_imgs, desc="Aligning"):
        align_stat["total"] += 1
        try:
            img = cv2.imread(str(src_img))
            if img is None:
                align_stat["failed"] += 1
                if len(debug_errs) < 5:
                    debug_errs.append("cv2.imread -> None")
                continue

            faces = app.get(img)
            if not faces:
                align_stat["no_face"] += 1
                continue

            face = largest_face(faces)
            if face is None or getattr(face, "kps", None) is None:
                align_stat["no_face"] += 1
                continue

            aligned = align_to_any_size(img, face.kps, target_size=args.size)

            rel = src_img.relative_to(dataset_src)
            dst_img = tmp_aligned / rel
            dst_img.parent.mkdir(parents=True, exist_ok=True)
            ok = cv2.imwrite(str(dst_img), aligned)
            if not ok:
                align_stat["failed"] += 1
                if len(debug_errs) < 5:
                    debug_errs.append("cv2.imwrite -> False")
                continue

            align_stat["aligned"] += 1

        except Exception as e:
            align_stat["failed"] += 1
            if len(debug_errs) < 5:
                debug_errs.append(f"{type(e).__name__}: {e}")

    # --- sort off: selesai setelah align; pindahkan hasil, tanpa gallery/probe ---
    if sort_mode == "off":
        # Bersihkan out_root kecuali _aligned jika sudah ada
        if out_root.exists():
            for child in out_root.iterdir():
                if child.name != "_aligned":
                    if child.is_dir():
                        shutil.rmtree(child)
                    else:
                        child.unlink(missing_ok=True)
        else:
            out_root.mkdir(parents=True, exist_ok=True)

        # Pindahkan isi _aligned ke out_root dan hapus _aligned
        if tmp_aligned.exists():
            for p in tmp_aligned.iterdir():
                target = out_root / p.name
                if target.exists():
                    if target.is_dir():
                        shutil.rmtree(target)
                    else:
                        target.unlink(missing_ok=True)
                shutil.move(str(p), str(target))
            shutil.rmtree(tmp_aligned, ignore_errors=True)

        # Ringkasan
        print("\n=== RINGKASAN ===")
        print(f"Dataset sumber  : {dataset_src}")
        print(f"Output root     : {out_root}")
        print(f"Sort            : {sort_mode}")
        print(f"Size (crop)     : {args.size}x{args.size}")
        print(f"Seed            : {args.seed}")
        print(f"Align stat      : {align_stat}")
        if debug_errs:
            print("\n[DEBUG] Contoh error (maks 5):")
            for s in debug_errs:
                print(f" - {s}")
        return

    # --- sort on: buat struktur (gallery per-ID, probe per-ID) ---
    gallery_dir = out_root / "gallery"
    probe_dir   = out_root / "probe"    # SEKARANG: per-ID (bukan flat)
    gallery_dir.mkdir(parents=True, exist_ok=True)
    probe_dir.mkdir(parents=True, exist_ok=True)

    # Kumpulkan hasil aligned per-ID (diasumsikan struktur top-level = ID)
    id2imgs: Dict[str, List[Path]] = {}
    if tmp_aligned.exists():
        for pid_dir in sorted([d for d in (tmp_aligned).glob("*") if d.is_dir()]):
            pid = pid_dir.name
            imgs = []
            for dp, _, fs in os.walk(pid_dir):
                for f in fs:
                    if f.lower().endswith(SUPPORTED_EXT):
                        imgs.append(Path(dp) / f)
            if imgs:
                id2imgs[pid] = sorted(imgs)

    wrote_gallery = 0
    wrote_probe = 0

    # Penentuan mode pembagian
    mode = "probe-per-id" if (args.probe_per_id is not None) else "gallery-per-id"
    P = max(0, int(args.probe_per_id)) if args.probe_per_id is not None else None
    G = max(0, int(args.gallery_per_id)) if args.gallery_per_id is not None else None

    # --- Pembagian file ---
    for pid, imgs in id2imgs.items():
        if not imgs:
            continue
        random.shuffle(imgs)

        if mode == "probe-per-id":
            # Maks P file ke PROBE/<ID>/, sisanya ke GALLERY/<ID>/
            take_p = min(P, len(imgs))
            probe_choice = imgs[:take_p]
            rest = imgs[take_p:]

            # PROBE per-ID (label tersimpan di struktur folder)
            for src_p in probe_choice:
                dst_p = probe_dir / pid / src_p.name
                dst_p = ensure_unique_path(dst_p)
                copy_file(src_p, dst_p)
                wrote_probe += 1

            # GALLERY per-ID
            for src_p in rest:
                dst_p = gallery_dir / pid / src_p.name
                dst_p = ensure_unique_path(dst_p)
                copy_file(src_p, dst_p)
                wrote_gallery += 1

        else:  # mode == "gallery-per-id"
            # Maks G file ke GALLERY/<ID>/, sisanya ke PROBE/<ID>/
            take_g = min(G, len(imgs))
            gallery_choice = imgs[:take_g]
            rest = imgs[take_g:]

            # GALLERY per-ID
            for src_p in gallery_choice:
                dst_p = gallery_dir / pid / src_p.name
                dst_p = ensure_unique_path(dst_p)
                copy_file(src_p, dst_p)
                wrote_gallery += 1

            # PROBE per-ID (label tersimpan di struktur folder)
            for src_p in rest:
                dst_p = probe_dir / pid / src_p.name
                dst_p = ensure_unique_path(dst_p)
                copy_file(src_p, dst_p)
                wrote_probe += 1

    # Hapus tmp aligned
    shutil.rmtree(tmp_aligned, ignore_errors=True)

    # Ringkasan
    print("\n=== RINGKASAN ===")
    print(f"Dataset sumber  : {dataset_src}")
    print(f"Output root     : {out_root}")
    print(f"Sort            : {sort_mode}")
    print(f"Size (crop)     : {args.size}x{args.size}")
    print(f"Seed            : {args.seed}")
    if mode == "probe-per-id":
        print(f"Probe/ID (maks) : {P} file (sisanya -> gallery)")
    else:
        print(f"Gallery/ID (maks): {G} file (sisanya -> probe)")
    print(f"Wrote gallery   : {wrote_gallery}")
    print(f"Wrote probe     : {wrote_probe}")
    print(f"Align stat      : {align_stat}")
    if debug_errs:
        print("\n[DEBUG] Contoh error (maks 5):")
        for s in debug_errs:
            print(f" - {s}")


if __name__ == "__main__":
    main()
