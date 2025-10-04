# 01_merapikan_dataset.py
# Tahap terpadu:
#   1) ALIGN & CROP semua gambar di dataset/raw -> dataset/fix_<size>
#   2) (Opsional) ORGANIZE:
#        - both    : gallery (1 gambar/ID), probe (sisa)
#        - gallery : gallery (1 gambar/ID), probe dibuat tapi kosong
#        - none    : tanpa pengorganisiran gallery/probe (struktur asli dipertahankan)
#
# Contoh:
#   python 01_merapikan_dataset.py --size 112x112 --mode both
#   python 01_merapikan_dataset.py --size 160 --mode gallery
#   python 01_merapikan_dataset.py --size 150x150 --mode none
#
# Dependensi:
#   pip install insightface onnxruntime opencv-python tqdm

import os
import re
import cv2
import sys
import glob
import math
import shutil
import random
import argparse
from pathlib import Path
from collections import defaultdict
from typing import Tuple, List, Dict

from tqdm import tqdm

# InsightFace (deteksi & align)
from insightface.app import FaceAnalysis
from insightface.utils.face_align import norm_crop


# ---------------------------
# Util parsing & path
# ---------------------------
def parse_size(s: str) -> int:
    """
    Terima '112x112' atau '112' dan kembalikan integer 112.
    Pastikan square & >0.
    """
    s = s.lower().strip()
    if "x" in s:
        a, b = s.split("x", 1)
        w = int(re.sub(r"\D", "", a))
        h = int(re.sub(r"\D", "", b))
        if w != h or w <= 0:
            raise ValueError("Ukuran harus square dan >0, contoh: 112x112 / 160x160 / 150x150.")
        return w
    else:
        n = int(re.sub(r"\D", "", s))
        if n <= 0:
            raise ValueError("Ukuran harus >0, contoh: 112 / 150 / 160.")
        return n


def is_image_file(name: str) -> bool:
    exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    return name.lower().endswith(exts)


def rel_top_id(raw_root: Path, file_path: Path) -> str:
    """
    Ambil nama folder level-1 (ID) relatif dari raw_root.
    dataset/raw/<ID>/.../file.jpg -> kembalikan '<ID>'
    Jika dataset lebih dalam, ambil segmen top-level pertama.
    """
    rel = file_path.relative_to(raw_root)
    parts = rel.parts
    return parts[0] if len(parts) > 0 else "UNKNOWN"


def prepare_face_app(det_size: Tuple[int, int] = (640, 640), prefer_gpu: bool = True) -> FaceAnalysis:
    """
    Inisialisasi FaceAnalysis dengan fallback:
      - coba GPU (CUDA) + CPU
      - kalau gagal, pakai CPU
    """
    if prefer_gpu:
        try:
            app = FaceAnalysis(name="buffalo_l", providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
            app.prepare(ctx_id=0, det_size=det_size)
            print("[INFO] FaceAnalysis siap (CUDA + CPU).")
            return app
        except Exception as e:
            print("[INFO] CUDA tidak tersedia, fallback ke CPU:", e)

    app = FaceAnalysis(name="buffalo_l", providers=['CPUExecutionProvider'])
    app.prepare(ctx_id=-1, det_size=det_size)
    print("[INFO] FaceAnalysis siap (CPU).")
    return app


# ---------------------------
# Tahap 1: ALIGN & CROP
# ---------------------------
def align_and_crop_dataset(
    raw_root: Path,
    out_dir: Path,
    image_size: int,
    det_size: Tuple[int, int] = (640, 640),
    prefer_gpu: bool = True
) -> Dict[str, int]:
    """
    Scan semua gambar di raw_root, deteksi wajah terbesar, lalu norm_crop ke image_size.
    Simpan ke out_dir dengan path relatif sama.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    app = prepare_face_app(det_size=det_size, prefer_gpu=prefer_gpu)

    all_imgs: List[Path] = []
    for dp, _, files in os.walk(raw_root):
        for f in files:
            if is_image_file(f):
                all_imgs.append(Path(dp) / f)

    stats = {"total": 0, "aligned": 0, "no_face": 0, "failed": 0}
    pbar = tqdm(all_imgs, desc=f"[ALIGN {image_size}x{image_size}]")

    for p in pbar:
        stats["total"] += 1
        try:
            img = cv2.imread(str(p))
            if img is None:
                stats["failed"] += 1
                continue

            faces = app.get(img)
            if not faces:
                stats["no_face"] += 1
                continue

            # pilih wajah terbesar
            face = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]))
            aligned = norm_crop(img, face.kps, image_size=image_size)

            out_path = out_dir / p.relative_to(raw_root)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            ok = cv2.imwrite(str(out_path), aligned)
            if not ok:
                stats["failed"] += 1
                continue

            stats["aligned"] += 1

        except Exception:
            stats["failed"] += 1

    print(f"[DONE] Align/Crop: {stats} -> {out_dir}")
    return stats


# ---------------------------
# Tahap 2: ORGANIZE (gallery/probe)
# ---------------------------
def organize_gallery_probe_from_aligned(
    aligned_root: Path,
    final_root: Path,
    mode: str,
    seed: int = 42,
    gallery_per_id: int = 1
) -> Dict[str, int]:
    """
    Dari hasil aligned_root, bentuk struktur akhir di final_root sesuai mode:
      - both   : gallery (1 gambar/ID acak), probe (sisa)
      - gallery: gallery (1 gambar/ID acak), probe kosong
      - none   : tidak dipanggil (caller akan skip)
    """
    random.seed(seed)
    final_root.mkdir(parents=True, exist_ok=True)

    # Kumpulkan file per ID (ID = top-level folder nama pertama)
    id2files: Dict[str, List[Path]] = defaultdict(list)
    for dp, _, files in os.walk(aligned_root):
        for f in files:
            if is_image_file(f):
                fp = Path(dp) / f
                pid = rel_top_id(aligned_root, fp)
                id2files[pid].append(fp)

    stats = {"ids": 0, "gallery": 0, "probe": 0, "skipped_ids": 0}
    gallery_dir = final_root / "gallery"
    probe_dir   = final_root / "probe"
    gallery_dir.mkdir(parents=True, exist_ok=True)
    probe_dir.mkdir(parents=True, exist_ok=True)  # dibuat walau mode=gallery (akan kosong)

    for pid, files in id2files.items():
        files = sorted(files)
        if len(files) == 0:
            stats["skipped_ids"] += 1
            continue

        stats["ids"] += 1
        # pilih 'gallery_per_id' acak/tetap (di sini 1)
        gk = min(gallery_per_id, len(files))
        gsel = random.sample(files, gk)

        # copy ke gallery
        for g in gsel:
            dst = gallery_dir / rel_path_without_top(aligned_root, g, keep_top=True)
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(g, dst)
            stats["gallery"] += 1

        if mode == "both":
            # sisa ke probe
            for p in files:
                if p in gsel:
                    continue
                dst = probe_dir / rel_path_without_top(aligned_root, p, keep_top=True)
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(p, dst)
                stats["probe"] += 1
        elif mode == "gallery":
            # probe dibiarkan kosong
            pass

    print(f"[DONE] Organize ({mode}) -> {final_root} | stats={stats}")
    return stats


def rel_path_without_top(base: Path, file_path: Path, keep_top: bool = True) -> Path:
    """
    Buat path relatif dari base.
    Jika keep_top=True, pertahankan segmen top-level (ID) agar hasilnya gallery/<ID>/nama.jpg dll.
    """
    rel = file_path.relative_to(base)
    parts = list(rel.parts)
    if not keep_top and len(parts) > 1:
        parts = parts[1:]
    return Path(*parts) if parts else Path(file_path.name)


# ---------------------------
# Main CLI
# ---------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Align & crop dataset, lalu optional organize ke gallery/probe."
    )
    ap.add_argument("--size", required=True,
                    help="Ukuran crop, contoh: 112x112 / 150x150 / 160 atau cukup '112'.")
    ap.add_argument("--mode", choices=["both", "gallery", "none"], default="both",
                    help="Cara organize: 'both' (gallery+probe), 'gallery' (probe kosong), 'none' (tanpa organize).")
    ap.add_argument("--raw-dir", default="dataset/raw",
                    help="Folder dataset mentah (default: dataset/raw).")
    ap.add_argument("--out-root", default="dataset",
                    help="Folder root keluaran (default: dataset). Akan dibuat subfolder fix_<N>.")
    ap.add_argument("--det-size", default="640x640",
                    help="Ukuran deteksi (detector) InsightFace, contoh: 640x640.")
    ap.add_argument("--seed", type=int, default=42, help="Seed untuk sampling gallery (default: 42).")
    ap.add_argument("--prefer-gpu", action="store_true",
                    help="Coba gunakan CUDA lebih dulu jika tersedia (default: CPU fallback).")
    ap.add_argument("--keep-temp", action="store_true",
                    help="Saat mode both/gallery: simpan folder aligned tmp (default: hapus setelah organize).")

    args = ap.parse_args()

    # Parse ukuran
    try:
        img_size = parse_size(args.size)
    except Exception as e:
        print(f"[ERROR] Ukuran tidak valid: {e}")
        sys.exit(1)

    # Parse det_size
    try:
        ds = args.det_size.lower().strip()
        if "x" in ds:
            a, b = ds.split("x", 1)
            det_w = int(re.sub(r"\D", "", a))
            det_h = int(re.sub(r"\D", "", b))
        else:
            det_w = det_h = int(re.sub(r"\D", "", ds))
        det_size = (det_w, det_h)
    except Exception as e:
        print(f"[ERROR] det-size tidak valid: {e}")
        sys.exit(1)

    raw_root = Path(args.raw_dir)
    if not raw_root.exists():
        # Backward-compat untuk struktur lama 'dataset_raw'
        alt = Path("dataset_raw")
        if alt.exists():
            print(f"[INFO] {raw_root} tidak ada. Memakai {alt} (kompatibilitas lama).")
            raw_root = alt
        else:
            print(f"[ERROR] Folder sumber tidak ditemukan: {raw_root}")
            sys.exit(1)

    out_root = Path(args.out_root)
    fix_dir = out_root / f"fix_{img_size}"
    fix_dir.mkdir(parents=True, exist_ok=True)

    # Jika mode none → langsung align ke fix_dir
    # Jika mode both/gallery → align dulu ke tmp, lalu organize → final ke fix_dir/(gallery,probe)
    if args.mode == "none":
        aligned_dir = fix_dir
    else:
        aligned_dir = fix_dir / "__aligned_tmp"
        aligned_dir.mkdir(parents=True, exist_ok=True)

    # Tahap 1: align & crop
    stats_align = align_and_crop_dataset(
        raw_root=raw_root,
        out_dir=aligned_dir,
        image_size=img_size,
        det_size=det_size,
        prefer_gpu=args.prefer_gpu
    )

    # Tahap 2: organize
    if args.mode == "none":
        print(f"[INFO] Mode 'none': hasil disimpan ke {fix_dir} tanpa struktur gallery/probe.")
        # selesai
    else:
        organize_gallery_probe_from_aligned(
            aligned_root=aligned_dir,
            final_root=fix_dir,
            mode=args.mode,
            seed=args.seed,
            gallery_per_id=1  # sama seperti program lama: 1 gambar/ID ke gallery
        )
        # Hapus tmp bila tidak diminta keep
        if not args.keep_temp:
            try:
                shutil.rmtree(aligned_dir)
            except Exception as e:
                print(f"[WARN] Gagal menghapus tmp: {aligned_dir} | {e}")

    # Ringkasan
    print("\n=== RINGKASAN ===")
    print(f"Raw input  : {raw_root}")
    print(f"Output root: {fix_dir}")
    print(f"Mode       : {args.mode}")
    print(f"Align stat : {stats_align}")
    if args.mode != "none":
        print("Struktur   :")
        print(f"  {fix_dir}/gallery/<ID>/*.jpg")
        print(f"  {fix_dir}/probe/<ID>/*.jpg (mode=gallery: kosong)")

if __name__ == "__main__":
    main()
