# 01_merapikan_dataset.py
import os
import sys
import cv2
import glob
import json
import shutil
import random
import argparse
from pathlib import Path
from typing import Dict, List
from tqdm import tqdm

from insightface.app import FaceAnalysis
from insightface.utils.face_align import norm_crop

DATASETS_DIR = Path("dataset")
SUPPORTED_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff")
_SUPPORTED_ALIGN_SIZES = {112, 224}

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
    if target_size in _SUPPORTED_ALIGN_SIZES:
        return norm_crop(img, kps, image_size=target_size)
    base = 112 if target_size <= 168 else 224
    aligned = norm_crop(img, kps, image_size=base)
    interp = cv2.INTER_AREA if target_size < base else cv2.INTER_CUBIC
    return cv2.resize(aligned, (target_size, target_size), interpolation=interp)

def copy_file(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)

def write_json(p: Path, obj: dict):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2), encoding="utf-8")

def main():
    ap = argparse.ArgumentParser(description="Merapikan dataset: align (crop) dan opsi mengorganisir gallery/probe.")
    ap.add_argument("--dataset-name", required=True,
                    help="Nama folder dataset sumber di dalam folder 'dataset'. Contoh: 'Sewa' -> dataset/Sewa")
    ap.add_argument("--size", type=int, required=True,
                    help="Ukuran sisi output (tanpa 'x'). Contoh: 112, 150, 160.")
    ap.add_argument("--mode", choices=["both", "gallery", "none"], default="both",
                    help="Cara mengorganisir output: 'both' (gallery & probe), 'gallery' (probe kosong), 'none' (tanpa struktur gallery/probe).")
    ap.add_argument("--gallery-per-id", type=int, default=1,
                    help="Jumlah file per-ID yang masuk folder gallery (hanya untuk mode != none). Default: 1")
    ap.add_argument("--seed", type=int, default=42, help="Seed random untuk pemilihan gallery. Default: 42")
    ap.add_argument("--det-size", type=int, default=640,
                    help="Ukuran deteksi wajah insightface (per sisi). Default: 640")

    args = ap.parse_args()
    random.seed(args.seed)

    # >>> Perbaikan di sini: gunakan dataset_name (underscore), bukan dataset-name
    dataset_src = DATASETS_DIR / args.dataset_name
    if not dataset_src.exists() or not dataset_src.is_dir():
        sys.exit(f"[ERROR] Dataset sumber tidak ditemukan: {dataset_src}")

    out_root = DATASETS_DIR / f"{args.dataset_name}_{args.size}"
    tmp_aligned = out_root / "_aligned"

    try:
        app = FaceAnalysis(name="buffalo_l", providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        app.prepare(ctx_id=0, det_size=(args.det_size, args.det_size))
    except Exception as e:
        print("[Info] CUDA tidak tersedia / gagal inisialisasi, fallback ke CPU:", e)
        app = FaceAnalysis(name="buffalo_l", providers=['CPUExecutionProvider'])
        app.prepare(ctx_id=-1, det_size=(args.det_size, args.det_size))

    all_imgs = collect_images(dataset_src)
    align_stat = {"total": 0, "aligned": 0, "no_face": 0, "failed": 0}
    debug_errs = []

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

    if args.mode == "none":
        if out_root.exists():
            for child in out_root.iterdir():
                if child.name != "_aligned":
                    if child.is_dir():
                        shutil.rmtree(child)
                    else:
                        child.unlink(missing_ok=True)
        else:
            out_root.mkdir(parents=True, exist_ok=True)

        if tmp_aligned.exists():
            for p in tmp_aligned.iterdir():
                target = out_root / p.name
                if target.exists():
                    shutil.rmtree(target) if target.is_dir() else target.unlink()
                shutil.move(str(p), str(target))
            shutil.rmtree(tmp_aligned, ignore_errors=True)

        print("\n=== RINGKASAN ===")
        print(f"Dataset sumber : {dataset_src}")
        print(f"Output root    : {out_root}")
        print(f"Mode           : {args.mode}")
        print(f"Size (crop)    : {args.size}x{args.size}")
        print(f"Seed           : {args.seed}")
        print(f"Align stat     : {align_stat}")
        if debug_errs:
            print("\n[DEBUG] Contoh error (maks 5):")
            for s in debug_errs:
                print(f" - {s}")
        return

    gallery_dir = out_root / "gallery"
    probe_dir   = out_root / "probe"
    gallery_dir.mkdir(parents=True, exist_ok=True)
    probe_dir.mkdir(parents=True, exist_ok=True)

    id2imgs: Dict[str, List[Path]] = {}
    if tmp_aligned.exists():
        for pid_dir in sorted([d for d in (tmp_aligned).glob("*") if d.is_dir()]):
            pid = pid_dir.name
            imgs = []
            for p in collect_images(pid_dir):
                imgs.append(p)
            if imgs:
                id2imgs[pid] = sorted(imgs)

    wrote_gallery = 0
    wrote_probe = 0

    for pid, imgs in id2imgs.items():
        if not imgs:
            continue
        random.shuffle(imgs)

        G = max(0, int(args.gallery_per_id))
        G = min(G, len(imgs))

        gal_choice = imgs[:G]
        rest = imgs[G:]

        for src_p in gal_choice:
            dst_p = gallery_dir / pid / src_p.name
            copy_file(src_p, dst_p)
            wrote_gallery += 1

        if args.mode == "both":
            for src_p in rest:
                dst_p = probe_dir / pid / src_p.name
                copy_file(src_p, dst_p)
                wrote_probe += 1

    shutil.rmtree(tmp_aligned, ignore_errors=True)

    print("\n=== RINGKASAN ===")
    print(f"Dataset sumber : {dataset_src}")
    print(f"Output root    : {out_root}")
    print(f"Mode           : {args.mode}")
    print(f"Size (crop)    : {args.size}x{args.size}")
    print(f"Seed           : {args.seed}")
    print(f"Gallery/ID     : {args.gallery_per_id} file")
    print(f"Wrote gallery  : {wrote_gallery}")
    print(f"Wrote probe    : {wrote_probe}")
    print(f"Align stat     : {align_stat}")
    if debug_errs:
        print("\n[DEBUG] Contoh error (maks 5):")
        for s in debug_errs:
            print(f" - {s}")

if __name__ == "__main__":
    main()
