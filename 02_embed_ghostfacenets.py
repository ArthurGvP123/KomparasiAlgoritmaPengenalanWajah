# 02_embed_ghostfacenets.py
# GhostFaceNets (Keras/TF) -> embedding .npz (512-dim, L2-normalized)
import os, sys, argparse
from pathlib import Path
import cv2, numpy as np
from tqdm import tqdm

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def force_device(device_flag: str):
    dev = (device_flag or "cpu").strip().lower()
    if dev == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    return dev

def preprocess_arcface_bgr(img_bgr, size=112):
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    if img.shape[:2] != (size, size):
        img = cv2.resize(img, (size, size), interpolation=cv2.INTER_LINEAR)
    img = img.astype(np.float32)
    img = (img - 127.5) / 128.0
    return img  # HWC float32

def find_embedding_head(model: keras.Model) -> keras.Model:
    # Cari layer dengan dim akhir 512 dan bukan logits/classifier
    for lyr in model.layers[::-1]:
        try:
            shp = lyr.output_shape
        except Exception:
            continue
        if isinstance(shp, (list, tuple)) and len(shp) > 0:
            last_dim = shp[-1] if isinstance(shp[-1], int) else (shp[-1][-1] if isinstance(shp[-1], (list, tuple)) else None)
            name_l = lyr.name.lower()
            if last_dim in (512,) and not any(k in name_l for k in ("logit","softmax","arcface","margin","classifier","dense_85742")):
                return keras.Model(inputs=model.inputs, outputs=lyr.output, name=model.name+"_emb")
    # fallback: layer -2
    if len(model.layers) >= 2:
        prev = model.layers[-2]
        return keras.Model(inputs=model.inputs, outputs=prev.output, name=model.name+"_emb_prev")
    return model

def build_from_backbones(repo_dir: str, width=1.3, strides=2) -> keras.Model:
    # Import GhostNet base dari file yang Anda kirim: backbones/ghost_model.py
    repo_dir = str(Path(repo_dir).resolve())
    if repo_dir not in sys.path:
        sys.path.insert(0, repo_dir)
    try:
        from backbones.ghost_model import GhostNet
    except Exception as e:
        raise ImportError(f"Gagal import backbones.ghost_model.GhostNet dari: {repo_dir}\n{e}")

    base = GhostNet(input_shape=(112,112,3), include_top=False, width=width, strides=strides, name="GhostNet_W1p3_S2")
    x = base.output  # ekspektasi: spatial 4x4 untuk input 112 pada W1.3 S2

    # Hitung kernel GDC dari ukuran spasial (seharusnya 4x4)
    ishape = x.shape
    if ishape[1] is None or ishape[2] is None:
        # fallback (umumnya 4x4)
        k = 4
    else:
        k = int(ishape[1])

    # GDC head: depthwise(k,k) -> Flatten -> Dense(512) -> BN (scale=False) -> L2
    x = layers.DepthwiseConv2D(kernel_size=(k, k), strides=1, padding="valid", use_bias=False, name="gdc_dw")(x)
    x = layers.Flatten(name="gdc_flatten")(x)
    x = layers.Dense(512, use_bias=False, name="gdc_fc")(x)
    x = layers.BatchNormalization(scale=False, name="gdc_bn")(x)
    # output FEATURES (tanpa softmax); L2 normalize saat inference
    emb = layers.Lambda(lambda t: tf.math.l2_normalize(t, axis=1), name="l2norm")(x)

    model = keras.Model(inputs=base.inputs, outputs=emb, name="GhostFaceNet_W1p3_S2_Emb")
    return model

def load_backbone(weights_path: str, repo_dir: str, device: str = "cpu"):
    dev = force_device(device)

    # A) coba load full model .h5
    try:
        print(f"[LOG] load_backbone: coba keras.models.load_model -> {weights_path}")
        full = keras.models.load_model(weights_path, compile=False)
        emb = find_embedding_head(full)
        print("[LOG] load_model berhasil (full model).")
        return emb
    except Exception as e:
        print(f"[INFO] load_model gagal (mungkin weights-only/custom_objects): {e}")

    # B) build dari backbones/ghost_model.py + GDC head, lalu load_weights by_name
    print(f"[LOG] build_from_backbones + load_weights(by_name, skip_mismatch=True) dari repo: {repo_dir}")
    model = build_from_backbones(repo_dir, width=1.3, strides=2)
    # muat dengan nama agar bagian yang cocok terisi
    model.load_weights(weights_path, by_name=True, skip_mismatch=True)
    print("[LOG] load_weights by_name selesai (skip_mismatch=True).")
    return model

@tf.function(jit_compile=False)
def _forward(model, batch_tensor):
    y = model(batch_tensor, training=False)
    return y  # sudah L2 di head

def embed_batch(model, batch_imgs):
    x = np.stack(batch_imgs, axis=0)  # NHWC
    x = tf.convert_to_tensor(x, dtype=tf.float32)
    y = _forward(model, x).numpy()
    # model head sudah L2-normalize; jaga-jaga normalisasi ulang ringan
    y = y / np.maximum(1e-12, np.linalg.norm(y, axis=1, keepdims=True))
    return y

def main():
    ap = argparse.ArgumentParser(description="GhostFaceNets embedding -> .npz (512-dim, L2-normalized)")
    ap.add_argument("--root", default=".\\crops_112")
    ap.add_argument("--weights", default=".\\weights\\GhostFaceNet_W1.3_S2_ArcFace.h5")
    ap.add_argument("--repo", default="D:\\KEVIN SUKA SAMBEL\\GhostFaceNets", help="folder yang berisi subfolder 'backbones'")
    ap.add_argument("--out", default=".\\embeds_ghostfacenet_w1p3_s2.npz")
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    print("== GhostFaceNets Embedding (W1.3 S2) ==")
    print("[LOG] args:", args)

    root = Path(args.root)
    exts = (".jpg", ".jpeg", ".png")
    paths = sorted([str(p) for p in root.rglob("*") if p.suffix.lower() in exts])
    print(f"[LOG] ditemukan {len(paths)} gambar di {args.root}")
    if args.limit and args.limit > 0 and len(paths) > args.limit:
        paths = paths[:args.limit]
        print(f"[LOG] MODE UJI: membatasi ke {len(paths)} gambar pertama.")
    if not paths:
        print("[ERR] Tidak ada gambar ditemukan. Pastikan 01_align.py sudah menghasilkan crops_112.")
        return

    model = load_backbone(args.weights, args.repo, args.device)

    # sanity check output shape
    dummy = np.zeros((1,112,112,3), dtype=np.float32)
    out = model(dummy, training=False)
    print(f"[LOG] model output shape: {tuple(out.shape)} (target [1, 512])")

    feats = {}
    rels = [str(Path(p).relative_to(root)).replace("\\", "/") for p in paths]
    B = max(1, int(args.batch))
    buf_imgs, buf_idx = [], []

    for i, p in enumerate(tqdm(paths, desc="Embedding[GhostFaceNets W1.3 S2]")):
        img = cv2.imread(p)
        if img is None:
            continue
        buf_imgs.append(preprocess_arcface_bgr(img, size=112))
        buf_idx.append(i)
        if len(buf_imgs) == B:
            F = embed_batch(model, buf_imgs)
            for j, ii in enumerate(buf_idx):
                feats[rels[ii]] = F[j]
            buf_imgs, buf_idx = [], []

    if buf_imgs:
        F = embed_batch(model, buf_imgs)
        for j, ii in enumerate(buf_idx):
            feats[rels[ii]] = F[j]

    out_path = Path(args.out); out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **feats)
    print(f"[OK] Saved {len(feats)} embeddings -> {out_path}")

if __name__ == "__main__":
    main()
