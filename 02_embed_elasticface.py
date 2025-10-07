# 02_embed_elasticface.py
# ElasticFace embedding (IResNet-100, 112x112), MENYIMPAN label DI DALAM NPZ (structured array: feat + label).
# Argumen konsisten:
#   --repo-name ".\algoritma\ElasticFace"
#   --dataset-name ".\dataset\Dosen_112"
#   --weights ".\algoritma\weights\elasticface_ir100_backbone.pth"
#   --arch ir100
#   --out ".\embeds\embeds_elasticface_ir100.npz"
#   --batch 128
#
# Output .npz:
#   key   : path relatif gambar (posix)
#   value : structured array shape (1,) dengan fields:
#           - 'feat'  : float32[emb_dim]
#           - 'label' : unicode (nama ID/kelas dari folder)
#
# Contoh akses:
#   E = np.load('embeds_elasticface_ir100.npz', allow_pickle=False)
#   arr = E['gallery/Ali/img_0001.jpg']   # -> structured array (1,)
#   feat, label = arr['feat'][0], arr['label'][0]

import os, sys, argparse
from pathlib import Path
import cv2, numpy as np, torch
from tqdm import tqdm

# ---------- Fallback IResNet (jika impor dari repo gagal) ----------
import torch.nn as nn

class _IBasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(inplanes, eps=2e-5, momentum=0.9)
        self.conv1 = nn.Conv2d(inplanes, planes, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, eps=2e-5, momentum=0.9)
        self.prelu = nn.PReLU(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes, eps=2e-5, momentum=0.9)
        self.downsample = downsample
    def forward(self, x):
        identity = x
        out = self.bn1(x); out = self.conv1(out)
        out = self.bn2(out); out = self.prelu(out)
        out = self.conv2(out); out = self.bn3(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        return out + identity

def _make_layer(block, inplanes_ref, planes, blocks, stride=1):
    inplanes, modules = inplanes_ref[0], []
    downsample = None
    if stride != 1 or inplanes != planes:
        downsample = nn.Sequential(
            nn.Conv2d(inplanes, planes, 1, stride, bias=False),
            nn.BatchNorm2d(planes, eps=2e-5, momentum=0.9),
        )
    modules.append(block(inplanes, planes, stride, downsample)); inplanes = planes
    for _ in range(1, blocks):
        modules.append(block(inplanes, planes))
    inplanes_ref[0] = inplanes
    return nn.Sequential(*modules)

class IResNet_Fallback(nn.Module):
    # 112x112 -> 7x7 spatial at the end
    def __init__(self, layers=(3,13,30,3), embedding_size=512):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(64, eps=2e-5, momentum=0.9)
        self.prelu = nn.PReLU(64)
        inref = [64]
        self.layer1 = _make_layer(_IBasicBlock, inref, 64,  layers[0], stride=2)
        self.layer2 = _make_layer(_IBasicBlock, inref, 128, layers[1], stride=2)
        self.layer3 = _make_layer(_IBasicBlock, inref, 256, layers[2], stride=2)
        self.layer4 = _make_layer(_IBasicBlock, inref, 512, layers[3], stride=2)
        self.bn2 = nn.BatchNorm2d(512, eps=2e-5, momentum=0.9)
        self.dropout = nn.Dropout2d(p=0.4, inplace=True)
        self.fc = nn.Linear(512 * 7 * 7, embedding_size)
        self.features = nn.BatchNorm1d(embedding_size, eps=2e-5, momentum=0.9)
        # init
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1); nn.init.constant_(m.bias, 0)
    def forward(self, x):
        x = self.conv1(x); x = self.bn1(x); x = self.prelu(x)
        x = self.layer1(x); x = self.layer2(x); x = self.layer3(x); x = self.layer4(x)
        x = self.bn2(x); x = self.dropout(x)
        x = x.reshape(x.size(0), -1)  # reshape (aman untuk non-contiguous)
        x = self.fc(x); x = self.features(x)
        return x

# ---------- Utils ----------
SUPPORTED_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff")

def preprocess_arcface(img_bgr, size=112):
    # RGB, (x-127.5)/128, CHW
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    if img.shape[:2] != (size, size):
        img = cv2.resize(img, (size, size), interpolation=cv2.INTER_LINEAR)
    img = img.astype(np.float32)
    img = (img - 127.5) / 128.0
    img = np.transpose(img, (2, 0, 1))
    return img

@torch.no_grad()
def embed_batch(model, batch_imgs, device):
    x = torch.from_numpy(np.stack(batch_imgs)).to(device).float()
    y = model(x)
    if isinstance(y, tuple):
        y = y[0]
    feat = y.detach().cpu().numpy()
    # L2-normalize per baris
    feat = feat / (np.linalg.norm(feat, axis=1, keepdims=True) + 1e-12)
    return feat

def collect_images(root: Path):
    return sorted([str(p) for p in root.rglob("*") if p.suffix.lower() in SUPPORTED_EXT])

def path_to_rel(root: Path, abs_path: Path) -> str:
    return abs_path.resolve().relative_to(root).as_posix()

def label_from_rel(rel_path: str) -> str:
    """
    Ambil label dari path relatif:
      - jika ada 'gallery'/'probe', pakai segmen setelahnya (jika bukan nama file)
      - jika tidak ada, gunakan nama folder induk
      - fallback: stem nama file
    """
    parts = rel_path.split("/")
    lowers = [s.lower() for s in parts]
    for anchor in ("gallery", "probe"):
        if anchor in lowers:
            i = lowers.index(anchor)
            if i + 1 < len(parts) and "." not in parts[i + 1]:
                return parts[i + 1]
    if len(parts) >= 2:
        return parts[-2]
    return Path(rel_path).stem

# ---------- Build & Load ----------
def build_model(elastic_repo: str|None):
    """
    Coba impor iresnet100 dari repo ElasticFace/backbones;
    bila gagal -> pakai fallback IResNet_Fallback() (512-D).
    """
    if elastic_repo:
        repo = Path(elastic_repo).resolve()
        if repo.exists():
            sys.path.insert(0, str(repo))
    try:
        from backbones.iresnet import iresnet100  # modul dari repo resmi ElasticFace
        model = iresnet100(num_classes=512)       # backbone-only (512-dim)
        print(f"[LOG] Loaded iresnet100 from repo.")
        return model
    except Exception as e:
        print(f"[WARN] Gagal impor iresnet100 dari repo: {e}")
        print("[LOG] Menggunakan fallback IResNet_Fallback (kompatibel 512-D).")
        return IResNet_Fallback()

def load_backbone(weight_path: str, elastic_repo: str|None, device: str):
    model = build_model(elastic_repo)
    # torch.load aman untuk PyTorch>=2.4
    try:
        ckpt = torch.load(weight_path, map_location="cpu", weights_only=True)  # type: ignore
    except TypeError:
        ckpt = torch.load(weight_path, map_location="cpu")

    # Ambil state_dict jika dibungkus
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        ckpt = ckpt["state_dict"]

    cleaned = {}
    if isinstance(ckpt, dict):
        for k, v in ckpt.items():
            nk = k
            for pref in ("module.", "model.", "backbone.", "features.", "encoder."):
                if nk.startswith(pref):
                    nk = nk[len(pref):]
            # lewati head klasifikasi
            if any(x in nk for x in ["head.", "margin", "kernel", "bias", "logits"]):
                continue
            cleaned[nk] = v

    # muat non-strict
    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    print(f"[Info] Loaded params: {len(cleaned)} | missing={len(missing)} | unexpected={len(unexpected)}")
    if missing or unexpected:
        print("[Info] load_state_dict non-strict. Contoh missing (<=10):")
        for i, n in enumerate(list(missing)[:10], 1):
            print(f"  {i:02d}. {n}")
    model.eval().to(device).float()
    return model

# ---------- Main ----------
def main():
    ap = argparse.ArgumentParser(description="ElasticFace (IResNet-100) embedder, menyimpan feat+label per key (NPZ).")
    ap.add_argument("--repo-name", required=False, default="",
                    help="Path folder repo ElasticFace (berisi backbones/iresnet.py). Contoh: .\\algoritma\\ElasticFace")
    ap.add_argument("--dataset-name", required=True,
                    help="Path folder dataset aligned (mis. .\\dataset\\Dosen_112)")
    ap.add_argument("--weights", required=True,
                    help="Path bobot backbone ElasticFace (.pth). Contoh: .\\algoritma\\weights\\elasticface_ir100_backbone.pth")
    ap.add_argument("--arch", default="ir100", choices=["ir100"],
                    help="Backbone (tetap ir100 untuk apple-to-apple).")
    ap.add_argument("--out", required=True,
                    help="Path file output .npz (mis. .\\embeds\\embeds_elasticface_ir100.npz)")
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu",
                    choices=["cpu","cuda"])
    ap.add_argument("--limit", type=int, default=0, help="Batasi jumlah gambar (debug, 0=semua)")
    args = ap.parse_args()

    dataset_root = Path(args.dataset_name).resolve()
    out_path = Path(args.out).resolve()
    device = "cuda" if (args.device == "cuda" and torch.cuda.is_available()) else "cpu"
    if args.device == "cuda" and device != "cuda":
        print("[WARN] CUDA diminta tapi tidak tersedia, pakai CPU.")

    print("== ElasticFace Embedding ==")
    print(f"[LOG] dataset_root : {dataset_root}")
    print(f"[LOG] out_path     : {out_path}")
    print(f"[LOG] weights      : {args.weights}")
    if args.repo_name:
        print(f"[LOG] repo-name    : {args.repo_name}")

    if not dataset_root.exists():
        sys.exit(f"[ERROR] Folder dataset tidak ditemukan: {dataset_root}")
    if not Path(args.weights).exists():
        sys.exit(f"[ERROR] Bobot tidak ditemukan: {args.weights}")

    # Kumpulkan path gambar
    paths = collect_images(dataset_root)
    print(f"[LOG] ditemukan {len(paths)} gambar di {dataset_root}")
    if not paths:
        sys.exit("[ERROR] Tidak ada gambar (jpg/jpeg/png/bmp/webp/tif/tiff) di folder dataset.")
    if args.limit and args.limit > 0:
        paths = paths[:args.limit]
        print(f"[LOG] MODE UJI: membatasi ke {len(paths)} gambar pertama.")

    # Muat backbone
    model = load_backbone(args.weights, args.repo_name or None, device)

    # ============ Sanity-check 1 sampel & ambil emb_dim ============
    try:
        smp = cv2.imread(paths[0]); assert smp is not None
        smp_pre = preprocess_arcface(smp, 112)
        smp_feat = embed_batch(model, [smp_pre], device)[0]
        emb_dim = int(smp_feat.shape[0])
        print(f"[LOG] sanity-check: 1 sample feat dim={emb_dim}, norm={np.linalg.norm(smp_feat):.6f}")
    except Exception as e:
        sys.exit(f"[ERROR] Sanity-check gagal: {e}")

    # Siapkan dtype structured untuk (feat + label)
    dtype_struct = np.dtype([('feat', np.float32, (emb_dim,)), ('label', 'U128')])

    # Proses embedding -> simpan structured array per key
    records = {}
    B = max(1, int(args.batch))
    buf_imgs, buf_idx = [], []
    rels = [str(Path(p).relative_to(dataset_root)).replace("\\", "/") for p in paths]

    proc = 0
    for i, p in enumerate(tqdm(paths, desc="Embedding[ElasticFace-ir100]")):
        img = cv2.imread(p)
        if img is None:
            continue
        buf_imgs.append(preprocess_arcface(img, 112))
        buf_idx.append(i)
        if len(buf_imgs) == B:
            F = embed_batch(model, buf_imgs, device)
            for j, ii in enumerate(buf_idx):
                rel = rels[ii]
                lbl = label_from_rel(rel)
                rec = np.empty((1,), dtype=dtype_struct)
                rec['feat'][0]  = F[j].astype(np.float32, copy=False)
                rec['label'][0] = lbl
                records[rel] = rec
            proc += len(buf_imgs)
            buf_imgs.clear(); buf_idx.clear()

    if buf_imgs:
        F = embed_batch(model, buf_imgs, device)
        for j, ii in enumerate(buf_idx):
            rel = rels[ii]
            lbl = label_from_rel(rel)
            rec = np.empty((1,), dtype=dtype_struct)
            rec['feat'][0]  = F[j].astype(np.float32, copy=False)
            rec['label'][0] = lbl
            records[rel] = rec
        proc += len(buf_imgs)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **records)
    print(f"[OK] Saved {len(records)} embeddings -> {out_path}")
    print(f"[DONE] processed={proc}/{len(paths)} images")

if __name__ == "__main__":
    torch.backends.cudnn.benchmark = False
    main()
