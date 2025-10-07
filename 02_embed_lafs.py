# 02_embed_lafs.py
# LAFS embedding with IResNet backbone (IR-100 / IR-50), NPZ berisi feat + label per key
# Output NPZ:
#   key   : path relatif (posix)
#   value : structured array shape (1,) dtypes:
#           - 'feat'  : float32[emb_dim]
#           - 'label' : unicode (ID/kelas dari nama folder)

import os, sys, argparse, zipfile
from io import BytesIO
from pathlib import Path
import numpy as np
import cv2, torch
from tqdm import tqdm

# --------------------------
# Minimal IResNet (ArcFace-style)
# --------------------------
import torch.nn as nn

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)

def conv1x1(in_planes, out_planes, stride=1):  # 1x1 conv
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class IBasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1):
        super().__init__()
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        self.bn1 = nn.BatchNorm2d(inplanes, eps=2e-05, momentum=0.9)
        self.conv1 = conv3x3(inplanes, planes)
        self.bn2 = nn.BatchNorm2d(planes, eps=2e-05, momentum=0.9)
        self.prelu = nn.PReLU(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn3 = nn.BatchNorm2d(planes, eps=2e-05, momentum=0.9)
        self.downsample = downsample
        self.stride = stride
    def forward(self, x):
        identity = x
        out = self.bn1(x)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.prelu(out)
        out = self.conv2(out)
        out = self.bn3(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return out

class IResNet(nn.Module):
    fc_scale = 7 * 7
    def __init__(self, block, layers, num_classes=512, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None):
        super().__init__()
        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes, eps=2e-05, momentum=0.9)
        self.prelu = nn.PReLU(self.inplanes)
        self.layer1 = self._make_layer(block, 64,  layers[0], stride=2)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.bn2 = nn.BatchNorm2d(512 * block.expansion, eps=2e-05, momentum=0.9)
        self.dropout = nn.Dropout2d(p=0.4, inplace=True)
        self.fc = nn.Linear(512 * block.expansion * self.fc_scale, num_classes)
        self.features = nn.BatchNorm1d(num_classes, eps=2e-05, momentum=0.9)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1); nn.init.constant_(m.bias, 0)
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, IBasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)
    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        downsample = None; previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride; stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion, eps=2e-05, momentum=0.9),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation))
        return nn.Sequential(*layers)
    def forward(self, x):
        x = self.conv1(x); x = self.bn1(x); x = self.prelu(x)
        x = self.layer1(x); x = self.layer2(x); x = self.layer3(x); x = self.layer4(x)
        x = self.bn2(x); x = self.dropout(x)
        x = x.reshape(x.size(0), -1)            # important: reshape (not view)
        x = self.fc(x)
        x = self.features(x)
        return x

def iresnet50(**kwargs):  return IResNet(IBasicBlock, [3, 4, 14, 3], **kwargs)
def iresnet100(**kwargs): return IResNet(IBasicBlock, [3, 13, 30, 3], **kwargs)

# --------------------------
# Utils
# --------------------------
def log(*a): print("[LOG]", *a)

SUPPORTED_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff")

def preprocess_arcface(img_bgr, size=112):
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
    if isinstance(y, (list, tuple)): y = y[0]
    elif isinstance(y, dict):
        for k in ("embeddings","feat","features","last_hidden_state","pooler_output","output"):
            if k in y: y = y[k]; break
        else:
            y = next(iter(y.values()))
    vec = y.detach().cpu().numpy()
    vec = vec / (np.linalg.norm(vec, axis=1, keepdims=True) + 1e-12)
    return vec

def _zip_compression(use_compress: bool):
    return zipfile.ZIP_DEFLATED if use_compress else zipfile.ZIP_STORED

def _write_record_to_npz(zf: zipfile.ZipFile, key: str, feat: np.ndarray, label: str, dtype_struct: np.dtype):
    """
    Tulis satu entry .npy (structured array shape (1,)) ke dalam ZIP.
    Ini membuat arsip .npz valid (zip berisi berkas .npy).
    """
    rec = np.empty((1,), dtype=dtype_struct)
    rec["feat"][0]  = feat.astype(np.float32, copy=False)
    rec["label"][0] = label
    buf = BytesIO()
    np.save(buf, rec)
    zf.writestr(f"{key}.npy", buf.getvalue())

def _list_images(root: Path):
    return sorted([str(p) for p in root.rglob("*") if p.suffix.lower() in SUPPORTED_EXT])

def _rel_posix(root: Path, p: Path) -> str:
    return p.resolve().relative_to(root).as_posix()

def _label_from_rel(rel_path: str) -> str:
    """
    Ambil label dari path relatif:
      - Jika ada 'gallery'/'probe' -> segmen setelahnya (bila bukan nama file)
      - Jika tidak, gunakan nama folder induk
      - Fallback: nama file (tanpa ekstensi)
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

def build_model(arch: str):
    arch = arch.lower()
    if arch in ("ir100","iresnet100","ir-100"):
        return iresnet100(num_classes=512)
    elif arch in ("ir50","iresnet50","ir-50"):
        return iresnet50(num_classes=512)
    else:
        raise ValueError(f"arch tidak dikenal: {arch} (pilih: ir100 / ir50)")

def load_backbone(weight_path: str, arch: str, device: str):
    model = build_model(arch)
    try:
        ckpt = torch.load(weight_path, map_location="cpu", weights_only=True)  # PyTorch>=2.4
    except TypeError:
        ckpt = torch.load(weight_path, map_location="cpu")
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        ckpt = ckpt["state_dict"]
    cleaned = {}
    skipped = []
    for k, v in (ckpt.items() if isinstance(ckpt, dict) else []):
        nk = k
        for pref in ("module.","model.","backbone.","features.module.","features."):
            if nk.startswith(pref):
                nk = nk[len(pref):]
        if any(nk.startswith(p) for p in ("head.","margin.","arcface.","logits.","classifier.","fc.weight","fc.bias")):
            skipped.append((k, "head_or_classifier")); continue
        cleaned[nk] = v
    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    print(f"[Info] Loaded params: {len(cleaned)-len(missing)} | missing={len(missing)} | unexpected={len(unexpected)}")
    if skipped:
        print("[Info] Skipped (head/fc), contoh <=10:")
        for i,(k,why) in enumerate(skipped[:10],1):
            print(f"  {i:02d}. {k} [{why}]")
    if missing:
        print("[Warn] Contoh missing <=10:")
        for i,k in enumerate(missing[:10],1):
            print(f"  {i:02d}. {k}")
    if unexpected:
        print("[Warn] Contoh unexpected <=10:")
        for i,k in enumerate(unexpected[:10],1):
            print(f"  {i:02d}. {k}")
    model.eval().to(device).float()
    return model

def main():
    ap = argparse.ArgumentParser(description="LAFS embedding (IResNet backbone) – simpan NPZ feat+label per key")
    # Argumen seragam
    ap.add_argument("--repo-name", default="", help="Folder repo LAFS (opsional, tidak wajib dipakai)")
    ap.add_argument("--dataset-name", required=True, help="Folder dataset aligned (mis. .\\dataset\\Dosen_112)")
    ap.add_argument("--weights", required=True, help="Path checkpoint LAFS (.pth/.pt)")
    ap.add_argument("--arch", default="ir100", choices=["ir100","ir50"], help="Backbone yang cocok")
    ap.add_argument("--out", required=True, help="Path output .npz")

    # Lainnya
    ap.add_argument("--batch",  type=int, default=128)
    ap.add_argument("--device", choices=["cpu","cuda"], default="cpu")
    ap.add_argument("--limit",  type=int, default=0, help="Uji sebagian gambar (0=semua)")
    ap.add_argument("--no-compress", action="store_true", help="NPZ tanpa kompresi (lebih cepat, file lebih besar)")
    args = ap.parse_args()

    device = "cuda" if (args.device=="cuda" and torch.cuda.is_available()) else "cpu"
    if args.device=="cuda" and device!="cuda":
        log("CUDA tidak tersedia, fallback ke CPU")

    dataset_root = Path(args.dataset_name).resolve()
    out = Path(args.out).resolve()
    if not dataset_root.exists():
        sys.exit(f"[ERROR] Folder dataset tidak ditemukan: {dataset_root}")
    if not Path(args.weights).exists():
        sys.exit(f"[ERROR] File weights tidak ditemukan: {args.weights}")

    paths = _list_images(dataset_root)
    log(f"dataset_root : {dataset_root}")
    log(f"out_path     : {out}")
    log(f"ditemukan {len(paths)} gambar di {dataset_root}")
    if not paths:
        print("[!] Tidak ada gambar. Pastikan 01_merapikan_dataset sudah jalan."); return
    if args.limit>0:
        paths = paths[:args.limit]; log(f"MODE UJI: {len(paths)} gambar.")

    log(f"load_backbone: {args.arch} | weights={args.weights}")
    model = load_backbone(args.weights, args.arch, device)

    # Siapkan ZIP (.npz) writer
    comp = _zip_compression(not args.no_compress)
    rels = [_rel_posix(dataset_root, Path(p)) for p in paths]
    B = max(1, int(args.batch))
    buf_imgs, buf_idx = [], []

    # dtype_struct akan ditentukan dinamis saat batch pertama (ambil emb_dim dari hasil model)
    dtype_struct = None

    out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(str(out), mode="w", compression=comp, allowZip64=True) as zf:
        # Proses batch
        for i, p in enumerate(tqdm(paths, desc=f"Embedding[LAFS-{args.arch} {device}]")):
            img = cv2.imread(p)
            if img is None: 
                continue
            buf_imgs.append(preprocess_arcface(img, 112))
            buf_idx.append(i)

            if len(buf_imgs) == B:
                F = embed_batch(model, buf_imgs, device)  # (N, D)
                if dtype_struct is None:
                    emb_dim = int(F.shape[1])
                    dtype_struct = np.dtype([("feat", np.float32, (emb_dim,)), ("label", "U128")])
                for j, ii in enumerate(buf_idx):
                    key = rels[ii]
                    label = _label_from_rel(key)
                    _write_record_to_npz(zf, key, F[j], label, dtype_struct)
                buf_imgs.clear(); buf_idx.clear()

        # Sisa buffer
        if buf_imgs:
            F = embed_batch(model, buf_imgs, device)
            if dtype_struct is None:
                emb_dim = int(F.shape[1])
                dtype_struct = np.dtype([("feat", np.float32, (emb_dim,)), ("label", "U128")])
            for j, ii in enumerate(buf_idx):
                key = rels[ii]
                label = _label_from_rel(key)
                _write_record_to_npz(zf, key, F[j], label, dtype_struct)

    log(f"[OK] Saved embeddings (feat+label) -> {out}")

if __name__ == "__main__":
    torch.backends.cudnn.benchmark = False
    try:
        torch.set_num_threads(min(8, os.cpu_count() or 8))
    except Exception:
        pass
    main()
