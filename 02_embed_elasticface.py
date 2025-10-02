# 02_embed_elasticface.py
import os, sys, argparse
from pathlib import Path
import cv2, numpy as np, torch
from tqdm import tqdm

HERE = Path(__file__).resolve().parent
WEIGHTS_DEFAULT = str(HERE / "weights" / "elasticface_ir100_backbone.pth")

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
        self.inplanes = 64
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
        x = x.reshape(x.size(0), -1)  # <- reshape (bukan view)
        x = self.fc(x); x = self.features(x)
        return x

# ---------- Build & Load ----------
def build_model(elastic_repo: str|None):
    """
    Coba impor iresnet100 dari repo ElasticFace/backbones;
    bila gagal -> pakai fallback IResNet_Fallback().
    """
    if elastic_repo:
        sys.path.append(str(Path(elastic_repo)))
    try:
        from backbones.iresnet import iresnet100  # repo resmi
        model = iresnet100(num_classes=512)
        return model
    except Exception:
        # fallback internal yang kompatibel shape 512-d
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
            # lewati head klasifikasi (kernel/t/bias dsb)
            if any(x in nk for x in ["head.", "margin", "kernel", "bias", "logits"]):
                continue
            cleaned[nk] = v

    # muat non-strict
    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    print(f"[Info] Loaded params: {len(cleaned)} | missing={len(missing)} | unexpected={len(unexpected)}")
    if missing or unexpected:
        sk = list(missing)[:10]
        print("[Info] Skipped (shape mismatch / not in model), contoh <=10:")
        for i, n in enumerate(sk, 1):
            sv = ckpt.get(n, "missing_in_model") if isinstance(ckpt, dict) else "missing_in_model"
            shape = tuple(sv.shape) if hasattr(sv, "shape") else "missing_in_model"
            print(f"  {i:02d}. {n}: ckpt{shape} vs modelmissing_in_model")
    model.eval().to(device).float()
    return model

# ---------- Preprocess & Embed ----------
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
    if isinstance(y, tuple):
        y = y[0]
    feat = y.detach().cpu().numpy()
    feat = feat / np.linalg.norm(feat, axis=1, keepdims=True)
    return feat

def main():
    ap = argparse.ArgumentParser(description="Embed wajah dengan ElasticFace (IResNet-100, 112x112)")
    ap.add_argument("--root", default=str(HERE/"crops_112"), help="root gambar aligned")
    ap.add_argument("--weights", default=WEIGHTS_DEFAULT, help="path .pth backbone ElasticFace")
    ap.add_argument("--elasticface-repo", default=str(HERE/"ElasticFace"), help="folder repo ElasticFace (untuk impor backbones)")
    ap.add_argument("--arch", default="ir100", choices=["ir100"], help="backbone (tetap ir100 untuk apple-to-apple)")
    ap.add_argument("--out", default=str(HERE/"embeds_elasticface_ir100.npz"))
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--limit", type=int, default=0, help="debug: batasi jumlah gambar (0=semua)")
    args = ap.parse_args()

    print("== ElasticFace Embedding ==")
    print("[LOG] args:", args)

    root = Path(args.root)
    exts = (".jpg", ".jpeg", ".png")
    paths = sorted([str(p) for p in root.rglob("*") if p.suffix.lower() in exts])
    print(f"[LOG] ditemukan {len(paths)} gambar di {root}")
    if args.limit and len(paths) > args.limit:
        paths = paths[:args.limit]
        print(f"[LOG] MODE UJI: membatasi ke {len(paths)} gambar pertama.")

    if not Path(args.weights).exists():
        raise FileNotFoundError(f"Bobot tidak ditemukan: {args.weights}")

    model = load_backbone(args.weights, args.elasticface_repo, args.device)

    rels = [str(Path(p).relative_to(root)).replace("\\", "/") for p in paths]
    feats = {}
    B = max(1, int(args.batch))
    buf_imgs, buf_idx = [], []

    for i, p in enumerate(tqdm(paths, desc=f"Embedding[ElasticFace-ir100]")):
        img = cv2.imread(p)
        if img is None:
            continue
        buf_imgs.append(preprocess_arcface(img, size=112))
        buf_idx.append(i)
        if len(buf_imgs) == B:
            F = embed_batch(model, buf_imgs, args.device)
            for j, ii in enumerate(buf_idx):
                feats[rels[ii]] = F[j]
            buf_imgs.clear(); buf_idx.clear()

    if buf_imgs:
        F = embed_batch(model, buf_imgs, args.device)
        for j, ii in enumerate(buf_idx):
            feats[rels[ii]] = F[j]

    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **feats)
    print(f"[OK] Saved {len(feats)} embeddings -> {out}")

if __name__ == "__main__":
    torch.backends.cudnn.benchmark = False
    main()
