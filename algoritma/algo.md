# Repositori Algoritma untuk Embedding

Dokumen ini merangkum **repositori resmi** yang digunakan, beserta perintah **clone** untuk menyiapkan pengujian embedding.

> **Prasyarat**:  
> - Sudah menginstal **Git**  
> - (Opsional) **Git LFS** untuk repo besar: `git lfs install`  
> - Jika repo punya submodule: `git submodule update --init --recursive`

---

## Clone Semua Sekaligus

# Clone cepat (shallow) agar lebih ringan. Hapus --depth=1 jika perlu riwayat penuh.
git clone --depth=1 https://github.com/HuangYG123/CurricularFace
git clone --depth=1 https://github.com/IrvingMeng/MagFace
git clone --depth=1 https://github.com/mk-minchul/AdaFace
git clone --depth=1 https://github.com/fdbtrs/ElasticFace
git clone --depth=1 https://github.com/DanJun6737/TransFace
git clone --depth=1 https://github.com/HamadYA/GhostFaceNets
git clone --depth=1 https://github.com/mk-minchul/CVLface
git clone --depth=1 https://github.com/szlbiubiubiu/LAFS_CVPR2024
git clone --depth=1 https://github.com/WakingHours-GitHub/EPL
git clone --depth=1 https://github.com/bytedance/LVFace
git clone --depth=1 https://github.com/davisking/dlib
git clone --depth=1 https://github.com/timesler/facenet-pytorch

# Perubahan dari Repositori Original

Dalam proyek ini ada beberapa perbaikan compatibility (penanganan tensor non-contiguous, inisialisasi bobot, dll).
Salin & timpa isi file berikut sesuai path di proyek kamu.

Catatan: Path yang ditulis di bawah mengacu ke struktur proyek lokal kamu (mis. folder algoritma/...). Jika struktur berbeda, sesuaikan lokasinya.

## 1. algoritma/CurricularFace/backbone/model_irse.py

import torch
import torch.nn as nn
from torch.nn import Linear, Conv2d, BatchNorm1d, BatchNorm2d, PReLU, ReLU, Sigmoid, Dropout, MaxPool2d, \
    AdaptiveAvgPool2d, Sequential, Module
from collections import namedtuple


### Support: ['IR_50', 'IR_101', 'IR_152', 'IR_SE_50', 'IR_SE_101', 'IR_SE_152']


class Flatten(Module):
    def forward(self, input):
        # was: return input.view(input.size(0), -1)
        # safer for non-contiguous tensors:
        return torch.flatten(input, 1)


def l2_norm(input, axis=1):
    norm = torch.norm(input, 2, axis, True)
    output = torch.div(input, norm)
    return output


class SEModule(Module):
    def __init__(self, channels, reduction):
        super(SEModule, self).__init__()
        self.avg_pool = AdaptiveAvgPool2d(1)
        self.fc1 = Conv2d(
            channels, channels // reduction, kernel_size=1, padding=0, bias=False)

        nn.init.xavier_uniform_(self.fc1.weight.data)

        self.relu = ReLU(inplace=True)
        self.fc2 = Conv2d(
            channels // reduction, channels, kernel_size=1, padding=0, bias=False)

        self.sigmoid = Sigmoid()

    def forward(self, x):
        module_input = x
        x = self.avg_pool(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return module_input * x


class bottleneck_IR(Module):
    def __init__(self, in_channel, depth, stride):
        super(bottleneck_IR, self).__init__()
        if in_channel == depth:
            self.shortcut_layer = MaxPool2d(1, stride)
        else:
            self.shortcut_layer = Sequential(
                Conv2d(in_channel, depth, (1, 1), stride, bias=False), BatchNorm2d(depth))
        self.res_layer = Sequential(
            BatchNorm2d(in_channel),
            Conv2d(in_channel, depth, (3, 3), (1, 1), 1, bias=False),
            PReLU(depth),
            Conv2d(depth, depth, (3, 3), stride, 1, bias=False),
            BatchNorm2d(depth))

    def forward(self, x):
        shortcut = self.shortcut_layer(x)
        res = self.res_layer(x)
        return res + shortcut


class bottleneck_IR_SE(Module):
    def __init__(self, in_channel, depth, stride):
        super(bottleneck_IR_SE, self).__init__()
        if in_channel == depth:
            self.shortcut_layer = MaxPool2d(1, stride)
        else:
            self.shortcut_layer = Sequential(
                Conv2d(in_channel, depth, (1, 1), stride, bias=False),
                BatchNorm2d(depth))
        self.res_layer = Sequential(
            BatchNorm2d(in_channel),
            Conv2d(in_channel, depth, (3, 3), (1, 1), 1, bias=False),
            PReLU(depth),
            Conv2d(depth, depth, (3, 3), stride, 1, bias=False),
            BatchNorm2d(depth),
            SEModule(depth, 16)
        )

    def forward(self, x):
        shortcut = self.shortcut_layer(x)
        res = self.res_layer(x)
        return res + shortcut


class Bottleneck(namedtuple('Block', ['in_channel', 'depth', 'stride'])):
    '''A named tuple describing a ResNet block.'''


def get_block(in_channel, depth, num_units, stride=2):
    return [Bottleneck(in_channel, depth, stride)] + [Bottleneck(depth, depth, 1) for i in range(num_units - 1)]


def get_blocks(num_layers):
    if num_layers == 50:
        blocks = [
            get_block(in_channel=64, depth=64, num_units=3),
            get_block(in_channel=64, depth=128, num_units=4),
            get_block(in_channel=128, depth=256, num_units=14),
            get_block(in_channel=256, depth=512, num_units=3)
        ]
    elif num_layers == 100:
        blocks = [
            get_block(in_channel=64, depth=64, num_units=3),
            get_block(in_channel=64, depth=128, num_units=13),
            get_block(in_channel=128, depth=256, num_units=30),
            get_block(in_channel=256, depth=512, num_units=3)
        ]
    elif num_layers == 152:
        blocks = [
            get_block(in_channel=64, depth=64, num_units=3),
            get_block(in_channel=64, depth=128, num_units=8),
            get_block(in_channel=128, depth=256, num_units=36),
            get_block(in_channel=256, depth=512, num_units=3)
        ]
    return blocks


class Backbone(Module):
    def __init__(self, input_size, num_layers, mode='ir'):
        super(Backbone, self).__init__()
        assert input_size[0] in [112, 224], "input_size should be [112, 112] or [224, 224]"
        assert num_layers in [50, 100, 152], "num_layers should be 50, 100 or 152"
        assert mode in ['ir', 'ir_se'], "mode should be ir or ir_se"
        blocks = get_blocks(num_layers)
        if mode == 'ir':
            unit_module = bottleneck_IR
        elif mode == 'ir_se':
            unit_module = bottleneck_IR_SE
        self.input_layer = Sequential(Conv2d(3, 64, (3, 3), 1, 1, bias=False),
                                      BatchNorm2d(64),
                                      PReLU(64))
        if input_size[0] == 112:
            self.output_layer = Sequential(BatchNorm2d(512),
                                           Dropout(0.4),
                                           Flatten(),
                                           Linear(512 * 7 * 7, 512),
                                           BatchNorm1d(512, affine=False))
        else:
            self.output_layer = Sequential(BatchNorm2d(512),
                                           Dropout(0.4),
                                           Flatten(),
                                           Linear(512 * 14 * 14, 512),
                                           BatchNorm1d(512, affine=False))

        modules = []
        for block in blocks:
            for bottleneck in block:
                modules.append(
                    unit_module(bottleneck.in_channel,
                                bottleneck.depth,
                                bottleneck.stride))
        self.body = Sequential(*modules)

        self._initialize_weights()

    def forward(self, x):
        x = self.input_layer(x)
        x = self.body(x)
        # was: conv_out = x.view(x.shape[0], -1)
        conv_out = x.reshape(x.shape[0], -1)   # safer than .view for non-contiguous tensors
        x = self.output_layer(x)
        return x, conv_out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    m.bias.data.zero_()

def IR_50(input_size):
    """Constructs a ir-50 model.
    """
    model = Backbone(input_size, 50, 'ir')
    return model


def IR_101(input_size):
    """Constructs a ir-101 model.
    """
    model = Backbone(input_size, 100, 'ir')
    return model


def IR_152(input_size):
    """Constructs a ir-152 model.
    """
    model = Backbone(input_size, 152, 'ir')
    return model


def IR_SE_50(input_size):
    """Constructs a ir_se-50 model.
    """
    model = Backbone(input_size, 50, 'ir_se')
    return model


def IR_SE_101(input_size):
    """Constructs a ir_se-101 model.
    """
    model = Backbone(input_size, 100, 'ir_se')
    return model


def IR_SE_152(input_size):
    """Constructs a ir_se-152 model.
    """
    model = Backbone(input_size, 152, 'ir_se')
    return model

## 2. algoritma/AdaFace/net.py

from collections import namedtuple
import torch
import torch.nn as nn
from torch.nn import Dropout, MaxPool2d, Sequential
from torch.nn import Conv2d, Linear, BatchNorm1d, BatchNorm2d
from torch.nn import ReLU, Sigmoid, Module, PReLU


def build_model(model_name='ir_50'):
    if model_name == 'ir_101':
        return IR_101(input_size=(112, 112))
    elif model_name == 'ir_50':
        return IR_50(input_size=(112, 112))
    elif model_name == 'ir_se_50':
        return IR_SE_50(input_size=(112, 112))
    elif model_name == 'ir_34':
        return IR_34(input_size=(112, 112))
    elif model_name == 'ir_18':
        return IR_18(input_size=(112, 112))
    else:
        raise ValueError('not a correct model name', model_name)


def initialize_weights(modules):
    """Kaiming init untuk Conv/Linear, BN diisi 1/0."""
    for m in modules:
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                m.bias.data.zero_()


class Flatten(Module):
    """Meratakan tensor NCHW -> N x (C*H*W) dengan aman untuk non-contiguous."""
    def forward(self, input):
        # gunakan reshape agar aman untuk non-contiguous
        return input.reshape(input.size(0), -1)


class LinearBlock(Module):
    """Conv + BN tanpa aktivasi."""
    def __init__(self, in_c, out_c, kernel=(1, 1), stride=(1, 1), padding=(0, 0), groups=1):
        super(LinearBlock, self).__init__()
        self.conv = Conv2d(in_c, out_c, kernel, stride, padding, groups=groups, bias=False)
        self.bn = BatchNorm2d(out_c)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class GNAP(Module):
    """Global Norm-Aware Pooling."""
    def __init__(self, in_c):
        super(GNAP, self).__init__()
        self.bn1 = BatchNorm2d(in_c, affine=False)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.bn2 = BatchNorm1d(in_c, affine=False)

    def forward(self, x):
        x = self.bn1(x)
        x_norm = torch.norm(x, p=2, dim=1, keepdim=True)
        x_norm_mean = torch.mean(x_norm)
        weight = x_norm_mean / x_norm
        x = x * weight
        x = self.pool(x)
        # aman untuk non-contiguous
        x = x.reshape(x.shape[0], -1)
        feature = self.bn2(x)
        return feature


class GDC(Module):
    """Global Depthwise Convolution block."""
    def __init__(self, in_c, embedding_size):
        super(GDC, self).__init__()
        self.conv_6_dw = LinearBlock(in_c, in_c, groups=in_c, kernel=(7, 7), stride=(1, 1), padding=(0, 0))
        self.conv_6_flatten = Flatten()
        self.linear = Linear(in_c, embedding_size, bias=False)
        self.bn = BatchNorm1d(embedding_size, affine=False)

    def forward(self, x):
        x = self.conv_6_dw(x)
        x = self.conv_6_flatten(x)
        x = self.linear(x)
        x = self.bn(x)
        return x


class SEModule(Module):
    """Squeeze-and-Excitation block."""
    def __init__(self, channels, reduction):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = Conv2d(channels, channels // reduction, kernel_size=1, padding=0, bias=False)
        nn.init.xavier_uniform_(self.fc1.weight.data)
        self.relu = ReLU(inplace=True)
        self.fc2 = Conv2d(channels // reduction, channels, kernel_size=1, padding=0, bias=False)
        self.sigmoid = Sigmoid()

    def forward(self, x):
        module_input = x
        x = self.avg_pool(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return module_input * x


class BasicBlockIR(Module):
    """BasicBlock untuk IRNet."""
    def __init__(self, in_channel, depth, stride):
        super(BasicBlockIR, self).__init__()
        if in_channel == depth:
            self.shortcut_layer = MaxPool2d(1, stride)
        else:
            self.shortcut_layer = Sequential(
                Conv2d(in_channel, depth, (1, 1), stride, bias=False),
                BatchNorm2d(depth))
        self.res_layer = Sequential(
            BatchNorm2d(in_channel),
            Conv2d(in_channel, depth, (3, 3), (1, 1), 1, bias=False),
            BatchNorm2d(depth),
            PReLU(depth),
            Conv2d(depth, depth, (3, 3), stride, 1, bias=False),
            BatchNorm2d(depth))

    def forward(self, x):
        shortcut = self.shortcut_layer(x)
        res = self.res_layer(x)
        return res + shortcut


class BottleneckIR(Module):
    """Bottleneck untuk IRNet."""
    def __init__(self, in_channel, depth, stride):
        super(BottleneckIR, self).__init__()
        reduction_channel = depth // 4
        if in_channel == depth:
            self.shortcut_layer = MaxPool2d(1, stride)
        else:
            self.shortcut_layer = Sequential(
                Conv2d(in_channel, depth, (1, 1), stride, bias=False),
                BatchNorm2d(depth))
        self.res_layer = Sequential(
            BatchNorm2d(in_channel),
            Conv2d(in_channel, reduction_channel, (1, 1), (1, 1), 0, bias=False),
            BatchNorm2d(reduction_channel),
            PReLU(reduction_channel),
            Conv2d(reduction_channel, reduction_channel, (3, 3), (1, 1), 1, bias=False),
            BatchNorm2d(reduction_channel),
            PReLU(reduction_channel),
            Conv2d(reduction_channel, depth, (1, 1), stride, 0, bias=False),
            BatchNorm2d(depth))

    def forward(self, x):
        shortcut = self.shortcut_layer(x)
        res = self.res_layer(x)
        return res + shortcut


class BasicBlockIRSE(BasicBlockIR):
    def __init__(self, in_channel, depth, stride):
        super(BasicBlockIRSE, self).__init__(in_channel, depth, stride)
        self.res_layer.add_module("se_block", SEModule(depth, 16))


class BottleneckIRSE(BottleneckIR):
    def __init__(self, in_channel, depth, stride):
        super(BottleneckIRSE, self).__init__(in_channel, depth, stride)
        self.res_layer.add_module("se_block", SEModule(depth, 16))


class Bottleneck(namedtuple('Block', ['in_channel', 'depth', 'stride'])):
    """Deskripsi blok ResNet."""


def get_block(in_channel, depth, num_units, stride=2):
    return [Bottleneck(in_channel, depth, stride)] + \
           [Bottleneck(depth, depth, 1) for _ in range(num_units - 1)]


def get_blocks(num_layers):
    if num_layers == 18:
        blocks = [
            get_block(in_channel=64, depth=64, num_units=2),
            get_block(in_channel=64, depth=128, num_units=2),
            get_block(in_channel=128, depth=256, num_units=2),
            get_block(in_channel=256, depth=512, num_units=2)
        ]
    elif num_layers == 34:
        blocks = [
            get_block(in_channel=64, depth=64, num_units=3),
            get_block(in_channel=64, depth=128, num_units=4),
            get_block(in_channel=128, depth=256, num_units=6),
            get_block(in_channel=256, depth=512, num_units=3)
        ]
    elif num_layers == 50:
        blocks = [
            get_block(in_channel=64, depth=64, num_units=3),
            get_block(in_channel=64, depth=128, num_units=4),
            get_block(in_channel=128, depth=256, num_units=14),
            get_block(in_channel=256, depth=512, num_units=3)
        ]
    elif num_layers == 100:
        blocks = [
            get_block(in_channel=64, depth=64, num_units=3),
            get_block(in_channel=64, depth=128, num_units=13),
            get_block(in_channel=128, depth=256, num_units=30),
            get_block(in_channel=256, depth=512, num_units=3)
        ]
    elif num_layers == 152:
        blocks = [
            get_block(in_channel=64, depth=256, num_units=3),
            get_block(in_channel=256, depth=512, num_units=8),
            get_block(in_channel=512, depth=1024, num_units=36),
            get_block(in_channel=1024, depth=2048, num_units=3)
        ]
    elif num_layers == 200:
        blocks = [
            get_block(in_channel=64, depth=256, num_units=3),
            get_block(in_channel=256, depth=512, num_units=24),
            get_block(in_channel=512, depth=1024, num_units=36),
            get_block(in_channel=1024, depth=2048, num_units=3)
        ]
    else:
        raise ValueError("Unsupported num_layers")
    return blocks


class Backbone(Module):
    def __init__(self, input_size, num_layers, mode='ir'):
        """Args:
            input_size: (H, W)
            num_layers: 18, 34, 50, 100, 152, 200
            mode: 'ir' atau 'ir_se'
        """
        super(Backbone, self).__init__()
        assert input_size[0] in [112, 224], "input_size should be [112,112] or [224,224]"
        assert num_layers in [18, 34, 50, 100, 152, 200], "num_layers should be 18,34,50,100,152,200"
        assert mode in ['ir', 'ir_se'], "mode should be ir or ir_se"

        self.input_layer = Sequential(
            Conv2d(3, 64, (3, 3), 1, 1, bias=False),
            BatchNorm2d(64),
            PReLU(64)
        )

        blocks = get_blocks(num_layers)
        if num_layers <= 100:
            unit_module = BasicBlockIR if mode == 'ir' else BasicBlockIRSE
            output_channel = 512
        else:
            unit_module = BottleneckIR if mode == 'ir' else BottleneckIRSE
            output_channel = 2048

        if input_size[0] == 112:
            self.output_layer = Sequential(
                BatchNorm2d(output_channel),
                Dropout(0.4),
                Flatten(),
                Linear(output_channel * 7 * 7, 512),
                BatchNorm1d(512, affine=False)
            )
        else:
            self.output_layer = Sequential(
                BatchNorm2d(output_channel),
                Dropout(0.4),
                Flatten(),
                Linear(output_channel * 14 * 14, 512),
                BatchNorm1d(512, affine=False)
            )

        modules = []
        for block in blocks:
            for bottleneck in block:
                modules.append(unit_module(bottleneck.in_channel,
                                           bottleneck.depth,
                                           bottleneck.stride))
        self.body = Sequential(*modules)

        initialize_weights(self.modules())

    def forward(self, x):
        x = self.input_layer(x)
        for module in self.body:
            x = module(x)
        x = self.output_layer(x)
        # L2-normalize (ArcFace-style)
        norm = torch.norm(x, p=2, dim=1, keepdim=True)
        output = torch.div(x, norm)
        return output, norm


def IR_18(input_size):
    return Backbone(input_size, 18, 'ir')


def IR_34(input_size):
    return Backbone(input_size, 34, 'ir')


def IR_50(input_size):
    return Backbone(input_size, 50, 'ir')


def IR_101(input_size):
    return Backbone(input_size, 100, 'ir')


def IR_152(input_size):
    return Backbone(input_size, 152, 'ir')


def IR_200(input_size):
    return Backbone(input_size, 200, 'ir')


def IR_SE_50(input_size):
    return Backbone(input_size, 50, 'ir_se')


def IR_SE_101(input_size):
    return Backbone(input_size, 100, 'ir_se')


def IR_SE_152(input_size):
    return Backbone(input_size, 152, 'ir_se')


def IR_SE_200(input_size):
    return Backbone(input_size, 200, 'ir_se')

## JIka masih ada kendala, unduh repositori lengkapnya yang sudah dimodifikasi melalui link drive berikut:

- [Google Drive – Repositori Algoritma](https://drive.google.com/drive/folders/1aSw_qF2TgxW3qsc-LJMKzUfFcJhi6DcA?usp=sharing)