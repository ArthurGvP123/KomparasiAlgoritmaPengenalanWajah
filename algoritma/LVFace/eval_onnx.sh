# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT
mkdir work_dirs/$1/$2
python3 torch2onnx_v1.py work_dirs/$1/$2.pt --output work_dirs/$1/$2/$2.onnx --network $3
CUDA_VISIBLE_DEVICES=$4 python3 onnx_ijbc.py --model-root work_dirs/$1/$2/ --image-path data/ijb/$5 --target $5
