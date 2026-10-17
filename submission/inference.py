"""[이미지 미션 제출용 inference.py] — 실제(0) vs 생성·변형(1) 딸기 이미지 판별.

채점기는 이 파일을 import 한 뒤 load_model / predict 두 함수만 호출한다. (이름·인자 순서 변경 금지)
모델 구조는 detector.py, 설정·가중치는 model/ (docs/전략.md 12절). 오프라인 동작: 다운로드·pip·네트워크 없음.
"""
import math
import os
import sys
import json
from pathlib import Path

os.environ.setdefault('HF_HUB_OFFLINE', '1')          # 방어용: 실수로 허브에 접속하려 해도 막는다
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')

import numpy as np
import torch
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))                          # 같은 폴더의 detector.py를 상대 경로로 import
from detector import Detector, load_weights            # noqa: E402

STANDARD_SIZES = {(1024, 768), (768, 1024)}            # 대회 이미지 규격 [설명서 2절]

# ⚠️ 백본을 .half()로 바꾸면 DINOv3 ViT-L이 T4에서 첫 블록부터 inf/NaN이 난다(10/2 실측). 가중치는 fp32로 두고
#    GPU에서는 autocast(fp16)로 연산만 줄인다(잔차는 fp32로 남아 안전, 속도는 순수 fp16과 같다). 학습도 같은 방식이다.
USE_AMP = True


def _standardize(img):
    """방어용 — 규격 밖 크기가 오면 12-6과 같이 4:3(또는 3:4)로 가운데 크롭 → 1024×768(768×1024) 리사이즈.
    정상 평가 이미지는 이미 규격이라 여기 오지 않는다(정상 이미지는 크롭·리사이즈 없이 그대로 쓴다)."""
    w, h = img.size
    tw, th = (1024, 768) if w >= h else (768, 1024)
    if w * th > h * tw:
        cw, ch = h * tw // th, h
    else:
        cw, ch = w, w * th // tw
    left, top = (w - cw) // 2, (h - ch) // 2
    return img.crop((left, top, left + cw, top + ch)).resize((tw, th), Image.LANCZOS)


def _load_image(path, mean, std):
    img = Image.open(path).convert('RGB')
    if img.size not in STANDARD_SIZES:
        img = _standardize(img)
    x = torch.from_numpy(np.asarray(img, dtype=np.float32) / 255.0).permute(2, 0, 1)
    return (x - mean) / std


def load_model(model_dir: str):
    """model_dir = model/ 절대경로. 그 안의 config.json + 가중치 파일만 읽는다."""
    model_dir = Path(model_dir)
    cfg = json.loads((model_dir / 'config.json').read_text(encoding='utf-8'))
    model = Detector(cfg)
    load_weights(model, model_dir / cfg['weights'])
    model.prepare(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))   # GPU 없으면 CPU
    model.device_ = next(model.common.parameters()).device
    model.mean_ = torch.tensor(cfg['mean'], dtype=torch.float32).view(3, 1, 1)
    model.std_ = torch.tensor(cfg['std'], dtype=torch.float32).view(3, 1, 1)
    return model


@torch.no_grad()
def predict(model, image_paths: list[str]) -> list[float]:
    """각 이미지가 라벨 1(생성·변형)일 확률. 입력과 같은 길이·같은 순서, 0.0~1.0.
    사진을 한 장씩 처리한다(조각 12장이 한 묶음) → 다른 이미지와 섞이지 않아 순서가 바뀌어도 결과가 같다."""
    probs = []
    amp = USE_AMP and model.device_.type == 'cuda'             # GPU에서만 fp16 연산(CPU면 그대로 fp32)
    for p in image_paths:
        x = _load_image(p, model.mean_, model.std_).unsqueeze(0).to(model.device_)
        with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=amp):
            tokens = model.get_tokens(x)                       # 백본은 fp16 연산
        common = model.common(x)                               # Common CNN은 fp32 — 강한 노이즈 사진에서 fp16 값이 크게 어긋남(10/6 실측)
        with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=amp):
            logit = model.combine(tokens, common)['logit']
        prob = torch.sigmoid(logit.float()).item()
        probs.append(prob if math.isfinite(prob) else 0.5)     # NaN·inf는 회차 실패 → 0.5로 방어
    return probs
