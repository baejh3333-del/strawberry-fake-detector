"""6단계: LaMa 부분 생성(⑤ 지우개형) 가짜 만들기 (Colab GPU).

실제 사진에서 부위 사각형(AI Hub bbox) 1~2개를 통째로 지우고 LaMa가 주변을 보고 메운다.
"일부 영역을 생성 모델로 바꾼" 유형 ⑤(부분 생성)에 해당. 전략.md 11-4의 "⑤ 지우개형 부분 생성", 게이팅 family는 E4(부분 편집).

- 모델: big-lama TorchScript 가중치. 원저작자 saic-mdal/lama(Places2/Places-Challenge로 학습, Apache-2.0)를
  enesmsahin/simple-lama-inpainting이 같은 라이선스로 GitHub 릴리스에 올려 둔 것을 직접 내려받는다.
  (simple-lama-inpainting **패키지**는 pillow<10·numpy<2.0을 요구해 Python 3.13/numpy 2.x인 Colab에서 설치가
  안 되므로 패키지는 안 쓰고, 전처리·후처리 코드만 그 패키지 방식대로 직접 구현했다.)
- 지울 영역은 03_make_composites.py의 objpaste와 같은 기준(사진 면적의 1~25%)으로 부위 사각형을 고르고,
  가장자리에 15% 여백을 더해 지운다 — 경계를 딱 붙여 지우면 LaMa가 윤곽 흔적을 그대로 남기기 쉽다.
- 500장으로 800장을 채워야 하면 사진 하나에 서로 다른 부위를 여러 번 지워 개수를 채운다(03과 같은 방식).
- 사진이 커서 GPU 메모리가 모자라면 절반 크기로 처리한 뒤 원래 크기로 늘린다(params에 기록).

Colab 준비 (셀 하나):
  from google.colab import drive; drive.mount('/content/drive')
실행:
  !python repo/data_pipeline/06_lama_inpaint_colab.py /content/drive/MyDrive/work --n 800
"""

import argparse, json, os, random, urllib.request
import numpy as np
import torch
from PIL import Image
from common import ManifestWriter, center_box, real_rows

MIN_FRAC, MAX_FRAC = 0.01, 0.25   # 03_make_composites.py의 objpaste와 같은 기준
PAD_RATIO = 0.15                  # 부위 사각형 가장자리에 더 지울 여백 (경계 흔적 방지)
MODEL_URL = 'https://github.com/enesmsahin/simple-lama-inpainting/releases/download/v0.1.0/big-lama.pt'


def load_lama(dev, cache_path=None):
    # torch.jit.load는 Windows에서 경로에 한글 등 비ASCII 문자가 있으면 못 연다 (errno 42) →
    # 한글 없는 위치(LOCALAPPDATA, 없으면 /content)에 캐시한다.
    cache_path = cache_path or os.path.join(os.environ.get('LOCALAPPDATA', '/content'), 'lama_cache', 'big-lama.pt')
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    if not os.path.exists(cache_path):
        print('big-lama.pt 다운로드 중 (약 200MB, Apache-2.0)...')
        urllib.request.urlretrieve(MODEL_URL, cache_path)
    model = torch.jit.load(cache_path, map_location=dev)
    model.eval()
    return model


def ceil_modulo(x, mod):
    return x if x % mod == 0 else (x // mod + 1) * mod


def pad_chw(a, mod=8):
    c, h, w = a.shape
    return np.pad(a, ((0, 0), (0, ceil_modulo(h, mod) - h), (0, ceil_modulo(w, mod) - w)), mode='symmetric')


@torch.no_grad()
def lama_infer(model, im, mask, dev):
    """simple-lama-inpainting과 같은 전처리(0~1 정규화, CHW, 8배수 패딩)·후처리 방식."""
    h, w = im.height, im.width
    img = pad_chw(np.transpose(np.asarray(im), (2, 0, 1)).astype(np.float32) / 255)
    msk = pad_chw(np.asarray(mask)[None].astype(np.float32) / 255)
    msk = (msk > 0).astype(np.float32)
    img_t = torch.from_numpy(img).unsqueeze(0).to(dev)
    msk_t = torch.from_numpy(msk).unsqueeze(0).to(dev)
    out = model(img_t, msk_t)[0].permute(1, 2, 0).cpu().numpy()
    out = np.clip(out * 255, 0, 255).astype(np.uint8)[:h, :w]
    return Image.fromarray(out)


def pick_boxes(boxes, area, rng, k):
    cands = [b for b in boxes if MIN_FRAC * area < b[2] * b[3] < MAX_FRAC * area]
    rng.shuffle(cands)
    return cands[:k]


def make_mask(size, boxes):
    m = Image.new('L', size, 0)
    w, h = size
    for x, y, bw, bh in boxes:
        px, py = bw * PAD_RATIO, bh * PAD_RATIO
        l, t = max(0, int(x - px)), max(0, int(y - py))
        r, b = min(w, int(x + bw + px)), min(h, int(y + bh + py))
        m.paste(255, (l, t, r, b))
    return m


def run_lama(model, im, mask, dev):
    try:
        return lama_infer(model, im, mask, dev), False
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        small = im.resize((im.width // 2, im.height // 2), Image.LANCZOS)
        small_mask = mask.resize(small.size, Image.NEAREST)
        out = lama_infer(model, small, small_mask, dev).resize(im.size, Image.LANCZOS)
        return out, True


def main(root, n, seed):
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = load_lama(dev)
    print('LaMa 로드 완료 (device:', dev, ')')

    boxes_all = json.load(open(os.path.join(root, 'raw', 'real_boxes.json'), encoding='utf-8'))
    pools = {s: real_rows(root, s) for s in ('train', 'val')}
    jobs = [r for s in pools for r in pools[s]]
    rng = random.Random(seed)
    rng.shuffle(jobs)

    mw = ManifestWriter(root, 'lama_partial')
    os.makedirs(os.path.join(root, 'raw', 'fake', 'lama_partial'), exist_ok=True)

    made, attempt = 0, 0
    while made < n and attempt < n * 4:   # 이미지 하나를 여러 번 우려써도 무한루프는 안 되게
        row = jobs[attempt % len(jobs)]
        pass_no = attempt // len(jobs)
        attempt += 1
        rel = f'fake/lama_partial/{row["source_id"]}__lama{pass_no}.png'
        if rel in mw.done:
            made += 1
            continue

        im = Image.open(os.path.join(root, 'raw', row['file'])).convert('RGB')
        cand_boxes = boxes_all.get(row['source_id'], [])
        rng_i = random.Random(f'{seed}-{row["source_id"]}-{pass_no}')
        picked = pick_boxes(cand_boxes, im.width * im.height, rng_i, rng_i.randint(1, 2))
        if not picked:
            continue   # 지울 만한 부위가 없으면 건너뜀 (개수는 다음 시도에서 채움)

        mask = make_mask(im.size, picked)
        out, downscaled = run_lama(model, im, mask, dev)
        if not torch.isfinite(torch.from_numpy(np.asarray(out).astype(np.float32))).all():
            continue   # NaN/inf 결과는 버림
        out.save(os.path.join(root, 'raw', rel))
        mw.add(file=rel, label=1, type='partial_gen', family='E4', generator='lama-erase',
               source_id=row['source_id'], group=row['group'], split=row['split'],
               params={'boxes': picked, 'downscaled': downscaled})
        made += 1
        if made % 100 == 0:
            print(f'{made}/{n}')
    mw.close()
    print(f'완료 {made}장 → 표: {mw.path}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('root')
    ap.add_argument('--n', type=int, default=800)
    ap.add_argument('--seed', type=int, default=4)
    a = ap.parse_args()
    main(a.root, a.n, a.seed)
