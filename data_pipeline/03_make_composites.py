"""3단계: ② 합성 가짜 만들기 (CPU — 로컬·Colab 어디서나).

문제설명서의 정의: "작물 영역 등 이미지의 일부를 다른 배경·영역과 합친 이미지".
공개샘플 합성 3장은 작물(잎 덩어리)을 오려 다른 사진의 배경(통로·바닥) 위에 붙였고, 가장자리에 흰 여백이 남기도 했다.
두 방식을 반씩 만든다.

  bgswap   : 사진 A의 작물(초록 영역)을 오려 사진 B의 배경 위에 붙임. B는 후보 중 초록이 가장 적은(통로·바닥 위주) 사진.
             일부는 아래쪽에 흰 띠를 남김.
  objpaste : 사진 A의 부위 사각형(AI Hub bbox) 1~3개를 작물 모양대로 오려 사진 B의 다른 위치에 붙임.

- 작물 영역은 "초록 정도(ExG = 2G−R−B)"로 자동 분리한다 (추가 패키지 없이 numpy·PIL만).
- A와 B는 같은 split(train/val)에서만 고른다 → 검증 사진의 조각이 학습에 섞이지 않게.
- 붙이는 위치는 normalize.py가 4:3으로 잘라낼 때 남는 영역 안으로 제한 (잘려서 가짜 흔적이 사라지지 않게).

사용법:
  py data_pipeline/03_make_composites.py ./data --n 1000
"""

import argparse, json, os, random
import numpy as np
from PIL import Image, ImageFilter
from common import ManifestWriter, center_box, fake_usage, least_used_first, read_manifest, real_rows


def plant_mask(im):
    """초록 정도(ExG)가 높은 픽셀 = 작물. 0~255 흑백 마스크를 돌려준다."""
    a = np.asarray(im.convert('RGB')).astype(np.float32) / 255
    exg = 2 * a[..., 1] - a[..., 0] - a[..., 2]
    hist, edges = np.histogram(exg, bins=256)
    # 오츠(Otsu) 방법: 두 무리(작물/배경)로 가장 잘 나뉘는 기준값을 자동으로 찾는다
    w = np.cumsum(hist); mu = np.cumsum(hist * edges[:-1])
    between = (mu[-1] * w - mu * w[-1]) ** 2 / (w * (w[-1] - w) + 1e-9)
    t = edges[np.argmax(between)]
    m = Image.fromarray(((exg > t) * 255).astype(np.uint8))
    m = m.filter(ImageFilter.MaxFilter(9)).filter(ImageFilter.MinFilter(9))    # 어두운 잎 때문에 생긴 구멍 메우기
    return m.filter(ImageFilter.MinFilter(3)).filter(ImageFilter.MaxFilter(3))  # 떨어진 작은 점 지우기


def green_ratio(r, load):
    """사진에서 작물(초록)이 차지하는 비율 — 작게 줄여서 빠르게 잰다."""
    im = load(r)
    im.thumbnail((256, 256))
    return np.asarray(plant_mask(im)).mean() / 255


def fit(im, size):
    """비율을 유지한 채 가운데를 잘라 size로 맞춘다 (찌그러뜨리지 않음)."""
    W, H = size
    w, h = im.size
    s = max(W / w, H / h)
    im = im.resize((max(W, round(w * s)), max(H, round(h * s))), Image.LANCZOS)
    l, t = (im.width - W) // 2, (im.height - H) // 2
    return im.crop((l, t, l + W, t + H))


def bgswap(donor, base, rng):
    bg = fit(base, donor.size)
    feather = rng.uniform(0.5, 4)
    m = plant_mask(donor).filter(ImageFilter.GaussianBlur(feather))   # 경계를 살짝 흐리게 → 자연스럽게
    out = Image.composite(donor, bg, m)
    params = {'feather': round(feather, 2), 'white_strip': 0}
    if rng.random() < 0.3:                                           # 공개샘플 yuag.jpg처럼 아래쪽 흰 여백
        l, t, r, b = center_box(*out.size)
        hh = int((b - t) * rng.uniform(0.03, 0.10))
        white = Image.new('RGB', out.size, (255, 255, 255))
        strip = Image.new('L', out.size, 0)
        strip.paste(255, (l, b - hh, r, b))
        strip = Image.composite(Image.new('L', out.size, 0), strip, m)   # 작물 위에는 칠하지 않음(잎이 띠를 덮게)
        out = Image.composite(white, out, strip)
        params['white_strip'] = hh
    return out, params


def objpaste(donor, base, boxes, rng):
    out = base.copy()
    l, t, r, b = center_box(*base.size)
    area = donor.width * donor.height
    cands = [bx for bx in boxes if 0.01 * area < bx[2] * bx[3] < 0.25 * area]
    rng.shuffle(cands)
    pasted = []
    for x, y, w, h in cands[:rng.randint(1, 3)]:
        piece = donor.crop((int(x), int(y), int(x + w), int(y + h)))
        s = rng.uniform(0.8, 1.25) * base.width / donor.width        # 두 사진 크기 차이 보정 + 약간의 확대·축소
        pw, ph = max(8, int(piece.width * s)), max(8, int(piece.height * s))
        if pw >= r - l or ph >= b - t:
            continue
        piece = piece.resize((pw, ph), Image.LANCZOS)
        m = plant_mask(piece)
        if np.asarray(m).mean() < 25:                                # 초록이 거의 없으면(열매·꽃 등) 사각형을 부드럽게
            m = Image.new('L', piece.size, 0)
            m.paste(255, (pw // 8, ph // 8, pw - pw // 8, ph - ph // 8))
        m = m.filter(ImageFilter.GaussianBlur(rng.uniform(0.5, 3)))
        px, py = rng.randint(l, r - pw), rng.randint(t, b - ph)      # 잘려 나가지 않는 영역 안에
        out.paste(piece, (px, py), m)
        pasted.append([px, py, pw, ph])
    return out, {'pasted': pasted}


def main(root, n, seed):
    rng = random.Random(seed)
    boxes = json.load(open(os.path.join(root, 'raw', 'real_boxes.json'), encoding='utf-8'))
    pools = {s: real_rows(root, s) for s in ('train', 'val')}
    mw = ManifestWriter(root, 'composite')
    os.makedirs(os.path.join(root, 'raw', 'fake', 'composite'), exist_ok=True)
    load = lambda r: Image.open(os.path.join(root, 'raw', r['file'])).convert('RGB')

    # 재료 고르기: 다른 가짜(VAE·LaMa·필터 등)에 덜 쓰인 원본부터 → 같은 사진이 여러 번 재사용되는 걸 줄인다
    done_src = {r['source_id'] for r in read_manifest(mw.path)}   # 이어서 하기: 이미 만든 원본은 빼고 이어서
    jobs = [r for s in pools for r in pools[s]]
    jobs = [r for r in least_used_first(jobs, fake_usage(root, exclude_step='composite'), rng)
            if r['source_id'] not in done_src]

    made = len(done_src)
    for i, base_row in enumerate(jobs):
        if made >= n:
            break
        mode = 'bgswap' if i % 2 == 0 else 'objpaste'
        rel = f'fake/composite/{base_row["source_id"]}__{mode}.png'
        rng_i = random.Random(f'{seed}-{base_row["source_id"]}')
        pool = [p for p in pools[base_row['split']] if p['group'] != base_row['group']]   # 다른 포기에서
        other = rng_i.sample(pool, min(12, len(pool)))
        if mode == 'bgswap':
            # 이 사진(base_row)의 작물을 오려서, 후보 12장 중 초록이 가장 적은(통로·바닥이 많은) 사진을 배경으로
            other_row = min(other, key=lambda p: green_ratio(p, load))
            out, params = bgswap(load(base_row), load(other_row), rng_i)
            params['background'] = other_row['source_id']
        else:
            # 다른 사진(other_row)의 부위를 오려 이 사진 위에 붙임
            other_row = other[0]
            out, params = objpaste(load(other_row), load(base_row), boxes.get(other_row['source_id'], []), rng_i)
            if not params['pasted']:
                continue                                             # 붙일 조각이 없으면 건너뜀
            params['donor'] = other_row['source_id']
        out.save(os.path.join(root, 'raw', rel))
        mw.add(file=rel, label=1, type='composite', family='E5', generator=f'composite-{mode}',
               source_id=base_row['source_id'], group=base_row['group'], split=base_row['split'], params=params)
        made += 1
        if made % 200 == 0:
            print(f'{made}/{n}')
    mw.close()
    print(f'완료 {made}장 → 표: {mw.path}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('root')
    ap.add_argument('--n', type=int, default=1000)
    ap.add_argument('--seed', type=int, default=2)
    a = ap.parse_args()
    main(a.root, a.n, a.seed)
