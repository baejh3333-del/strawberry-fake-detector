"""2단계: ③ 지정 영상처리 필터 가짜 만들기 (CPU — 로컬·Colab 어디서나).

문제설명서의 정의: "제작 단계에서 블러, 가우시안 노이즈, 소금·후추 잡음 또는 JPEG 압축률 변경을 의도적으로 적용한 이미지".
→ 실제 사진 원본(raw)에 필터 하나를 걸어 raw 가짜로 저장. 규격 맞추기는 나중(05_finalize.py)에 실제·가짜 똑같이.

강도는 비공개라 넓게 흩뿌리고, **거의 안 보이는 약한 것**도 포함한다 (전략.md 결정 ⑥).
저장: 블러·노이즈·소금후추는 PNG(추가 압축 없음), JPEG 압축률 변경은 그 품질의 JPEG.

사용법:
  py data_pipeline/02_make_filters.py ./data --n 1000
"""

import argparse, os, random
import numpy as np
from PIL import Image, ImageFilter
from common import ManifestWriter, fake_usage, least_used_first, read_manifest, real_rows

KINDS = ['blur', 'noise', 'saltpepper', 'jpeg']


def apply(im, kind, rng):
    if kind == 'blur':
        s = round(rng.uniform(0.3, 3.0), 2)                       # 0.3이면 눈으로 거의 안 보임
        return im.filter(ImageFilter.GaussianBlur(s)), {'sigma': s}
    a = np.asarray(im).astype(np.float32)
    nrng = np.random.default_rng(rng.randrange(2**31))
    if kind == 'noise':
        s = round(rng.uniform(2, 25), 1)                          # 0~255 기준 표준편차
        a = a + nrng.normal(0, s, a.shape)
        return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8)), {'std': s}
    if kind == 'saltpepper':
        amt = round(rng.uniform(0.001, 0.05), 4)                  # 점이 찍히는 픽셀 비율
        m = nrng.random(a.shape[:2])
        a[m < amt / 2] = 0
        a[(m >= amt / 2) & (m < amt)] = 255
        return Image.fromarray(a.astype(np.uint8)), {'amount': amt}
    raise ValueError(kind)


def main(root, n, seed):
    rng = random.Random(seed)
    mw = ManifestWriter(root, 'filter')
    os.makedirs(os.path.join(root, 'raw', 'fake', 'filter'), exist_ok=True)

    # 재료 고르기: 다른 가짜(VAE·LaMa 등)에 덜 쓰인 원본부터 → 같은 사진이 여러 번 재사용되는 걸 줄인다
    done_src = {r['source_id'] for r in read_manifest(mw.path)}   # 이어서 하기: 이미 만든 원본은 빼고 이어서
    reals = least_used_first(real_rows(root), fake_usage(root, exclude_step='filter'), rng)
    reals = [r for r in reals if r['source_id'] not in done_src]
    start = len(done_src)

    for i, r in enumerate(reals[:max(n - start, 0)], start=start):
        kind = KINDS[i % len(KINDS)]                              # 네 종류를 고르게
        ext = 'jpg' if kind == 'jpeg' else 'png'
        rel = f'fake/filter/{r["source_id"]}__{kind}.{ext}'
        if rel in mw.done:
            continue
        rng_i = random.Random(f'{seed}-{r["source_id"]}')         # 사진마다 고정 난수 → 다시 돌려도 같은 결과(재현)
        im = Image.open(os.path.join(root, 'raw', r['file'])).convert('RGB')
        dst = os.path.join(root, 'raw', rel)
        if kind == 'jpeg':
            q = rng_i.randint(10, 90)                             # 압축률 변경. 90 근처는 거의 티 안 남
            im.save(dst, 'JPEG', quality=q)
            params = {'quality': q}
        else:
            out, params = apply(im, kind, rng_i)
            out.save(dst)
        mw.add(file=rel, label=1, type='filter', family='E5', generator=f'filter-{kind}',
               source_id=r['source_id'], group=r['group'], split=r['split'], params=params)
        if (i + 1) % 200 == 0:
            print(f'{i + 1}/{n}')
    mw.close()
    print(f'완료 → 표: {mw.path}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('root')
    ap.add_argument('--n', type=int, default=1000)
    ap.add_argument('--seed', type=int, default=1)
    a = ap.parse_args()
    main(a.root, a.n, a.seed)
