"""10단계: SDXL 인페인팅 부분 생성(⑤ 새로 그려 넣기형) 가짜 만들기 — 게이팅 family E4(부분 편집). Colab A100/L4.

06(LaMa)은 부위를 **지우고 메우는** 방식, 이 스크립트는 같은 부위 사각형(AI Hub bbox 1~2개)을 **새로 그려 넣는** 방식이다.
대회의 부분 생성이 어느 쪽인지 모르므로 두 방식을 섞는다(전략 12-2c). 실제 사진이 필요하다.

- 모델: diffusers/stable-diffusion-xl-1.0-inpainting-0.1 (허깅페이스, CreativeML Open RAIL++-M) + madebyollin/sdxl-vae-fp16-fix(MIT, fp16 NaN 방지).
- 규격: 실제 사진(AI Hub, 크기·비율 제각각)을 **먼저** 4:3 가운데 크롭 → LANCZOS로 1024×768(가로)/768×1024(세로)로 맞춘 뒤 인페인팅한다.
  normalize.py와 같은 방식이라 실제 사진이 규격화에서 거치는 것과 똑같이 한 번만 축소되고, 결과는 그 크기 그대로 저장해 추가 리사이즈가 없다.
- 부위 사각형은 원본 좌표라서 크롭·축소에 맞춰 변환하고, 사진 면적의 1~25%인 것만 쓴다(06과 같은 기준). 가장자리에 15% 여백을 더한다.
- 두 가지 결과를 절반씩: 'full' = 인페인팅 결과 그대로(사진 전체가 VAE를 한 번 거침), 'pasted' = 마스크 안쪽만 원본에 붙임(나머지는 원본 그대로, 경계를 살짝 흐림).
- 마스크 안에 그릴 내용은 성장 단계와 상관없이 부위에 맞게: 잎 위주, 꽃·열매는 소량(prompts.py의 v5 원칙 — 잎만 있는 부위에 열매를 그리지 않는 쪽으로 잎 비중을 높게).
- 07·08·09와 같은 규칙: 표(manifest), 이어서 하기, --sync_to 로 로컬 디스크에 먼저 쓰고 Drive에 주기 복사.

준비물(real_root 안): raw/real/*.jpg, raw/real_boxes.json, manifests/real.csv  ← 01_extract_real.py 결과. 개인 Drive에만 두고 공유 금지(AI Hub 이용조건).

Colab 실행:
  !python 10_sdxl_inpaint.py /content/drive/MyDrive/work --work /content/work_inp --n 10 --sync_to /content/drive/MyDrive/work
"""

import argparse, json, os, random, time
import numpy as np
from PIL import Image, ImageFilter
from common import ManifestWriter, Syncer, read_manifest, split_of

REPO = 'diffusers/stable-diffusion-xl-1.0-inpainting-0.1'
VAE = 'madebyollin/sdxl-vae-fp16-fix'
NAME = 'sdxl_inpaint'
MIN_FRAC, MAX_FRAC = 0.01, 0.25   # 06_lama_inpaint_colab.py와 같은 기준
PAD_RATIO = 0.15
STEPS, GUIDANCE, STRENGTH = 30, 7.5, 0.99

# 마스크 안에 그릴 내용 (가중치). 잎 위주 — 꽃·열매는 소량
CONTENT = [
    ('dark green serrated strawberry leaves', 0.60),
    ('green strawberry leaves and thin stems', 0.15),
    ('small white strawberry flowers among green leaves', 0.10),
    ('a ripe red strawberry', 0.08),
    ('a small unripe green strawberry', 0.07),
]
NEGATIVE = 'painting, illustration, cartoon, 3d render, text, watermark, logo, deformed, blurry, low quality'


def center_box(w, h):
    """normalize.py가 4:3(또는 3:4)로 가운데를 잘라낼 때 남는 영역 (left, top, right, bottom)."""
    W, H = (4, 3) if w >= h else (3, 4)
    if w * H > h * W:
        cw, ch = h * W // H, h
    else:
        cw, ch = w, w * H // W
    left, top = (w - cw) // 2, (h - ch) // 2
    return left, top, left + cw, top + ch


def standardize(im):
    """(규격 이미지, 크롭 영역, 배율). 가로 1024×768 / 세로 768×1024."""
    w, h = im.size
    box = center_box(w, h)
    tw, th = (1024, 768) if w >= h else (768, 1024)
    out = im.crop(box).resize((tw, th), Image.LANCZOS)
    scale = tw / (box[2] - box[0])
    return out, box, scale


def transform_boxes(boxes, crop, scale, size):
    """원본 좌표 (x, y, bw, bh) → 규격 이미지 좌표. 크롭 밖으로 나간 부분은 잘라 내고, 면적이 1~25%인 것만 남긴다."""
    left, top = crop[0], crop[1]
    W, H = size
    out = []
    for x, y, bw, bh in boxes:
        x0, y0 = max(0.0, (x - left) * scale), max(0.0, (y - top) * scale)
        x1, y1 = min(float(W), (x + bw - left) * scale), min(float(H), (y + bh - top) * scale)
        if x1 - x0 < 8 or y1 - y0 < 8:
            continue
        area = (x1 - x0) * (y1 - y0)
        if MIN_FRAC * W * H < area < MAX_FRAC * W * H:
            out.append((int(x0), int(y0), int(x1 - x0), int(y1 - y0)))
    return out


def make_mask(size, boxes):
    m = Image.new('L', size, 0)
    W, H = size
    for x, y, bw, bh in boxes:
        px, py = bw * PAD_RATIO, bh * PAD_RATIO
        m.paste(255, (max(0, int(x - px)), max(0, int(y - py)), min(W, int(x + bw + px)), min(H, int(y + bh + py))))
    return m


def pick_content(rng):
    x, acc = rng.random(), 0.0
    for text, w in CONTENT:
        acc += w
        if x < acc:
            return text
    return CONTENT[0][0]


def paste_masked(orig, gen, mask, feather=6):
    """마스크 안쪽만 gen, 나머지는 orig 그대로. 경계는 살짝 흐려 붙인다."""
    soft = mask.filter(ImageFilter.GaussianBlur(feather)) if feather else mask
    return Image.composite(gen, orig, soft)


def load_pipe():
    import torch
    from diffusers import AutoPipelineForInpainting, AutoencoderKL
    vae = AutoencoderKL.from_pretrained(VAE, torch_dtype=torch.float16)
    try:
        pipe = AutoPipelineForInpainting.from_pretrained(REPO, vae=vae, torch_dtype=torch.float16, variant='fp16')
    except Exception:
        pipe = AutoPipelineForInpainting.from_pretrained(REPO, vae=vae, torch_dtype=torch.float16)
    pipe.to('cuda')
    pipe.set_progress_bar_config(disable=True)
    return pipe


def main(real_root, work, n, seed, sync_to, sync_every):
    import torch
    if not torch.cuda.is_available():
        raise SystemExit('GPU가 없으면 중단합니다.')
    boxes_all = json.load(open(os.path.join(real_root, 'raw', 'real_boxes.json'), encoding='utf-8'))
    rows = [r for r in read_manifest(os.path.join(real_root, 'manifests', 'real.csv')) if r['label'] == '0']
    rng = random.Random(seed)
    rng.shuffle(rows)
    pipe = load_pipe()

    syncer = Syncer(work, sync_to, NAME) if sync_to else None
    if syncer:
        syncer.pull_manifest()
    mw = ManifestWriter(work, NAME)
    os.makedirs(os.path.join(work, 'raw', 'fake', NAME), exist_ok=True)
    made, attempt, t0 = 0, 0, time.time()
    while made < n and attempt < n * 6:
        row = rows[attempt % len(rows)]
        pass_no = attempt // len(rows)
        attempt += 1
        rel = f'fake/{NAME}/{row["source_id"]}__sdxlinp{pass_no}.png'
        if rel in mw.done:
            made += 1
            continue
        rng_i = random.Random(f'{seed}-{row["source_id"]}-{pass_no}')
        im = Image.open(os.path.join(real_root, 'raw', row['file'])).convert('RGB')
        std, crop, scale = standardize(im)
        cands = transform_boxes(boxes_all.get(row['source_id'], []), crop, scale, std.size)
        if not cands:
            continue
        rng_i.shuffle(cands)
        picked = cands[:rng_i.randint(1, 2)]
        mask = make_mask(std.size, picked)
        content = pick_content(rng_i)
        prompt = f'{content}, natural greenhouse photo, realistic, same lighting as the surroundings'
        mode = 'pasted' if rng_i.random() < 0.5 else 'full'
        g = torch.Generator('cpu').manual_seed(rng_i.randint(0, 2**31 - 1))
        gen = pipe(prompt=prompt, negative_prompt=NEGATIVE, image=std, mask_image=mask, width=std.width, height=std.height,
                   num_inference_steps=STEPS, guidance_scale=GUIDANCE, strength=STRENGTH, generator=g).images[0]
        if gen.size != std.size:
            gen = gen.resize(std.size, Image.LANCZOS)   # 기대하지 않음 — 크기가 다르면 기록해 둔다
        out = paste_masked(std, gen, mask) if mode == 'pasted' else gen
        out.save(os.path.join(work, 'raw', rel))
        mw.add(file=rel, label=1, type='partial_gen', family='E4', generator=f'sdxl-inpaint-{mode}',
               source_id=row['source_id'], group=row['group'], split=row['split'],
               params={'repo': REPO, 'vae': VAE, 'mode': mode, 'prompt': prompt, 'boxes': picked, 'size': list(std.size),
                       'steps': STEPS, 'guidance': GUIDANCE, 'strength': STRENGTH, 'crop': list(crop), 'scale': round(scale, 5)})
        made += 1
        if syncer and made % sync_every == 0:
            syncer.push()
        if made % 10 == 0 or made == n:
            per = (time.time() - t0) / made
            print(f'{made}/{n} (장당 {per:.0f}초, 남은 시간 약 {per * (n - made) / 60:.0f}분)', flush=True)
    mw.close()
    if syncer:
        ok = syncer.push(retries=5)
        print('Drive 복사 완료' if ok else f'!! Drive 복사 실패 — 결과는 {work} 에 남아 있으니 Drive 연결 후 다시 실행하면 이어서 복사됩니다', flush=True)
    print(f'완료 {made}장 → 표: {mw.path}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('real_root', help='raw/real, raw/real_boxes.json, manifests/real.csv 가 있는 폴더(개인 Drive)')
    ap.add_argument('--work', required=True, help='결과를 먼저 쓰는 로컬 폴더(Colab: /content/work_inp)')
    ap.add_argument('--n', type=int, default=10)
    ap.add_argument('--seed', type=int, default=5)
    ap.add_argument('--sync_to', default=None)
    ap.add_argument('--sync_every', type=int, default=50)
    a = ap.parse_args()
    main(a.real_root, a.work, a.n, a.seed, a.sync_to, a.sync_every)
