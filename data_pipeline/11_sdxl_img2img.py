"""11단계: SDXL img2img 전체 스타일 변환(④) 가짜 만들기 — 게이팅 family E1(확산 U-Net). Colab T4/A100.

실제 사진 한 장 전체를 SDXL로 다시 그린다. 그림체(유화·애니 등)가 아니라 육안으로 구분이 안 될 정도로 사실적인 사진 그대로 재생성한다.
strength가 클수록 원본에서 많이 멀어진다. 대회 ④의 강도를 모르므로 0.15~0.45 범위로 흩뿌린다(전략 결정 ⑥과 같은 발상).

- 모델: stabilityai/stable-diffusion-xl-base-1.0 (허깅페이스, CreativeML Open RAIL++-M) + madebyollin/sdxl-vae-fp16-fix(MIT).
- 규격: 10_sdxl_inpaint.py와 같다 — 실제 사진을 먼저 4:3 가운데 크롭 → 1024×768/768×1024로 맞춘 뒤 변환하고, 그 크기 그대로 저장.
- 결과 PNG는 raw/fake/sdxl_img2img/ 에, 표는 manifests/sdxl_img2img.csv 에 쌓인다(type=full_style, family=E1).
- 12_partial_style.py(⑥)가 이 결과를 재료로 쓴다.

--set paint (10/4 추가): 공개샘플 설명 PDF가 전체 스타일 변환을 "일러스트처럼"·"회화적인 색감"이라 적었다 → 화풍판.
  prompts.PAINT_STYLES 6종(flat_illustration·anime_bg·watercolor·oil·gouache·digital_paint), strength 0.50~0.70, 결과는 raw/fake/sdxl_paint/ ·
  manifests/sdxl_paint.csv (generator = sdxl-paint-<화풍>). 기존 사실적판(--set realistic, 기본값)과 폴더·표가 따로다.
  12_partial_style.py --src sdxl_paint 로 부분 스타일(⑥)도 만든다.

Colab 실행:
  !python 11_sdxl_img2img.py /content/rr --work /content/work_i2i --n 10 --sync_to /content/drive/MyDrive/work
  !python 11_sdxl_img2img.py /content/rr --work /content/work_paint --set paint --n 500 --sync_to /content/drive/MyDrive/work
"""

import argparse, os, random, time
from PIL import Image
from common import ManifestWriter, Syncer, read_manifest
import prompts as P

REPO = 'stabilityai/stable-diffusion-xl-base-1.0'
VAE = 'madebyollin/sdxl-vae-fp16-fix'
NAME = 'sdxl_img2img'
STEPS, GUIDANCE = 30, 7.0
STRENGTH_RANGE = (0.15, 0.45)

# (이름, 프롬프트 뒷부분) — 그림체가 아니라 육안으로 구분 안 되는 사실적 사진 변형(질감·색감·조명만 미묘하게 달라진다)
STYLES = [
    ('natural', 'photorealistic, natural daylight, sharp focus, realistic textures'),
    ('warm', 'photorealistic, warm afternoon light, realistic color grading'),
    ('overcast', 'photorealistic, soft overcast light, neutral color, fine detail'),
    ('bright', 'photorealistic, bright greenhouse lighting, high detail, natural colors'),
    ('muted', 'photorealistic, slightly muted colors, subtle film tone, realistic'),
    ('crisp', 'photorealistic, crisp detail, natural contrast, DSLR photo'),
]
NEGATIVE = 'painting, illustration, drawing, cartoon, anime, cgi, 3d render, text, watermark, logo, deformed, blurry, low quality'


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
    return out, box, tw / (box[2] - box[0])


def load_pipe():
    import torch
    from diffusers import StableDiffusionXLImg2ImgPipeline, AutoencoderKL
    vae = AutoencoderKL.from_pretrained(VAE, torch_dtype=torch.float16)
    try:
        pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(REPO, vae=vae, torch_dtype=torch.float16, variant='fp16')
    except Exception:
        pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(REPO, vae=vae, torch_dtype=torch.float16)
    pipe.to('cuda')
    pipe.set_progress_bar_config(disable=True)
    return pipe


def style_config(style_set):
    """화풍 묶음별 설정. realistic = 기존 ④(사실적), paint = 화풍판(10/4)."""
    if style_set == 'paint':
        return dict(name='sdxl_paint', styles=[(nm, txt) for nm, txt, _ in P.PAINT_STYLES], strength=(0.50, 0.70),
                    negative=P.PAINT_NEGATIVE, make_prompt=P.make_paint_prompt, gen='sdxl-paint', tag='pnt')
    return dict(name=NAME, styles=STYLES, strength=STRENGTH_RANGE, negative=NEGATIVE,
                make_prompt=lambda t: f'strawberry plants in a greenhouse, {t}', gen='sdxl-img2img', tag='i2i')


def main(real_root, work, n, seed, sync_to, sync_every, style_set='realistic'):
    import torch
    cfg = style_config(style_set)
    name = cfg['name']
    if not torch.cuda.is_available():
        raise SystemExit('GPU가 없으면 중단합니다.')
    rows = [r for r in read_manifest(os.path.join(real_root, 'manifests', 'real.csv')) if r['label'] == '0']
    rng = random.Random(seed)
    rng.shuffle(rows)
    pipe = load_pipe()

    syncer = Syncer(work, sync_to, name) if sync_to else None
    if syncer:
        syncer.pull_manifest()
    mw = ManifestWriter(work, name)
    os.makedirs(os.path.join(work, 'raw', 'fake', name), exist_ok=True)
    made, attempt, t0, new = 0, 0, time.time(), 0
    while made < n and attempt < n * 3:
        row = rows[attempt % len(rows)]
        pass_no = attempt // len(rows)
        attempt += 1
        rel = f'fake/{name}/{row["source_id"]}__{cfg["tag"]}{pass_no}.png'
        if rel in mw.done:
            made += 1
            continue
        rng_i = random.Random(f'{seed}-{row["source_id"]}-{pass_no}')
        style, style_text = rng_i.choice(cfg['styles'])
        strength = round(rng_i.uniform(*cfg['strength']), 2)
        im = Image.open(os.path.join(real_root, 'raw', row['file'])).convert('RGB')
        std, crop, scale = standardize(im)
        prompt = cfg['make_prompt'](style_text)
        g = torch.Generator('cpu').manual_seed(rng_i.randint(0, 2**31 - 1))
        gen = pipe(prompt=prompt, negative_prompt=cfg['negative'], image=std, strength=strength, num_inference_steps=STEPS,
                   guidance_scale=GUIDANCE, generator=g).images[0]
        if gen.size != std.size:
            gen = gen.resize(std.size, Image.LANCZOS)
        gen.save(os.path.join(work, 'raw', rel))
        mw.add(file=rel, label=1, type='full_style', family='E1', generator=f'{cfg["gen"]}-{style}',
               source_id=row['source_id'], group=row['group'], split=row['split'],
               params={'repo': REPO, 'vae': VAE, 'style': style, 'prompt': prompt, 'strength': strength, 'steps': STEPS,
                       'guidance': GUIDANCE, 'size': list(std.size), 'crop': list(crop), 'scale': round(scale, 5)})
        made += 1
        new += 1
        if syncer and new % sync_every == 0:
            syncer.push()
        if new % 10 == 0 or made == n:
            per = (time.time() - t0) / new
            print(f'{made}/{n} (장당 {per:.1f}초, 남은 시간 약 {per * (n - made) / 60:.0f}분)', flush=True)
    mw.close()
    if syncer:
        ok = syncer.push(retries=5)
        print('Drive 복사 완료' if ok else f'!! Drive 복사 실패 — 결과는 {work} 에 남아 있으니 다시 실행하면 이어서 복사됩니다', flush=True)
    print(f'완료 {made}장 → 표: {mw.path}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('real_root', help='raw/real, manifests/real.csv 가 있는 폴더(개인 Drive 또는 그 바로가기 폴더)')
    ap.add_argument('--work', required=True, help='결과를 먼저 쓰는 로컬 폴더(Colab: /content/work_i2i)')
    ap.add_argument('--n', type=int, default=10)
    ap.add_argument('--seed', type=int, default=4)
    ap.add_argument('--sync_to', default=None)
    ap.add_argument('--sync_every', type=int, default=50)
    ap.add_argument('--set', dest='style_set', choices=['realistic', 'paint'], default='realistic', help='realistic=기존 ④, paint=화풍판(일러스트·회화)')
    a = ap.parse_args()
    main(a.real_root, a.work, a.n, a.seed, a.sync_to, a.sync_every, a.style_set)
