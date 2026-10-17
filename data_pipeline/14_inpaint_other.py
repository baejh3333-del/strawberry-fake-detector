"""14단계: 다른 모델로 국소 인페인팅(⑤ 보강) 가짜 만들기 — 게이팅 family E4, type partial_gen. Colab A100/L4.

왜: LOGO 검증(10/4)에서 부분 편집은 처음 보는 방식이면 거의 못 잡았다. 지금 E4의 인페인팅은 SDXL(U-Net, 4ch VAE) 하나뿐이다.
흔적이 다른 모델을 더해 "편집됐다"는 공통점을 배우게 한다. 영역은 13단계와 같은 방식(AI Hub 부위 사각형 1~2개 + 사각형·타원·불규칙 모양).

--model kandinsky   Kandinsky 2.2 인페인팅 (kandinsky-community/kandinsky-2-2-decoder-inpaint, Apache-2.0, 압축기 = MoVQ(VQ), U-Net)
                    'full'(결과 그대로 — 사진 전체가 MoVQ를 거침) / 'pasted'(마스크 안쪽만 원본에 붙임) 절반씩
--model sd15        Stable Diffusion 1.5 인페인팅 (stable-diffusion-v1-5/stable-diffusion-inpainting, CreativeML Open RAIL-M, 4ch VAE·512 해상도)
                    영역 주변 정사각형 창을 잘라 512로 줄여 인페인팅 → 원래 크기로 키워 마스크 안쪽만 붙임('crop'). 편집 도구에서 흔한 방식이고
                    영역 안에만 축소·확대 흔적이 남는다. safety_checker는 끈다(딸기 사진이 오탐으로 검게 나오는 것 방지).

규격·재료: 10·13단계와 같다(원본 실제 사진을 4:3 가운데 크롭 → 1024×768/768×1024, PNG 저장, 이후 05_finalize.py가 q92 JPEG로 통일).
표는 모델마다 따로: manifests/kandinsky_inpaint.csv, manifests/sd15_inpaint.csv.

Colab 실행 (13단계처럼 /content/rr 를 먼저 만든다 — raw/real, raw/real_boxes.json, manifests/real.csv):
  !python 14_inpaint_other.py /content/rr --model kandinsky --work /content/work_k --n 300 --sync_to /content/drive/MyDrive/work
  !python 14_inpaint_other.py /content/rr --model sd15 --work /content/work_s --n 300 --sync_to /content/drive/MyDrive/work
"""

import argparse, importlib.util, json, os, random, time
from PIL import Image, ImageFilter
from common import ManifestWriter, Syncer, fake_usage, least_used_first, read_manifest

MODELS = {
    'kandinsky': {'name': 'kandinsky_inpaint', 'repo': 'kandinsky-community/kandinsky-2-2-decoder-inpaint', 'license': 'Apache-2.0'},
    'sd15': {'name': 'sd15_inpaint', 'repo': 'stable-diffusion-v1-5/stable-diffusion-inpainting', 'license': 'CreativeML Open RAIL-M'},
}
NEGATIVE = 'painting, illustration, cartoon, 3d render, text, watermark, logo, deformed, blurry, low quality'


def _load(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(os.path.dirname(os.path.abspath(__file__)), name + '.py'))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_pipe(model):
    import torch
    from diffusers import AutoPipelineForInpainting
    repo = MODELS[model]['repo']
    kw = {'torch_dtype': torch.float16}
    if model == 'sd15':
        kw.update(safety_checker=None, requires_safety_checker=False)
    try:
        pipe = AutoPipelineForInpainting.from_pretrained(repo, variant='fp16', **kw)
    except Exception:
        pipe = AutoPipelineForInpainting.from_pretrained(repo, **kw)
    pipe.to('cuda')
    pipe.set_progress_bar_config(disable=True)
    return pipe


def run_kandinsky(pipe, std, mask, prompt, rng, invert):
    import torch
    m = mask if not invert else mask.point(lambda v: 255 - v)
    g = torch.Generator('cpu').manual_seed(rng.randint(0, 2**31 - 1))
    return pipe(prompt=prompt, negative_prompt=NEGATIVE, image=std, mask_image=m, height=std.height, width=std.width,
                num_inference_steps=50, guidance_scale=4.0, generator=g).images[0]


def run_sd15(pipe, std, mask, prompt, rng):
    """마스크 bbox 주변 정사각형 창을 512로 줄여 인페인팅 → 창 크기로 되돌린 결과를 전체 크기 캔버스에 올려 돌려준다(창 밖은 원본)."""
    import torch
    W, H = std.size
    x0, y0, x1, y1 = mask.getbbox()
    side = int(min(min(W, H), max(256, max(x1 - x0, y1 - y0) * 1.6)))
    cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
    left = max(0, min(W - side, cx - side // 2))
    top = max(0, min(H - side, cy - side // 2))
    box = (left, top, left + side, top + side)
    img512 = std.crop(box).resize((512, 512), Image.LANCZOS)
    msk512 = mask.crop(box).resize((512, 512), Image.NEAREST)
    g = torch.Generator('cpu').manual_seed(rng.randint(0, 2**31 - 1))
    out = pipe(prompt=prompt, negative_prompt=NEGATIVE, image=img512, mask_image=msk512, height=512, width=512,
               num_inference_steps=30, guidance_scale=7.5, strength=1.0, generator=g).images[0]
    canvas = std.copy()
    canvas.paste(out.resize((side, side), Image.LANCZOS), (left, top))
    return canvas, list(box)


def main(real_root, work, n, seed, sync_to, sync_every, model, invert):
    import torch
    if not torch.cuda.is_available():
        raise SystemExit('GPU가 없으면 중단합니다.')
    info = MODELS[model]
    NAME = info['name']
    inp = _load('10_sdxl_inpaint')       # standardize, transform_boxes, pick_content 재사용
    le = _load('13_local_edits')         # region_mask(사각형·타원·불규칙), pick_shape 재사용
    boxes_all = json.load(open(os.path.join(real_root, 'raw', 'real_boxes.json'), encoding='utf-8'))
    rows = [r for r in read_manifest(os.path.join(real_root, 'manifests', 'real.csv')) if r['label'] == '0']
    rng = random.Random(seed)
    rows = least_used_first(rows, fake_usage(real_root, exclude_step=NAME), rng)
    pipe = load_pipe(model)

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
        rel = f'fake/{NAME}/{row["source_id"]}__{model}{pass_no}.png'
        if rel in mw.done:
            made += 1
            continue
        rng_i = random.Random(f'{seed}-{row["source_id"]}-{model}-{pass_no}')
        im = Image.open(os.path.join(real_root, 'raw', row['file'])).convert('RGB')
        std, crop, scale = inp.standardize(im)
        cands = inp.transform_boxes(boxes_all.get(row['source_id'], []), crop, scale, std.size)
        if not cands:
            continue
        rng_i.shuffle(cands)
        picked = cands[:1] if model == 'sd15' else cands[:rng_i.randint(1, 2)]    # sd15는 창 하나에 영역 하나
        shape = le.pick_shape(rng_i)
        mask = Image.new('L', std.size, 0)
        for box in picked:
            m1 = le.region_mask(std.size, box, shape, rng_i)
            mask = Image.composite(Image.new('L', std.size, 255), mask, m1)       # 영역 합치기
        content = inp.pick_content(rng_i)
        prompt = f'{content}, natural greenhouse photo, realistic, same lighting as the surroundings'
        feather = rng_i.uniform(3, 8)
        params = {'repo': info['repo'], 'license': info['license'], 'prompt': prompt, 'boxes': picked, 'shape': shape, 'size': list(std.size),
                  'crop': list(crop), 'scale': round(scale, 5), 'feather': round(feather, 1)}
        if model == 'kandinsky':
            mode = 'pasted' if rng_i.random() < 0.5 else 'full'
            gen = run_kandinsky(pipe, std, mask, prompt, rng_i, invert)
            if gen.size != std.size:
                gen = gen.resize(std.size, Image.LANCZOS)
            out = Image.composite(gen, std, mask.filter(ImageFilter.GaussianBlur(feather))) if mode == 'pasted' else gen
            generator = f'kandinsky-inpaint-{mode}'
            params.update(mode=mode, steps=50, guidance=4.0, invert_mask=invert)
        else:
            canvas, win = run_sd15(pipe, std, mask, prompt, rng_i)
            out = Image.composite(canvas, std, mask.filter(ImageFilter.GaussianBlur(feather)))
            generator = 'sd15-inpaint-crop'
            params.update(mode='crop', window=win, steps=30, guidance=7.5)
        out.save(os.path.join(work, 'raw', rel))
        mw.add(file=rel, label=1, type='partial_gen', family='E4', generator=generator, source_id=row['source_id'], group=row['group'],
               split=row['split'], params=params)
        made += 1
        if syncer and made % sync_every == 0:
            syncer.push()
        if made % 10 == 0 or made == n:
            per = (time.time() - t0) / made
            print(f'{made}/{n} (장당 {per:.1f}초, 남은 시간 약 {per * (n - made) / 60:.0f}분)', flush=True)
    mw.close()
    if syncer:
        ok = syncer.push(retries=5)
        print('Drive 복사 완료' if ok else f'!! Drive 복사 실패 — 결과는 {work} 에 남아 있으니 Drive 연결 후 다시 실행하면 이어서 복사됩니다', flush=True)
    print(f'완료 {made}장 → 표: {mw.path}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('real_root', help='raw/real, raw/real_boxes.json, manifests/real.csv 가 있는 폴더')
    ap.add_argument('--model', required=True, choices=list(MODELS))
    ap.add_argument('--work', required=True)
    ap.add_argument('--n', type=int, default=10)
    ap.add_argument('--seed', type=int, default=14)
    ap.add_argument('--sync_to', default=None)
    ap.add_argument('--sync_every', type=int, default=50)
    ap.add_argument('--invert_mask', action='store_true', help='kandinsky에서 마스크 흑백이 반대로 해석될 때(버전 차이) — 시험 10장으로 확인')
    a = ap.parse_args()
    main(a.real_root, a.work, a.n, a.seed, a.sync_to, a.sync_every, a.model, a.invert_mask)
