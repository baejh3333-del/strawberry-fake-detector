"""16단계: FLUX.2 [klein] 4B로 (a) 전체 생성 v6 (b) 전체 스타일 화풍판 (c) 부분 생성 지시형 편집 — 한 모델로 세 가지. Colab L4/A100.

모델: black-forest-labs/FLUX.2-klein-4B (Apache-2.0, 증류 모델 — 4 step, guidance 1.0). 텍스트→이미지와 사진 편집(이미지 입력 + 지시문)을 한 모델이 한다.
설치: `pip install -U git+https://github.com/huggingface/diffusers.git` (모델 카드 기준. Flux2KleinPipeline이 들어 있는 diffusers가 필요하다 —
      `from diffusers import Flux2KleinPipeline`이 되면 설치 생략). 가중치는 HF에서 처음 한 번 내려받는다(생성용, 제출물과 무관).
⚠️ 10/4 작성 시 GPU에서 돌려 보지 못했다(로컬은 GTX 1660 Ti). **처음에는 --n 3 으로 돌려 에러와 그림을 눈으로 확인**한 뒤 늘린다.

--mode t2i    전체 생성(①) — prompts.make_prompt_v6. type=AI_gen, family=E2, 표 klein_t2i. 크기는 768×1024(75%)·1024×768(25%).
--mode style  전체 스타일 변환(④) 화풍판 — 실제 사진을 규격으로 맞춘 뒤 prompts.PAINT_STYLES 6종 중 하나로 다시 그리게 한다(구조·글씨 보존 지시).
              type=full_style, family=E2, 표 klein_style. 12_partial_style.py --src klein_style 로 부분 스타일(⑥)도 만든다.
--mode edit   부분 생성(⑤) 지시형 편집 — prompts.EDIT_INSTRUCTIONS(물체 추가·삭제·교체) 중 하나를 준다. 결과를 두 방식으로 저장:
                full   모델이 돌려준 사진 전체(편집기는 안 바꾼 곳도 다시 그려서 전체에 흔적이 남는다)
                pasted 원본과 달라진 영역만 원본에 붙인 것(포토샵 생성형 채우기처럼 바깥은 원본 그대로)
              대상이 사진에 없으면 변화가 거의 없다 → 변화량 0.5% 미만이면 버리고 다른 지시로 다시 시도. type=partial_gen, family=E4, 표 klein_edit.

실행 (Colab):
  !python 16_klein.py /content/rr --mode t2i   --work /content/work_k_t2i   --n 400 --sync_to /content/drive/MyDrive/work
  !python 16_klein.py /content/rr --mode style --work /content/work_k_style --n 500 --sync_to /content/drive/MyDrive/work
  !python 16_klein.py /content/rr --mode edit  --work /content/work_k_edit  --n 600 --sync_to /content/drive/MyDrive/work
  (root는 t2i에서는 쓰지 않지만 자리를 채워야 한다. style·edit에서는 raw/real, manifests/real.csv가 있는 폴더)
"""

import argparse, importlib.util, os, random, time
import numpy as np
from PIL import Image, ImageFilter
from common import ManifestWriter, Syncer, fake_usage, least_used_first, read_manifest, split_of
import prompts as P

REPO = 'black-forest-labs/FLUX.2-klein-4B'
STEPS, GUIDANCE = 4, 1.0
NAMES = {'t2i': 'klein_t2i', 'style': 'klein_style', 'edit': 'klein_edit'}
MIN_CHANGE, MAX_PASTE_CHANGE = 0.005, 0.45       # 변화 영역 비율: 이보다 작으면 편집 실패로 버림 / 이보다 크면 pasted 버전 불가(전체 저장)


def _load(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(os.path.dirname(os.path.abspath(__file__)), name + '.py'))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_pipe():
    import torch
    try:
        from diffusers import Flux2KleinPipeline
    except ImportError:
        raise SystemExit('diffusers에 Flux2KleinPipeline이 없습니다. 위 설치 줄(git+https://github.com/huggingface/diffusers.git)을 먼저 실행하세요.')
    pipe = Flux2KleinPipeline.from_pretrained(REPO, torch_dtype=torch.bfloat16)
    pipe.enable_model_cpu_offload()      # 4B라 L4(24GB)에서도 돈다. A100이면 pipe.to('cuda')로 바꿔도 된다
    pipe.set_progress_bar_config(disable=True)
    return pipe


def change_mask(orig, edited, blur=5, thr=24):
    """원본과 편집 결과가 달라진 영역(0/255 L 마스크)과 그 비율. 약한 차이는 무시하고 작은 구멍·점을 정리한다."""
    import cv2
    a = np.asarray(orig.convert('RGB')).astype(np.int16)
    b = np.asarray(edited.convert('RGB')).astype(np.int16)
    d = np.abs(a - b).max(axis=2).astype(np.uint8)
    d = cv2.GaussianBlur(d, (0, 0), blur)
    m = (d > thr).astype(np.uint8) * 255
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)
    m = cv2.dilate(m, k)
    return Image.fromarray(m), float((m > 0).mean())


def paste_changed(orig, edited, mask, feather=5):
    soft = mask.filter(ImageFilter.GaussianBlur(feather))
    return Image.composite(edited, orig, soft)


def run(root, mode, work, n, seed, sync_to, sync_every):
    import torch
    if not torch.cuda.is_available():
        raise SystemExit('GPU가 없으면 중단합니다.')
    name = NAMES[mode]
    pipe = load_pipe()
    inp = _load('10_sdxl_inpaint') if mode != 't2i' else None    # standardize 재사용
    rows = None
    if mode != 't2i':
        rows = [r for r in read_manifest(os.path.join(root, 'manifests', 'real.csv')) if r['label'] == '0']
        rows = least_used_first(rows, fake_usage(root, exclude_step=name), random.Random(seed))

    syncer = Syncer(work, sync_to, name) if sync_to else None
    if syncer:
        syncer.pull_manifest()
    mw = ManifestWriter(work, name)
    os.makedirs(os.path.join(work, 'raw', 'fake', name), exist_ok=True)
    made, attempt, t0 = 0, 0, time.time()
    max_attempt = n * (6 if mode == 'edit' else 3)
    while made < n and attempt < max_attempt:
        i = attempt
        attempt += 1
        if mode == 't2i':
            rel = f'fake/{name}/{name}_{i:04d}.png'
            if rel in mw.done:
                made += 1
                continue
            rng = random.Random(f'{name}:{i}')
            prompt, tag, _ = P.make_prompt_v6(rng)
            w, h = (768, 1024) if rng.random() < 0.75 else (1024, 768)
            img = pipe(prompt=prompt, height=h, width=w, guidance_scale=GUIDANCE, num_inference_steps=STEPS,
                       generator=torch.Generator('cpu').manual_seed(i)).images[0]
            if img.size != (w, h):
                img = img.resize((w, h), Image.LANCZOS)
            img.save(os.path.join(work, 'raw', rel))
            group = f'{name}_{i:04d}'
            mw.add(file=rel, label=1, type='AI_gen', family='E2', generator='flux2-klein-4b', source_id='', group=group, split=split_of(group),
                   params={'repo': REPO, 'prompt': prompt, 'prompt_version': 'v6', 'stage': tag, 'seed': i, 'size': [w, h],
                           'steps': STEPS, 'cfg': GUIDANCE})
        else:
            row = rows[i % len(rows)]
            pass_no = i // len(rows)
            rel = f'fake/{name}/{row["source_id"]}__k{mode[0]}{pass_no}.png'
            if rel in mw.done:
                made += 1
                continue
            rng = random.Random(f'{seed}-{row["source_id"]}-{mode}-{pass_no}')
            im = Image.open(os.path.join(root, 'raw', row['file'])).convert('RGB')
            std, crop, scale = inp.standardize(im)
            g = torch.Generator('cpu').manual_seed(rng.randint(0, 2**31 - 1))
            base = {'repo': REPO, 'steps': STEPS, 'cfg': GUIDANCE, 'size': list(std.size), 'crop': list(crop), 'scale': round(scale, 5)}
            if mode == 'style':
                style, _, klein_text = P.PAINT_STYLES[made % len(P.PAINT_STYLES)]    # 만든 장수 기준으로 6종이 고르게 돈다
                prompt = P.make_style_instruction(klein_text)
                out = pipe(image=std, prompt=prompt, height=std.height, width=std.width, guidance_scale=GUIDANCE,
                           num_inference_steps=STEPS, generator=g).images[0]
                if out.size != std.size:
                    out = out.resize(std.size, Image.LANCZOS)
                out.save(os.path.join(work, 'raw', rel))
                mw.add(file=rel, label=1, type='full_style', family='E2', generator=f'flux2-klein-style-{style}',
                       source_id=row['source_id'], group=row['group'], split=row['split'],
                       params={**base, 'style': style, 'prompt': prompt})
            else:   # edit
                kind, prompt = P.make_edit_instruction(rng)
                out = pipe(image=std, prompt=prompt, height=std.height, width=std.width, guidance_scale=GUIDANCE,
                           num_inference_steps=STEPS, generator=g).images[0]
                if out.size != std.size:
                    out = out.resize(std.size, Image.LANCZOS)
                mask, frac = change_mask(std, out)
                if frac < MIN_CHANGE:
                    print(f'[건너뜀] 변화 없음({frac:.3%}): {prompt[:60]}', flush=True)
                    continue
                variant = 'pasted' if (rng.random() < 0.5 and frac <= MAX_PASTE_CHANGE) else 'full'
                if variant == 'pasted':
                    out = paste_changed(std, out, mask)
                out.save(os.path.join(work, 'raw', rel))
                mw.add(file=rel, label=1, type='partial_gen', family='E4', generator=f'flux2-klein-edit-{kind}',
                       source_id=row['source_id'], group=row['group'], split=row['split'],
                       params={**base, 'kind': kind, 'prompt': prompt, 'variant': variant, 'change_frac': round(frac, 4)})
        made += 1
        if syncer and made % sync_every == 0:
            syncer.push()
        if made % 10 == 0 or made == n:
            per = (time.time() - t0) / made
            print(f'{made}/{n} (장당 {per:.1f}초, 남은 시간 약 {per * (n - made) / 60:.0f}분)', flush=True)
    mw.close()
    if syncer:
        ok = syncer.push(retries=5)
        print('Drive 복사 완료' if ok else f'!! Drive 복사 실패 — 결과는 {work} 에 남아 있으니 다시 실행하면 이어서 복사됩니다', flush=True)
    print(f'완료 {made}장 (시도 {attempt}) → 표: {mw.path}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('root', help='raw/real, manifests/real.csv 가 있는 폴더 (t2i는 쓰지 않음)')
    ap.add_argument('--mode', choices=list(NAMES), required=True)
    ap.add_argument('--work', required=True, help='결과를 먼저 쓰는 로컬 폴더(Colab: /content/work_k_<mode>)')
    ap.add_argument('--n', type=int, default=3, help='이번에 만들 장수(처음엔 3)')
    ap.add_argument('--seed', type=int, default=16)
    ap.add_argument('--sync_to', default=None)
    ap.add_argument('--sync_every', type=int, default=50)
    a = ap.parse_args()
    run(a.root, a.mode, a.work, a.n, a.seed, a.sync_to, a.sync_every)
