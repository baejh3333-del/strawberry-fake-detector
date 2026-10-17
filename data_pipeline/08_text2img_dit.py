"""8단계: 글로 전체 생성(유형 ①) — 확산 트랜스포머 계열(E2): FLUX.1-schnell, Z-Image-Turbo.

07_text2img.py와 같은 규칙(프롬프트 재료·크기·표·이어서 하기)이고 모델만 다르다. 두 파일을 같은 폴더에 둘 것.
- flux : black-forest-labs/FLUX.1-schnell (Apache-2.0, **gated** — 허깅페이스에서 동의 + Colab 보안 비밀 HF_TOKEN 필요), 4단계, guidance 0
- zimage: Tongyi-MAI/Z-Image-Turbo (Apache-2.0, gated 아님 — 2026-09-29 모델 카드 확인), 9단계(=DiT 8회), guidance 0
- 두 모델 모두 negative prompt를 쓰지 않는다(증류 모델). 크기는 16의 배수 768×1024 / 1024×768 그대로.
- FLUX(12B)는 L4 24GB에 통째로 안 들어가 --offload 필요.

실행 예 (Colab):
  !python 08_text2img_dit.py /content/drive/MyDrive/work --model zimage --n 10
  !python 08_text2img_dit.py /content/drive/MyDrive/work --model flux --n 10 --offload
"""

import argparse, os, random, time
import torch
from common import ManifestWriter, Syncer, split_of
import prompts as t2i07   # 프롬프트 재료는 07과 같은 prompts.py

MODELS = {
    'flux':   dict(repo='black-forest-labs/FLUX.1-schnell', name='flux_schnell', steps=4, cfg=0.0),
    'zimage': dict(repo='Tongyi-MAI/Z-Image-Turbo',          name='zimage_turbo', steps=9, cfg=0.0),
}


def load_pipe(kind, repo, offload):
    if kind == 'flux':
        from diffusers import FluxPipeline
        pipe = FluxPipeline.from_pretrained(repo, torch_dtype=torch.bfloat16)
    else:
        from diffusers import ZImagePipeline
        pipe = ZImagePipeline.from_pretrained(repo, torch_dtype=torch.bfloat16)
    if offload:
        pipe.enable_model_cpu_offload()
    else:
        pipe.to('cuda')
    return pipe


def main(root, kind, n, offload, family, sync_to=None, sync_every=50, prompt_ver='v2', suffix=''):
    if not torch.cuda.is_available():
        raise SystemExit('GPU가 없으면 중단합니다.')
    cfg = MODELS[kind]
    name, repo = cfg['name'] + suffix, cfg['repo']
    pipe = load_pipe(kind, repo, offload)
    pipe.set_progress_bar_config(disable=True)

    syncer = Syncer(root, sync_to, name) if sync_to else None
    if syncer:
        syncer.pull_manifest()   # Drive에 이미 만든 번호를 이어받는다
    mw = ManifestWriter(root, name)
    os.makedirs(os.path.join(root, 'raw', 'fake', name), exist_ok=True)
    made, t0, i = 0, time.time(), 0
    while made < n:
        rel = f'fake/{name}/{name}_{i:04d}.png'
        seed_i, i = i, i + 1
        if rel in mw.done:
            continue
        rng = random.Random(f'{name}:{seed_i}')
        if prompt_ver == 'v5':
            prompt, stage = t2i07.make_prompt_v5(rng)
        elif prompt_ver == 'v6':
            prompt, stage, _ = t2i07.make_prompt_v6(rng)   # FLUX·Z-Image는 negative를 쓰지 않는다. stage 칸에 '단계/샷' 태그
        else:
            prompt, stage = t2i07.make_prompt(rng), None
        w, h = (768, 1024) if rng.random() < 0.75 else (1024, 768)
        kw = dict(prompt=prompt, width=w, height=h, num_inference_steps=cfg['steps'], guidance_scale=cfg['cfg'],
                  generator=torch.Generator('cpu').manual_seed(seed_i))
        if kind == 'flux':
            kw['max_sequence_length'] = 256
        img = pipe(**kw).images[0]
        img.save(os.path.join(root, 'raw', rel))
        group = f'{name}_{seed_i:04d}'
        mw.add(file=rel, label=1, type='AI_gen', family=family, generator=name, source_id='',
               group=group, split=split_of(group),
               params={'repo': repo, 'prompt': prompt, 'prompt_version': prompt_ver if prompt_ver in ('v5', 'v6') else t2i07.PROMPT_VERSION, 'stage': stage, 'seed': seed_i, 'size': [w, h],
                       'steps': cfg['steps'], 'cfg': cfg['cfg']})
        made += 1
        if syncer and made % sync_every == 0:
            syncer.push()   # 실패해도 생성은 계속, 다음 복사 때 다시 시도
        if made % 10 == 0 or made == n:
            per = (time.time() - t0) / made
            print(f'{made}/{n} (장당 {per:.0f}초, 남은 시간 약 {per * (n - made) / 60:.0f}분)', flush=True)
    mw.close()
    if syncer:
        ok = syncer.push(retries=5)
        print('Drive 복사 완료' if ok else f'!! Drive 복사 실패 — 결과는 {root} 에 남아 있으니 Drive 연결 후 다시 실행하면 이어서 복사됩니다', flush=True)
    print(f'완료 → 표: {mw.path}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('root')
    ap.add_argument('--model', choices=list(MODELS), required=True)
    ap.add_argument('--n', type=int, default=10, help='이번에 새로 만들 장수')
    ap.add_argument('--sync_to', default=None, help='Drive 결과 폴더. 주면 root(로컬 디스크)에 먼저 쓰고 sync_every장마다 여기로 복사')
    ap.add_argument('--sync_every', type=int, default=50)
    ap.add_argument('--prompt', choices=['v2', 'v5', 'v6'], default='v2', help='v2=기존, v5=성장 단계별(싹/잎/꽃/열매), v6=근접 폰 사진+노란 트랩(10/4)')
    ap.add_argument('--offload', action='store_true', help='GPU 메모리가 모자랄 때(FLUX on L4)')
    ap.add_argument('--family', default='E2')
    ap.add_argument('--suffix', default='', help='이름 뒤에 붙일 말. v6는 기존 flux_schnell·zimage_turbo와 섞이지 않게 `--suffix _v6`을 쓴다')
    a = ap.parse_args()
    main(a.root, a.model, a.n, a.offload, a.family, a.sync_to, a.sync_every, a.prompt, a.suffix)
