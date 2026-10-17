"""7단계: 글로 전체 생성(유형 ①) — 확산 모델 U-Net 계열(E1). 기본은 RealVisXL V5.0 (Colab L4 또는 GPU 있는 PC).

실제 사진이 필요 없다(글만 보고 그림). 결과는 ROOT/raw/fake/<name>/ 에 PNG로, 표는 ROOT/manifests/<name>.csv 에 쌓인다.
- 모델: SG161222/RealVisXL_V5.0 (허깅페이스, 접근 승인 불필요, 라이선스 openrail++ — 2026-09-29 모델 카드 확인). SDXL 계열이라 E1.
- 크기: 768×1024(75%) 또는 1024×768(25%)로 **처음부터 대회 규격 크기로 생성**한다. 리사이즈가 없으니 생성물에만 보간 흔적이 생기지 않는다(전략 12-2b).
- 프롬프트 재료는 `prompts.py` (공개샘플을 보고 쓴 목표 모습·배운 점이 거기 적혀 있다). 08_text2img_dit.py도 같은 파일을 쓴다.
- 이어서 하기: 표에 있는 파일은 건너뛴다. 세션이 끊겨도 표에 남은 것까지는 유지. 번호·시드가 고정돼 있어 같은 번호는 같은 그림.
- VAE는 madebyollin/sdxl-vae-fp16-fix(MIT)로 바꿔 쓴다 — fp16에서 SDXL 기본 VAE가 NaN(검은 그림)을 내는 문제 방지.
- GPU 메모리가 6GB 정도면 --offload 는 되지만 GTX 16 시리즈는 fp16이 깨져 검은 그림이 나온다(9/29 확인) → 쓰지 말 것.

Colab 준비:
  !pip -q install diffusers transformers accelerate
실행 (시험 10장 → 눈으로 확인 → 본 생성):
  !python 07_text2img.py /content/drive/MyDrive/work --n 10
"""

import argparse, os, random, time
import torch
from common import ManifestWriter, Syncer, split_of

from prompts import make_prompt, make_negative, make_prompt_v5, make_negative_v5, make_prompt_v6, PROMPT_VERSION


def main(root, n, repo, vae_repo, name, family, steps, cfg, offload, sync_to=None, sync_every=50, prompt_ver='v2'):
    from diffusers import StableDiffusionXLPipeline, AutoencoderKL, DPMSolverMultistepScheduler
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    if dev == 'cpu':
        raise SystemExit('GPU가 없으면 너무 느려서 중단합니다 (CPU는 장당 수십 분).')
    vae = AutoencoderKL.from_pretrained(vae_repo, torch_dtype=torch.float16)
    try:
        pipe = StableDiffusionXLPipeline.from_pretrained(repo, vae=vae, torch_dtype=torch.float16,
                                                         variant='fp16', use_safetensors=True)
    except Exception:
        pipe = StableDiffusionXLPipeline.from_pretrained(repo, vae=vae, torch_dtype=torch.float16, use_safetensors=True)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config, use_karras_sigmas=True)  # DPM++ 2M Karras
    if offload:
        pipe.enable_model_cpu_offload()
        pipe.vae.enable_tiling()
    else:
        pipe.to(dev)
    pipe.set_progress_bar_config(disable=True)

    syncer = Syncer(root, sync_to, name) if sync_to else None
    if syncer:
        syncer.pull_manifest()   # Drive에 이미 만든 번호를 이어받는다
    mw = ManifestWriter(root, name)
    os.makedirs(os.path.join(root, 'raw', 'fake', name), exist_ok=True)
    made, t0 = 0, time.time()
    i = 0
    while made < n:
        rel = f'fake/{name}/{name}_{i:04d}.png'
        seed_i, i = i, i + 1
        if rel in mw.done:
            continue
        rng = random.Random(f'{name}:{seed_i}')
        if prompt_ver == 'v5':
            prompt, stage = make_prompt_v5(rng)
            neg = make_negative_v5(stage)
        elif prompt_ver == 'v6':
            prompt, stage, neg = make_prompt_v6(rng)     # stage 칸에 '단계/샷' 태그가 들어간다
        else:
            prompt, stage, neg = make_prompt(rng), None, make_negative(None)
        w, h = (768, 1024) if rng.random() < 0.75 else (1024, 768)   # 공개샘플은 세로가 많다(21장 중 17장)
        img = pipe(prompt=prompt, negative_prompt=neg, width=w, height=h, num_inference_steps=steps,
                   guidance_scale=cfg, generator=torch.Generator('cpu').manual_seed(seed_i)).images[0]
        img.save(os.path.join(root, 'raw', rel))
        group = f'{name}_{seed_i:04d}'   # 재료 사진이 없으니 파일 하나가 한 그룹
        mw.add(file=rel, label=1, type='AI_gen', family=family, generator=name, source_id='',
               group=group, split=split_of(group),
               params={'repo': repo, 'vae': vae_repo, 'prompt': prompt, 'prompt_version': prompt_ver if prompt_ver in ('v5', 'v6') else PROMPT_VERSION, 'stage': stage,
                       'seed': seed_i, 'size': [w, h], 'steps': steps, 'cfg': cfg, 'scheduler': 'dpmpp2m-karras'})
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
    ap.add_argument('--n', type=int, default=600, help='이번에 새로 만들 장수')
    ap.add_argument('--repo', default='SG161222/RealVisXL_V5.0')
    ap.add_argument('--vae', default='madebyollin/sdxl-vae-fp16-fix')
    ap.add_argument('--name', default='realvisxl')
    ap.add_argument('--family', default='E1')
    ap.add_argument('--steps', type=int, default=40)
    ap.add_argument('--cfg', type=float, default=5.0)
    ap.add_argument('--sync_to', default=None, help='Drive 결과 폴더. 주면 root(로컬 디스크)에 먼저 쓰고 sync_every장마다 여기로 복사')
    ap.add_argument('--sync_every', type=int, default=50)
    ap.add_argument('--prompt', choices=['v2', 'v5', 'v6'], default='v2', help='v2=기존, v5=성장 단계별(싹/잎/꽃/열매), v6=근접 폰 사진+노란 트랩(10/4)')
    ap.add_argument('--offload', action='store_true', help='GPU 메모리가 작을 때(6GB급) — GTX 16 시리즈는 fp16이 깨져 검은 그림')
    a = ap.parse_args()
    main(a.root, a.n, a.repo, a.vae, a.name, a.family, a.steps, a.cfg, a.offload, a.sync_to, a.sync_every, a.prompt)
