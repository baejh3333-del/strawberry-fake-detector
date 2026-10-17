"""9단계: 글로 전체 생성(유형 ①) — 자기회귀 모델 Infinity-2B (E3). 프롬프트는 v5(성장 단계별)를 쓴다.

⚠️ 아직 Colab에서 돌려 보지 않은 초안이다. **반드시 --n 3 으로 먼저 시험**하고, 안 되면 아래 '알려진 위험'부터 본다.

- 모델: FoundationVision/Infinity (MIT). infinity_2b_reg.pth + infinity_vae_d32reg.pth (허깅페이스), 텍스트 인코더 google/flan-t5-xl (Apache-2.0).
  공식 코드 저장소 https://github.com/FoundationVision/Infinity (MIT)를 /content/Infinity 에 받아 그 안의 tools/run_infinity.py 함수를 그대로 쓴다.
- 해상도: 1M(약 100만 화소)에서 세로 1152×864, 가로 1344×1008로 나온다. 우리 규격(768×1024·1024×768)과 달라서
  **LANCZOS로 한 번 줄인다**(tools/normalize.py도 LANCZOS이므로 실제·가짜 모두 같은 방식). params에 원래 크기를 기록한다.
- 07·08과 같은 규칙: 표(manifest) 기록, 이어서 하기, --sync_to 로 로컬 디스크에 먼저 쓰고 Drive에 주기 복사.
- 프롬프트 v5: 싹만 15% / 잎만 55% / 꽃 15% / 열매 15%. 어린 묘에는 열매·꽃을 달지 않는다. params.stage에 기록.

알려진 위험
1. 저장소의 requirements.txt를 그대로 설치하면 torch 버전이 바뀔 수 있다 → 설치하지 말고, 없는 패키지만 하나씩 깐다(에러 메시지 보고).
2. 공식 코드가 flash-attn·FlexAttention(torch>=2.5.1)을 쓸 수 있다. use_flex_attn=0 으로 시작한다.
3. gen_one_img 인자 이름이 저장소 최신판과 다르면 TypeError가 난다 → 저장소의 interactive_infer.ipynb 호출부를 보고 맞춘다.

Colab 준비 (A100 권장):
  !git clone --depth 1 https://github.com/FoundationVision/Infinity /content/Infinity
  !pip -q install huggingface_hub   (이미 있음)  + 실행 중 나오는 ModuleNotFoundError 패키지만 설치
실행:
  !python 09_text2img_infinity.py /content/work --n 3 --sync_to /content/drive/MyDrive/work
"""

import argparse, os, random, subprocess, sys, time
import numpy as np
import torch
from PIL import Image
from common import ManifestWriter, Syncer, split_of
import prompts as P

REPO_URL = 'https://github.com/FoundationVision/Infinity'
HF_REPO = 'FoundationVision/Infinity'
NAME = 'infinity_2b'


def setup_repo(repo_dir):
    if not os.path.isdir(repo_dir):
        subprocess.run(['git', 'clone', '--depth', '1', REPO_URL, repo_dir], check=True)
    if repo_dir not in sys.path:
        sys.path.insert(0, repo_dir)


def load_infinity(repo_dir, weights_dir):
    from huggingface_hub import hf_hub_download
    setup_repo(repo_dir)
    os.makedirs(weights_dir, exist_ok=True)
    model_path = hf_hub_download(HF_REPO, 'infinity_2b_reg.pth', local_dir=weights_dir)
    vae_path = hf_hub_download(HF_REPO, 'infinity_vae_d32reg.pth', local_dir=weights_dir)
    from tools.run_infinity import load_tokenizer, load_visual_tokenizer, load_transformer
    args = argparse.Namespace(
        pn='1M', model_path=model_path, cfg_insertion_layer=0, vae_type=32, vae_path=vae_path,
        add_lvl_embeding_only_first_block=1, use_bit_label=1, model_type='infinity_2b',
        rope2d_each_sa_layer=1, rope2d_normalized_by_hw=2, use_scale_schedule_embedding=0,
        sampling_per_bits=1, text_encoder_ckpt='google/flan-t5-xl', text_channels=2048,
        apply_spatial_patchify=0, h_div_w_template=1.000, use_flex_attn=0, cache_dir='/dev/shm',
        checkpoint_type='torch', seed=0, bf16=1, save_file='/tmp/tmp.jpg',
        enable_model_cache=0, enable_positive_prompt=0, cfg='3', tau=0.5, prompt='a dog')   # load_transformer가 읽는 인자(공식 기본값)
    text_tokenizer, text_encoder = load_tokenizer(t5_path=args.text_encoder_ckpt)
    vae = load_visual_tokenizer(args)
    infinity = load_transformer(vae, args)
    return args, text_tokenizer, text_encoder, vae, infinity


def generate(args, tok, enc, vae, infinity, prompt, seed, portrait, cfg=3.0, tau=0.5):
    from tools.run_infinity import gen_one_img
    from infinity.utils.dynamic_resolution import dynamic_resolution_h_w, h_div_w_templates
    h_div_w = 4 / 3 if portrait else 3 / 4          # 세로 = 높이/너비 1.333, 가로 = 0.75
    tmpl = h_div_w_templates[int(np.argmin(np.abs(h_div_w_templates - h_div_w)))]
    scale_schedule = [(1, h, w) for (_, h, w) in dynamic_resolution_h_w[tmpl][args.pn]['scales']]
    native_hw = dynamic_resolution_h_w[tmpl][args.pn]['pixel']
    out = gen_one_img(infinity, vae, tok, enc, prompt, g_seed=seed, gt_leak=0, gt_ls_Bl=None,
                      cfg_list=cfg, tau_list=tau, scale_schedule=scale_schedule,
                      cfg_insertion_layer=[args.cfg_insertion_layer], vae_type=args.vae_type,
                      sampling_per_bits=args.sampling_per_bits, enable_positive_prompt=0)
    arr = out.cpu().numpy()[:, :, ::-1]             # BGR → RGB
    return Image.fromarray(np.ascontiguousarray(arr)), tuple(native_hw)


def main(root, n, sync_to, sync_every, repo_dir, weights_dir, family='E3'):
    if not torch.cuda.is_available():
        raise SystemExit('GPU가 없으면 중단합니다.')
    args, tok, enc, vae, infinity = load_infinity(repo_dir, weights_dir)

    syncer = Syncer(root, sync_to, NAME) if sync_to else None
    if syncer:
        syncer.pull_manifest()
    mw = ManifestWriter(root, NAME)
    os.makedirs(os.path.join(root, 'raw', 'fake', NAME), exist_ok=True)
    made, t0, i = 0, time.time(), 0
    while made < n:
        rel = f'fake/{NAME}/{NAME}_{i:04d}.png'
        seed_i, i = i, i + 1
        if rel in mw.done:
            continue
        rng = random.Random(f'{NAME}:{seed_i}')
        prompt, stage = P.make_prompt_v5(rng)
        portrait = rng.random() < 0.75
        target = (768, 1024) if portrait else (1024, 768)          # (너비, 높이)
        img, native_hw = generate(args, tok, enc, vae, infinity, prompt, seed_i, portrait)
        if img.size != target:
            img = img.resize(target, Image.LANCZOS)              # normalize.py와 같은 방식 — 실제·가짜 모두 이 한 번
        img.save(os.path.join(root, 'raw', rel))
        group = f'{NAME}_{seed_i:04d}'
        mw.add(file=rel, label=1, type='AI_gen', family=family, generator=NAME, source_id='',
               group=group, split=split_of(group),
               params={'repo': HF_REPO, 'ckpt': 'infinity_2b_reg.pth', 'vae': 'infinity_vae_d32reg.pth',
                       'prompt': prompt, 'prompt_version': 'v5', 'stage': stage, 'seed': seed_i,
                       'size': list(target), 'native_hw': list(native_hw), 'resize': 'LANCZOS', 'cfg': 3.0, 'tau': 0.5})
        made += 1
        if syncer and made % sync_every == 0:
            syncer.push()
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
    ap.add_argument('--n', type=int, default=3, help='이번에 새로 만들 장수(처음엔 3)')
    ap.add_argument('--sync_to', default=None)
    ap.add_argument('--sync_every', type=int, default=50)
    ap.add_argument('--repo_dir', default='/content/Infinity')
    ap.add_argument('--weights_dir', default='/content/infinity_weights')
    a = ap.parse_args()
    main(a.root, a.n, a.sync_to, a.sync_every, a.repo_dir, a.weights_dir)
