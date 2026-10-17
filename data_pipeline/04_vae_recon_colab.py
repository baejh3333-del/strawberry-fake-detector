"""4단계: VAE 재구성 가짜 만들기 (Colab GPU).

실제 사진을 생성 모델의 압축기(VAE)에 "넣었다 빼기"만 한 이미지. 생성은 안 했지만 **압축기 흔적**이 남는다.
전략.md 11-4·11-5의 "① 보조(압축기 흔적)". 기본은 FLUX VAE(16ch, E2 계열).
실제 사진과 내용이 같고 흔적만 다른 짝이라 대조 학습(SupCon)에도 좋다 (12-3).

- 모델: black-forest-labs/FLUX.1-schnell 의 vae 폴더 (Apache-2.0). **반드시 schnell** — dev 저장소 VAE는 비상업 라이선스.
  허깅페이스에서 연락처 공유 동의(gated) 후 받을 수 있다 → Colab에서 토큰 로그인 필요.
- 크기를 바꾸지 않는다: 16의 배수가 되게 가장자리를 거울처럼 채워 넣고(reflect pad), 복원 후 원래 크기로 잘라낸다.
- 사진이 커서 GPU 메모리가 모자라면 자동으로 조각(tiling) 모드로 다시 한다 (params에 기록).
- 다른 VAE도 --repo/--family로 가능: 예) --repo stabilityai/sdxl-vae --subfolder "" --family E1 --name sdxl-vae

Colab 준비 (셀 하나):
  !pip -q install diffusers
  from google.colab import drive; drive.mount('/content/drive')
  from huggingface_hub import login; login()          # 읽기 토큰 입력
  !git clone <저장소> repo   (또는 이 파일만 업로드)
실행:
  !python repo/data_pipeline/04_vae_recon_colab.py /content/drive/MyDrive/work --n 500
"""

import argparse, os, random
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from common import ManifestWriter, real_rows


@torch.no_grad()
def reconstruct(vae, im, dev):
    x = torch.from_numpy(np.array(im)).permute(2, 0, 1)[None].float().div(127.5).sub(1).to(dev)   # 0~255 → -1~1
    h, w = x.shape[-2:]
    ph, pw = (-h) % 16, (-w) % 16
    x = F.pad(x, (0, pw, 0, ph), mode='reflect')
    y = vae.decode(vae.encode(x).latent_dist.mode()).sample[..., :h, :w]
    if not torch.isfinite(y).all():
        raise RuntimeError('NaN/inf 발생')
    y = ((y.clamp(-1, 1) + 1) * 127.5).round().byte()[0].permute(1, 2, 0).cpu().numpy()
    return Image.fromarray(y)


def main(root, n, seed, repo, subfolder, family, name):
    from diffusers import AutoencoderKL
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    vae = AutoencoderKL.from_pretrained(repo, subfolder=subfolder or None, torch_dtype=torch.float32).to(dev).eval()
    print(f'{repo} 로드 완료 ({dev}), 잠재 채널 {vae.config.latent_channels}')

    reals = real_rows(root)
    random.Random(seed).shuffle(reals)
    mw = ManifestWriter(root, name)
    os.makedirs(os.path.join(root, 'raw', 'fake', name), exist_ok=True)

    for i, r in enumerate(reals[:n]):
        rel = f'fake/{name}/{r["source_id"]}__{name}.png'
        if rel in mw.done:
            continue
        im = Image.open(os.path.join(root, 'raw', r['file'])).convert('RGB')
        tiled = False
        try:
            out = reconstruct(vae, im, dev)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            vae.enable_tiling()
            out, tiled = reconstruct(vae, im, dev), True
            vae.disable_tiling()
        out.save(os.path.join(root, 'raw', rel))
        mw.add(file=rel, label=1, type='vae_recon', family=family, generator=name,
               source_id=r['source_id'], group=r['group'], split=r['split'],
               params={'repo': repo, 'tiled': tiled, 'latent_channels': vae.config.latent_channels})
        if (i + 1) % 50 == 0:
            print(f'{i + 1}/{min(n, len(reals))}')
    mw.close()
    print(f'완료 → 표: {mw.path}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('root')
    ap.add_argument('--n', type=int, default=500)
    ap.add_argument('--seed', type=int, default=3)
    ap.add_argument('--repo', default='black-forest-labs/FLUX.1-schnell')
    ap.add_argument('--subfolder', default='vae')
    ap.add_argument('--family', default='E2')
    ap.add_argument('--name', default='flux_vae')
    a = ap.parse_args()
    main(a.root, a.n, a.seed, a.repo, a.subfolder, a.family, a.name)
