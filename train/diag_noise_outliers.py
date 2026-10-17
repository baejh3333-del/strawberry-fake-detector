"""강한 노이즈 사진에서 Common CNN 특징이 비정상적으로 커져 가짜를 '진짜'로 확신하며 틀리는 문제 점검 (10/6, 진행상황.md).

제출 형식 폴더(inference.py + model/)를 불러 filter-noise 사진마다 (a) Common 특징의 최대 절댓값 (b) 최종 확률 p를 잰다.
특징 최대값이 5를 넘는 사진('튀는 사진')의 수와 그 사진들의 p, 노이즈 세기(σ) 구간별 recall을 출력한다.
기준(exp06, 로컬 filter-noise 250장): 튀는 사진 6장(>20은 3장) 모두 학습셋, p≈0. 정상 사진의 특징 최대값은 중앙값 1~2, 최대 약 4.

사용법 (로컬, 사진당 약 5초):
  py train/diag_noise_outliers.py --team_dir <제출 폴더> --data ./data/out/C [--n 250]
"""
import argparse, csv, json, sys
from pathlib import Path
import numpy as np
import torch

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
ap = argparse.ArgumentParser()
ap.add_argument('--team_dir', required=True); ap.add_argument('--data', required=True)
ap.add_argument('--n', type=int, default=250); ap.add_argument('--generator', default='filter-noise')
a = ap.parse_args()

sys.path.insert(0, str(Path(a.team_dir).resolve()))
import inference as inf                                       # noqa: E402

model = inf.load_model(str(Path(a.team_dir).resolve() / 'model')); dev = model.device_
amp = dev.type == 'cuda'
D = Path(a.data)
rows = [r for r in csv.DictReader(open(D / 'manifest.csv', encoding='utf-8')) if r['generator'] == a.generator][:a.n]
out = []
for i, r in enumerate(rows):
    x = inf._load_image(D / r['file'], model.mean_, model.std_).unsqueeze(0).to(dev)
    with torch.no_grad():
        with torch.autocast('cuda', dtype=torch.float16, enabled=amp):
            tokens = model.get_tokens(x)
        common = model.common(x)                              # 제출 predict와 같은 순서(Common은 fp32)
        with torch.autocast('cuda', dtype=torch.float16, enabled=amp):
            logit = model.combine(tokens, common)['logit'].float().item()
    out.append((json.loads(r['params']).get('std', 0.0), common.abs().max().item(), logit, r['split'], r['file']))
    if i % 25 == 0:
        print(i, flush=True)
sig, cmax, logit = (np.array([o[k] for o in out]) for k in (0, 1, 2))
p = 1 / (1 + np.exp(-np.clip(logit, -50, 50)))
big = cmax > 5
print(f'\n{a.generator} {len(out)}장 | 정상 특징 최대값 중앙값 {np.median(cmax[~big]):.2f}')
print(f'튀는 사진(특징 최대값 > 5): {big.sum()}장, > 20: {(cmax > 20).sum()}장')
print('  그 사진들의 (σ, 특징최대, 로짓, p, split):')
for o, pp in sorted(zip(out, p), key=lambda t: -t[0][1]):
    if o[1] > 5:
        print(f'   σ={o[0]:.1f} cmax={o[1]:.1f} logit={o[2]:.1f} p={pp:.3f} {o[3]}')
print('σ 구간별 recall@0.5:')
for lo, hi in [(0, 10), (10, 15), (15, 20), (20, 26)]:
    m = (sig >= lo) & (sig < hi)
    if m.sum():
        print(f'   σ {lo}-{hi}: n={m.sum()} recall {(p[m] >= .5).mean():.3f}')
