"""val 점수(diag.py가 만든 diag_val.csv)로 판정 문턱 T를 훑는다. 학습·GPU 없음(로컬 가능). 공개샘플은 안 쓴다.
사용: python -I train/diag_threshold.py <diag_val.csv> [--extra T1,T2,...] [--boot 500]
(10/17 exp18에서 처음 씀 — 아래 선택 규칙은 그때 결과를 보기 전에 docs/진행상황.md에 먼저 적어 둔 것)

선택 규칙:
  후보 T = 0.5, 0.4, 0.3, 0.25, 0.2, 0.15, 0.1, 0.07, 0.05  (S = ln((1-T)/T)를 마지막 층 bias에 더하는 것과 같음)
  제약: 실제 정확도 ≥ 0.970 (v5가 0.973으로 리더보드 +0.02를 얻은 선례를 하한으로)
  제약을 통과한 T 중 '처음 본 방식 균형 정확도(bal_unseen)'가 가장 높은 T. 0.002 이내 차이면 더 큰 T(보수적) 선택.
  채택 최소 조건: 그 T의 bal_unseen이 T=0.5보다 +0.5%p 이상 높고, 군집 부트스트랩(같은 원본 묶음) 95% 구간의 하한이 0 이상.
균형 정확도 = 0.5 × 실제 정확도 + 0.5 × 유형별 가짜 recall 평균 (대회처럼 유형을 같은 비중으로).
채택하면 굽기: python train/diag_usage.py --weights <가중치 폴더> --bake_to <새 weights.safetensors> --shift S
"""
import argparse, math, sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, str(Path(__file__).resolve().parent))
from diag_usage import _metrics, HOLD_DEFAULT          # 처음 본 방식 목록·균형 정확도 계산은 지금까지 쓰던 것과 같은 코드

ap = argparse.ArgumentParser()
ap.add_argument('csv')
ap.add_argument('--extra', default='', help='후보에 더할 T(쉼표). 예: 0.12,0.08 (규칙을 바꾸는 게 아니라 격자를 촘촘히 보는 용도 — 선택 규칙은 미리 정한 것)')
ap.add_argument('--boot', type=int, default=500)
ap.add_argument('--fpr', type=float, default=0.01)
a = ap.parse_args()

d = pd.read_csv(a.csv, usecols=['file', 'y', 'generator', 'type', 'p', 'split', 'source_id'])
n_nan = int(d['p'].isna().sum())
if n_nan:
    print(f'⚠️ p가 NaN인 사진 {n_nan}장 → 0.5로 채움')
    d['p'] = d['p'].fillna(0.5)
d['type'] = d['type'].fillna('')
y = d['y'].values.astype(int)
p = d['p'].values.astype(np.float64)
typ = d['type'].values.astype(str)
hold = tuple(s.strip() for s in HOLD_DEFAULT.split(',') if s.strip())
unseen = np.array([(yy == 1) and str(g).startswith(hold) for yy, g in zip(y, d['generator'])])
print(f'{len(d)}장 (실제 {(y == 0).sum()} / 가짜 {(y == 1).sum()} · 그중 처음 본 방식 {unseen.sum()}) | split: {dict(d["split"].value_counts())}')

pc = np.clip(p, 1e-12, 1 - 1e-12)
logit = np.log(pc / (1 - pc))
Ts = sorted({0.5, 0.4, 0.3, 0.25, 0.2, 0.15, 0.1, 0.07, 0.05} | {float(t) for t in a.extra.split(',') if t.strip()}, reverse=True)


def bal(pp, thr, kind):
    """kind = 'all'|'unseen' — 균형 정확도(실제 정확도와 유형별 가짜 recall 평균의 평균). _metrics와 같은 정의, 부트스트랩용으로 가볍게."""
    real = y == 0
    ra = float((pp[real] < thr).mean())
    m = (y == 1) & (unseen if kind == 'unseen' else True)
    rec = [float((pp[m & (typ == t)] >= thr).mean()) for t in sorted(set(typ[m]))]
    return 0.5 * ra + 0.5 * float(np.mean(rec))


rows = []
base = None
for T in Ts:
    S = math.log((1 - T) / T)
    m = _metrics(logit + S, y, typ, unseen, a.fpr)
    # 같은 정의인지 확인(부트스트랩에 쓰는 가벼운 함수와 _metrics가 일치해야 함)
    assert abs(bal(p, T, 'unseen') - m['bal_unseen']) < 1e-9 and abs(bal(p, T, 'all') - m['bal_all']) < 1e-9, (T, bal(p, T, 'unseen'), m['bal_unseen'])
    fk = y == 1
    m['acc'] = float(((p >= T) == (y == 1)).mean())
    m['fake_recall'] = float((p[fk] >= T).mean())
    m['T'], m['S'] = T, S
    rows.append(m)
    if T == 0.5:
        base = m

print(f'\nAUROC 전체 {rows[0]["auroc_all"]:.4f} · 처음 본 방식 {rows[0]["auroc_unseen"]:.4f} (문턱과 무관) | 처음 본 방식 균형(오탐 {a.fpr:.0%} 고정, 문턱 무관) balf_unseen {rows[0]["balf_unseen"]:.4f}')
print(f'\n{"T":>5}{"S":>8}{"실제":>7}{"가짜rec":>9}{"acc":>7}{"bal_all":>9}{"bal_un":>8}{"Δun(%p)":>9}{"Δall(%p)":>9}')
for m in rows:
    print(f'{m["T"]:>5.2f}{m["S"]:>8.4f}{m["real_acc"]:>7.3f}{m["fake_recall"]:>9.3f}{m["acc"]:>7.3f}{m["bal_all"]:>9.4f}{m["bal_unseen"]:>8.4f}'
          f'{(m["bal_unseen"] - base["bal_unseen"]) * 100:>+9.2f}{(m["bal_all"] - base["bal_all"]) * 100:>+9.2f}')

types = sorted(base['rec_all'])
print(f'\n유형별 가짜 recall (전체 / 처음 본 방식)')
print(f'{"T":>5}' + ''.join(f'{t[:11]:>14}' for t in types))
for m in rows:
    print(f'{m["T"]:>5.2f}' + ''.join(f'{m["rec_all"][t]:>7.3f}/{m["rec_unseen"].get(t, float("nan")):<6.3f}' for t in types))

# 선택 규칙 적용
ok = [m for m in rows if m['real_acc'] >= 0.970]
best = max(ok, key=lambda m: (round(m['bal_unseen'] / 0.002), m['T']))       # 0.002 이내는 같은 값으로 보고 큰 T 우선
print(f'\n제약(실제 ≥ 0.970)을 통과한 T: {[m["T"] for m in ok]}')
print(f'→ 선택 T = {best["T"]} (S = {best["S"]:.6f}): 실제 {best["real_acc"]:.3f} · bal_unseen {best["bal_unseen"]:.4f} (T=0.5 대비 {(best["bal_unseen"] - base["bal_unseen"]) * 100:+.2f}%p) · bal_all {best["bal_all"]:.4f} ({(best["bal_all"] - base["bal_all"]) * 100:+.2f}%p)')

# 군집 부트스트랩: 같은 source_id를 한 묶음으로 다시 뽑기(실제·가짜가 같은 원본을 공유하므로 사진 단위 부트스트랩은 구간이 너무 좁다)
rng = np.random.default_rng(0)
src = d['source_id'].fillna(d['file']).astype(str).values
groups = {}
for i, s in enumerate(src):
    groups.setdefault(s, []).append(i)
keys = list(groups)
diffs = {'unseen': [], 'all': []}
real_diff = []
for _ in range(a.boot):
    pick = rng.choice(len(keys), size=len(keys), replace=True)
    ii = np.concatenate([groups[keys[k]] for k in pick])
    y0, p0, t0, u0 = y, p, typ, unseen
    y, p, typ, unseen = y0[ii], p0[ii], t0[ii], u0[ii]
    try:
        diffs['unseen'].append(bal(p, best['T'], 'unseen') - bal(p, 0.5, 'unseen'))
        diffs['all'].append(bal(p, best['T'], 'all') - bal(p, 0.5, 'all'))
        real_diff.append(float((p[y == 0] < best['T']).mean()) - float((p[y == 0] < 0.5).mean()))
    except ValueError:        # 드물게 한 유형이 통째로 빠지면 평균을 못 냄 → 그 표본은 건너뜀
        pass
    y, p, typ, unseen = y0, p0, t0, u0
for k, v in diffs.items():
    v = np.array(v) * 100
    print(f'부트스트랩({len(v)}회, 원본 묶음 {len(keys)}개 단위) T={best["T"]} − 0.5 의 균형 정확도 차이 [{k}]: 평균 {v.mean():+.2f}%p · 95% 구간 [{np.percentile(v, 2.5):+.2f}, {np.percentile(v, 97.5):+.2f}]')
rd = np.array(real_diff) * 100
print(f'  실제 정확도 차이: 평균 {rd.mean():+.2f}%p · 95% 구간 [{np.percentile(rd, 2.5):+.2f}, {np.percentile(rd, 97.5):+.2f}]')
lo = np.percentile(np.array(diffs['unseen']) * 100, 2.5)
gain = (best['bal_unseen'] - base['bal_unseen']) * 100
verdict = best['T'] != 0.5 and gain >= 0.5 and lo >= 0
print(f'\n채택 최소 조건 (T≠0.5, 이득 ≥ +0.5%p, 부트스트랩 하한 ≥ 0): {"PASS" if verdict else "FAIL"}  (이득 {gain:+.2f}%p, 하한 {lo:+.2f}%p)')
