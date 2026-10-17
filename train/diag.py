"""학습된 모델 진단 — 사진마다 최종 확률·Gate·Expert 점수·Common을 0으로 막은 확률을 뽑고, 약한 곳의 원인을 표로 본다.
학습은 하지 않는다(읽기 전용). 10/4 공개샘플 진단(docs/진행상황.md)을 val 전체로 넓힌 것.

볼 것 (표 A~E):
  A. 합치는 방식별 정확도·AUROC — 최종 / Common 끔 / Expert 평균 / Expert max / E2 뺀 max / Gate 가중합
  B. 생성기별 정확도 — 최종 vs Common 끔 vs Expert max   (Common이 가짜를 뒤집는 곳, max가 도움 되는 곳)
  C. 실제 사진에서 Expert별 점수 — E2가 실제를 가짜라고 하는지 (공개샘플 실제 3장은 0.72·0.50·0.96이었음)
  D. Gate 확신도별 정확도 — Gate 최대값이 낮을수록(균등) 틀리는지
  E. Common 뒤집기 — Common 때문에 맞던 가짜가 틀려진 수(veto) / 틀리던 가짜가 맞아진 수(rescue)

사용법 (Colab, 저장소 루트에서; 이미 로컬로 복사해 둔 데이터면 --local_cache 그대로):
  python train/diag.py --data /content/drive/MyDrive/work/out/C --local_cache /content/data_C \\
      --weights /content/drive/MyDrive/work/runs/full1 --split val --out /content/drive/MyDrive/work/runs/full1/diag_val.csv
  (가중치 폴더 = config.json + weights.safetensors, 제출 형식. 학습셋 진단은 --split train --per_generator 60)
이미 만든 CSV만 다시 분석하려면: python train/diag.py --analyze_only <csv>   (GPU 필요 없음, 로컬 가능)

leave-one-generator-out(처음 보는 생성기 점수): train.py --holdout <앞부분>으로 학습한 가중치에 대해
  python train/diag.py ... --weights <holdout 런> --only_generator sdxl-img2img --out <csv>
  → 테스트 = 빼 둔 생성기의 가짜 전부(train+val, 학습에서 안 봄) + val 실제. 표 F는 같은 원본의 실제·가짜 짝 비교(장면 지름길 제거).
  ⚠️ holdout하지 않은 가중치에 쓰면 '본 적 있는' 점수가 된다.
"""
import argparse, csv, json, os, random, sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


def run_model(args):
    import torch
    from train import read_manifest, make_loader, encode, ensure_local     # train.py는 import해도 학습이 시작되지 않는다
    from detector import Detector, load_weights

    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    wdir = Path(args.weights)
    cfg = json.loads((wdir / 'config.json').read_text(encoding='utf-8'))
    model = Detector(cfg)
    load_weights(model, wdir / cfg['weights'])
    model.prepare(dev)
    mean = torch.tensor(cfg['mean'], device=dev).view(1, 3, 1, 1); std = torch.tensor(cfg['std'], device=dev).view(1, 3, 1, 1)

    data_dir = ensure_local(args.data, args.local_cache) if args.local_cache else args.data
    args.data_dir = data_dir
    rows = read_manifest(os.path.join(data_dir, 'manifest.csv'), args.family_map)      # 학습 때 쓴 --family_map과 같은 값을 줘야 표의 family가 학습 때와 같다
    if args.only_generator:
        pre = tuple(p.strip() for p in args.only_generator.split(',') if p.strip())
        rows = [r for r in rows if (r['y'] == 1 and r['generator'].startswith(pre)) or (r['y'] == 0 and r['split'] == 'val')]
        assert any(r['y'] == 1 for r in rows), f'--only_generator {args.only_generator}에 맞는 가짜가 없음'
    elif args.split != 'all':
        rows = [r for r in rows if r['split'] == args.split]
    if args.per_generator:
        by = defaultdict(list)
        for r in rows:
            by[r['generator']].append(r)
        rg = random.Random(0)
        rows = [r for g in sorted(by) for r in (rg.sample(by[g], args.per_generator) if len(by[g]) > args.per_generator else by[g])]
    print(f'{args.only_generator or args.split} {len(rows)}장 진단 시작 (실제 {sum(r["y"] == 0 for r in rows)} / 가짜 {sum(r["y"] == 1 for r in rows)}, 장치 {dev})', flush=True)

    bs = args.batch
    loader = make_loader(data_dir, rows, [list(range(i, min(i + bs, len(rows)))) for i in range(0, len(rows), bs)], False, args.workers)
    amp = dev.type == 'cuda'
    out = []
    with torch.no_grad():
        for n, batch in enumerate(loader):
            with torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=amp):
                tokens, common = encode(model, batch, dev, mean, std)
                o = model.combine(tokens, common)
                o0 = model.combine(tokens, torch.zeros_like(common))       # Common을 0으로 막은 경우 (head가 본 적 없는 입력이라 방향만 참고)
            p = torch.sigmoid(o['logit'].float()).tolist(); p0 = torch.sigmoid(o0['logit'].float()).tolist()
            g = o['gate'].float().tolist(); e = torch.sigmoid(o['expert_logits'].float()).tolist()
            for k, i in enumerate(batch['idx']):
                r = rows[i]
                out.append([r['file'], r['y'], r['family'], r['generator'], r.get('type', ''), p[k], p0[k]] + g[k] + e[k] + [r['split'], r.get('source_id', '')])
            if (n + 1) % 20 == 0:
                print(f'  {len(out)}/{len(rows)}', flush=True)
    n_exp = len(model.experts)        # Expert 5개(기본) 또는 8개(--family_map type8로 학습한 가중치)
    cols = ['file', 'y', 'family', 'generator', 'type', 'p', 'p_nocommon'] + [f'g{k}' for k in range(1, n_exp + 1)] + [f'e{k}' for k in range(1, n_exp + 1)] + ['split', 'source_id']
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, 'w', encoding='utf-8', newline='') as f:
        w = csv.writer(f); w.writerow(cols); w.writerows(out)
    print('저장 →', args.out)
    return args.out


def analyze(path):
    import pandas as pd
    from sklearn.metrics import roc_auc_score
    d = pd.read_csv(path)
    n_nan = int(d['p'].isna().sum())
    if n_nan:       # 최종 확률이 NaN인 사진(Common CNN fp16 넘침 등) — 제출용 predict처럼 0.5로 막고 몇 장인지 알린다
        print(f'⚠️ p가 NaN인 사진 {n_nan}장 → 0.5로 채움 (generator: {dict(d[d.p.isna()].generator.value_counts())})')
        d['p'] = d['p'].fillna(0.5)
    y = d['y'].values
    n_exp = sum(1 for c in d.columns if c[0] == 'e' and c[1:].isdigit())      # CSV에 든 Expert 수 (5 또는 8)
    E = d[[f'e{k}' for k in range(1, n_exp + 1)]].values; G = d[[f'g{k}' for k in range(1, n_exp + 1)]].values
    d['mean_exp'] = E.mean(1); d['max_exp'] = E.max(1); d['max_noE2'] = np.delete(E, 1, axis=1).max(1)
    d['gated'] = 1 / (1 + np.exp(-(G * np.log(np.clip(E, 1e-6, 1 - 1e-6) / np.clip(1 - E, 1e-6, 1))).sum(1)))   # Gate 가중합(로짓 기준)
    d['gate_max'] = G.max(1)
    print(f'\n=== {Path(path).name} : {len(d)}장 (실제 {int((y == 0).sum())} / 가짜 {int((y == 1).sum())}) ===')

    def acc(s, m=None):
        m = np.ones(len(d), bool) if m is None else m
        return float(((s[m] >= .5) == (y[m] == 1)).mean()) if m.any() else float('nan')

    print('\n[A] 합치는 방식별 (문턱 0.5)')
    print(f'{"방식":<14}{"acc":>7}{"AUROC":>8}{"실제정확도":>10}{"가짜recall":>10}')
    for name, col in [('최종', 'p'), ('Common 끔', 'p_nocommon'), ('Expert 평균', 'mean_exp'), ('Expert max', 'max_exp'),
                      ('E2 뺀 max', 'max_noE2'), ('Gate 가중합', 'gated')]:
        s = d[col].values
        print(f'{name:<14}{acc(s):>7.3f}{roc_auc_score(y, s):>8.3f}{acc(s, y == 0):>10.3f}{acc(s, y == 1):>10.3f}')

    print('\n[B] 생성기별 정확도 (실제는 real 행)')
    print(f'{"generator":<26}{"family":>7}{"n":>6}{"최종":>7}{"Common끔":>9}{"Expert max":>11}{"p평균":>7}')
    for (fam, gen), g in d.groupby(['family', 'generator']):
        m = (d['family'] == fam) & (d['generator'] == gen)
        print(f'{gen:<26}{fam:>7}{int(m.sum()):>6}{acc(d.p.values, m.values):>7.2f}{acc(d.p_nocommon.values, m.values):>9.2f}'
              f'{acc(d.max_exp.values, m.values):>11.2f}{d.p[m].mean():>7.2f}')

    print('\n[C] 실제 사진에서 Expert별 점수 (낮아야 정상)')
    r = d[d.y == 0]
    print(f'{"":<10}' + ''.join(f'{"E" + str(k):>8}' for k in range(1, n_exp + 1)))
    print(f'{"평균":<10}' + ''.join(f'{r[f"e{k}"].mean():>8.2f}' for k in range(1, n_exp + 1)))
    print(f'{"≥0.5 비율":<8}' + ''.join(f'{(r[f"e{k}"] >= .5).mean():>8.2f}' for k in range(1, n_exp + 1)))

    print('\n[D] Gate 확신도(최대값)별 — 낮을수록 균등')
    print(f'{"구간":<12}{"n":>6}{"실제":>6}{"가짜":>6}{"실제정확도":>10}{"가짜recall":>10}{"Gate가 계열 맞힘":>16}')
    for lo, hi in [(0, .4), (.4, .8), (.8, 1.01)]:
        m = ((d.gate_max >= lo) & (d.gate_max < hi)).values
        fk = m & (y == 1)
        fam = d.family.map({f'E{k}': k - 1 for k in range(1, n_exp + 1)}).fillna(-1).astype(int).values
        top1 = float((G[fk].argmax(1) == fam[fk]).mean()) if fk.any() else float('nan')
        print(f'{f"{lo:.1f}~{min(hi, 1):.1f}":<12}{int(m.sum()):>6}{int((m & (y == 0)).sum()):>6}{int(fk.sum()):>6}'
              f'{acc(d.p.values, m & (y == 0)):>10.3f}{acc(d.p.values, fk):>10.3f}{top1:>16.3f}')

    print('\n[E] Common 뒤집기 (가짜만) — veto: Common 끄면 가짜로 맞는데 켜면 틀림 / rescue: 반대')
    print(f'{"generator":<26}{"n":>6}{"veto":>6}{"rescue":>8}')
    fk = d[d.y == 1]
    for gen, g in fk.groupby('generator'):
        veto = int(((g.p_nocommon >= .5) & (g.p < .5)).sum()); resc = int(((g.p_nocommon < .5) & (g.p >= .5)).sum())
        print(f'{gen:<26}{len(g):>6}{veto:>6}{resc:>8}')
    print(f'{"전체":<26}{len(fk):>6}{int(((fk.p_nocommon >= .5) & (fk.p < .5)).sum()):>6}{int(((fk.p_nocommon < .5) & (fk.p >= .5)).sum()):>8}')
    paired(d)


def paired(d):
    """[F] 같은 원본(source_id)의 실제·가짜 짝(둘 다 val = 둘 다 학습에서 안 봄)에서 가짜 점수가 실제보다 높은 비율.
    장면이 같으니 '장면이 이상하면 가짜' 지름길이 통하지 않는다 → 진짜 흔적을 봤는지의 지표."""
    if 'source_id' not in d.columns:
        return
    v = d[(d['split'] == 'val') & d.source_id.notna()]
    real = v[v.y == 0].drop_duplicates('source_id').set_index('source_id').p
    fk = v[(v.y == 1) & v.source_id.isin(real.index)].copy()
    if fk.empty:
        print('\n[F] 짝 비교: 같은 원본의 val 실제·가짜 짝이 없음')
        return
    fk['win'] = [1.0 if p > real[s] else (0.5 if p == real[s] else 0.0) for p, s in zip(fk.p, fk.source_id)]
    print('\n[F] 같은 원본 짝에서 가짜 점수 > 실제 점수 비율 (val 짝만, 50%면 못 가름)')
    print(f'{"generator":<26}{"짝 수":>6}{"win":>7}')
    for gen, g in fk.groupby('generator'):
        print(f'{gen:<26}{len(g):>6}{g.win.mean():>7.2f}')
    print(f'{"전체":<26}{len(fk):>6}{fk.win.mean():>7.2f}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default=''); ap.add_argument('--local_cache', default='')
    ap.add_argument('--weights', default='', help='config.json + weights.safetensors가 있는 폴더 (예: runs/full1)')
    ap.add_argument('--split', default='val', choices=['val', 'train', 'all', 'extra'], help="extra = 학습·검증에 안 쓰고 따로 둔 '실제 오탐 점검용' 사진(추가 실제 사진을 합치는 스크립트가 만든 split='extra')")
    ap.add_argument('--only_generator', default='', help='쉼표로 구분한 generator 앞부분. 주면 그 가짜 전부(train+val) + val 실제만 테스트 (leave-one-generator-out용)')
    ap.add_argument('--per_generator', type=int, default=0, help='0이면 전부. 주면 생성기마다 이만큼만(학습셋은 60 정도 권장)')
    ap.add_argument('--batch', type=int, default=8); ap.add_argument('--workers', type=int, default=2)
    ap.add_argument('--out', default='diag.csv')
    ap.add_argument('--analyze_only', default='', help='이미 만든 CSV만 다시 분석')
    ap.add_argument('--family_map', default='none', choices=['none', 'type8', 'gen24'], help='학습 때 쓴 값과 같게 (type8로 학습한 가중치면 type8). 표의 family·Gate 정답이 이걸로 정해진다')
    args = ap.parse_args()
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')       # Windows 콘솔(cp949)에서 —·≥ 같은 기호가 에러 나지 않게
    if args.analyze_only:
        analyze(args.analyze_only)
        return
    assert args.data and args.weights, '--data 와 --weights 를 주세요'
    analyze(run_model(args))


if __name__ == '__main__':
    main()
