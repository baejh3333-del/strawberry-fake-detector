"""죽은 Expert를 빼도 점수가 변하는지 본다 — 학습은 하지 않는다(읽기 전용). 10/16 "Expert가 너무 많아서 정확도가 떨어지나?" 확인용.

exp11b(= Groot v3)는 11방식의 가짜를 학습에서 뺐다(--holdout). 그 11방식의 Expert는 가짜를 한 장도 못 보고 실제만 봐서 사실상 죽은 Expert다.
같은 가중치에서 죽은 Expert 11개를 추론 때만 빼고 val을 다시 채점한다 (백본 토큰은 한 번만 계산해 변형 3가지에 같이 쓴다).

변형
  base     : 그대로 (24개 Gate 확률 그대로)
  renorm   : 죽은 Expert의 Gate 확률을 0으로 하고 산 13개끼리 합이 1이 되게 다시 나눔  ← 13개로 재학습한 모델과 가장 가까운 근사
  drop     : 죽은 Expert의 Gate 확률만 0으로 (다시 나누지 않음 → 가중합이 작아짐)
⚠️ head는 24개 구조로 학습돼서 renorm도 재학습 모델과 같지는 않다. "죽은 Expert가 점수에 끼치는 영향"을 보는 시험이지, 13개 모델의 점수 예측이 아니다.

사용법 (Colab, 저장소 루트에서; diag_usage.py와 같은 데이터·가중치):
  python train/diag_mask.py --data /content/data_probe --weights <exp11b 가중치 폴더> --batch 16 --workers 4 --out mask_diag.csv
이미 만든 CSV만 다시 보려면: python train/diag_mask.py --analyze_only mask_diag.csv   (GPU 필요 없음)
"""
import argparse, json, os, sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

# exp11b가 학습에서 뺀 11방식 (train.py GEN_METHODS의 두 번째 값). 진행상황.md exp11 A/B 참고
DEAD = ['realvisxl', 'flux_schnell', 'infinity', 'flux_vae', 'objpaste', 'sd15_inpaint', 'sdxl_partial', 'local_filter',
        'klein_t2i', 'klein_edit', 'partial_classic']
VARIANTS = ['base', 'renorm', 'drop']
# 처음 본 방식 generator 앞부분 — diag_usage.py의 HOLD_DEFAULT와 같다 (Drive의 옛 diag_usage.py에도 없어서 여기에 둔다)
HOLD = ('realvisxl', 'flux_schnell', 'infinity_2b', 'flux_vae', 'composite-objpaste', 'sd15-inpaint', 'sdxl-partial-', 'local-filter-',
        'flux2-klein-4b', 'flux2-klein-edit-', 'partial-classic-')


def dead_index():
    from train import GEN_METHODS
    names = [m for _, m in GEN_METHODS]
    return [names.index(n) for n in DEAD]


def masked_logit(model, tokens, common, dead, mode):
    """Detector.combine과 같은 계산인데 Gate 확률만 바꾼다."""
    import torch
    gate = model.gate(tokens)
    if mode != 'base':
        keep = torch.ones(gate.shape[1], device=gate.device); keep[dead] = 0
        gate = gate * keep
        if mode == 'renorm':
            gate = gate / gate.sum(1, keepdim=True).clamp_min(1e-9)
    outs = [e(tokens) for e in model.experts]
    feats = torch.stack([o[0] for o in outs], 1); scores = torch.stack([o[1] for o in outs], 1)
    mixed = (gate.unsqueeze(-1) * feats).sum(1); mixed_score = (gate * scores).sum(1, keepdim=True)
    if len(model.fusion_norm):
        mixed, common = model.fusion_norm[0](mixed), model.fusion_norm[1](common)
    return model.head(torch.cat([mixed, mixed_score, common], 1)).squeeze(1)


def run_model(args):
    import torch
    from train import read_manifest, make_loader, encode
    from detector import Detector, load_weights

    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    wdir = Path(args.weights)
    cfg = json.loads((wdir / 'config.json').read_text(encoding='utf-8'))
    assert cfg['n_experts'] == 24, f'n_experts={cfg["n_experts"]} — gen24 가중치(exp11b)에만 쓴다'
    model = Detector(cfg); load_weights(model, wdir / cfg['weights']); model.prepare(dev)
    mean = torch.tensor(cfg['mean'], device=dev).view(1, 3, 1, 1); std = torch.tensor(cfg['std'], device=dev).view(1, 3, 1, 1)
    dead = dead_index()

    rows = [r for r in read_manifest(os.path.join(args.data, 'manifest.csv'), 'gen24') if r['split'] == 'val']
    print(f'val {len(rows)}장 (실제 {sum(r["y"] == 0 for r in rows)} / 가짜 {sum(r["y"] == 1 for r in rows)}), 죽은 Expert {len(dead)}개 인덱스 {dead}, 장치 {dev}', flush=True)
    bs = args.batch
    loader = make_loader(args.data, rows, [list(range(i, min(i + bs, len(rows)))) for i in range(0, len(rows), bs)], False, args.workers)
    amp = dev.type == 'cuda'
    out = []
    with torch.no_grad():
        for n, batch in enumerate(loader):
            with torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=amp):
                tokens, common = encode(model, batch, dev, mean, std)
                lg = {m: masked_logit(model, tokens, common, dead, m).float() for m in VARIANTS}
                gate = model.gate(tokens).float()
            p = {m: torch.sigmoid(v).tolist() for m, v in lg.items()}
            dead_mass = gate[:, dead].sum(1).tolist()
            for k, i in enumerate(batch['idx']):
                r = rows[i]
                out.append([r['file'], r['y'], r['generator'], r.get('type', ''), dead_mass[k]] + [p[m][k] for m in VARIANTS])
            if (n + 1) % 20 == 0:
                print(f'  {len(out)}/{len(rows)}', flush=True)
    import csv
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, 'w', encoding='utf-8', newline='') as f:
        w = csv.writer(f); w.writerow(['file', 'y', 'generator', 'type', 'dead_gate'] + [f'p_{m}' for m in VARIANTS]); w.writerows(out)
    print('저장 →', args.out)
    return args.out


def is_unseen(gen):
    """val 가짜가 '처음 본 방식'인가 (diag_usage.py의 HOLD_DEFAULT와 같은 기준)."""
    return gen.startswith(HOLD)


def analyze(path):
    import pandas as pd
    from sklearn.metrics import roc_auc_score
    d = pd.read_csv(path)
    y = d['y'].values
    unseen = np.array([bool(g) and y[i] == 1 and is_unseen(str(g)) for i, g in enumerate(d['generator'])])
    print(f'\n=== {Path(path).name} : {len(d)}장 (실제 {int((y == 0).sum())} / 가짜 {int((y == 1).sum())}, 처음 본 방식 가짜 {int(unseen.sum())}) ===')
    print(f'죽은 Expert의 Gate 확률 합 평균: 실제 {d[d.y == 0].dead_gate.mean():.3f} · 가짜 {d[d.y == 1].dead_gate.mean():.3f} · 처음 본 방식 가짜 {d[unseen].dead_gate.mean():.3f}')

    def bal(p):
        """균형 = 0.5 × 실제 정확도 + 0.5 × type별 가짜 recall 평균 (대회처럼 실제:가짜 = 50:50, 유형 균등)"""
        real = float((p[y == 0] < .5).mean())
        rec = [float((p[(y == 1) & (d.type.values == t)] >= .5).mean()) for t in sorted(d[d.y == 1].type.dropna().unique())]
        return 0.5 * real + 0.5 * float(np.mean(rec))

    print(f'\n{"변형":<8}{"acc":>7}{"AUROC":>8}{"실제":>7}{"가짜recall":>10}{"균형":>7}{"처음본recall":>12}{"본적있는recall":>14}')
    for m in VARIANTS:
        p = d[f'p_{m}'].values
        seen = (y == 1) & ~unseen
        print(f'{m:<8}{float(((p >= .5) == (y == 1)).mean()):>7.3f}{roc_auc_score(y, p):>8.3f}{float((p[y == 0] < .5).mean()):>7.3f}'
              f'{float((p[y == 1] >= .5).mean()):>10.3f}{bal(p):>7.3f}{float((p[unseen] >= .5).mean()):>12.3f}{float((p[seen] >= .5).mean()):>14.3f}')

    print('\n사진 단위로 바뀐 수 (base → renorm): 문턱 0.5를 넘나든 장')
    b, r = d.p_base.values >= .5, d.p_renorm.values >= .5
    for name, mask in [('실제', y == 0), ('가짜(본 적 있는)', (y == 1) & ~unseen), ('가짜(처음 본)', unseen)]:
        print(f'  {name:<16} 가짜로 넘어감 {int((~b & r & mask).sum()):>4} · 실제로 넘어감 {int((b & ~r & mask).sum()):>4} (n={int(mask.sum())})')
    print('\n해석: 세 변형의 acc·균형이 ±0.005 안이면 죽은 Expert는 점수에 영향이 없다 → 13개로 줄여도(재학습해도) 개수 때문에 오를 근거는 약함.\n'
          '      renorm이 뚜렷이 오르면 죽은 Expert가 Gate 확률을 새게 해서 손해를 줬다는 뜻 → 13개 모델 재학습 후보.')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default=''); ap.add_argument('--weights', default='', help='exp11b 폴더 (config.json + weights.safetensors, n_experts 24)')
    ap.add_argument('--batch', type=int, default=8); ap.add_argument('--workers', type=int, default=2)
    ap.add_argument('--out', default='mask_diag.csv'); ap.add_argument('--analyze_only', default='', help='이미 만든 CSV만 다시 분석')
    args = ap.parse_args()
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    if args.analyze_only:
        analyze(args.analyze_only)
        return
    assert args.data and args.weights, '--data 와 --weights 를 주세요'
    analyze(run_model(args))


if __name__ == '__main__':
    main()
