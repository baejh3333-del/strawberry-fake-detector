"""최종 head가 Expert 쪽과 Common 쪽을 얼마나 쓰는지 본다 — 학습은 하지 않는다(읽기 전용).
10-3 할일 4번 "먼저 확인: 최종 MLP가 Expert 쪽을 실제로 쓰는지" 용.

방법: 사진마다 head 입력(feat 1345 = Expert 가중합 64 + Gate 점수 1 + Common 1280)을 한 번만 뽑아 저장(--dump)하고,
head(작은 MLP)만 CPU에서 다시 돌려 한쪽 입력을 바꿨을 때 정확도가 얼마나 떨어지는지 본다(--analyze).
 - 평균으로 대체: 그 쪽을 val 전체 평균 벡터로 바꿈 (0으로 막는 것보다 head가 덜 낯설어한다)
 - 섞기(permutation): 그 쪽을 다른 사진의 것으로 바꿈 (정보만 지우고 값의 분포는 유지) — 3번 평균
 - 0으로 막기: diag.py의 "Common 끔"과 같은 방식 (참고용)
 많이 떨어지는 쪽 = head가 실제로 쓰는 쪽. 거의 안 떨어지는 쪽 = 있으나 마나.

사용법 (저장소 루트에서, 제출 폴더 가중치 = config.json + weights.safetensors):
  python train/diag_usage.py --data /content/drive/MyDrive/work/out/C --local_cache /content/data_C --weights /content/drive/MyDrive/work/runs/full1 --n_real 300 --per_gen 20 --out /content/drive/MyDrive/work/runs/full1/usage_feat.npz
  python train/diag_usage.py --analyze .../usage_feat.npz --weights <가중치 폴더>     (GPU 필요 없음, 로컬 가능)

비중 조정(10/16) — 학습 없이 Common·Expert 입력의 상대 비중만 바꿔 본다 (다른 건 그대로):
  python train/diag_usage.py --data /content/data_probe --weights <가중치 폴더> --n_real 99999 --per_gen 99999 --batch 16 --workers 4 --out usage_feat.npz   (val 전부, GPU)
  python train/diag_usage.py --reweight usage_feat.npz --weights <가중치 폴더>      격자 탐색 + 교차 검증 + 채택 기준 판정 (GPU 필요 없음)
  python train/diag_usage.py --weights <가중치 폴더> --bake_to <새 weights.safetensors> --alpha A --beta B [--check usage_feat.npz]
  입력을 s배 하는 것 = head.0.weight의 해당 열을 s배 하는 것 → 구운 가중치는 inference.py·detector.py를 안 고쳐도 그대로 로드된다.
  --shift S 를 주면 마지막 층 bias에 S를 더한다(판정 기준 이동, 예: p≥0.05 = 2.9444). alpha·beta는 기본 1이라 shift만 굽는 데도 쓴다.
"""
import argparse, json, os, random, sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
ED = 64          # expert_dim: feat[:, :64] = Expert 가중합
SC = 65          # feat[:, 64] = Gate 가중합 점수, feat[:, 65:] = Common


def dump(args):
    import torch
    from train import read_manifest, make_loader, encode
    from detector import Detector, load_weights

    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    wdir = Path(args.weights)
    cfg = json.loads((wdir / 'config.json').read_text(encoding='utf-8'))
    model = Detector(cfg); load_weights(model, wdir / cfg['weights']); model.prepare(dev)
    mean = torch.tensor(cfg['mean'], device=dev).view(1, 3, 1, 1); std = torch.tensor(cfg['std'], device=dev).view(1, 3, 1, 1)

    rows = [r for r in read_manifest(os.path.join(args.data, 'manifest.csv')) if r['split'] == 'val']
    rg = random.Random(0)
    real = [r for r in rows if r['y'] == 0]; by = defaultdict(list)
    for r in rows:
        if r['y'] == 1:
            by[r['generator']].append(r)
    pick = rg.sample(real, min(args.n_real, len(real))) + [x for g in sorted(by) for x in rg.sample(by[g], min(args.per_gen, len(by[g])))]
    print(f'val에서 {len(pick)}장 선택 (실제 {sum(r["y"] == 0 for r in pick)} / 가짜 {sum(r["y"] == 1 for r in pick)}, 장치 {dev})', flush=True)

    if args.local_cache:        # Drive(파일 하나 약 1초)에서 뽑은 사진만 Colab 로컬 디스크로 복사 — 전체(11454장)를 옮기지 않는다
        from concurrent.futures import ThreadPoolExecutor
        import shutil

        def one(r):
            dst = os.path.join(args.local_cache, r['file'])
            if not os.path.exists(dst):
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                shutil.copyfile(os.path.join(args.data, r['file']), dst + '.part'); os.replace(dst + '.part', dst)
        with ThreadPoolExecutor(32) as ex:
            list(ex.map(one, pick))
        args.data = args.local_cache
    bs = args.batch
    loader = make_loader(args.data, pick, [list(range(i, min(i + bs, len(pick)))) for i in range(0, len(pick), bs)], False, args.workers)
    feats, logits = [], []
    amp = dev.type == 'cuda'
    with torch.no_grad():
        for n, batch in enumerate(loader):
            with torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=amp):
                tokens, common = encode(model, batch, dev, mean, std)
                o = model.combine(tokens, common)
            feats.append(o['feat'].float().cpu().numpy()); logits.append(o['logit'].float().cpu().numpy())
            if (n + 1) % 10 == 0:
                print(f'  {sum(len(f) for f in feats)}/{len(pick)}', flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez(args.out, feat=np.concatenate(feats), logit=np.concatenate(logits),
             y=np.array([r['y'] for r in pick]), family=np.array([r['family'] for r in pick]), generator=np.array([r['generator'] for r in pick]),
             files=np.array([r['file'] for r in pick]), type=np.array([r.get('type', '') for r in pick]), group=np.array([r.get('group', '') for r in pick]))   # files·type·group은 --reweight(유형별·포기별 교차 검증)용. ⚠️ 키 이름을 file로 하면 np.savez의 첫 인자 이름과 충돌한다
    print('저장 →', args.out)


def head_forward(W1, b1, W2, b2, x):
    from math import erf
    z = x @ W1.T + b1
    h = 0.5 * z * (1 + np.vectorize(erf)(z / np.sqrt(2)))        # GELU (head 안의 nn.GELU, 기본 exact)
    return (h @ W2.T + b2).reshape(-1)


def analyze(args):
    from safetensors.numpy import load_file
    from sklearn.metrics import roc_auc_score
    cfg = json.loads((Path(args.weights) / 'config.json').read_text(encoding='utf-8'))
    sd = load_file(str(Path(args.weights) / cfg['weights']))
    W1, b1, W2, b2 = (sd[k].astype(np.float32) for k in ['head.0.weight', 'head.0.bias', 'head.3.weight', 'head.3.bias'])
    d = np.load(args.analyze)
    X, y, gen, fam = d['feat'].astype(np.float32), d['y'], d['generator'], d['family']
    print(f'\n=== {len(y)}장 (실제 {int((y == 0).sum())} / 가짜 {int((y == 1).sum())}), feat {X.shape[1]}차원 ===')
    base = head_forward(W1, b1, W2, b2, X)
    print(f'확인: 저장 때 logit과 다시 계산한 logit의 최대 차이 {np.abs(base - d["logit"]).max():.4f} (autocast 오차 정도면 정상)')

    exp_idx = np.arange(0, SC); com_idx = np.arange(SC, X.shape[1])
    mu = X.mean(0)

    def variant(idx, how, seed=0):
        Z = X.copy()
        if how == 'zero':
            Z[:, idx] = 0
        elif how == 'mean':
            Z[:, idx] = mu[idx]
        else:
            Z[:, idx] = X[np.random.RandomState(seed).permutation(len(X))][:, idx]
        return Z

    def stats(logit):
        p = 1 / (1 + np.exp(-logit)); pred = p >= .5
        return (pred == (y == 1)).mean(), roc_auc_score(y, p), (~pred[y == 0]).mean(), pred[y == 1].mean(), p

    print(f'\n[A] 한쪽 입력을 바꿨을 때 (문턱 0.5)   Expert = feat 0~64 (가중합 64 + Gate 점수 1 = {len(exp_idx)}차원), Common = feat 65~1344 ({len(com_idx)}차원)')
    print(f'{"방식":<26}{"acc":>7}{"AUROC":>8}{"실제정확도":>10}{"가짜recall":>10}')
    res = {}
    for name, idx in [('Expert', exp_idx), ('Common', com_idx)]:
        for how, label in [('mean', '평균으로 대체'), ('shuffle', '다른 사진 것으로 섞기'), ('zero', '0으로 막기(참고)')]:
            runs = [stats(head_forward(W1, b1, W2, b2, variant(idx, how, s))) for s in (range(3) if how == 'shuffle' else [0])]
            a = [np.mean([r[i] for r in runs]) for i in range(4)]
            res[(name, how)] = runs[0][4]
            print(f'{(name + " " + label):<26}{a[0]:>7.3f}{a[1]:>8.3f}{a[2]:>10.3f}{a[3]:>10.3f}')
        if name == 'Expert':
            print(f'{"(원본)":<26}' + '{:>7.3f}{:>8.3f}{:>10.3f}{:>10.3f}'.format(*stats(base)[:4]))

    print('\n[B] head 첫 층이 각 입력에 두는 비중 (입력 단위로 정규화)')
    print('    비중 = 그 입력 덩어리가 첫 층 출력(256개)에 만드는 변동의 크기: |W[:,덩어리]| × 그 입력의 사진 간 표준편차')
    Wf = {'Expert 가중합(64)': slice(0, ED), 'Gate 점수(1)': slice(ED, SC), 'Common(1280)': slice(SC, X.shape[1])}
    sd_x = X.std(0)
    tot = {k: float(np.sqrt(((W1[:, s] * sd_x[s]) ** 2).sum())) for k, s in Wf.items()}
    allt = sum(v ** 2 for v in tot.values())
    for k, s in Wf.items():
        print(f'  {k:<20} 가중치 평균 크기 {np.abs(W1[:, s]).mean():.4f} · 입력 표준편차 평균 {sd_x[s].mean():.4f} · 변동 비중 {tot[k] ** 2 / allt * 100:5.1f}%')

    print('\n[C] 유형별 정확도 — 원본 vs Expert 평균으로 대체 vs Common 평균으로 대체  (어느 쪽이 그 유형을 맡고 있나)')
    print(f'{"generator":<26}{"family":>7}{"n":>5}{"원본":>7}{"Expert없이":>10}{"Common없이":>11}')
    pc = {'원본': 1 / (1 + np.exp(-base)), 'E': res[('Expert', 'mean')], 'C': res[('Common', 'mean')]}    # res에는 이미 확률이 들어 있다
    rows = [('real', '-', y == 0)] + [(g, str(fam[gen == g][0]), (gen == g) & (y == 1)) for g in sorted(set(gen[y == 1]))]
    for g, f, m in rows:
        if not m.any():
            continue
        def a(p):
            return float((p[m] < .5).mean()) if y[m][0] == 0 else float((p[m] >= .5).mean())
        print(f'{g:<26}{f:>7}{int(m.sum()):>5}{a(pc["원본"]):>7.2f}{a(pc["E"]):>10.2f}{a(pc["C"]):>11.2f}')


HOLD_DEFAULT = ('realvisxl,flux_schnell,infinity_2b,flux_vae,composite-objpaste,sd15-inpaint,sdxl-partial-,local-filter-,'
                'flux2-klein-4b,flux2-klein-edit-,partial-classic-')      # exp11b가 학습에서 뺀 11방식 = val에서 "처음 본 방식"


def load_head(wdir):
    """제출 폴더(config.json + weights.safetensors)에서 head 텐서 4개만 읽는다(백본 0.6GB는 안 읽음)."""
    from safetensors import safe_open
    cfg = json.loads((Path(wdir) / 'config.json').read_text(encoding='utf-8'))
    with safe_open(str(Path(wdir) / cfg['weights']), framework='pt') as f:
        return {k: f.get_tensor(k).float() for k in ('head.0.weight', 'head.0.bias', 'head.3.weight', 'head.3.bias')}


def head_logit(H, X, alpha=1.0, beta=1.0):
    """X [N,1345] → logit [N]. beta = Expert 쪽(feat 0~64) 입력 배수, alpha = Common 쪽(65~) 입력 배수.
    입력을 s배 하는 것은 head.0.weight의 같은 열을 s배 하는 것과 정확히 같다(--bake_to가 이걸 가중치에 굽는다)."""
    import torch
    s = torch.ones(X.shape[1]); s[:SC] = beta; s[SC:] = alpha
    z = (X * s) @ H['head.0.weight'].T + H['head.0.bias']
    return (torch.nn.functional.gelu(z) @ H['head.3.weight'].T + H['head.3.bias']).reshape(-1)


def _metrics(logit, y, typ, unseen, fpr):
    """한 (alpha, beta) 칸의 성적. 균형 정확도 = 0.5 × 실제 정확도 + 0.5 × 유형별 가짜 recall 평균(대회처럼 유형을 같은 비중으로).
    balf = 같은 계산인데 문턱을 '실제 오탐 fpr'로 맞춘 값 → 이득이 문턱 이동뿐인지 진짜 분리력인지 가린다."""
    from sklearn.metrics import roc_auc_score
    logit = logit.astype(np.float64)
    p = 1 / (1 + np.exp(-logit))
    real = y == 0
    m = {'real_acc': float((p[real] < .5).mean()), 'thr_fpr': float(np.quantile(p[real], 1 - fpr))}

    def per_type(mask, thr):
        return {t: float((p[mask & (typ == t)] >= thr).mean()) for t in sorted(set(typ[mask]))}
    for name, mask in (('all', y == 1), ('unseen', (y == 1) & unseen)):
        rec, rec_f = per_type(mask, .5), per_type(mask, m['thr_fpr'])
        m['rec_' + name] = rec
        m['bal_' + name] = 0.5 * m['real_acc'] + 0.5 * float(np.mean(list(rec.values())))
        m['balf_' + name] = 0.5 * (1 - fpr) + 0.5 * float(np.mean(list(rec_f.values())))
    sel = real | unseen
    m['auroc_all'] = float(roc_auc_score(y, logit))
    m['auroc_unseen'] = float(roc_auc_score(y[sel], logit[sel]))
    return m


def _checks(m, base):
    """채택 기준 중 칸 하나씩 보는 제약 3개(진행상황 10/16 계획 B3 ②③④)."""
    return {'실제 정확도 ≥ 0.985': m['real_acc'] >= min(0.985, base['real_acc']),      # 기준이 이미 0.985 미만인 경우(시험용 데이터)엔 기준보다 나빠지지만 않게
            '처음 본 방식 AUROC −0.003 이내': m['auroc_unseen'] >= base['auroc_unseen'] - 0.003,
            '유형별 recall −5%p 이내': all(m['rec_all'][t] >= base['rec_all'][t] - 0.05 for t in base['rec_all'])}


def _grid_eval(H, X, y, typ, unseen, alphas, betas, fpr):
    return {(a, b): _metrics(head_logit(H, X, a, b).numpy(), y, typ, unseen, fpr) for a in alphas for b in betas}


def _pick(res, base):
    """제약을 통과한 칸 중 '처음 본 방식' 균형 정확도가 가장 높은 칸. 동점이면 (1,1)에 가까운 칸. 통과한 칸이 없으면 기준 칸 (1,1)."""
    best, key = None, None
    for (a, b), m in res.items():
        if all(_checks(m, base).values()):
            k = (round(m['bal_unseen'], 4), -abs(np.log(a)) - abs(np.log(b)))
            if key is None or k > key:
                best, key = (a, b), k
    return best if best is not None else (1.0, 1.0)


def reweight(args):
    """Common·Expert 입력 비중만 바꿔 가며(다른 건 그대로) 가장 나은 칸을 찾는다. 학습·GPU 없음. 선택 기준은 미리 정한 것(아래 verdict)."""
    import hashlib, torch
    from sklearn.metrics import roc_auc_score
    H = load_head(args.weights)
    d = np.load(args.reweight)
    X = torch.from_numpy(d['feat'].astype(np.float32)); y = d['y'].astype(int); gen = d['generator']
    typ = d['type'] if 'type' in d.files else d['family']
    grp = d['group'] if 'group' in d.files else d['generator']
    hold = tuple(p.strip() for p in args.holdout.split(',') if p.strip())
    unseen = np.array([(yy == 1) and str(g).startswith(hold) for yy, g in zip(y, gen)])
    grid = [float(v) for v in args.grid.split(',')]
    assert 1.0 in grid, '--grid에 1이 있어야 기준 칸(1,1)이 생김'
    print(f'\n=== {len(y)}장 (실제 {int((y == 0).sum())} / 가짜 {int((y == 1).sum())} · 그중 처음 본 방식 {int(unseen.sum())}) | 문턱 0.5, 실제 오탐 {args.fpr:.0%} 맞춘 문턱은 balf ===')
    base_logit = head_logit(H, X).numpy()
    print(f'확인: 저장 때 logit과 head 재계산 logit의 최대 차이 {np.abs(base_logit - d["logit"]).max():.4f} (autocast 오차 정도면 정상)')

    mu = X.mean(0); sel = (y == 0) | unseen
    Xc = X.clone(); Xc[:, SC:] = mu[SC:]
    Xe = X.clone(); Xe[:, :SC] = mu[:SC]

    def auc(Z, mask=None):
        lg = head_logit(H, Z).numpy()
        return roc_auc_score(y, lg) if mask is None else roc_auc_score(y[mask], lg[mask])
    sd = X.std(0)
    print('\n[0] 블록 효과 — 한쪽을 평균으로 대체했을 때 AUROC (전체 / 처음 본 방식). 많이 떨어지는 쪽 = 머리가 더 쓰는 쪽')
    print(f'  원본 {auc(X):.3f} / {auc(X, sel):.3f} | Common 대체 {auc(Xc):.3f} / {auc(Xc, sel):.3f} | Expert 대체 {auc(Xe):.3f} / {auc(Xe, sel):.3f}')
    print(f'  입력 표준편차 평균: Expert 특징 {sd[:ED].mean():.3f} · Gate 점수 {sd[ED]:.3f} · Common {sd[SC:].mean():.3f}')

    base = _metrics(base_logit, y, typ, unseen, args.fpr)
    res = _grid_eval(H, X, y, typ, unseen, grid, grid, args.fpr)
    best1 = _pick(res, base)
    mults = [0.75, 0.87, 1.0, 1.15, 1.33]                      # 2회차: 1회차 최고 칸 주변만 세분(최대 2회 — 과적합 방지)
    a2 = sorted({round(best1[0] * m, 3) for m in mults}); b2 = sorted({round(best1[1] * m, 3) for m in mults})
    res.update(_grid_eval(H, X, y, typ, unseen, a2, b2, args.fpr))
    best = _pick(res, base)

    print(f'\n[1] 격자(알파=Common 배수 · 베타=Expert 배수) — "처음 본 방식 균형 정확도" 기준 − 기준칸(1,1) 차이(%p)   기준 {base["bal_unseen"]:.3f}')
    print('       beta→ ' + ''.join(f'{b:>7g}' for b in grid))
    for a in grid:
        print(f'  alpha {a:>5g} ' + ''.join(f'{(res[(a, b)]["bal_unseen"] - base["bal_unseen"]) * 100:>+7.1f}' for b in grid))
    print('  (같은 칸의 실제 정확도)')
    for a in grid:
        print(f'  alpha {a:>5g} ' + ''.join(f'{res[(a, b)]["real_acc"]:>7.3f}' for b in grid))

    print('\n[2] 후보 상위 8칸 (제약 통과한 칸만)   칸 | 처음본 균형 | 전체 균형 | 실제 | 처음본 AUROC | 처음본 balf(오탐 고정)')
    ok_cells = [(k, m) for k, m in res.items() if all(_checks(m, base).values())]
    for (a, b), m in sorted(ok_cells, key=lambda kv: -kv[1]['bal_unseen'])[:8]:
        print(f'  α={a:<6g} β={b:<6g} | {m["bal_unseen"]:.3f} ({(m["bal_unseen"] - base["bal_unseen"]) * 100:+.1f}) | {m["bal_all"]:.3f} ({(m["bal_all"] - base["bal_all"]) * 100:+.1f}) | '
              f'{m["real_acc"]:.3f} | {m["auroc_unseen"]:.3f} | {m["balf_unseen"]:.3f} ({(m["balf_unseen"] - base["balf_unseen"]) * 100:+.1f})')

    fold = np.array([int(hashlib.md5(str(g).encode('utf-8')).hexdigest(), 16) % 2 for g in grp])
    cf = []
    for f in (0, 1):                                              # 한 조각에서 고르고 다른 조각에서 확인 (포기 단위로 나눠 같은 포기가 양쪽에 안 걸침)
        tr, te = fold == f, fold != f
        r_tr = _grid_eval(H, X[tr], y[tr], typ[tr], unseen[tr], grid, grid, args.fpr)
        pk = _pick(r_tr, r_tr[(1.0, 1.0)])
        m_te = _metrics(head_logit(H, X[te], *pk).numpy(), y[te], typ[te], unseen[te], args.fpr)
        b_te = _metrics(head_logit(H, X[te]).numpy(), y[te], typ[te], unseen[te], args.fpr)
        cf.append((f, pk, m_te['bal_unseen'] - b_te['bal_unseen'], m_te['bal_all'] - b_te['bal_all']))
    print('\n[3] 교차 검증 (한 조각에서 고른 칸을 다른 조각에서 시험)')
    for f, pk, g_un, g_all in cf:
        print(f'  조각{f}에서 고름 α={pk[0]:g} β={pk[1]:g} → 반대 조각 처음본 균형 {g_un * 100:+.1f}%p · 전체 균형 {g_all * 100:+.1f}%p')

    mb = res[best]
    print(f'\n[4] 선택된 칸 α={best[0]:g}(Common) β={best[1]:g}(Expert)  ←→ 기준 (1,1)')
    print(f'  {"유형":<14}{"전체 recall 기준→선택":>24}{"처음본 recall 기준→선택":>26}')
    for t in sorted(base['rec_all']):
        un = f'{base["rec_unseen"][t]:.3f} → {mb["rec_unseen"][t]:.3f}' if t in base['rec_unseen'] else '-'
        print(f'  {t:<14}{base["rec_all"][t]:>10.3f} → {mb["rec_all"][t]:<8.3f}{un:>26}')

    near = lambda p, q: max(abs(np.log(p[0] / q[0])), abs(np.log(p[1] / q[1]))) <= np.log(1.5) + 1e-9
    v = {'① 처음 본 방식 균형 +1.5%p 이상': mb['bal_unseen'] - base['bal_unseen'] >= 0.015,
         '② 전체 균형 −0.5%p 이내': mb['bal_all'] - base['bal_all'] >= -0.005,
         **_checks(mb, base),
         '⑤ 교차 검증 양쪽에서 개선(+) 이고 고른 칸이 한 칸 이내': all(g > 0 for _, _, g, _ in cf) and all(near(pk, best) for _, pk, _, _ in cf)}
    print('\n[5] 채택 기준(미리 정함) — 전부 만족해야 채택')
    for k, ok in v.items():
        print(f'  {"PASS" if ok else "FAIL"}  {k}')
    thr_only = (mb['bal_unseen'] - base['bal_unseen'] >= 0.015) and (mb['balf_unseen'] - base['balf_unseen'] < 0.005)
    if thr_only:
        print('  ⚠️ 이득이 문턱 이동뿐으로 보임(오탐을 고정하면 처음 본 방식 recall이 안 오름) → 비중 조정이 아니라 판정 문턱 조정임')
    adopt = all(v.values()) and not thr_only and best != (1.0, 1.0)
    print('\n결론: ' + (f'채택 → --bake_to로 구움 (--alpha {best[0]:g} --beta {best[1]:g})' if adopt else '기준 미달(또는 기준 칸이 최선) → v3 그대로'))


def bake(args):
    """alpha·beta를 head.0.weight의 열에 곱해 새 weights.safetensors로 저장한다. 나머지 텐서는 바이트 그대로(백본·Expert·Gate·Common 불변)."""
    import torch
    from safetensors.torch import load_file, save_file
    cfg = json.loads((Path(args.weights) / 'config.json').read_text(encoding='utf-8'))
    sd = load_file(str(Path(args.weights) / cfg['weights']))
    W = sd['head.0.weight']
    assert W.dtype == torch.float32 and W.shape[1] > SC, (W.dtype, tuple(W.shape))      # Common 열 수는 1280(기본) 말고 스펙트럼을 붙인 1312(→ 입력 1377)도 가능
    s = torch.ones(W.shape[1]); s[:SC] = args.beta; s[SC:] = args.alpha
    out = dict(sd); out['head.0.weight'] = (W * s).contiguous()
    changed = {'head.0.weight'}
    if args.shift:          # 판정 기준 이동: 마지막 층 bias에 shift를 더하면 logit이 shift만큼 커진다 → sigmoid(logit+shift) ≥ 0.5 ⇔ 원래 p ≥ 1/(1+e^shift). 순서가 안 바뀌어 AUROC는 같다
        assert sd['head.3.bias'].dtype == torch.float32 and sd['head.3.bias'].numel() == 1
        out['head.3.bias'] = (sd['head.3.bias'] + args.shift).contiguous(); changed.add('head.3.bias')
    save_file(out, args.bake_to)
    chk = load_file(args.bake_to)
    same = set(chk) == set(sd) and all(torch.equal(chk[k], sd[k]) for k in sd if k not in changed)
    print(f'굽기 완료 → {args.bake_to} | Expert 열(0~64) ×{args.beta:g} · Common 열(65~) ×{args.alpha:g} · logit {args.shift:+g} | {sorted(changed)} 외 {len(sd) - len(changed)}개 텐서 동일: {same}')
    assert same
    if args.check:
        Ho = load_head(args.weights)
        Hb = {k: chk[k].float() for k in Ho}
        X = torch.from_numpy(np.load(args.check)['feat'].astype(np.float32))
        diff = (head_logit(Ho, X, args.alpha, args.beta) + args.shift - head_logit(Hb, X)).abs().max().item()
        print(f'동일성: "스케일한 입력 + 원래 head + shift"와 "원래 입력 + 구운 head"의 logit 최대 차이 {diff:.2e} (1e-4 미만이면 정상)')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--data'); ap.add_argument('--weights', required=True); ap.add_argument('--out', default='usage_feat.npz')
    ap.add_argument('--n_real', type=int, default=150); ap.add_argument('--per_gen', type=int, default=8)
    ap.add_argument('--batch', type=int, default=4); ap.add_argument('--workers', type=int, default=0)
    ap.add_argument('--local_cache', default='', help='Colab: Drive 데이터를 이 폴더로 복사해 읽는다 (train.py와 같은 옵션)')
    ap.add_argument('--analyze', help='저장된 npz만 다시 분석 (GPU 필요 없음)')
    ap.add_argument('--reweight', help='저장된 npz로 Common·Expert 입력 비중(alpha·beta) 격자 탐색 (GPU 필요 없음)')
    ap.add_argument('--grid', default='0.25,0.35,0.5,0.7,1,1.4,2,2.8,4', help='alpha·beta 후보(쉼표). 1이 꼭 있어야 함')
    ap.add_argument('--holdout', default=HOLD_DEFAULT, help='"처음 본 방식" generator 앞부분(쉼표) — 학습에서 뺀 방식과 같게')
    ap.add_argument('--fpr', type=float, default=0.01, help='문턱을 맞출 실제 오탐률')
    ap.add_argument('--bake_to', default='', help='alpha·beta를 head.0.weight 열에 곱해 새 weights.safetensors로 저장')
    ap.add_argument('--alpha', type=float, default=1.0, help='Common 열 배수'); ap.add_argument('--beta', type=float, default=1.0, help='Expert 열(가중합 64 + Gate 점수 1) 배수')
    ap.add_argument('--shift', type=float, default=0.0, help='--bake_to와 함께: 마지막 층 bias에 더할 값(판정 기준 이동). p>=t를 가짜로 하려면 ln((1-t)/t), t=0.05면 2.9444')
    ap.add_argument('--check', default='', help='--bake_to와 함께: npz로 굽기 동일성 확인')
    a = ap.parse_args()
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')       # Windows 콘솔(cp949)에서 —·≥ 같은 기호가 에러 나지 않게 (diag.py와 같음)
    if a.analyze:
        analyze(a)
    elif a.reweight:
        reweight(a)
    elif a.bake_to:
        bake(a)
    else:
        assert a.data, '--data 필요'
        dump(a)
