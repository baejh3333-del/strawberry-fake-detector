"""판별 모델 학습 스크립트 — 1~3단계 (docs/전략.md 12-3, docs/아키텍처.md 5절).

백본(DINOv3)은 끝까지 얼려 두고(학습 안 함), 그 위의 머리(Expert 5[`--family_map type8`이면 8]·Gate·Common CNN·최종 MLP)만 학습한다.

  1단계  Expert 5(또는 8) + Common(CNN)   각 Expert = '실제 + 자기 계열 가짜'만 보고 진짜/가짜 학습 (BCE)
                                  Common = 전부로 학습 (임시 선형 머리 1280→1, 3단계에서 버림)
  2단계  Gate만                   '이 사진은 어느 계열 흔적인가' 맞히기 (CE). 실제 사진은 균등 분배(uniform)가 목표
  3단계  전체, 낮은 학습률         BCE(최종) + λ·CE(Gate) + μ·SupCon(대조 학습)

배치는 '실제:가짜 = 1:1, 가짜 안에서는 계열 균등'으로 뽑는다(결정 ⑪). 3단계는 같은 원본의 실제·가짜 쌍을 한 배치에 넣는다.

⚠️ 학습 증강은 좌우 뒤집기뿐이다 — 크기 조정·재압축·블러는 ③ 필터 흔적과 겹쳐서 쓰지 않는다(전략 9절 S4).

사용법 (Colab, 저장소 루트에서):
  python train/train.py --data /content/drive/MyDrive/work/out/C --local_cache /content/data_C \\
      --pretrained /content/drive/MyDrive/work/pretrained --out /content/drive/MyDrive/work/runs/run1 --stage all
시험(작은 모델, 몇 걸음만 — 코드가 도는지만 본다):
  python train/train.py --data <DATA_DIR> --out <폴더> --stage all --smoke
Expert 배정을 대회 유형 기준으로 바꿔 학습(exp09) — 목록표 파일은 그대로 두고 읽을 때 family를 다시 정한다:
  python train/train.py ... --family_map type8        (Expert 8개. 내보내는 config.json의 n_experts도 8로 쓴다)
생성 방식마다 Expert 하나(분석용, exp10) — 1단계만 돌려 "대회 가짜가 어느 방식 지문과 닮았나"를 본다:
  python train/train.py ... --family_map gen24 --stage 1 --steps1 4000
"""
import argparse, copy, json, math, os, random, shutil, sys, time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'submission'))
from detector import Detector, load_pretrained_backbone, load_pretrained_common, save_weights   # noqa: E402

FAM = {f'E{k}': k - 1 for k in range(1, 33)}              # Gate 정답 (실제 사진은 -1). E6~E8은 --family_map type8, E9~E24는 gen24에서만 나온다

# --family_map: 목록표(manifest)의 family(어느 Expert가 맡나)를 '읽을 때' 다시 정한다. 목록표 파일은 안 고친다. 쓴 값은 run_config.json에 남는다.
#   none  = 목록표 그대로 (exp02~exp08, 기본). Expert 5개.
#   type8 = 대회 유형 기준 (exp09). ① 전체 생성(AI_gen·vae_recon)은 생성기 계열 E1~E3을 그대로 둔다(대회 생성기를 모르니 다양성 유지).
#           나머지는 type마다 Expert를 따로 둔다: ④ full_style E4 · ⑤ partial_gen E5 · ⑥ partial_style E6 · ② composite E7 · ③ filter E8. Expert 8개.
#   gen24 = 생성 방식(모델·도구)마다 Expert 하나 (exp10, 분석용 — "대회 가짜가 어느 방식 지문과 닮았나"를 보려고). 아래 GEN_METHODS 순서가 E1~E24.
FAMILY_MAPS = {
    'none': None,
    'type8': {'full_style': 'E4', 'partial_gen': 'E5', 'partial_style': 'E6', 'composite': 'E7', 'filter': 'E8'},
    'gen24': 'gen',
}
# gen24: generator 이름 앞부분 → 방식. 위에서부터 처음 맞는 것. (v6 같은 프롬프트 변형은 같은 방식으로 묶는다)
GEN_METHODS = [
    ('realvisxl', 'realvisxl'), ('flux_schnell', 'flux_schnell'), ('zimage_turbo', 'zimage_turbo'), ('infinity_2b', 'infinity'), ('flux_vae', 'flux_vae'),
    ('composite-bgswap', 'bgswap'), ('composite-objpaste', 'objpaste'), ('filter-', 'filter'),
    ('sdxl-img2img-', 'sdxl_img2img'), ('lama-erase', 'lama'), ('sdxl-inpaint-', 'sdxl_inpaint'), ('sd15-inpaint-', 'sd15_inpaint'), ('clone-', 'clone'),
    ('sdxl-partial-', 'sdxl_partial'), ('local-filter-', 'local_filter'),
    ('flux2-klein-4b', 'klein_t2i'), ('flux2-klein-style-', 'klein_style'), ('classic-', 'classic_style'), ('sdxl-paint-', 'sdxl_paint'),
    ('flux2-klein-edit-', 'klein_edit'), ('kandinsky-inpaint-', 'kandinsky_inpaint'),
    ('partial-flux2-klein-style-', 'partial_klein'), ('partial-sdxl-paint-', 'partial_sdxl_paint'), ('partial-classic-', 'partial_classic'),
]
N_EXPERTS = {'none': 5, 'type8': 8, 'gen24': len(GEN_METHODS)}       # 내보내는 config.json의 n_experts


# ───────────────────────── 데이터 ─────────────────────────

def method_index(generator):
    for i, (prefix, _) in enumerate(GEN_METHODS):
        if generator.startswith(prefix):
            return i
    return -1


def apply_family_map(rows, family_map='none'):
    """가짜의 type(type8) 또는 generator(gen24)를 보고 family를 다시 정한다. type8에서 표에 없는 type(AI_gen·vae_recon)은 목록표의 family를 그대로 둔다. 실제 사진은 안 건드린다."""
    fmap = FAMILY_MAPS[family_map]
    for r in rows:
        if fmap and r['y'] == 1:
            if fmap == 'gen':
                i = method_index(r.get('generator', ''))
                r['family'] = f'E{i + 1}' if i >= 0 else 'unknown'
            else:
                r['family'] = fmap.get(r.get('type', ''), r['family'])
        r['fam'] = FAM.get(r['family'], -1)
    if fmap:        # 가짜인데 Expert를 못 정한 사진이 있으면 조용히 실제처럼 취급되므로 여기서 멈춘다
        bad = Counter((r.get('type', ''), r.get('generator', ''), r['family']) for r in rows if r['y'] == 1 and r['fam'] < 0)
        assert not bad, f'--family_map {family_map}: Expert를 못 정한 가짜가 있음 (type, generator, family): {dict(bad)}'
    return rows


def read_manifest(path, family_map='none'):
    import csv
    with open(path, encoding='utf-8', newline='') as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r['y'] = int(r['label'])
    return apply_family_map(rows, family_map)


class ImageSet(torch.utils.data.Dataset):
    """사진을 uint8 텐서 [3,H,W]로 읽는다(정규화는 GPU에서). 학습에서만 좌우 뒤집기."""

    def __init__(self, root, rows, train, flip=True):
        self.root, self.rows, self.train, self.flip = root, rows, train, flip

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        im = Image.open(os.path.join(self.root, self.rows[i]['file'])).convert('RGB')
        x = torch.from_numpy(np.asarray(im).copy()).permute(2, 0, 1)
        if self.train and self.flip and random.random() < 0.5:
            x = x.flip(-1)
        return x, i


def collate(items):
    """가로·세로 사진은 한 텐서로 못 쌓으므로 모양별로 묶는다 (둘 다 조각 12장이라 토큰 수는 같다)."""
    xs, idx = zip(*items)
    groups = defaultdict(list)
    for pos, x in enumerate(xs):
        groups[tuple(x.shape)].append(pos)
    return {'groups': [(p, torch.stack([xs[i] for i in p])) for p in groups.values()], 'idx': list(idx)}


def worker_init(wid):
    s = torch.initial_seed() % 2 ** 32
    random.seed(s); np.random.seed(s)


def make_loader(root, rows, batches, train, workers, flip=True):
    ds = ImageSet(root, rows, train, flip)
    return torch.utils.data.DataLoader(ds, batch_sampler=batches, num_workers=workers, collate_fn=collate, worker_init_fn=worker_init,
                                       pin_memory=torch.cuda.is_available(), persistent_workers=False, prefetch_factor=2 if workers else None)


NEW_REAL_BOOST = 1.0      # --new_real_boost: 새로 합친 자료(group이 'n:'로 시작)의 실제 사진을 뽑을 확률 배수. 실제 전체 몫(1/2)은 그대로 두고 그 안에서만 비중을 바꾼다


def sample_weights(rows):
    """실제:가짜 = 1:1, 가짜 안에서는 계열 균등, 계열 안에서는 사진마다 균등. (--new_real_boost가 1보다 크면 새 자료의 실제 사진을 그만큼 더 자주 뽑는다)"""
    boost = np.array([NEW_REAL_BOOST if (r['y'] == 0 and str(r.get('group', '')).startswith('n:')) else 1.0 for r in rows])
    real_total = sum(boost[i] for i, r in enumerate(rows) if r['y'] == 0)
    fam = Counter(r['fam'] for r in rows if r['y'] == 1)
    return np.array([0.5 * boost[i] / real_total if r['y'] == 0 else 0.5 / (len(fam) * fam[r['fam']]) for i, r in enumerate(rows)])


def weighted_batches(rows, n_batches, bs, rng):
    p = sample_weights(rows); p = p / p.sum()
    idx = rng.choice(len(rows), size=n_batches * bs, replace=True, p=p)
    return [idx[i * bs:(i + 1) * bs].tolist() for i in range(n_batches)]


def pair_batches(rows, n_batches, bs, n_pairs, rng):
    """3단계용: 같은 원본(source_id)의 실제·가짜 쌍 n_pairs개 + 나머지는 가중 샘플링. SupCon이 '내용은 같고 흔적만 다른' 쌍을 본다."""
    real_of = {r['source_id']: i for i, r in enumerate(rows) if r['y'] == 0 and r['source_id']}
    fakes = defaultdict(list)
    for i, r in enumerate(rows):
        if r['y'] == 1 and r['source_id'] in real_of:
            fakes[r['source_id']].append(i)
    srcs = sorted(fakes)
    if not srcs or n_pairs == 0:
        return weighted_batches(rows, n_batches, bs, rng)
    rest = weighted_batches(rows, n_batches, bs - 2 * n_pairs, np.random.RandomState(rng.randint(2 ** 31))) if bs > 2 * n_pairs else [[] for _ in range(n_batches)]
    out = []
    for b in range(n_batches):
        pick = rng.choice(len(srcs), size=n_pairs, replace=False)
        batch = []
        for k in pick:
            s = srcs[k]
            batch += [real_of[s], int(rng.choice(fakes[s]))]
        out.append(batch + rest[b])
    return out


# ───────────────────────── 모델 쪽 도우미 ─────────────────────────

def encode(model, batch, dev, mean, std, need_common=True):
    """토큰(백본, 기울기 없음)과 Common 특징을 만든다. 가로·세로 묶음을 합쳐 원래 순서로 되돌린다."""
    order, toks, coms = [], [], []
    for pos, x in batch['groups']:
        x = (x.to(dev, non_blocking=True).float().div_(255) - mean) / std
        with torch.no_grad():
            toks.append(model.get_tokens(x))
        if need_common:
            with torch.autocast(device_type=dev.type, enabled=False):    # Common CNN은 fp32 — 강한 노이즈 사진에서 fp16 값이 크게 어긋남(10/6). 제출 inference.py와 같게
                coms.append(model.common(x))
        order += pos
    inv = torch.argsort(torch.tensor(order, device=dev))
    tokens = torch.cat(toks)[inv]
    common = torch.cat(coms)[inv] if need_common else None
    return tokens, common


def bce_balanced(logit, y, mask):
    """마스크로 고른 사진에서 '실제 평균'과 '가짜 평균'을 절반씩 섞어 BCE (Expert는 실제가 가짜보다 훨씬 많이 걸리므로)."""
    if mask.sum() == 0:
        return logit.sum() * 0
    l = F.binary_cross_entropy_with_logits(logit[mask].float(), y[mask], reduction='none')
    ym = y[mask]
    terms = [l[ym == 0].mean()] if (ym == 0).any() else []
    terms += [l[ym == 1].mean()] if (ym == 1).any() else []
    return sum(terms) / len(terms)


def gate_ce(gate, fam, real_gate):
    """Gate 확률(합 1) vs 정답. 가짜 = 자기 계열 one-hot, 실제 = 균등 분배(uniform) 또는 무시(ignore)."""
    n = gate.shape[1]
    target = torch.full_like(gate, 1.0 / n)
    is_fake = fam >= 0
    target[is_fake] = F.one_hot(fam[is_fake], n).float()
    ce = -(target * torch.log(gate.float().clamp_min(1e-8))).sum(1)
    keep = torch.ones_like(is_fake) if real_gate == 'uniform' else is_fake
    return ce[keep].mean() if keep.any() else gate.sum() * 0


def supcon(z, y, tau=0.1):
    """지도 대조 학습(SupCon): 같은 라벨끼리 가깝게, 다른 라벨은 멀게. 같은 원본의 실제·가짜가 한 배치에 있어 '내용 대신 흔적'을 보게 된다."""
    with torch.autocast(device_type=z.device.type, enabled=False):      # 혼합 정밀도 안이어도 fp32로 계산(fp16이면 -1e9 마스킹이 넘친다)
        z = F.normalize(z.float(), dim=1)
        n = len(z)
        if n < 2:
            return z.sum() * 0
        sim = z @ z.t() / tau
        eye = torch.eye(n, dtype=torch.bool, device=z.device)
        sim = sim.masked_fill(eye, -1e9)
        logp = sim - torch.logsumexp(sim, dim=1, keepdim=True)
        pos = (y.unsqueeze(0) == y.unsqueeze(1)) & ~eye
        cnt = pos.sum(1)
        ok = cnt > 0
        if not ok.any():
            return z.sum() * 0
        return (-(logp * pos).sum(1)[ok] / cnt[ok]).mean()


def trainable_state(model, extra):
    sd = {k: v.detach().cpu() for k, v in model.state_dict().items() if not k.startswith('backbone.')}
    return {'model': sd, **{k: m.state_dict() for k, m in extra.items()}}


def load_trainable(model, extra, ck):
    own = model.state_dict()
    # 머리(head)는 3단계에서 처음 학습한다. head_hidden을 바꿔 비교할 때 1·2단계 결과를 재사용할 수 있도록, 크기가 다른 head만 건너뛴다
    skip = [k for k, v in ck['model'].items() if k.startswith('head.') and k in own and own[k].shape != v.shape]
    if skip:
        print(f'  크기가 다른 head 가중치는 불러오지 않음(새로 초기화): {skip}')
    sd = {k: v for k, v in ck['model'].items() if k not in skip}
    miss = model.load_state_dict(sd, strict=False)
    bad = [k for k in miss.missing_keys if not k.startswith(('backbone.', 'spec.')) and k not in skip] + list(miss.unexpected_keys)    # spec.* = --spectrum으로 새로 생긴 모듈(없던 체크포인트에서 이어받을 때 새로 초기화)
    assert not bad, f'체크포인트 키가 안 맞음: {bad[:5]}'
    for k, m in extra.items():
        if k in ck:
            try:
                m.load_state_dict(ck[k])
            except RuntimeError:        # 입력 크기가 바뀐 보조 머리(tmp_head·proj)는 1·2단계 결과를 재사용할 때 새로 초기화한다(3단계가 처음 학습)
                print(f'  {k}: 크기가 달라 불러오지 않음(새로 초기화)')


# ───────────────────────── 평가 ─────────────────────────

@torch.no_grad()
def evaluate(model, extra, args, rows, dev, mean, std, stage, tag, n=None):
    """검증 세트(또는 생성기별로 고르게 뽑은 일부)를 돌려 정확도·AUROC·생성기별 표를 만든다."""
    if n:
        by = defaultdict(list)
        for i, r in enumerate(rows):
            by[r['generator']].append(i)
        quota = max(5, n // len(by)); rg = random.Random(123)
        sel = sorted(i for g in by.values() for i in (rg.sample(g, quota) if len(g) > quota else g))
    else:
        sel = list(range(len(rows)))
    sub = [rows[i] for i in sel]
    bs = args.batch
    loader = make_loader(args.data_dir, sub, [list(range(i, min(i + bs, len(sub)))) for i in range(0, len(sub), bs)], False, args.workers)
    model.eval()
    P = defaultdict(list)
    for batch in loader:
        with torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=args.amp):
            tokens, common = encode(model, batch, dev, mean, std, need_common=(stage != 2))
            if stage == 1:
                o = model.combine(tokens, torch.zeros(len(tokens), model.common.num_features, device=dev))
                P['common'] += torch.sigmoid(extra['tmp_head'](common).squeeze(1).float()).tolist()
            elif stage == 2:
                o = model.combine(tokens, torch.zeros(len(tokens), model.common.num_features, device=dev))
            else:
                o = model.combine(tokens, common)
        P['expert'] += torch.sigmoid(o['expert_logits'].float()).tolist()
        P['gate'] += o['gate'].float().tolist()
        P['final'] += torch.sigmoid(o['logit'].float()).tolist()
    model.train(); model.backbone.eval()
    y = np.array([r['y'] for r in sub]); gen = np.array([r['generator'] for r in sub]); fam = np.array([r['fam'] for r in sub])
    res = {'tag': tag, 'stage': stage, 'n': len(sub)}
    gate, expert = np.array(P['gate']), np.array(P['expert'])
    if stage in (1, 3):
        p = np.array(P['common' if stage == 1 else 'final'])
        res['acc'] = float(((p >= 0.5) == (y == 1)).mean())
        try:
            from sklearn.metrics import roc_auc_score
            res['auroc'] = float(roc_auc_score(y, p))
        except Exception:
            res['auroc'] = None
        res['acc_by_generator'] = {g: float(((p[gen == g] >= 0.5) == (y[gen == g] == 1)).mean()) for g in sorted(set(gen))}
        res['acc_by_family'] = {f'E{f + 1}' if f >= 0 else 'real': float(((p[fam == f] >= 0.5) == (y[fam == f] == 1)).mean()) for f in sorted(set(fam))}
    if stage == 1:     # Expert k는 '실제 + 자기 계열'에서 몇 %를 맞히나
        res['expert_acc'] = {}
        for k in range(expert.shape[1]):
            m = (y == 0) | (fam == k)
            res['expert_acc'][f'E{k + 1}'] = float(((expert[m, k] >= 0.5) == (y[m] == 1)).mean()) if m.any() else None
    if stage in (2, 3):
        fk = fam >= 0
        res['gate_top1_acc'] = float((gate[fk].argmax(1) == fam[fk]).mean()) if fk.any() else None
        res['gate_mean_by_generator'] = {g: np.round(gate[gen == g].mean(0), 3).tolist() for g in sorted(set(gen))}
    return res


# ───────────────────────── 학습 ─────────────────────────

def lr_at(step, total, warm):
    if step < warm:
        return (step + 1) / warm
    return 0.5 * (1 + math.cos(math.pi * (step - warm) / max(1, total - warm)))


def run_stage(stage, model, extra, args, train_rows, val_rows, dev, mean, std, log):
    out = Path(args.out)
    final, last = out / f'stage{stage}.pt', out / f'stage{stage}_last.pt'
    steps = getattr(args, f'steps{stage}')
    if final.exists() and not args.force:
        print(f'[{stage}단계] 이미 끝남 → 체크포인트만 불러옴: {final}')
        load_trainable(model, extra, torch.load(final, map_location='cpu', weights_only=False)); return
    prev = out / f'stage{stage - 1}.pt'
    if stage > 1 and prev.exists():
        load_trainable(model, extra, torch.load(prev, map_location='cpu', weights_only=False)); print(f'[{stage}단계] {prev.name}에서 이어받음')

    # 이 단계에서 학습할 것만 켠다
    for p in model.parameters():
        p.requires_grad = False
    for m in extra.values():
        for p in m.parameters():
            p.requires_grad = False
    groups, mods = [], {}
    if stage == 1:
        mods = {'experts': model.experts, 'common': model.common, 'tmp_head': extra['tmp_head']}
        lrs = {'experts': args.lr_head, 'common': args.lr_cnn, 'tmp_head': args.lr_head}
    elif stage == 2:
        mods = {'gate': model.gate}; lrs = {'gate': args.lr_head}
    else:
        mods = {'gate': model.gate, 'experts': model.experts, 'common': model.common, 'head': model.head, 'proj': extra['proj']}
        lrs = {'gate': args.lr3_head, 'experts': args.lr3_head, 'head': args.lr3_head, 'proj': args.lr3_head, 'common': args.lr3_cnn}
        if len(model.fusion_norm):                  # --fusion_norm일 때만 있음 (LayerNorm 두 개 — head와 같은 학습률)
            mods['fusion_norm'] = model.fusion_norm; lrs['fusion_norm'] = args.lr3_head
    if hasattr(model, 'spec') and stage != 2:      # --spectrum: 스펙트럼 MLP는 1·3단계에서 head와 같은 학습률로 학습
        mods['spec'] = model.spec; lrs['spec'] = args.lr_head if stage == 1 else args.lr3_head
    for k, m in mods.items():
        ps = list(m.parameters())
        for p in ps:
            p.requires_grad = True
        groups.append({'params': ps, 'lr': lrs[k], 'base': lrs[k], 'weight_decay': args.wd})
    model.train(); model.backbone.eval()
    model.mod_drop = args.mod_drop if stage == 3 else 0.0      # 3단계에서만 (1·2단계는 combine에 0 벡터를 넣으므로 쓰면 안 됨)
    if stage == 2:
        model.common.eval(); model.experts.eval()
    opt = torch.optim.AdamW(groups)
    scaler = torch.amp.GradScaler('cuda', enabled=args.amp)

    start = 0
    if last.exists() and not args.force:
        ck = torch.load(last, map_location='cpu', weights_only=False)
        load_trainable(model, extra, ck); opt.load_state_dict(ck['opt']); scaler.load_state_dict(ck['scaler']); start = ck['step']
        print(f'[{stage}단계] 중간 저장({start}걸음)에서 이어서 시작')

    rng = np.random.RandomState(args.seed * 100 + stage)
    batches = pair_batches(train_rows, steps, args.batch, args.pairs, rng) if stage == 3 else weighted_batches(train_rows, steps, args.batch, rng)
    loader = make_loader(args.data_dir, train_rows, batches[start:], True, args.workers, flip=not args.no_flip)
    y_all = torch.tensor([r['y'] for r in train_rows], dtype=torch.float32); f_all = torch.tensor([r['fam'] for r in train_rows])
    n_exp = len(model.experts)
    t0, acc = time.time(), defaultdict(float)
    print(f'[{stage}단계] {steps}걸음 시작 (배치 {args.batch}, 학습 파라미터 {sum(p.numel() for g in groups for p in g["params"]) / 1e6:.2f}M)')

    for step, batch in enumerate(loader, start=start):
        for g in opt.param_groups:
            g['lr'] = g['base'] * lr_at(step, steps, args.warmup)
        y, fam = y_all[batch['idx']].to(dev), f_all[batch['idx']].to(dev)
        with torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=args.amp):
            tokens, common = encode(model, batch, dev, mean, std, need_common=(stage != 2))   # CNN도 autocast 안에서(T4 속도·메모리)
            if stage == 1:
                o = model.combine(tokens, torch.zeros(len(tokens), model.common.num_features, device=dev))
                l_exp = sum(bce_balanced(o['expert_logits'][:, k], y, (y == 0) | (fam == k)) for k in range(n_exp))
                l_com = F.binary_cross_entropy_with_logits(extra['tmp_head'](common).squeeze(1).float(), y)
                loss = l_exp + l_com; parts = {'experts': l_exp, 'common': l_com}
            elif stage == 2:
                o = model.combine(tokens, torch.zeros(len(tokens), model.common.num_features, device=dev))
                loss = gate_ce(o['gate'], fam, args.real_gate); parts = {'gate': loss}
            else:
                o = model.combine(tokens, common)
                l_bce = F.binary_cross_entropy_with_logits(o['logit'].float(), y)
                l_gate = gate_ce(o['gate'], fam, args.real_gate)
                l_sc = supcon(extra['proj'](o['feat']), y.long()) if args.mu > 0 else o['logit'].sum() * 0
                loss = l_bce + args.lam * l_gate + args.mu * l_sc; parts = {'bce': l_bce, 'gate': l_gate, 'supcon': l_sc}
                if args.w_expert > 0:
                    l_aux = sum(bce_balanced(o['expert_logits'][:, k], y, (y == 0) | (fam == k)) for k in range(n_exp)) / n_exp
                    loss = loss + args.w_expert * l_aux; parts['aux'] = l_aux
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_([p for g in groups for p in g['params']], 1.0)
        scaler.step(opt); scaler.update()
        for k, v in parts.items():
            acc[k] += float(v.detach())
        done = step + 1
        if done % args.log_every == 0:
            msg = {k: round(v / args.log_every, 4) for k, v in acc.items()}; acc.clear()
            msg.update(stage=stage, step=done, lr=round(opt.param_groups[0]['lr'], 6), sec_per_step=round((time.time() - t0) / (done - start), 2))
            print(msg, flush=True); log.write(json.dumps(msg, ensure_ascii=False) + '\n'); log.flush()
        if args.eval_every and done % args.eval_every == 0 and done < steps:
            r = evaluate(model, extra, args, val_rows, dev, mean, std, stage, f'stage{stage}_step{done}', n=args.val_n)
            print('  검증(일부)', {k: v for k, v in r.items() if k in ('acc', 'auroc', 'expert_acc', 'gate_top1_acc')}, flush=True); log.write(json.dumps(r, ensure_ascii=False) + '\n')
        if done % args.ckpt_every == 0 and done < steps:
            torch.save({**trainable_state(model, extra), 'opt': opt.state_dict(), 'scaler': scaler.state_dict(), 'step': done}, last)

    r = evaluate(model, extra, args, val_rows, dev, mean, std, stage, f'stage{stage}_final', n=args.val_max or None)   # 단계 끝: 검증 전체(기본)
    print(f'[{stage}단계] 끝 — 검증 전체', json.dumps({k: v for k, v in r.items() if k not in ('gate_mean_by_generator',)}, ensure_ascii=False), flush=True)
    (out / f'metrics_stage{stage}.json').write_text(json.dumps(r, ensure_ascii=False, indent=2), encoding='utf-8')
    torch.save(trainable_state(model, extra), final)
    if last.exists():
        last.unlink()


def ensure_local(src, dst, threads=32):
    """Drive(느림)에 있는 데이터를 Colab 로컬 디스크로 한 번 복사한다. 이미 있는 파일은 건너뜀(이어서 하기).
    Drive는 파일 하나를 읽는 데 약 1초가 걸려서(10/2 실측) 한 장씩 복사하면 8900장에 2시간이다 → 32개 동시에 읽으면 5분이다.
    내용만 복사하고(copy2 아님) 파일 크기를 Drive에 다시 묻지 않는다(그 질문도 1초씩 걸린다)."""
    from concurrent.futures import ThreadPoolExecutor
    rows = read_manifest(os.path.join(src, 'manifest.csv'))
    os.makedirs(dst, exist_ok=True)
    shutil.copyfile(os.path.join(src, 'manifest.csv'), os.path.join(dst, 'manifest.csv'))
    todo = [r['file'] for r in rows if not os.path.exists(os.path.join(dst, r['file']))]
    for d in {os.path.dirname(os.path.join(dst, f)) for f in todo}:
        os.makedirs(d, exist_ok=True)

    def one(f):
        b = os.path.join(dst, f)
        shutil.copyfile(os.path.join(src, f), b + '.part')      # 다 받은 파일만 이름을 바꾼다 → 중간에 끊겨도 반쪽 파일이 남지 않음
        os.replace(b + '.part', b)

    t0 = time.time()
    with ThreadPoolExecutor(threads) as ex:
        for i, _ in enumerate(ex.map(one, todo), 1):
            if i % 1000 == 0:
                print(f'  복사 {i}/{len(todo)} ({time.time() - t0:.0f}초)', flush=True)
    print(f'로컬 복사 완료: 새로 {len(todo)}장 / 전체 {len(rows)}장 → {dst} ({time.time() - t0:.0f}초)')
    return dst


def main():
    global NEW_REAL_BOOST
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True, help='manifest.csv와 사진이 있는 폴더 (out/C)')
    ap.add_argument('--local_cache', default='', help='주면 데이터를 이 로컬 폴더로 복사해서 읽는다(Colab Drive가 느릴 때)')
    ap.add_argument('--out', required=True)
    ap.add_argument('--pretrained', default='', help='dinov3_...safetensors와 efficientnet_...safetensors가 있는 폴더')
    ap.add_argument('--config', default=str(ROOT / 'submission' / 'model' / 'config.json'))
    ap.add_argument('--stage', default='all', help='1, 2, 3, 또는 all')
    ap.add_argument('--steps1', type=int, default=1500); ap.add_argument('--steps2', type=int, default=400); ap.add_argument('--steps3', type=int, default=1500)
    ap.add_argument('--batch', type=int, default=8)
    ap.add_argument('--lr_head', type=float, default=1e-3, help='1·2단계: Expert·Gate 학습률')
    ap.add_argument('--lr_cnn', type=float, default=1e-4, help='1단계: Common CNN 학습률(사전학습이라 작게)')
    ap.add_argument('--lr3_head', type=float, default=2e-4, help='3단계 머리 학습률(낮게)')
    ap.add_argument('--lr3_cnn', type=float, default=3e-5)
    ap.add_argument('--wd', type=float, default=0.01); ap.add_argument('--warmup', type=int, default=100)
    ap.add_argument('--lam', type=float, default=0.2, help='3단계 Gate CE 가중치 λ (0.1~0.3)')
    ap.add_argument('--mu', type=float, default=0.1, help='3단계 SupCon 가중치 μ')
    ap.add_argument('--pairs', type=int, default=2, help='3단계: 한 배치에 넣을 같은 원본 실제·가짜 쌍 수')
    ap.add_argument('--w_expert', type=float, default=0.0, help='3단계에서 Expert 개별 BCE를 계속 줄지(기본 끔)')
    ap.add_argument('--real_gate', default='uniform', choices=['uniform', 'ignore'], help='실제 사진의 Gate 목표')
    ap.add_argument('--no_flip', action='store_true')
    ap.add_argument('--seed', type=int, default=0); ap.add_argument('--workers', type=int, default=2)
    ap.add_argument('--val_n', type=int, default=400, help='중간 검증에 쓸 사진 수(생성기별 균등)')
    ap.add_argument('--val_max', type=int, default=0, help='단계 끝 검증에 쓸 사진 수 (0 = 검증 전체)')
    ap.add_argument('--eval_every', type=int, default=500); ap.add_argument('--ckpt_every', type=int, default=250); ap.add_argument('--log_every', type=int, default=25)
    ap.add_argument('--force', action='store_true', help='끝난 단계도 다시 한다')
    ap.add_argument('--device', default='auto')
    ap.add_argument('--smoke', action='store_true', help='작은 백본·사전학습 없음 — 코드가 도는지만 확인')
    ap.add_argument('--fusion_norm', action='store_true', help='합치기 전에 Expert·Common 벡터에 각각 LayerNorm (exp08). 내보내는 config.json에도 fusion_norm: true가 들어간다')
    ap.add_argument('--common_topk', type=float, default=0.0, help='Common CNN 풀링을 평균 대신 top-k(채널별 상위 이 비율 위치의 평균)와 섞기. 0=끔(기본), 예: 0.1. 부분 편집처럼 작은 영역의 반응이 평균에 묻히는 문제 대응. config.json의 common.pool에도 기록된다')
    ap.add_argument('--common_topk_mix', type=float, default=0.5, help='--common_topk일 때 top-k 비중(0=평균만, 1=top-k만)')
    ap.add_argument('--mod_drop', type=float, default=0.0, help='3단계에서 최종 MLP 입력의 Expert 블록 / Common 블록을 사진마다 이 확률씩 0으로 만든다(한 사진에서 둘이 같이 꺼지지는 않음). 0=끔(기본), 예: 0.15. 학습용 장치라 config·추론은 그대로')
    ap.add_argument('--spectrum', type=int, default=0, help='사진 한 장의 흑백 전역 스펙트럼 요약(FFT, 48개 숫자 → MLP)을 이 차원의 벡터로 만들어 Common 출력 뒤에 붙인다(최종 MLP 입력이 그만큼 늘어남). 0=끔(기본), 예: 32. config.json의 spectrum에도 기록된다')
    ap.add_argument('--grad_ckpt', action='store_true', help='Common CNN에 gradient checkpointing — 활성값을 저장하지 않고 역전파 때 다시 계산해 메모리를 아낀다(T4 15GB에서 batch 8이 안 들어갈 때). 결과는 같고(BN 이동평균만 두 번 갱신) 느려진다')
    ap.add_argument('--common_highpass', type=float, default=0.0, help='Common CNN 입력에 고주파 잔차(RGB − 3×3 평균)를 gain배 해 3채널 더 붙임(입력 6채널). 0=끔(기본), 예: 8. 블러·노이즈·인페인팅 경계 같은 미세 흔적 대응. config.json의 common.highpass에도 기록된다')
    ap.add_argument('--holdout', default='', help='leave-one-generator-out: 쉼표로 구분한 generator 이름 앞부분(예: sdxl-img2img). 이 가짜를 학습에서 통째로 뺀다 (검증 표에는 그대로 남아 처음 보는 생성기 점수가 나온다)')
    ap.add_argument('--family_map', default='none', choices=list(FAMILY_MAPS), help='가짜를 어느 Expert가 맡을지 정하는 규칙(목록표는 안 고치고 읽을 때 다시 정함). none=목록표 그대로 Expert 5개 / type8=대회 유형 기준 Expert 8개 (exp09) / gen24=생성 방식마다 Expert 24개 (exp10, 분석용). 내보내는 config.json의 n_experts도 맞춰 쓴다')
    ap.add_argument('--new_real_boost', type=float, default=1.0, help="추가 자료로 합친 새 자료(group이 'n:'로 시작)의 실제 사진을 뽑을 확률 배수 (기본 1 = 기존과 같음). 새 실제 사진만 늘려 이어서 학습할 때 4 안팎")
    args = ap.parse_args()
    NEW_REAL_BOOST = args.new_real_boost

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    dev = torch.device('cuda' if (args.device == 'auto' and torch.cuda.is_available()) else ('cpu' if args.device == 'auto' else args.device))
    args.amp = dev.type == 'cuda'
    os.makedirs(args.out, exist_ok=True)
    cfg = json.loads(Path(args.config).read_text(encoding='utf-8'))
    if args.fusion_norm:
        cfg['fusion_norm'] = True
    if args.common_topk > 0:
        cfg['common'].update(pool='topk', frac=args.common_topk, mix=args.common_topk_mix)
    if args.common_highpass > 0:
        cfg['common'].update(highpass={'gain': args.common_highpass})
    if args.spectrum > 0:
        cfg['spectrum'] = {'dim': args.spectrum}
    if args.family_map != 'none':      # Expert 수가 config와 다르면 Gate 출력·Expert 머리 개수가 안 맞으므로 여기서 맞춘다 (내보낼 때도 같게 씀)
        cfg['n_experts'] = N_EXPERTS[args.family_map]
    if args.smoke:
        cfg['backbone'].update(hidden_size=64, intermediate_size=128, num_hidden_layers=2, num_attention_heads=4)
        cfg['pool_dim'] = 32
    args.data_dir = ensure_local(args.data, args.local_cache) if args.local_cache else args.data
    rows = read_manifest(os.path.join(args.data_dir, 'manifest.csv'), args.family_map)
    train_rows = [r for r in rows if r['split'] == 'train']; val_rows = [r for r in rows if r['split'] == 'val']
    if args.holdout:
        prefixes = tuple(p.strip() for p in args.holdout.split(',') if p.strip())
        out_gen = [r for r in train_rows if r['y'] == 1 and r['generator'].startswith(prefixes)]
        assert out_gen, f'--holdout {args.holdout}에 맞는 학습용 가짜가 없음 (generator 이름 확인)'
        train_rows = [r for r in train_rows if not (r['y'] == 1 and r['generator'].startswith(prefixes))]    # 실제 사진은 그대로, 해당 가짜만 제거
        print(f'holdout {prefixes}: 학습에서 가짜 {len(out_gen)}장 제거 {dict(Counter(r["generator"] for r in out_gen))}')
    print('장치', dev, '/ 학습', len(train_rows), '장 (실제', sum(r['y'] == 0 for r in train_rows), '/ 가짜', sum(r['y'] == 1 for r in train_rows), ') / 검증', len(val_rows), '장')
    print(f'family_map={args.family_map} (Expert {cfg["n_experts"]}개) / 학습 계열별 가짜:', dict(sorted(Counter(r['family'] for r in train_rows if r['y'] == 1).items())))
    if args.family_map == 'gen24':      # 분석용: Expert 번호가 어느 방식인지 남겨 둔다 (표를 읽을 때 필요)
        names = [m for _, m in GEN_METHODS]
        (Path(args.out) / 'expert_names.json').write_text(json.dumps(names, ensure_ascii=False), encoding='utf-8')
        print('Expert ↔ 방식:', {f'E{i + 1}': m for i, m in enumerate(names)})

    model = Detector(cfg)
    if not args.smoke:
        assert args.pretrained, '--pretrained 폴더를 주세요 (사전학습 가중치)'
        load_pretrained_backbone(model, os.path.join(args.pretrained, 'dinov3_vitl16_pretrain_lvd1689m.safetensors'))
        load_pretrained_common(model, os.path.join(args.pretrained, 'efficientnet_b0_ra_in1k.safetensors'))
    model.to(dev)          # ⚠️ 백본을 .half()로 바꾸지 않는다: DINOv3 ViT-L은 fp16에서 첫 블록부터 NaN(10/2 실측). 연산만 autocast(fp16)로 줄인다.
    if args.grad_ckpt:
        model.common.set_grad_checkpointing(True)
    for p in model.backbone.parameters():
        p.requires_grad = False
    extra = {'tmp_head': nn.Linear(model.common.num_features, 1).to(dev),
             'proj': nn.Sequential(nn.Linear(model.head[0].in_features, 256), nn.ReLU(), nn.Linear(256, 128)).to(dev)}
    mean = torch.tensor(cfg['mean'], device=dev).view(1, 3, 1, 1); std = torch.tensor(cfg['std'], device=dev).view(1, 3, 1, 1)
    (Path(args.out) / 'run_config.json').write_text(json.dumps({'args': {k: v for k, v in vars(args).items()}, 'model_config': cfg}, ensure_ascii=False, indent=2), encoding='utf-8')

    stages = [1, 2, 3] if args.stage == 'all' else [int(args.stage)]
    with open(Path(args.out) / 'log.jsonl', 'a', encoding='utf-8') as log:
        for s in stages:
            run_stage(s, model, extra, args, train_rows, val_rows, dev, mean, std, log)
    if 3 in stages:
        save_weights(model, Path(args.out) / cfg['weights'])
        if args.fusion_norm or args.family_map != 'none' or args.common_topk > 0 or args.common_highpass > 0 or args.spectrum > 0:        # 제출용 config.json에 fusion_norm·n_experts 표시 (없으면 가중치 키가 안 맞아 로드 실패). smoke가 바꾼 크기는 넣지 않는다
            base = json.loads(Path(args.config).read_text(encoding='utf-8'))
            if args.fusion_norm:
                base['fusion_norm'] = True
            if args.family_map != 'none':
                base['n_experts'] = N_EXPERTS[args.family_map]
            if args.common_topk > 0:
                base['common'].update(pool='topk', frac=args.common_topk, mix=args.common_topk_mix)
            if args.common_highpass > 0:
                base['common'].update(highpass={'gain': args.common_highpass})
            if args.spectrum > 0:
                base['spectrum'] = {'dim': args.spectrum}
            (Path(args.out) / 'config.json').write_text(json.dumps(base, ensure_ascii=False, indent=2), encoding='utf-8')
        else:
            shutil.copy2(args.config, Path(args.out) / 'config.json')
        print('내보내기 완료 →', Path(args.out) / cfg['weights'], '(제출용 형식: 백본 fp16 + 나머지 fp32)')


if __name__ == '__main__':
    main()
