"""train.py의 핵심 함수(배치 뽑기·손실)가 의도대로인지 보는 작은 시험. 데이터·GPU 없이 몇 초 안에 끝난다.

  py train/test_train.py
"""
import math, sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train as T   # noqa: E402


def fake_rows():
    """실제 60 + 가짜(E1 20 · E2 10 · E4 30 · E5 40). 가짜 중 E4·E5는 실제와 같은 원본(source_id)을 쓴다."""
    rows = []
    for i in range(60):
        rows.append({'file': f'r{i}', 'y': 0, 'fam': -1, 'family': 'real', 'source_id': f's{i}', 'generator': '-'})
    for i in range(20):
        rows.append({'file': f'a{i}', 'y': 1, 'fam': 0, 'family': 'E1', 'source_id': '', 'generator': 'sdxl'})
    for i in range(10):
        rows.append({'file': f'b{i}', 'y': 1, 'fam': 1, 'family': 'E2', 'source_id': '', 'generator': 'flux'})
    for i in range(30):
        rows.append({'file': f'c{i}', 'y': 1, 'fam': 3, 'family': 'E4', 'source_id': f's{i}', 'generator': 'lama'})
    for i in range(40):
        rows.append({'file': f'd{i}', 'y': 1, 'fam': 4, 'family': 'E5', 'source_id': f's{i}', 'generator': 'filter'})
    return rows


def test_sampling_is_balanced():
    rows = fake_rows()
    batches = T.weighted_batches(rows, 4000, 8, np.random.RandomState(0))
    idx = np.array([i for b in batches for i in b])
    real = np.mean([rows[i]['y'] == 0 for i in idx])
    fam = Counter(rows[i]['fam'] for i in idx if rows[i]['y'] == 1)
    tot = sum(fam.values())
    assert abs(real - 0.5) < 0.02, f'실제 비율 {real}'
    for f in (0, 1, 3, 4):                       # 가짜 계열 4개가 각각 1/4
        assert abs(fam[f] / tot - 0.25) < 0.02, (f, fam[f] / tot)
    print('OK 배치 뽑기: 실제 %.3f, 가짜 계열 비율' % real, {k: round(v / tot, 3) for k, v in sorted(fam.items())})


def test_pairs_share_source():
    rows = fake_rows()
    batches = T.pair_batches(rows, 200, 8, 2, np.random.RandomState(1))
    for b in batches:
        assert len(b) == 8
        for k in (0, 2):                         # 앞 2쌍: (실제, 같은 원본의 가짜)
            r, f = rows[b[k]], rows[b[k + 1]]
            assert r['y'] == 0 and f['y'] == 1 and r['source_id'] == f['source_id'] != ''
        assert len({rows[i]['source_id'] for i in (b[0], b[2])}) == 2      # 한 배치의 두 쌍은 서로 다른 원본
    print('OK 3단계 쌍 배치: 200배치 모두 같은 원본의 (실제, 가짜) 쌍')


def test_supcon_matches_naive_loop():
    torch.manual_seed(0)
    z = torch.randn(10, 6); y = torch.tensor([0, 0, 0, 1, 1, 1, 1, 0, 1, 0])
    got = T.supcon(z, y, tau=0.2).item()
    zn = torch.nn.functional.normalize(z, dim=1); tot, cnt = 0.0, 0
    for i in range(10):
        pos = [j for j in range(10) if j != i and y[j] == y[i]]
        den = sum(math.exp(float(zn[i] @ zn[a]) / 0.2) for a in range(10) if a != i)
        tot += -sum(math.log(math.exp(float(zn[i] @ zn[p]) / 0.2) / den) for p in pos) / len(pos); cnt += 1
    assert abs(got - tot / cnt) < 1e-5, (got, tot / cnt)
    zz = torch.tensor([[1., 0], [1, 0.01], [-1, 0], [-1, 0.01]]); yy = torch.tensor([0, 0, 1, 1])
    assert T.supcon(zz, yy).item() < 0.2, T.supcon(zz, yy).item()          # 같은 라벨끼리 모이고 다른 라벨은 반대 방향 → 손실이 거의 0
    print('OK SupCon: 직접 짠 반복문과 일치, 잘 갈라진 경우 손실 %.4f' % T.supcon(zz, yy).item())


def test_gate_ce():
    gate = torch.tensor([[0.9, 0.025, 0.025, 0.025, 0.025], [0.2] * 5, [0.025, 0.025, 0.025, 0.9, 0.025]])
    fam = torch.tensor([0, -1, 3])               # 가짜 E1, 실제, 가짜 E4
    ce = T.gate_ce(gate, fam, 'uniform').item()
    want = (-math.log(0.9) + -math.log(0.2) + -math.log(0.9)) / 3          # 실제의 정답은 균등(0.2)이라 로그(0.2)
    assert abs(ce - want) < 1e-5, (ce, want)
    only_fake = T.gate_ce(gate, fam, 'ignore').item()
    assert abs(only_fake - (-math.log(0.9)) ) < 1e-5, only_fake              # 실제 무시 → 가짜 둘만
    print('OK Gate CE: uniform·ignore 모두 손으로 계산한 값과 일치')


def test_bce_balanced():
    logit = torch.tensor([2.0, -1.0, 0.5, 0.3, -0.7]); y = torch.tensor([0., 0., 1., 1., 1.])
    mask = torch.tensor([True, True, True, False, True])
    b = torch.nn.functional.binary_cross_entropy_with_logits
    want = 0.5 * b(logit[:2], y[:2]) + 0.5 * b(logit[[2, 4]], y[[2, 4]])
    assert abs(T.bce_balanced(logit, y, mask).item() - want.item()) < 1e-6
    assert T.bce_balanced(logit, y, torch.zeros(5, dtype=torch.bool)).item() == 0.0       # 해당 사진이 없으면 0
    print('OK Expert BCE: 실제 평균·가짜 평균을 절반씩 섞은 값과 일치')


def test_family_map():
    import copy
    rows = []

    def add(t, fam, y=1):
        rows.append({'y': y, 'type': t, 'family': fam, 'generator': 'g'})
    add('real', 'real', 0)
    add('AI_gen', 'E1'); add('AI_gen', 'E2'); add('AI_gen', 'E3'); add('vae_recon', 'E2')
    add('full_style', 'E1'); add('partial_gen', 'E4'); add('partial_gen', 'E5')           # exp06 목록표는 복제(partial_gen)를 E5로 고쳐 둔 상태
    add('partial_style', 'E4'); add('partial_style', 'E5'); add('composite', 'E5'); add('filter', 'E5')
    base = T.apply_family_map(copy.deepcopy(rows), 'none')
    assert [r['family'] for r in base] == [r['family'] for r in rows], '기본(none)은 목록표 family를 그대로 둬야 함'
    assert [r['fam'] for r in base] == [-1, 0, 1, 2, 1, 0, 3, 4, 3, 4, 4, 4]
    t8 = T.apply_family_map(copy.deepcopy(rows), 'type8')
    assert [r['family'] for r in t8] == ['real', 'E1', 'E2', 'E3', 'E2', 'E4', 'E5', 'E5', 'E6', 'E6', 'E7', 'E8'], [r['family'] for r in t8]
    assert [r['fam'] for r in t8] == [-1, 0, 1, 2, 1, 3, 4, 4, 5, 5, 6, 7]
    assert T.N_EXPERTS['type8'] == 8 and max(r['fam'] for r in t8) == 7
    try:        # Expert를 못 정한 가짜는 조용히 '실제'처럼 지나가지 않고 멈춘다
        T.apply_family_map([{'y': 1, 'type': 'AI_gen', 'family': 'E99', 'generator': 'x'}], 'type8')
        raise SystemExit('멈췄어야 하는데 지나감')
    except AssertionError:
        pass
    print('OK family_map: none은 목록표 그대로, type8은 유형별 Expert 8개(① 전체 생성은 생성기 계열 유지), 모르는 family는 멈춤')


def test_family_map_gen24():
    names = [m for _, m in T.GEN_METHODS]
    assert len(names) == len(set(names)) == T.N_EXPERTS['gen24'] == 24
    cases = [('realvisxl', 'realvisxl'), ('realvisxl_v6', 'realvisxl'), ('flux_schnell', 'flux_schnell'), ('flux_schnell_v6', 'flux_schnell'),
             ('zimage_turbo', 'zimage_turbo'), ('zimage_turbo_v6', 'zimage_turbo'), ('infinity_2b', 'infinity'), ('flux_vae', 'flux_vae'),
             ('composite-bgswap', 'bgswap'), ('composite-objpaste', 'objpaste'), ('filter-noise', 'filter'), ('filter-saltpepper', 'filter'),
             ('sdxl-img2img-warm', 'sdxl_img2img'), ('lama-erase', 'lama'), ('sdxl-inpaint-pasted', 'sdxl_inpaint'), ('sd15-inpaint-crop', 'sd15_inpaint'),
             ('clone-alpha', 'clone'), ('sdxl-partial-muted', 'sdxl_partial'), ('local-filter-blur', 'local_filter'), ('flux2-klein-4b', 'klein_t2i'),
             ('flux2-klein-style-oil', 'klein_style'), ('classic-cartoon', 'classic_style'), ('sdxl-paint-watercolor', 'sdxl_paint'),
             ('flux2-klein-edit-add', 'klein_edit'), ('kandinsky-inpaint-full', 'kandinsky_inpaint'), ('partial-flux2-klein-style-oil', 'partial_klein'),
             ('partial-sdxl-paint-oil', 'partial_sdxl_paint'), ('partial-classic-oil', 'partial_classic')]
    rows = [{'y': 1, 'type': 'x', 'family': 'E1', 'generator': g} for g, _ in cases] + [{'y': 0, 'type': 'real', 'family': 'real', 'generator': '-'}]
    out = T.apply_family_map(rows, 'gen24')
    got = [names[r['fam']] for r in out[:-1]]
    assert got == [m for _, m in cases], [(g, a, b) for (g, b), a in zip(cases, got) if a != b]
    assert out[-1]['fam'] == -1 and set(r['fam'] for r in out[:-1]) == set(range(24)), '24방식이 전부 한 번 이상 나와야 함'
    try:
        T.apply_family_map([{'y': 1, 'type': 'x', 'family': 'E1', 'generator': 'unknown-gen'}], 'gen24')
        raise SystemExit('멈췄어야 하는데 지나감')
    except AssertionError:
        pass
    print('OK gen24: generator 28가지 이름이 24방식으로 맞게 묶임, 모르는 generator는 멈춤')


def test_sampling_with_8_families():
    rows = [{'file': f'r{i}', 'y': 0, 'fam': -1, 'family': 'real', 'source_id': '', 'generator': '-'} for i in range(80)]
    for k in range(8):
        rows += [{'file': f'f{k}_{i}', 'y': 1, 'fam': k, 'family': f'E{k + 1}', 'source_id': '', 'generator': f'g{k}'} for i in range(10 + 7 * k)]
    batches = T.weighted_batches(rows, 4000, 8, np.random.RandomState(0))
    fam = Counter(rows[i]['fam'] for b in batches for i in b if rows[i]['y'] == 1)
    tot = sum(fam.values())
    for k in range(8):                         # 가짜 계열 8개가 각각 1/8 (장수가 달라도)
        assert abs(fam[k] / tot - 0.125) < 0.02, (k, fam[k] / tot)
    gate = torch.full((2, 8), 0.125)
    ce = T.gate_ce(gate, torch.tensor([7, -1]), 'uniform').item()
    assert abs(ce - math.log(8)) < 1e-5, ce
    print('OK 8계열: 가짜 계열별 비율', {k: round(v / tot, 3) for k, v in sorted(fam.items())}, '/ Gate CE(균등) = ln 8')


def test_new_real_boost():
    rows = [{'file': f'r{i}', 'y': 0, 'fam': -1, 'family': 'real', 'source_id': '', 'generator': '-', 'group': 'old'} for i in range(60)]
    rows += [{'file': f'n{i}', 'y': 0, 'fam': -1, 'family': 'real', 'source_id': '', 'generator': '-', 'group': 'n:x'} for i in range(20)]
    rows += [{'file': f'f{i}', 'y': 1, 'fam': 0, 'family': 'E1', 'source_id': '', 'generator': 'g', 'group': 'g'} for i in range(30)]
    try:
        T.NEW_REAL_BOOST = 1.0
        w = T.sample_weights(rows)
        assert abs(w[:80].sum() - 0.5) < 1e-9 and abs(w[80:].sum() - 0.5) < 1e-9
        assert abs(w[0] - w[60]) < 1e-12, '1배면 새·옛 실제 사진이 같은 확률'
        T.NEW_REAL_BOOST = 4.0
        w = T.sample_weights(rows)
        assert abs(w[:80].sum() - 0.5) < 1e-9 and abs(w[80:].sum() - 0.5) < 1e-9, '실제 전체 몫(1/2)은 그대로'
        assert abs(w[60] / w[0] - 4.0) < 1e-9
        batches = T.weighted_batches(rows, 4000, 8, np.random.RandomState(0))
        idx = np.array([i for b in batches for i in b]); real = idx[idx < 80]
        share = float((real >= 60).mean())                      # 새 실제의 몫 = 4*20 / (60 + 4*20) = 0.571
        assert abs(share - 80 / 140) < 0.03, share
    finally:
        T.NEW_REAL_BOOST = 1.0
    print('OK new_real_boost: 실제 전체 몫 1/2 유지, 새 실제 사진은 4배 확률(뽑힌 몫 %.3f, 기대 0.571)' % share)


def tiny_cfg(**common):
    """작은 백본(--smoke와 같은 크기)으로 Detector를 만들 때 쓰는 설정."""
    import json
    cfg = json.loads((Path(__file__).resolve().parent.parent / 'submission' / 'model' / 'config.json').read_text(encoding='utf-8'))
    cfg['backbone'].update(hidden_size=64, intermediate_size=128, num_hidden_layers=2, num_attention_heads=4)
    cfg['pool_dim'] = 32
    cfg['common'].update(common)
    return cfg


def test_highpass_input():
    import tempfile
    import timm
    from safetensors.torch import save_file
    import detector as DET
    torch.manual_seed(0)
    x = torch.randn(2, 3, 32, 48)
    # 1) 입력 변환: 앞 3채널은 원본 그대로, 뒤 3채널은 gain × (x − 3×3 평균). 상수 사진의 잔차는 0
    y = DET.with_highpass(x, 8.0)
    assert y.shape == (2, 6, 32, 48) and torch.equal(y[:, :3], x)
    pad = torch.nn.functional.pad(x, (1, 1, 1, 1), mode='reflect')
    box = sum(pad[:, :, i:i + 32, j:j + 48] for i in range(3) for j in range(3)) / 9
    assert torch.allclose(y[:, 3:], 8.0 * (x - box), atol=1e-5)
    assert DET.with_highpass(torch.full((1, 3, 16, 16), 0.7), 8.0)[:, 3:].abs().max() < 1e-6
    # 2) 기본(highpass 없음)은 기존과 같다: 3채널·훅 없음
    base = DET.Detector(tiny_cfg())
    assert base.common.conv_stem.weight.shape[1] == 3 and len(base.common._forward_pre_hooks) == 0
    # 3) highpass: 6채널 + 훅 1개. 사전학습 가중치를 넣으면 시작 시점에 RGB 전용 모델과 출력이 같다(새 채널 가중치 0)
    hp = DET.Detector(tiny_cfg(highpass={'gain': 8.0}))
    assert hp.common.conv_stem.weight.shape[1] == 6 and len(hp.common._forward_pre_hooks) == 1
    ref = timm.create_model('efficientnet_b0', pretrained=False, num_classes=0).eval()
    with tempfile.TemporaryDirectory() as d:
        p = str(Path(d) / 'eff.safetensors')
        save_file({k: v.contiguous() for k, v in ref.state_dict().items()}, p)
        DET.load_pretrained_common(hp, p)
    hp.common.eval()
    img = torch.randn(2, 3, 64, 96)
    with torch.no_grad():
        a, b = ref(img), hp.common(img)
    assert torch.allclose(a, b, atol=1e-4), float((a - b).abs().max())
    # 4) 새 채널 가중치를 키우면 출력이 실제로 바뀐다(잔차가 모델에 들어간다)
    with torch.no_grad():
        hp.common.conv_stem.weight[:, 3:] += 0.5
        c = hp.common(img)
    assert (a - c).abs().max() > 1e-2
    print('OK highpass: 입력 6채널 변환 정확 · 기본 경로 불변 · 사전학습 로드 직후 RGB 전용 모델과 동일(최대 차이 %.1e) · 새 채널을 키우면 출력 변화' % float((a - b).abs().max()))


def test_modality_dropout():
    import detector as DET
    torch.manual_seed(0)
    m = DET.Detector(tiny_cfg())
    m.head = torch.nn.Identity()                                  # 최종 MLP 입력을 그대로 꺼내 본다
    B, d = 3000, m.common.num_features
    n_e = 1 + 64                                                  # Expert 블록 = 섞인 특징 64 + 점수 1
    tokens, common = torch.randn(B, 6, m.backbone.config.hidden_size), torch.ones(B, d)
    m.eval()
    with torch.no_grad():
        o_eval = m.combine(tokens, common)
    assert torch.equal(o_eval['logit'], o_eval['feat']) and o_eval['feat'].shape == (B, n_e + d)
    m.train()
    with torch.no_grad():
        o0 = m.combine(tokens, common)                            # mod_drop 0(기본) → 기존과 동일
        assert torch.equal(o0['logit'], o0['feat'])
        m.mod_drop = 0.2
        o = m.combine(tokens, common)
    x = o['logit']
    e_off, c_off = (x[:, :n_e] == 0).all(1), (x[:, n_e:] == 0).all(1)
    assert not (e_off & c_off).any(), '한 사진에서 둘이 같이 꺼지면 안 됨'
    assert abs(float(e_off.float().mean()) - 0.2) < 0.03 and abs(float(c_off.float().mean()) - 0.2) < 0.03, (e_off.float().mean(), c_off.float().mean())
    keep = ~(e_off | c_off)
    assert torch.equal(x[keep], o['feat'][keep]), '안 꺼진 사진은 값이 그대로'
    assert torch.equal(o['feat'], o_eval['feat']), 'SupCon이 쓰는 feat는 드롭아웃 전 값이어야 함'
    m.eval()
    with torch.no_grad():
        assert torch.equal(m.combine(tokens, common)['logit'], o_eval['logit']), '평가·추론에선 안 꺼짐'
    print('OK modality dropout: Expert %.3f · Common %.3f 꺼짐(기대 0.2씩), 동시에 꺼짐 0, feat(SupCon용)·평가 경로 불변' % (float(e_off.float().mean()), float(c_off.float().mean())))


def _smooth_image(h, w, seed=0):
    """질감이 있는 합성 사진(여러 크기의 부드러운 무늬 합) — [0,1] 범위 numpy [H,W,3]."""
    import numpy as np
    rng = np.random.RandomState(seed)
    img = np.zeros((h, w, 3), dtype=np.float32)
    for s in (4, 8, 16, 32):
        small = rng.rand(h // s + 1, w // s + 1, 3).astype(np.float32)
        t = torch.from_numpy(small).permute(2, 0, 1)[None]
        up = torch.nn.functional.interpolate(t, size=(h, w), mode='bilinear', align_corners=False)[0].permute(1, 2, 0).numpy()
        img += up / s ** 0.5
    img = (img - img.min()) / (img.max() - img.min())
    return img


def _to_norm(imgs, mean, std):
    x = torch.from_numpy(__import__('numpy').stack(imgs)).permute(0, 3, 1, 2)
    return (x - torch.tensor(mean).view(1, 3, 1, 1)) / torch.tensor(std).view(1, 3, 1, 1)


def test_spectrum_features():
    import io
    import numpy as np
    from PIL import Image
    import detector as DET
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    sp = DET.SpectrumHead(mean, std, 32)
    h, w = 768, 1024
    base = _smooth_image(h, w, 1)
    # 1) 모양·유한값: 가로형·세로형 모두 [B,50], NaN 없음(반지름 칸이 비지 않음)
    for (hh, ww) in ((768, 1024), (1024, 768)):
        f = sp.features(_to_norm([_smooth_image(hh, ww, 2), _smooth_image(hh, ww, 3)], mean, std))
        assert f.shape == (2, DET.SpectrumHead.N_FEAT) and torch.isfinite(f).all(), (hh, ww, f.shape)
        assert sp(_to_norm([_smooth_image(hh, ww, 2)], mean, std)).shape == (1, 32)
        assert DET.SpectrumHead.N_FEAT == 48
    # 2) 블러는 고주파 반지름 칸을 낮추고, 노이즈는 높인다 (반지름 칸 마지막 8개 평균)
    t = torch.from_numpy(base).permute(2, 0, 1)[None]
    blur = torch.nn.functional.avg_pool2d(torch.nn.functional.pad(t, (2, 2, 2, 2), mode='reflect'), 5, 1)[0].permute(1, 2, 0).numpy()
    noisy = np.clip(base + np.random.RandomState(0).randn(*base.shape).astype(np.float32) * 0.03, 0, 1)
    fb, f0, fn = (sp.features(_to_norm([im], mean, std))[0, :DET.SpectrumHead.N_RAD][-8:].mean().item() for im in (blur, base, noisy))
    assert fb < f0 - 0.3 and fn > f0 + 0.3, (fb, f0, fn)
    # 3) JPEG 품질이 낮을수록 블록 격자 비율(뒤 4개)이 커진다
    def jpeg(img, q):
        buf = io.BytesIO(); Image.fromarray((img * 255).astype(np.uint8)).save(buf, 'JPEG', quality=q); buf.seek(0)
        return np.asarray(Image.open(buf).convert('RGB'), dtype=np.float32) / 255.0
    pk = lambda im: sp.features(_to_norm([im], mean, std))[0, -DET.SpectrumHead.N_BLOCK:].mean().item()
    p20, p95, p0 = pk(jpeg(base, 20)), pk(jpeg(base, 95)), pk(base)
    assert p20 > p95 + 0.3 and p20 > p0 + 0.3, (p20, p95, p0)
    # 4) 밝기만 바뀌면 같다(평균 제거), 순서를 바꿔도 사진마다 같다
    xa = _to_norm([base, noisy], mean, std)
    xb = _to_norm([np.clip(base + 0.1, 0, 1.2), noisy], mean, std)
    assert torch.allclose(sp.features(xa)[0], sp.features(xb)[0], atol=1e-3)
    assert torch.allclose(sp.features(xa)[[1, 0]], sp.features(xa.flip(0)), atol=1e-5)
    print('OK spectrum: 가로·세로 모양 정상, 블러 %.2f < 원본 %.2f < 노이즈 %.2f(고주파 칸), JPEG 블록 격자 q20 %.2f > q95 %.2f(원본 %.2f), 밝기·순서 불변' % (fb, f0, fn, p20, p95, p0))


def test_detector_spectrum():
    import detector as DET
    torch.manual_seed(0)
    cfg = tiny_cfg(highpass={'gain': 6.0})
    cfg['spectrum'] = {'dim': 32}
    m = DET.Detector(cfg).eval()
    assert m.common.num_features == 1280 + 32 and m.head[0].in_features == 65 + 1280 + 32
    x = _to_norm([_smooth_image(768, 1024, 4), _smooth_image(768, 1024, 5)], [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    with torch.no_grad():
        out = m.common(x)                                          # [CNN 1280 | 스펙트럼 32]
        assert out.shape == (2, 1312)
        assert torch.allclose(out[:, 1280:], m.spec(x), atol=1e-5), 'highpass(6채널)와 함께여도 스펙트럼은 RGB 앞 3채널에서 계산돼야 함'
        o = m.combine(torch.randn(2, 6, m.backbone.config.hidden_size), out)
        assert o['logit'].shape == (2,) and torch.isfinite(o['logit']).all()
        z = m.combine(torch.randn(2, 6, m.backbone.config.hidden_size), torch.zeros(2, m.common.num_features))   # 1·2단계처럼 0 벡터를 넣어도 크기가 맞음
        assert z['logit'].shape == (2,)
    base = DET.Detector(tiny_cfg())                               # 기본(spectrum 없음)은 기존과 같다
    assert base.common.num_features == 1280 and not hasattr(base, 'spec') and len(base.common._forward_hooks) == 0
    print('OK detector spectrum: common 출력 1312(= 1280 + 32), head 입력 1377, highpass와 함께 동작, 기본 경로 불변')


if __name__ == '__main__':
    for f in (test_sampling_is_balanced, test_pairs_share_source, test_supcon_matches_naive_loop, test_gate_ce, test_bce_balanced,
              test_family_map, test_family_map_gen24, test_sampling_with_8_families, test_new_real_boost, test_highpass_input, test_modality_dropout, test_spectrum_features, test_detector_spectrum):
        f()
    print('전부 통과')
