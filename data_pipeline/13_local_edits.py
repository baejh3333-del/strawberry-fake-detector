"""13단계: 국소 편집(⑤⑥ 보강) 가짜 만들기 — 생성 모델 없이, CPU만으로. 게이팅 family E4(부분 편집).

왜: leave-one-generator-out(10/4, 진행상황.md)에서 부분 편집은 처음 보는 방식이면 거의 못 잡았다(AUROC 0.71~0.82).
지금 E4는 LaMa·SDXL 두 가지뿐이라 그 둘의 흔적을 외운 것. 방식을 넓혀 "편집됐다"는 공통점을 배우게 한다.
전부 실제 사진의 **일부 영역만** 바꾸고 나머지는 원본 그대로다. 영역은 10단계와 같은 AI Hub 부위 사각형(1~2개, 면적 1~25%)을 기준으로
하되 모양을 사각형·타원·불규칙 덩어리 중에서 고른다(사각형만 쓰면 "사각형 경계 = 가짜"를 배운다).

방식 (generator 이름)                           하는 일
  local-filter-blur / noise / jpeg / sharpen    영역만 블러(반경 1.5~4)·가우시안 노이즈(σ 6~22)·JPEG 재압축(품질 15~55)·샤프닝
  local-color-hue / tone / warm                 영역만 색상 이동(±8~25°)·밝기 감마·대비 / 색온도(R·B 이득)
  clone-alpha / clone-seamless                  같은 사진의 다른 곳을 복사해 붙임(좌우반전·회전·배율 변형) — 가장자리를 흐려 붙임 / Poisson 합성(cv2.seamlessClone, 경계 불연속을 숨김)

type: 필터·색은 partial_style, 복제는 partial_gen (공개샘플 표기와 같게 쓰되 실제 생성 모델은 아님 — params에 방식 기록).
규격: 10단계와 같다 — 실제 사진을 4:3 가운데 크롭 → 1024×768(768×1024)로 맞춘 뒤 편집, PNG 저장. 이후 05_finalize.py가 q92 JPEG로 통일.
한 번 실행에 방식을 같은 비율로 돌려 가며(round-robin) 만든다. 덜 쓰인 원본 사진부터 고른다(common.least_used_first).

사용법:
  py data_pipeline/13_local_edits.py ./data --work ./work_local_edits --n 40        (로컬 시험)
  Colab: !python 13_local_edits.py /content/drive/MyDrive/work --work /content/work_le --n 1500 --sync_to /content/drive/MyDrive/work
"""

import argparse, importlib.util, io, json, math, os, random, time
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageDraw
from common import ManifestWriter, Syncer, fake_usage, least_used_first, read_manifest

NAME = 'local_edits'
METHODS = ['local-filter-blur', 'local-filter-noise', 'local-filter-jpeg', 'local-filter-sharpen',
           'local-color-hue', 'local-color-tone', 'local-color-warm',
           'clone-alpha', 'clone-seamless']
PAD_RATIO = 0.10
SHAPES = [('rect', 0.30), ('ellipse', 0.35), ('blob', 0.35)]


def _load(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(os.path.dirname(os.path.abspath(__file__)), name + '.py'))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ───────────── 마스크 ─────────────

def pick_shape(rng):
    x, acc = rng.random(), 0.0
    for s, w in SHAPES:
        acc += w
        if x < acc:
            return s
    return 'rect'


def region_mask(size, box, shape, rng):
    """box(x, y, w, h) 안에 모양을 그린 L 마스크(0/255). 경계 흐리기는 붙일 때 따로 한다."""
    W, H = size
    x, y, bw, bh = box
    px, py = bw * PAD_RATIO, bh * PAD_RATIO
    x0, y0 = max(0, int(x - px)), max(0, int(y - py))
    x1, y1 = min(W, int(x + bw + px)), min(H, int(y + bh + py))
    m = Image.new('L', size, 0)
    d = ImageDraw.Draw(m)
    if shape == 'rect':
        d.rectangle((x0, y0, x1, y1), fill=255)
    elif shape == 'ellipse':
        d.ellipse((x0, y0, x1, y1), fill=255)
    else:   # blob: 가장자리가 울퉁불퉁한 덩어리 — 중심에서 방향마다 반지름이 부드럽게 변하는 다각형
        cx, cy, rx, ry = (x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0) / 2, (y1 - y0) / 2
        n = 24
        k = [rng.uniform(0.55, 1.0) for _ in range(6)]
        pts = []
        for i in range(n):
            a = 2 * math.pi * i / n
            r = 0.78 + 0.22 * math.sin(2 * a + k[0] * 6) * k[1] + 0.12 * math.sin(3 * a + k[2] * 6) * k[3] + 0.08 * math.sin(5 * a + k[4] * 6) * k[5]
            r = min(1.0, max(0.5, r))
            pts.append((cx + rx * r * math.cos(a), cy + ry * r * math.sin(a)))
        d.polygon(pts, fill=255)
    return m


def soft(mask, feather):
    return mask.filter(ImageFilter.GaussianBlur(feather)) if feather else mask


def bbox_of(mask):
    b = mask.getbbox()
    return b if b else (0, 0, mask.width, mask.height)


# ───────────── 방식 ─────────────

def edit_filter(im, kind, rng):
    if kind == 'blur':
        r = rng.uniform(1.5, 4.0)
        return im.filter(ImageFilter.GaussianBlur(r)), {'radius': round(r, 2)}
    if kind == 'noise':
        s = rng.uniform(6, 22)
        a = np.asarray(im, dtype=np.float32)
        n = np.random.default_rng(rng.randint(0, 2**31 - 1)).normal(0, s, a.shape[:2] + ((1,) if rng.random() < 0.5 else (3,)))
        return Image.fromarray(np.clip(a + n, 0, 255).astype(np.uint8)), {'sigma': round(s, 1), 'mono': n.shape[2] == 1}
    if kind == 'jpeg':
        q = rng.randint(15, 55)
        buf = io.BytesIO()
        im.save(buf, 'JPEG', quality=q)
        return Image.open(io.BytesIO(buf.getvalue())).convert('RGB'), {'quality': q}
    # sharpen
    amt = rng.uniform(1.5, 3.5)
    return im.filter(ImageFilter.UnsharpMask(radius=rng.uniform(1.5, 3.0), percent=int(amt * 100), threshold=2)), {'percent': int(amt * 100)}


def edit_color(im, kind, rng):
    if kind == 'hue':
        shift = rng.uniform(8, 25) * rng.choice([-1, 1])
        hsv = np.asarray(im.convert('HSV'), dtype=np.int16).copy()
        hsv[..., 0] = (hsv[..., 0] + int(shift / 360 * 256)) % 256
        return Image.fromarray(hsv.astype(np.uint8), 'HSV').convert('RGB'), {'hue_shift_deg': round(shift, 1)}
    if kind == 'tone':
        g = rng.uniform(0.7, 1.4)
        c = rng.uniform(0.8, 1.3)
        a = np.asarray(im, dtype=np.float32) / 255.0
        a = np.clip(a ** g, 0, 1)
        out = Image.fromarray((a * 255).astype(np.uint8))
        out = ImageEnhance.Contrast(out).enhance(c)
        return out, {'gamma': round(g, 2), 'contrast': round(c, 2)}
    # warm: 색온도 — R·B 이득을 반대로
    t = rng.uniform(0.08, 0.22) * rng.choice([-1, 1])
    a = np.asarray(im, dtype=np.float32)
    a[..., 0] *= 1 + t
    a[..., 2] *= 1 - t
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8)), {'warmth': round(t, 3)}


def edit_clone(std, mask, rng, mode):
    """마스크 영역에, 같은 사진의 겹치지 않는 다른 곳을 복사해 붙인다. (결과, params)"""
    W, H = std.size
    bx0, by0, bx1, by1 = bbox_of(mask)
    bw, bh = bx1 - bx0, by1 - by0
    s = rng.uniform(0.85, 1.15)
    sw, sh = max(16, min(W - 2, int(bw * s))), max(16, min(H - 2, int(bh * s)))
    for _ in range(40):                                  # 대상과 겹치지 않는 원본 위치 찾기
        sx, sy = rng.randint(0, W - sw), rng.randint(0, H - sh)
        if sx + sw < bx0 or sx > bx1 or sy + sh < by0 or sy > by1:
            break
    else:
        sx, sy = (0 if bx0 > W // 2 else W - sw), (0 if by0 > H // 2 else H - sh)
    patch = std.crop((sx, sy, sx + sw, sy + sh)).resize((bw, bh), Image.LANCZOS)
    flip = rng.random() < 0.5
    ang = rng.uniform(-15, 15)
    if flip:
        patch = patch.transpose(Image.FLIP_LEFT_RIGHT)
    patch = patch.rotate(ang, resample=Image.BICUBIC, expand=False)
    params = {'src_box': [sx, sy, sw, sh], 'scale': round(s, 2), 'flip': flip, 'angle': round(ang, 1), 'blend': mode}
    full = Image.new('RGB', std.size)
    full.paste(patch, (bx0, by0))
    if mode == 'alpha':
        return Image.composite(full, std, soft(mask, rng.uniform(3, 9))), params
    import cv2
    m = np.asarray(mask.crop((bx0, by0, bx1, by1)), dtype=np.uint8).copy()
    m[:2, :], m[-2:, :], m[:, :2], m[:, -2:] = 0, 0, 0, 0     # seamlessClone은 마스크가 가장자리에 닿으면 안 된다
    src = cv2.cvtColor(np.asarray(patch), cv2.COLOR_RGB2BGR)
    dst = cv2.cvtColor(np.asarray(std), cv2.COLOR_RGB2BGR)
    center = (bx0 + bw // 2, by0 + bh // 2)
    try:
        out = cv2.seamlessClone(src, dst, m, center, cv2.NORMAL_CLONE)
    except cv2.error:
        return Image.composite(full, std, soft(mask, 6)), {**params, 'blend': 'alpha-fallback'}
    return Image.fromarray(cv2.cvtColor(out, cv2.COLOR_BGR2RGB)), params


def make_one(std, boxes, method, rng):
    """(결과 이미지, params). 영역 1~2개를 골라 같은 방식을 적용한다."""
    picked = list(boxes)
    rng.shuffle(picked)
    picked = picked[:rng.randint(1, 2)]
    shape = pick_shape(rng)
    out = std
    regions = []
    for box in picked:
        mask = region_mask(std.size, box, shape, rng)
        feather = rng.uniform(2, 8)
        if method.startswith('local-filter-'):
            edited, p = edit_filter(out, method.split('-')[-1], rng)
            out = Image.composite(edited, out, soft(mask, feather))
        elif method.startswith('local-color-'):
            edited, p = edit_color(out, method.split('-')[-1], rng)
            out = Image.composite(edited, out, soft(mask, feather))
        else:
            out, p = edit_clone(out, mask, rng, 'seamless' if method == 'clone-seamless' else 'alpha')
        regions.append({'box': list(box), 'shape': shape, 'feather': round(feather, 1), **p})
    return out, {'regions': regions, 'size': list(std.size)}


def main(real_root, work, n, seed, sync_to, sync_every, methods):
    inp = _load('10_sdxl_inpaint')       # standardize, transform_boxes 재사용 (GPU 코드는 함수 안이라 import해도 안 불림)
    boxes_all = json.load(open(os.path.join(real_root, 'raw', 'real_boxes.json'), encoding='utf-8'))
    rows = [r for r in read_manifest(os.path.join(real_root, 'manifests', 'real.csv')) if r['label'] == '0']
    rng = random.Random(seed)
    rows = least_used_first(rows, fake_usage(real_root, exclude_step=NAME), rng)

    syncer = Syncer(work, sync_to, NAME) if sync_to else None
    if syncer:
        syncer.pull_manifest()
    mw = ManifestWriter(work, NAME)
    os.makedirs(os.path.join(work, 'raw', 'fake', NAME), exist_ok=True)
    made, attempt, t0 = 0, 0, time.time()
    while made < n and attempt < n * 6:
        row = rows[attempt % len(rows)]
        method = methods[made % len(methods)]       # 건너뛴 사진이 있어도 방식이 고르게 돌게 '만든 장수' 기준
        pass_no = attempt // len(rows)
        attempt += 1
        rel = f'fake/{NAME}/{row["source_id"]}__{method}{pass_no}.png'
        if rel in mw.done:
            made += 1
            continue
        rng_i = random.Random(f'{seed}-{row["source_id"]}-{method}-{pass_no}')
        im = Image.open(os.path.join(real_root, 'raw', row['file'])).convert('RGB')
        std, crop, scale = inp.standardize(im)
        cands = inp.transform_boxes(boxes_all.get(row['source_id'], []), crop, scale, std.size)
        if not cands:
            continue
        out, params = make_one(std, cands, method, rng_i)
        out.save(os.path.join(work, 'raw', rel))
        mw.add(file=rel, label=1, type='partial_gen' if method.startswith('clone') else 'partial_style', family='E4', generator=method,
               source_id=row['source_id'], group=row['group'], split=row['split'], params=params)
        made += 1
        if syncer and made % sync_every == 0:
            syncer.push()
        if made % 100 == 0 or made == n:
            per = (time.time() - t0) / made
            print(f'{made}/{n} (장당 {per:.1f}초, 남은 시간 약 {per * (n - made) / 60:.0f}분)', flush=True)
    mw.close()
    if syncer:
        ok = syncer.push(retries=5)
        print('Drive 복사 완료' if ok else f'!! Drive 복사 실패 — 결과는 {work} 에 남아 있으니 Drive 연결 후 다시 실행하면 이어서 복사됩니다', flush=True)
    print(f'완료 {made}장 → 표: {mw.path}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('real_root', help='raw/real, raw/real_boxes.json, manifests/real.csv 가 있는 폴더(개인 Drive 또는 로컬)')
    ap.add_argument('--work', required=True, help='결과를 먼저 쓰는 로컬 폴더')
    ap.add_argument('--n', type=int, default=40)
    ap.add_argument('--seed', type=int, default=13)
    ap.add_argument('--sync_to', default=None)
    ap.add_argument('--sync_every', type=int, default=100)
    ap.add_argument('--methods', default=','.join(METHODS), help='쉼표로 구분한 방식 이름(기본: 전부)')
    a = ap.parse_args()
    ms = [m.strip() for m in a.methods.split(',') if m.strip()]
    assert all(m in METHODS for m in ms), f'알 수 없는 방식: {ms}'
    main(a.real_root, a.work, a.n, a.seed, a.sync_to, a.sync_every, ms)
