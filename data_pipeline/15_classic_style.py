"""15단계: 전체 스타일 변환(④) 화풍판 — 모델 없이 OpenCV·numpy만으로, CPU에서 장당 1초 안팎. 게이팅 family E5(영상처리).

왜: 대회 문제 설명서가 전체·부분 스타일 변환을 "일러스트처럼", "회화적인 색감", "매끈하게 표현된 명암"으로 설명해서,
우리 ④(SDXL img2img, 사실적)는 이런 설명과 달라서 보완이 필요했다.
고전 영상처리 필터는 구조·글씨를 완벽히 보존하면서 색면은 매끈해지고 윤곽은 강해지는 화풍을 만든다.

방식 (generator 이름)          하는 일
  classic-stylization          cv2.stylization — 매끈한 색면 + 윤곽 (수채화·일러스트 느낌), 채도 약간 올림
  classic-edgepreserve         cv2.edgePreservingFilter — 윤곽은 두고 질감만 지움 (회화적으로 매끈한 명암)
  classic-cartoon              bilateral 필터 여러 번 + 색 단계 줄이기 + 어두운 윤곽선 (일러스트·만화 느낌)
  classic-oil                  Kuwahara 필터 — 붓 터치처럼 뭉치는 유화 느낌

type=full_style, family=E5. 규격: 10단계와 같다 — 실제 사진을 4:3 가운데 크롭 → 1024×768(768×1024)로 맞춘 뒤 변환, PNG 저장.
이후 05_finalize.py가 q92 JPEG로 통일. 12_partial_style.py --src classic_style 로 부분 스타일(⑥)도 만든다.
한 번 실행에 방식을 같은 비율로 돌려 가며(round-robin) 만든다. 덜 쓰인 원본 사진부터 고른다(common.least_used_first).

사용법:
  py data_pipeline/15_classic_style.py ./data --work ./work_classic --n 20           (로컬 시험)
  Colab: !python 15_classic_style.py /content/rr --work /content/work_cs --n 500 --sync_to /content/drive/MyDrive/work
"""

import argparse, importlib.util, os, random, time
import cv2
import numpy as np
from PIL import Image
from common import ManifestWriter, Syncer, fake_usage, least_used_first, read_manifest

NAME = 'classic_style'
METHODS = ['classic-stylization', 'classic-edgepreserve', 'classic-cartoon', 'classic-oil']


def _load(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(os.path.dirname(os.path.abspath(__file__)), name + '.py'))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def boost_saturation(bgr, gain):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 1] = np.clip(hsv[..., 1] * gain, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def f_stylization(bgr, rng):
    p = {'sigma_s': round(rng.uniform(15, 60), 1), 'sigma_r': round(rng.uniform(0.08, 0.4), 2), 'sat': round(rng.uniform(1.0, 1.3), 2)}
    out = cv2.stylization(bgr, sigma_s=p['sigma_s'], sigma_r=p['sigma_r'])
    return boost_saturation(out, p['sat']), p


def f_edgepreserve(bgr, rng):
    p = {'sigma_s': round(rng.uniform(10, 50), 1), 'sigma_r': round(rng.uniform(0.1, 0.35), 2), 'passes': rng.randint(1, 2),
         'sat': round(rng.uniform(1.0, 1.3), 2)}
    out = bgr
    for _ in range(p['passes']):
        out = cv2.edgePreservingFilter(out, flags=cv2.RECURS_FILTER, sigma_s=p['sigma_s'], sigma_r=p['sigma_r'])
    return boost_saturation(out, p['sat']), p


def f_cartoon(bgr, rng):
    p = {'bilateral_passes': rng.randint(1, 3), 'sigma_color': round(rng.uniform(25, 70), 1), 'levels': rng.randint(12, 24),
         'edge_c': round(rng.uniform(7, 11), 1), 'line_strength': round(rng.uniform(0.15, 0.45), 2)}
    smooth = bgr
    for _ in range(p['bilateral_passes']):
        smooth = cv2.bilateralFilter(smooth, 9, p['sigma_color'], 7)
    step = 256 // p['levels']
    quant = np.minimum((smooth.astype(np.int16) // step) * step + step // 2, 255).astype(np.float32)   # uint8로 더하면 255를 넘어 초록색으로 뒤집힌다
    gray = cv2.medianBlur(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), 5)
    edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, p['edge_c']).astype(np.float32) / 255.0
    line = (1.0 - p['line_strength']) + p['line_strength'] * edges      # 윤곽선 자리는 어둡게
    out = np.clip(quant * line[..., None], 0, 255).astype(np.uint8)
    return out, p


def f_oil(bgr, rng):
    """Kuwahara: 픽셀마다 주변 4개 사분면 중 밝기 분산이 가장 작은 곳의 평균색을 쓴다 → 붓 터치처럼 뭉친다."""
    r = rng.randint(2, 4)
    p = {'radius': r, 'passes': 1, 'sat': round(rng.uniform(1.0, 1.3), 2)}
    out = bgr
    k = r + 1
    anchors = [(0, 0), (k - 1, 0), (0, k - 1), (k - 1, k - 1)]
    for _ in range(p['passes']):
        f = out.astype(np.float32)
        g = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY).astype(np.float32)
        means, variances = [], []
        for ax, ay in anchors:
            m = cv2.boxFilter(f, -1, (k, k), anchor=(ax, ay), borderType=cv2.BORDER_REFLECT)
            mg = cv2.boxFilter(g, -1, (k, k), anchor=(ax, ay), borderType=cv2.BORDER_REFLECT)
            mg2 = cv2.boxFilter(g * g, -1, (k, k), anchor=(ax, ay), borderType=cv2.BORDER_REFLECT)
            means.append(m)
            variances.append(mg2 - mg * mg)
        idx = np.argmin(np.stack(variances), axis=0)                       # (H, W)
        chosen = np.take_along_axis(np.stack(means), idx[None, :, :, None], axis=0)[0]
        out = np.clip(chosen, 0, 255).astype(np.uint8)
    return boost_saturation(out, p['sat']), p


FUNCS = {'classic-stylization': f_stylization, 'classic-edgepreserve': f_edgepreserve, 'classic-cartoon': f_cartoon, 'classic-oil': f_oil}


def stylize(pil_img, method, rng):
    bgr = cv2.cvtColor(np.asarray(pil_img.convert('RGB')), cv2.COLOR_RGB2BGR)
    out, params = FUNCS[method](bgr, rng)
    return Image.fromarray(cv2.cvtColor(out, cv2.COLOR_BGR2RGB)), params


def main(real_root, work, n, seed, sync_to, sync_every, methods):
    inp = _load('10_sdxl_inpaint')       # standardize 재사용 (GPU 코드는 함수 안이라 import해도 안 불림)
    rows = [r for r in read_manifest(os.path.join(real_root, 'manifests', 'real.csv')) if r['label'] == '0']
    rng = random.Random(seed)
    rows = least_used_first(rows, fake_usage(real_root, exclude_step=NAME), rng)

    syncer = Syncer(work, sync_to, NAME) if sync_to else None
    if syncer:
        syncer.pull_manifest()
    mw = ManifestWriter(work, NAME)
    os.makedirs(os.path.join(work, 'raw', 'fake', NAME), exist_ok=True)
    made, attempt, t0 = 0, 0, time.time()
    while made < n and attempt < n * 3:
        row = rows[attempt % len(rows)]
        method = methods[made % len(methods)]
        pass_no = attempt // len(rows)
        attempt += 1
        rel = f'fake/{NAME}/{row["source_id"]}__{method}{pass_no}.png'
        if rel in mw.done:
            made += 1
            continue
        rng_i = random.Random(f'{seed}-{row["source_id"]}-{method}-{pass_no}')
        im = Image.open(os.path.join(real_root, 'raw', row['file'])).convert('RGB')
        std, crop, scale = inp.standardize(im)
        out, params = stylize(std, method, rng_i)
        out.save(os.path.join(work, 'raw', rel))
        mw.add(file=rel, label=1, type='full_style', family='E5', generator=method,
               source_id=row['source_id'], group=row['group'], split=row['split'],
               params={**params, 'size': list(std.size), 'crop': list(crop), 'scale': round(scale, 5), 'style': method})
        made += 1
        if syncer and made % sync_every == 0:
            syncer.push()
        if made % 50 == 0 or made == n:
            per = (time.time() - t0) / made
            print(f'{made}/{n} (장당 {per:.1f}초, 남은 시간 약 {per * (n - made) / 60:.0f}분)', flush=True)
    mw.close()
    if syncer:
        ok = syncer.push(retries=5)
        print('Drive 복사 완료' if ok else f'!! Drive 복사 실패 — 결과는 {work} 에 남아 있으니 다시 실행하면 이어서 복사됩니다', flush=True)
    print(f'완료 {made}장 → 표: {mw.path}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('real_root', help='raw/real, manifests/real.csv 가 있는 폴더(개인 Drive 또는 로컬)')
    ap.add_argument('--work', required=True, help='결과를 먼저 쓰는 로컬 폴더')
    ap.add_argument('--n', type=int, default=20)
    ap.add_argument('--seed', type=int, default=15)
    ap.add_argument('--sync_to', default=None)
    ap.add_argument('--sync_every', type=int, default=100)
    ap.add_argument('--methods', default=','.join(METHODS), help='쉼표로 구분한 방식 이름(기본: 전부)')
    a = ap.parse_args()
    ms = [m.strip() for m in a.methods.split(',') if m.strip()]
    assert all(m in METHODS for m in ms), f'알 수 없는 방식: {ms}'
    main(a.real_root, a.work, a.n, a.seed, a.sync_to, a.sync_every, ms)
