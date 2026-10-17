"""submission/ 뼈대 점검 — 학습된 가중치 없이, 무작위 가중치로 '구조가 돌아가는지'만 본다 (진행상황 M1: 조각 12장 + 추론 시간 실측).

하는 일
  1) submission/을 임시 폴더로 복사하고 무작위 가중치(weights.safetensors)를 만든다 → 저장소에는 가중치가 남지 않는다.
  2) 조각 개수(가로 1024×768 → 12장), 파라미터 수, 가중치 파일 크기 출력
  3) 합성 이미지(가로·세로·규격 밖 크기)로 load_model / predict 를 호출해 길이·범위·순서 독립성 확인
  4) 사진 1장당 추론 시간 → 524장 예상 시간 (이 PC 기준. 채점은 Colab T4이므로 M4 전에 T4에서 다시 잴 것)

사용법:  py tools/smoke_test_submission.py [--keep] [--no_amp]
         --keep   : 임시 폴더를 지우지 않고 self_check용으로 남김
         --no_amp : GPU에서도 fp16(autocast) 없이 순수 fp32로 돌림. GTX 1660 Ti처럼 fp16이 오히려 느린 GPU(텐서코어 없음)에서 시간을 재려면 필요.
                    제출 코드(inference.py)는 GPU면 autocast(fp16) — T4는 텐서코어가 있어 빠르다.
                    ⚠️ 백본을 .half()로 바꾸면 실제 가중치에서 NaN이 난다(10/2 실측). 무작위 가중치로는 이 문제가 안 보이니
                    실제 가중치로 만든 팀 폴더에서도 한 번 돌려 볼 것.
이어서:  py <contest-provided>/self_check.py --team_dir <출력된 임시 폴더>
"""
import argparse, importlib.util, shutil, sys, tempfile, time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / 'submission'


def load_inference(team_dir):
    sys.path.insert(0, str(team_dir))      # 채점기(inference_worker.py)와 같은 방식
    spec = importlib.util.spec_from_file_location('inference', team_dir / 'inference.py')
    mod = importlib.util.module_from_spec(spec)
    sys.modules['inference'] = mod
    spec.loader.exec_module(mod)
    return mod


def fake_photo(path, size, seed):
    rng = np.random.default_rng(seed)
    w, h = size
    base = np.linspace(0, 255, w, dtype=np.float32)[None, :, None] * np.ones((h, 1, 3), np.float32)
    img = np.clip(base + rng.normal(0, 25, (h, w, 3)), 0, 255).astype(np.uint8)
    Image.fromarray(img).save(path, 'JPEG', quality=92)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--keep', action='store_true')
    ap.add_argument('--no_amp', action='store_true')
    ap.add_argument('--n', type=int, default=8, help='시간 재는 사진 수')
    a = ap.parse_args()

    tmp = Path(tempfile.mkdtemp(prefix='submission_smoke_'))
    team = tmp / 'team'
    shutil.copytree(SRC, team, ignore=shutil.ignore_patterns('__pycache__', '*.safetensors'))
    print(f'임시 팀 폴더: {team}')

    # ── 1) 무작위 가중치 만들기 (구조는 config.json 그대로)
    sys.path.insert(0, str(team))
    import json
    from detector import Detector, save_weights, tile_image
    cfg = json.loads((team / 'model' / 'config.json').read_text(encoding='utf-8'))
    torch.manual_seed(0)
    m = Detector(cfg)
    n_params = sum(p.numel() for p in m.parameters())
    n_bb = sum(p.numel() for p in m.backbone.parameters())
    print(f'파라미터 {n_params/1e6:.1f}M (백본 {n_bb/1e6:.1f}M / 그 밖 {(n_params-n_bb)/1e6:.1f}M)')
    save_weights(m, team / 'model' / cfg['weights'])
    size_gb = sum(f.stat().st_size for f in team.rglob('*') if f.is_file()) / 1e9
    print(f'팀 폴더 용량 {size_gb:.3f}GB (한도 1GB)')
    del m

    # ── 2) 조각 개수
    print('조각 수: 가로', tile_image(torch.zeros(1, 3, 768, 1024), 256).shape[0], '/ 세로', tile_image(torch.zeros(1, 3, 1024, 768), 256).shape[0], '(둘 다 12여야 함)')

    # ── 3) load_model / predict
    imgs = tmp / 'imgs'; imgs.mkdir()
    sizes = [(1024, 768), (768, 1024)] * (a.n // 2) + [(1600, 1200), (900, 900)]   # 마지막 둘은 규격 밖(방어 경로)
    paths = []
    for i, s in enumerate(sizes):
        p = imgs / f'img_{i:02d}.jpg'; fake_photo(p, s, i); paths.append(str(p))

    inf = load_inference(team)
    t0 = time.perf_counter(); model = inf.load_model(str(team / 'model')); t1 = time.perf_counter()
    if a.no_amp:
        inf.USE_AMP = False
    print(f'device={model.device_}  백본 가중치 {next(model.backbone.parameters()).dtype}  연산 {"fp32" if (a.no_amp or model.device_.type != "cuda") else "autocast fp16"}  load_model {t1-t0:.1f}초')
    inf.predict(model, paths[:1])            # 워밍업(첫 호출은 느림)
    t2 = time.perf_counter(); probs = inf.predict(model, paths); t3 = time.perf_counter()
    per = (t3 - t2) / len(paths)
    print(f'사진 {len(paths)}장 {t3-t2:.1f}초 → 장당 {per:.2f}초 → 524장 약 {per*524/60:.1f}분 (+로딩 {t1-t0:.0f}초; 한도 5분)')

    ok = True
    def check(cond, msg):
        nonlocal ok
        print(('PASS ' if cond else 'FAIL ') + msg); ok &= bool(cond)
    check(len(probs) == len(paths), '반환 길이 = 입력 길이')
    check(all(isinstance(p, float) and 0.0 <= p <= 1.0 and np.isfinite(p) for p in probs), '모든 값이 0~1 실수(NaN·inf 없음)')
    order = list(reversed(range(len(paths))))
    probs2 = inf.predict(model, [paths[i] for i in order])
    diff = max(abs(probs[i] - probs2[k]) for k, i in enumerate(order))
    check(diff < 1e-5, f'순서를 바꿔도 같은 결과 (최대 차이 {diff:.2e})')
    with torch.no_grad():
        out = model(inf._load_image(paths[0], model.mean_, model.std_).unsqueeze(0).to(model.device_))
    check(abs(float(out['gate'].sum()) - 1.0) < 1e-4 and out['gate'].shape[-1] == 5, f"Gate 합 1, 전문가 5명 (gate={[round(v, 3) for v in out['gate'][0].tolist()]})")

    if a.keep:
        print(f'\n[남김] self_check: py <contest-provided>/self_check.py --team_dir "{team}"')
    else:
        shutil.rmtree(tmp, ignore_errors=True)
    print('\n결과:', '전부 통과' if ok else '실패 있음')
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
