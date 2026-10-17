"""5단계: raw(원본)에서 학습용 세 벌 A/B/C 만들기 — 전략.md 12-6 "규격 비교 실험".

  A  처리 없음 (raw 그대로 복사)                         = 규정만 따른 상태 (학습 형식 규정은 없음)
  B  4:3 크롭 → 1024×768 → JPEG quality=95, 실제·가짜 똑같이 = 설명서의 "두 클래스 공통 크롭·크기 조정·JPEG 저장" 흉내
  C  B와 같되 quality=92                                  = 공개샘플에서 역산한 값까지 맞춤

B·C의 val 부분은 그대로 "평가 형식 검증 세트" 두 벌(q95·q92)이 된다 → 세 버전 모두 이걸로 비교.
결과: ROOT/out/<A|B|C>/ 아래 raw와 같은 폴더 구조 + manifest.csv (file 칸이 그 폴더 기준 경로)

⚠️ A에는 AI Hub 원본이 그대로 들어 있다 → 개인 폴더에만. 공유 Drive에는 B·C만 (전략 결정 ⑧).

사용법:
  py data_pipeline/05_finalize.py ./data              (A·B·C 전부)
  py data_pipeline/05_finalize.py ./data --versions C
"""

import argparse, csv, os, shutil, sys
from concurrent.futures import ProcessPoolExecutor
from common import COLUMNS, read_all

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'tools'))
from normalize import normalize   # tools/normalize.py

QUALITY = {'B': 95, 'C': 92}


def one(v, src, dst):
    if os.path.exists(dst):
        return                                    # 이미 만든 것은 건너뜀 (이어서 하기)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if v == 'A':
        shutil.copyfile(src, dst)
    else:
        normalize(src, dst, quality=QUALITY[v])


def main(root, versions):
    rows = read_all(root)
    print(f'manifest 합계 {len(rows)}장 (실제 {sum(r["label"] == "0" for r in rows)} / 가짜 {sum(r["label"] == "1" for r in rows)})')
    for v in versions:
        out_dir = os.path.join(root, 'out', v)
        jobs, out_rows = [], []
        for r in rows:
            rel = r['file'] if v == 'A' else os.path.splitext(r['file'])[0] + '.jpg'
            jobs.append((v, os.path.join(root, 'raw', r['file']), os.path.join(out_dir, rel)))
            out_rows.append({**r, 'file': rel})
        fails = 0
        with ProcessPoolExecutor() as ex:
            futs = [(j[1], ex.submit(one, *j)) for j in jobs]
            for i, (src, f) in enumerate(futs, 1):
                try:
                    f.result()
                except Exception as e:
                    fails += 1
                    print(f'실패: {src} ({e})')
                if i % 1000 == 0:
                    print(f'[{v}] {i}/{len(jobs)}')
        out_rows = [r for r, j in zip(out_rows, jobs) if os.path.exists(j[2])]   # 실패한 사진은 표에서 뺀다
        with open(os.path.join(out_dir, 'manifest.csv'), 'w', encoding='utf-8', newline='') as f:
            w = csv.DictWriter(f, fieldnames=COLUMNS)
            w.writeheader()
            w.writerows(out_rows)
        print(f'[{v}] 완료 {len(jobs) - fails}장, 실패 {fails}장 → {out_dir}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('root')
    ap.add_argument('--versions', default='A,B,C')
    a = ap.parse_args()
    main(a.root, a.versions.split(','))
