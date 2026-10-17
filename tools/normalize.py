"""학습 이미지를 평가 이미지와 같은 규격으로 통일하는 스크립트 (결정 ③).

규격: RGB, 1024×768(가로) 또는 768×1024(세로), JPEG quality=92(4:2:0), EXIF 없음.
      = 공개샘플 21장과 같은 규격 (docs/전략.md 3-3).

하는 일 (사진 한 장마다)
  1) 카메라 회전 정보(EXIF)대로 바로 세우기
  2) 가로가 길면 4:3, 세로가 길면 3:4 비율로 가운데를 잘라내기
  3) 1024×768 / 768×1024로 크기 맞추기
  4) quality=92 JPEG로 저장 (EXIF는 저장하지 않으므로 자동으로 빠진다)

⚠️ 실제·가짜 모두 이 스크립트를 똑같이 통과시킨다. 가짜는 만드는 작업을 다 끝낸 뒤 마지막에 돌린다.
⚠️ 입력 폴더는 읽기만 한다(원본 보존). 결과는 출력 폴더에 같은 하위 폴더 구조로 저장.

사용법 (PIL 필요 → 데스크톱에서):
  python tools/normalize.py <입력폴더> <출력폴더>
확인:
  python tools/jpegscan.py <출력폴더의 하위폴더>   → 전부 (1024,768) 또는 (768,1024), q=6c3c42bf87 이어야 함
"""

import os, sys
from concurrent.futures import ProcessPoolExecutor
from PIL import Image, ImageOps

EXTS = ('.jpg', '.jpeg', '.png', '.webp', '.bmp')


def normalize(src, dst, quality=92):
    im = ImageOps.exif_transpose(Image.open(src)).convert('RGB')
    w, h = im.size
    W, H = (1024, 768) if w >= h else (768, 1024)
    # 목표 비율(W:H)에 맞게 가운데를 잘라낸다
    if w * H > h * W:              # 목표보다 옆으로 넓다 → 좌우를 자름
        cw, ch = h * W // H, h
    else:                          # 목표보다 위아래로 길다 → 위아래를 자름
        cw, ch = w, w * H // W
    left, top = (w - cw) // 2, (h - ch) // 2
    # ponytail: 리사이즈 방식(LANCZOS)은 대회 쪽 방식을 몰라 고른 것. 리더보드 차이가 크면 BICUBIC 등과 비교
    im = im.crop((left, top, left + cw, top + ch)).resize((W, H), Image.LANCZOS)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    im.save(dst, 'JPEG', quality=quality)


def main(src_dir, dst_dir):
    jobs = []
    for root, _, files in os.walk(src_dir):
        for fn in files:
            if fn.lower().endswith(EXTS):
                rel = os.path.relpath(os.path.join(root, fn), src_dir)
                jobs.append((os.path.join(src_dir, rel), os.path.join(dst_dir, os.path.splitext(rel)[0] + '.jpg')))
    print(f'{len(jobs)}장 변환 시작')
    fails = 0
    with ProcessPoolExecutor() as ex:   # CPU 코어를 전부 써서 병렬 처리
        futures = [(s, ex.submit(normalize, s, d)) for s, d in jobs]
        for i, (s, f) in enumerate(futures, 1):
            try:
                f.result()
            except Exception as e:     # 깨진 파일 하나 때문에 전체가 멈추지 않도록 건너뛰고 기록
                fails += 1
                print(f'실패: {s} ({e})')
            if i % 1000 == 0:
                print(f'{i}/{len(jobs)}')
    print(f'완료: {len(jobs) - fails}장 성공, {fails}장 실패')


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
