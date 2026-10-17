"""JPEG 파일의 저장 규격을 확인하는 도구 (추가 패키지 없이 동작).

이미지 폴더를 넘기면 파일마다 크기·채널·메타데이터·JPEG 양자화 테이블 해시를 출력합니다.
양자화 테이블 해시가 같으면 같은 인코더·같은 품질로 저장된 것입니다.

용도
  1) 공개샘플의 저장 규격 확인 (실제·가짜가 같은 설정인지)
  2) 우리가 만든 학습 데이터가 같은 규격으로 통일됐는지 검사

사용법 (저장소 루트에서):
  py tools/jpegscan.py "<sample_dir>"

자세한 배경은 docs/전략.md 3-3 참고.
"""

import os, struct, hashlib, sys

d = sys.argv[1]
for fn in sorted(os.listdir(d)):
    if not fn.lower().endswith(('.jpg', '.jpeg')):
        continue
    b = open(os.path.join(d, fn), 'rb').read()
    i = 2
    dqt = []
    apps = []
    sof = None
    prog = None
    while i < len(b) - 1:
        if b[i] != 0xFF:
            i += 1
            continue
        m = b[i + 1]
        if m in (0xD8, 0xD9, 0x01) or 0xD0 <= m <= 0xD7:
            i += 2
            continue
        L = struct.unpack('>H', b[i + 2:i + 4])[0]
        seg = b[i + 4:i + 2 + L]
        if m == 0xDB:                          # 양자화 테이블 (= 저장 품질)
            dqt.append(seg.hex())
        elif 0xE0 <= m <= 0xEF:                # APPn (EXIF·JFIF 등 메타데이터)
            apps.append((hex(m), seg[:20].split(b'\x00')[0][:18]))
        elif m in (0xC0, 0xC1, 0xC2):          # SOF (크기·채널)
            prog = (m == 0xC2)
            h, w = struct.unpack('>HH', seg[1:5])
            sof = (w, h, seg[5])
        elif m == 0xDA:                        # 이미지 데이터 시작 → 여기서 끝
            break
        i += 2 + L
    qh = hashlib.md5(''.join(dqt).encode()).hexdigest()[:10]
    print(f"{fn} {sof} prog={prog} ntab={len(dqt)} q={qh} apps={apps} size={len(b)//1024}KB")
