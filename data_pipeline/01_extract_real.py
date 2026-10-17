"""1단계: AI Hub zip에서 실제 사진을 뽑아 ROOT/raw/real/ 에 **원본 그대로** 저장 (로컬 데스크톱에서 실행).

- zip을 통째로 풀지 않고 필요한 사진만 꺼낸다 (원천 zip 5개 약 25GB).
- 다시 저장(재인코딩)하지 않는다 — 바이트 그대로 복사. 규격 맞추기는 05_finalize.py에서.
- 병해 5종(정상·역병·시들음병·잎끝마름·황화)에서 종류마다 --per_class 장씩 무작위로.
  --max_per_group N 을 주면 같은 포기에서 종류당 N장까지만 (반복 촬영 줄이기. 안 주면 제한 없음).
- 라벨 zip의 부위 사각형(bbox)도 같이 뽑아 raw/real_boxes.json 에 저장 → 03_make_composites.py가 씀.
- 파일 이름은 한글 대신 영문 ID: c<종류번호>_<포기번호>_<촬영시각>.jpg (Colab·Windows 인코딩 문제 방지)

⚠️ AI Hub 이용조건(본인 활용만, 제3자 제공 금지) — ROOT는 개인 폴더. 공유 Drive·GitHub에 올리지 않는다.

사용법:
  py data_pipeline/01_extract_real.py <AI_HUB_DIR> ./data --per_class 1500
"""

import argparse, glob, json, os, random, re, zipfile
from common import ManifestWriter, split_of


def fixed_name(info):
    """AI Hub zip은 파일 이름이 cp949(한글 윈도우)로 들어 있는데 파이썬은 cp437로 읽는다 → 되돌린다."""
    if info.flag_bits & 0x800:          # UTF-8로 저장된 경우는 그대로
        return info.filename
    try:
        return info.filename.encode('cp437').decode('cp949')
    except UnicodeError:
        return info.filename


def parse(name, cls):
    """'딸기_설향_정상_13_001_220909091738.jpg' → group '13_001', source_id 'c01_13_001_220909091738'"""
    stem = os.path.splitext(os.path.basename(name))[0]
    p = stem.split('_')
    group = f'{p[-3]}_{p[-2]}'
    return f'c{cls}_{group}_{p[-1]}', group, stem


def main(aihub_dir, root, per_class, seed, max_per_group=None, classes=None):
    rng = random.Random(seed)
    out_dir = os.path.join(root, 'raw', 'real')
    os.makedirs(out_dir, exist_ok=True)
    mw = ManifestWriter(root, 'real')
    boxes_path = os.path.join(root, 'raw', 'real_boxes.json')
    boxes = json.load(open(boxes_path, encoding='utf-8')) if os.path.exists(boxes_path) else {}

    for zpath in sorted(glob.glob(os.path.join(aihub_dir, '01.원천데이터', '*.zip'))):
        cls = re.search(r'_(\d\d)\.[^.]+\.zip$', os.path.basename(zpath)).group(1)   # 01~05
        if classes and cls not in classes:  # 필요한 종류만 (예: 01 = 정상)
            continue
        zf = zipfile.ZipFile(zpath)
        items = [i for i in zf.infolist() if i.filename.lower().endswith('.jpg')]
        rng.shuffle(items)
        if max_per_group:                   # 같은 포기에서 최대 N장 (반복 촬영 사진 줄이기)
            counts, picked = {}, []
            for info in items:
                g = parse(fixed_name(info), cls)[1]
                if counts.get(g, 0) >= max_per_group:
                    continue
                counts[g] = counts.get(g, 0) + 1
                picked.append(info)
                if len(picked) >= per_class:
                    break
        else:
            picked = items[:per_class]

        # 같은 종류의 라벨 zip (VS_ → VL_)
        lpath = os.path.join(aihub_dir, '02.라벨링데이터', os.path.basename(zpath).replace('VS_', 'VL_'))
        lz = zipfile.ZipFile(lpath) if os.path.exists(lpath) else None
        labels = {}
        if lz:
            for i in lz.infolist():
                if i.filename.endswith('.json'):
                    labels[os.path.splitext(os.path.basename(fixed_name(i)))[0]] = i

        n = 0
        for info in picked:
            source_id, group, stem = parse(fixed_name(info), cls)
            rel = f'real/{source_id}.jpg'
            if rel in mw.done:
                continue
            with open(os.path.join(root, 'raw', rel), 'wb') as f:
                f.write(zf.read(info))                     # 바이트 그대로 = 원본 보존
            if stem in labels:
                d = json.loads(lz.read(labels[stem]).decode('utf-8'))
                boxes[source_id] = [a['bbox'] for a in d.get('annotations', [])]
            mw.add(file=rel, label=0, type='real', family='real', generator='-', source_id=source_id,
                   group=group, split=split_of(group), params={'aihub_class': cls})
            n += 1
        json.dump(boxes, open(boxes_path, 'w', encoding='utf-8'))   # 종류마다 저장 (중간에 끊겨도 남게)
        print(f'종류 {cls}: {n}장 추가 (zip 안 {len(items)}장 중)')

    mw.close()
    print(f'완료 → {out_dir}, 표: {mw.path}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('aihub_dir', help='AI Hub Validation 폴더 (01.원천데이터·02.라벨링데이터가 들어 있는 곳)')
    ap.add_argument('root', help='개인 작업 폴더 (ROOT)')
    ap.add_argument('--per_class', type=int, default=1500)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--max_per_group', type=int, default=None,
                    help='같은 포기(group)에서 종류당 최대 몇 장까지 (기본: 제한 없음). 새 데이터를 받을 때만 쓴다')
    ap.add_argument('--classes', nargs='+', default=None, help='뽑을 종류 번호만 (예: 01 = 정상). 기본: 전부')
    a = ap.parse_args()
    main(a.aihub_dir, a.root, a.per_class, a.seed, a.max_per_group, a.classes)
