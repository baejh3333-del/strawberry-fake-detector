"""1b단계: AI Hub 596(지능형 수직농장 통합 데이터-딸기)의 **금실** 사진을 실제 풀에 추가한다 (로컬 데스크톱에서 실행).

왜 금실만?  설향(VS_2.설향*)은 분홍색 LED 수직농장 사진이라 공개샘플과 전혀 다르다(전략.md 참고, 10/12 확인).
            금실(AIF005)은 흰 베드·자갈·드립 호스·번호표로 공개샘플 실제 사진과 같은 장면이다.

01_extract_real.py(71451)와 같은 곳에 이어서 쓴다:
- ROOT/raw/real/g596_<이름>.jpg   원본 바이트 그대로 (재인코딩 없음). 이름 앞 g596_ = 71451 사진과 안 겹치게.
- ROOT/manifests/real.csv          줄을 덧붙임 (이미 있는 파일은 건너뜀)
- ROOT/raw/real_boxes.json         부위 bbox [x, y, w, h] 를 합침 (03_make_composites.py가 씀)

같은 위치를 1시간마다 찍은 사진이라 반복이 심하다 →
  group = 촬영 위치(예: C22_B02_003)당 최대 --max_per_group 장,
  촬영 시각 순으로 늘어놓고 **고르게 띄엄띄엄** 뽑아서 생육 단계(정식기~수확기)가 골고루 섞이게 한다.
  group 상한은 모든 tar를 합쳐서 센다(같은 위치가 여러 tar에 걸쳐 있으므로).
  다시 돌려도 안전: 이미 real.csv에 있는 사진은 건너뛰고, 상한은 기존 장수를 빼고 계산한다.

⚠️ AI Hub 이용조건(본인 활용만, 제3자 제공 금지) — ROOT는 개인 폴더. 공유 Drive·GitHub에 올리지 않는다.

사용법 (먼저 --dry_run 으로 뽑힐 장수만 확인):
  py data_pipeline/01b_extract_real_596.py <AI_HUB_DIR> ./data --max_per_group 10 --dry_run
  py data_pipeline/01b_extract_real_596.py <AI_HUB_DIR> ./data --max_per_group 10
"""

import argparse, collections, glob, json, os, tarfile
from common import ManifestWriter, read_manifest, split_of

PREFIX = 'g596_'


def read_labels(lpath):
    """라벨 tar → {이름(확장자 뺀 것): json 딕셔너리}"""
    out = {}
    with tarfile.open(lpath) as tf:
        for m in tf:
            if m.isfile() and m.name.lower().endswith('.json'):
                stem = os.path.splitext(os.path.basename(m.name))[0]
                out[stem] = json.loads(tf.extractfile(m).read().decode('utf-8'))
    return out


def spread(items, k):
    """시각 순으로 정렬된 items에서 k개를 고르게 띄엄띄엄 뽑는다."""
    n = len(items)
    if k >= n:
        return list(items)
    return [items[round(i * (n - 1) / (k - 1))] if k > 1 else items[n // 2] for i in range(k)]


def main(aihub_dir, root, max_per_group, min_side, pattern, dry_run):
    vs_paths = sorted(glob.glob(os.path.join(aihub_dir, '원천데이터', pattern)))
    if not vs_paths:
        raise SystemExit(f'원천 tar를 못 찾음: {os.path.join(aihub_dir, "원천데이터", pattern)}')

    mw = None if dry_run else ManifestWriter(root, 'real')
    real_csv = os.path.join(root, 'manifests', 'real.csv')
    existing = collections.Counter(r['group'] for r in read_manifest(real_csv) if r['group'].startswith(PREFIX))
    done = {r['file'] for r in read_manifest(real_csv)}

    # 1) 후보 모으기 (라벨 JSON만 읽어서 가볍다)
    cands = collections.defaultdict(list)    # group -> [dict]
    skipped_small = 0
    for vs in vs_paths:
        lpath = os.path.join(aihub_dir, '라벨링데이터', os.path.basename(vs).replace('VS_', 'VL_'))
        if not os.path.exists(lpath):
            print(f'라벨 tar 없음, 건너뜀: {os.path.basename(lpath)}')
            continue
        labels = read_labels(lpath)
        with tarfile.open(vs) as tf:
            names = [m.name for m in tf if m.isfile() and m.name.lower().endswith('.jpg')]
        for name in names:
            stem = os.path.splitext(os.path.basename(name))[0]          # C22_B02_006_415743
            lab = labels.get(stem)
            if lab is None:
                continue
            im = lab['images']
            if max(int(im['width']), int(im['height'])) < min_side:      # 긴 변이 작으면 늘려야 해서 흐려짐 → 제외
                skipped_small += 1
                continue
            grp = '_'.join(stem.split('_')[:-1])                         # C22_B02_006 (촬영 위치)
            cands[PREFIX + grp].append(dict(
                tar=vs, name=name, stem=stem, date=im.get('date_captured') or '', stage=im.get('growth_stage'),
                farm=im.get('farm_id'), boxes=[a['bbox'] for a in lab.get('annotations') or []]))

    # 2) 위치(group)마다 상한만큼 시각 순으로 고르게 뽑기
    chosen = []
    for g, items in sorted(cands.items()):
        items.sort(key=lambda d: d['date'])
        todo = [d for d in items if f'real/{PREFIX}{d["stem"]}.jpg' not in done]
        budget = max(0, max_per_group - existing[g])
        chosen += [(g, d) for d in spread(todo, min(budget, len(todo)))]

    stages = collections.Counter(d['stage'] for _, d in chosen)
    splits = collections.Counter(split_of(g) for g, _ in chosen)
    print(f'원천 tar {len(vs_paths)}개 · 후보 위치(group) {len(cands)}개 · 긴 변 {min_side} 미만 제외 {skipped_small}장')
    print(f'이번에 뽑을 사진 {len(chosen)}장 (기존 596 금실 {sum(existing.values())}장은 별도)')
    print('  생육 단계:', dict(stages))
    print('  split:', dict(splits))
    if dry_run:
        print('--dry_run: 아무것도 쓰지 않았음')
        return

    # 3) 꺼내서 저장 (tar마다 한 번만 연다)
    out_dir = os.path.join(root, 'raw', 'real')
    os.makedirs(out_dir, exist_ok=True)
    boxes_path = os.path.join(root, 'raw', 'real_boxes.json')
    boxes = json.load(open(boxes_path, encoding='utf-8')) if os.path.exists(boxes_path) else {}
    by_tar = collections.defaultdict(list)
    for g, d in chosen:
        by_tar[d['tar']].append((g, d))
    n = 0
    try:
        for vs, lst in by_tar.items():
            with tarfile.open(vs) as tf:
                for g, d in lst:
                    source_id = PREFIX + d['stem']
                    rel = f'real/{source_id}.jpg'
                    with open(os.path.join(root, 'raw', rel), 'wb') as f:
                        f.write(tf.extractfile(d['name']).read())        # 바이트 그대로 = 원본 보존
                    boxes[source_id] = d['boxes']
                    mw.add(file=rel, label=0, type='real', family='real', generator='-', source_id=source_id,
                           group=g, split=split_of(g),
                           params={'dataset': 'aihub596', 'variety': '금실', 'farm': d['farm'],
                                   'growth_stage': d['stage'], 'date_captured': d['date']})
                    n += 1
            print(f'  {os.path.basename(vs)}: 누적 {n}장')
    finally:
        json.dump(boxes, open(boxes_path, 'w', encoding='utf-8'))        # 중간에 끊겨도 bbox는 남게
        mw.close()
    print(f'완료: {n}장 추가 → {out_dir}, 표: {mw.path}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('aihub_dir', help='596 Validation 폴더 (원천데이터·라벨링데이터가 들어 있는 곳)')
    ap.add_argument('root', help='개인 작업 폴더 (ROOT)')
    ap.add_argument('--max_per_group', type=int, default=10, help='촬영 위치(group)당 최대 장수 (기본 10)')
    ap.add_argument('--min_side', type=int, default=1024, help='긴 변이 이보다 작은 사진은 제외 (기본 1024)')
    ap.add_argument('--pattern', default='VS_1.금실*.tar', help='원천 tar 이름 패턴. 설향(VS_2.*)은 쓰지 않는다')
    ap.add_argument('--dry_run', action='store_true', help='뽑힐 장수만 보고 아무것도 쓰지 않음')
    a = ap.parse_args()
    main(a.aihub_dir, a.root, a.max_per_group, a.min_side, a.pattern, a.dry_run)
