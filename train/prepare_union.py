"""기존 학습자료(data_probe)와 새로 받은 자료(extra_real_fake, 2026-10-08 팀 구성)를 합쳐 train.py가 읽는 목록표 한 벌로 만든다.

넣고 뺀 기준 (CLAUDE.md '학습자료' 절, docs/추가생성_지시서.md):
  - pool == provided (대회 배포 실제 샘플 PNG 30장)와 그 파생 가짜는 학습에서 뺀다. 제공 자료는 학습에 쓰지 않는다는 팀 방침(전략 6절).
    실제 30장은 split='extra'로 따로 남겨 '실제 오탐 점검용'으로만 쓴다(diag.py --split extra).
  - license == flux-1-dev-non-commercial (FLUX.1 Kontext/Krea dev) 가짜는 뺀다. 비상업 라이선스 모델은 쓰지 않는 팀 방침.
  - edit_area_exceeded == True (바뀐 영역이 60%를 넘은 '부분' 편집)는 뺀다.
  - 비교용 세트(실제 120장에 모든 생성기를 적용한 aihub305_normal)는 같은 원본·같은 유형의 가짜를 --per_src_type장까지만 둔다. 같은 장면이 너무 반복되지 않게.
  - 합성(plant_transplant)은 다른 포기의 배경을 쓰므로, 원본 포기와 상대 포기의 split이 다르면 뺀다(검증 누출 방지).
나누기: group 단위(같은 포기의 사진이 train·val에 걸치지 않게).
  - AI Hub 305 포기는 기존 자료(AI Hub 305 Validation)에 같은 포기 번호가 있으면 그쪽 split을 따른다(기존 val 점수를 깨끗하게 유지).
  - 나머지는 'n:'+group_id를 해시해 15%를 val로.
실제 사진 쪽 가짜의 source_id: 글→이미지(method C)는 원본 픽셀이 없으므로 비운다(3단계 쌍 배치가 엉뚱한 쌍을 만들지 않게).

사용법 (Colab):
  python train/prepare_union.py --old /content/data_probe --new "/content/drive/MyDrive/work/plus/extra_real_fake" --out /content/data_U
  --dry_run 이면 장수만 세고 아무것도 복사하지 않는다. 사진이 아직 덜 올라왔으면 --allow_missing 으로 있는 것만 쓴다(최종 학습엔 쓰지 말 것).
"""
import argparse, collections, csv, hashlib, json, os, re, subprocess, time
from concurrent.futures import ThreadPoolExecutor

COLUMNS = ['file', 'label', 'type', 'family', 'generator', 'source_id', 'group', 'split', 'params']
VAL_PERCENT = 15
FAMILY_OF_TYPE = {'AI_gen': 'E2', 'full_style': 'E1', 'partial_gen': 'E4', 'partial_style': 'E4', 'composite': 'E5', 'filter': 'E5'}   # none 모드 호환값. type8에서는 AI_gen만 이 값을 그대로 쓴다(DiT·flow 계열 = E2)
DEV_LICENSE = 'flux-1-dev-non-commercial'


def split_of(group):
    return 'val' if int(hashlib.md5(group.encode('utf-8')).hexdigest(), 16) % 100 < VAL_PERCENT else 'train'


def read_csv(path):
    with open(path, encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def plant_key(group_id):
    """'aihub305_normal:딸기_설향_정상_19_004' → '19_004' (기존 자료의 group 형식과 같다)"""
    m = re.search(r'_(\d+_\d+)$', group_id)
    return m.group(1) if m else None


def tag_of(r):
    m = r['method']
    if m == 'B':
        return 's' + str(int(round(float(r['strength']) * 100))).zfill(3) if r['strength'] else 'img2img'
    if m == 'C':
        return 't2i'
    if m == 'E':
        return r['fake_type'] or r['competition_type']
    if m == 'F':
        return r['filter_kind']
    if m == 'M':
        return r['composite_kind']
    return m


def cp_retry(src, dst, tries=4, timeout=60):
    """Drive(느림)에서 로컬로 한 장 복사. 시간제한과 재시도, 다 받은 파일만 이름을 바꾼다."""
    if os.path.exists(dst) and os.path.getsize(dst) > 0:
        return None
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    for a in range(tries):
        try:
            subprocess.run(['cp', src, dst + '.part'], check=True, timeout=timeout, stderr=subprocess.DEVNULL)
            os.replace(dst + '.part', dst)
            return None
        except Exception as e:
            try:
                os.remove(dst + '.part')
            except OSError:
                pass
            if a == tries - 1:
                return (src, repr(e))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--old', required=True, help='기존 자료 폴더(manifest.csv + real/ fake/). 예: /content/data_probe')
    ap.add_argument('--new', required=True, help='새 자료 폴더(manifest.csv + real/ fake/)')
    ap.add_argument('--out', required=True)
    ap.add_argument('--per_src_type', type=int, default=2)
    ap.add_argument('--reals_only', action='store_true', help='새 자료에서 실제 사진만 쓴다(가짜는 하나도 안 합침). 같은 장면의 가짜가 없어 "이 장면 = 실제" 지름길이 생길 수 있으니 diag로 꼭 확인')
    ap.add_argument('--dry_run', action='store_true')
    ap.add_argument('--allow_missing', action='store_true', help='새 자료 중 아직 안 올라온 사진은 건너뜀(시험용)')
    ap.add_argument('--threads', type=int, default=16)
    a = ap.parse_args()

    old = read_csv(os.path.join(a.old, 'manifest.csv'))
    new = read_csv(os.path.join(a.new, 'manifest.csv'))
    plant_split = {}
    for r in old:
        if r['label'] == '0' and re.fullmatch(r'\d+_\d+', r['group'] or ''):
            plant_split[r['group']] = r['split']

    def split_for(gid):
        if gid.startswith('aihub305'):
            k = plant_key(gid)
            if k in plant_split:
                return plant_split[k]
        return split_of('n:' + gid)

    # 1) 새 자료 거르기
    rows, drop = [], collections.Counter()
    for r in new:
        if a.reals_only and r['label'] == 'fake':
            drop['가짜(--reals_only)'] += 1
            continue
        if r['pool'] == 'provided':
            if r['label'] == 'real':
                rows.append((r, 'extra'))
            else:
                drop['provided 파생 가짜'] += 1
            continue
        if r['label'] == 'fake':
            if r['license'] == DEV_LICENSE:
                drop['FLUX.1-dev 비상업'] += 1; continue
            if r['edit_area_exceeded'] in ('True', 'true', '1'):
                drop['편집 영역 60% 초과'] += 1; continue
        rows.append((r, split_for(r['group_id'])))

    # 2) 합성: 상대 포기의 split이 다르면 뺀다
    kept = []
    for r, sp in rows:
        if r['label'] == 'fake' and r['composite_kind'] == 'plant_transplant' and r['partner_group_id'] and sp != 'extra':
            if split_for(r['partner_group_id']) != sp:
                drop['합성 상대 포기 split 다름'] += 1; continue
        kept.append((r, sp))
    rows = kept

    # 3) 비교용 세트 상한: 같은 원본·같은 유형 가짜는 per_src_type장까지, 생성기가 겹치지 않게 먼저 고른다
    comp = collections.defaultdict(list)
    for i, (r, sp) in enumerate(rows):
        if r['label'] == 'fake' and r['pool'] == 'aihub305_normal':
            comp[(r['source_id'], r['competition_type'])].append(i)
    remove = set()
    for key, idx in comp.items():
        idx.sort(key=lambda i: hashlib.md5(rows[i][0]['image_id'].encode()).hexdigest())
        chosen, used = [], set()
        for i in idx:
            g = rows[i][0]['generator']
            if g not in used and len(chosen) < a.per_src_type:
                chosen.append(i); used.add(g)
        for i in idx:
            if i not in chosen and len(chosen) < a.per_src_type:
                chosen.append(i)
        remove |= set(idx) - set(chosen)
    drop['비교용 세트 상한'] = len(remove)
    rows = [x for i, x in enumerate(rows) if i not in remove]

    # 4) 있는 사진만(허용 시)
    miss = [r['file'] for r, _ in rows if not os.path.exists(os.path.join(a.new, r['file']))]
    if miss and not a.allow_missing:
        raise SystemExit(f'새 자료 사진이 {len(miss)}장 없음 (예: {miss[:2]}). 업로드가 끝났는지 확인하거나 --allow_missing으로 있는 것만 쓰세요.')
    if miss:
        ms = set(miss)
        rows = [(r, sp) for r, sp in rows if r['file'] not in ms]
        drop['아직 없는 사진'] = len(miss)

    # 5) 목록표 행 만들기
    out_rows = []
    for r, sp in rows:
        real = r['label'] == 'real'
        typ = 'real' if real else r['competition_type']
        gen = '-' if real else f'n-{r["generator"]}-{tag_of(r)}'
        src = r['source_id'] if (real or r['method'] != 'C') else ''
        params = {'set': 'team-1008', 'pool': r['pool'], 'license': r['license'], 'method': r['method']}
        for k in ('strength', 'edit_area_frac', 'filter_kind', 'filter_param', 'composite_kind', 'partner_group_id'):
            if r.get(k):
                params[k] = r[k]
        out_rows.append({'file': 'new/' + r['file'], 'label': 0 if real else 1, 'type': typ, 'family': 'real' if real else FAMILY_OF_TYPE[typ],
                         'generator': gen, 'source_id': src, 'group': 'n:' + r['group_id'], 'split': sp, 'params': json.dumps(params, ensure_ascii=False)})

    c_split = collections.Counter((x['split'], x['label']) for x in out_rows)
    print('새 자료에서 뺀 것:', dict(drop))
    print('새 자료 사용:', {f'{s}/{"real" if l == 0 else "fake"}': n for (s, l), n in sorted(c_split.items())})
    print('새 가짜 유형별(train+val):', dict(collections.Counter(x['type'] for x in out_rows if x['label'] == 1)))
    print('새 가짜 생성기별(train+val):', dict(collections.Counter(x['generator'] for x in out_rows if x['label'] == 1)))
    val_old = collections.Counter(r['split'] for r in old)
    print('기존:', dict(val_old), '→ 합친 뒤 train/val:', dict(collections.Counter(r['split'] for r in old) + collections.Counter(x['split'] for x in out_rows)))
    if a.dry_run:
        print('--dry_run: 아무것도 만들지 않았음')
        return

    # 6) 폴더 만들기: 기존 자료는 링크, 새 사진은 로컬로 복사
    os.makedirs(os.path.join(a.out, 'fake'), exist_ok=True)
    for name in ('real',):
        dst = os.path.join(a.out, name)
        if not os.path.lexists(dst):
            os.symlink(os.path.realpath(os.path.join(a.old, name)), dst)
    for d in os.listdir(os.path.join(a.old, 'fake')):
        dst = os.path.join(a.out, 'fake', d)
        if not os.path.lexists(dst):
            os.symlink(os.path.realpath(os.path.join(a.old, 'fake', d)), dst)
    todo = [(os.path.join(a.new, x['file'][len('new/'):]), os.path.join(a.out, x['file'])) for x in out_rows]
    done, fail, t0 = [0], [], time.time()

    def one(j):
        r = cp_retry(*j)
        if r:
            fail.append(r)
        done[0] += 1
        if done[0] % 500 == 0:
            print(f'  새 사진 복사 {done[0]}/{len(todo)} ({time.time() - t0:.0f}초) 실패 {len(fail)}', flush=True)
    with ThreadPoolExecutor(a.threads) as ex:
        list(ex.map(one, todo))
    assert not fail, f'복사 실패 {len(fail)}장: {fail[:3]}'
    with open(os.path.join(a.out, 'manifest.csv'), 'w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        for r in old:
            w.writerow({c: r.get(c, '') for c in COLUMNS})
        w.writerows(out_rows)
    print(f'DONE {a.out}/manifest.csv: 기존 {len(old)} + 새 {len(out_rows)} = {len(old) + len(out_rows)}행 ({time.time() - t0:.0f}초)')


if __name__ == '__main__':
    main()
