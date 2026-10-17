"""데이터 생성 파이프라인 공통 도구 — 폴더 규칙, manifest(꼬리표 표) 읽기/쓰기, train/val 나누기.

폴더 규칙 (ROOT = 개인 작업 폴더. 로컬 예: ./data, Colab 예: /content/drive/MyDrive/work)
  ROOT/raw/real/...           AI Hub 원본 그대로 (재인코딩 없음)            ← 개인 보관만 (AI Hub 이용조건)
  ROOT/raw/fake/<종류>/...    가짜 원본(raw). 가공 전 상태 그대로 PNG 등
  ROOT/manifests/<단계>.csv   단계마다 자기 표를 따로 쓴다 (서로 덮어쓰지 않게)
  ROOT/out/A|B|C/...          05_finalize.py 결과 (전략.md 12-6의 규격 비교 실험)

manifest 한 줄 = 사진 한 장. 칸 설명은 COLUMNS 참고 (전략.md 12-3).
"""

import csv, glob, hashlib, json, os

COLUMNS = [
    'file',        # ROOT/raw 기준 상대경로 (예: real/c01_13_001_220909091738.jpg)
    'label',       # 0 = 실제, 1 = 가짜
    'type',        # 공개샘플 표기와 같게: real / AI_gen / composite / filter / full_style / partial_gen / partial_style / vae_recon
    'family',      # 게이팅 정답: real / E1~E5 (전략.md 12-2 표)
    'generator',   # 무엇으로 만들었나 (예: filter-blur, composite-bgswap, flux-vae)
    'source_id',   # 재료가 된 실제 사진 ID (전체 생성이면 빈칸)
    'group',       # train/val을 나누는 단위 = 같은 포기(식물) ID. 같은 포기의 다른 날 사진이 양쪽에 걸치지 않게
    'split',       # train / val
    'params',      # 강도 등 만든 설정 (JSON 문자열) — 재현용
]

VAL_PERCENT = 15   # 포기 단위로 15%를 검증용으로


def split_of(group):
    """같은 group(포기 ID)은 언제 어디서 돌려도 같은 split이 나오도록 해시로 정한다."""
    h = int(hashlib.md5(group.encode('utf-8')).hexdigest(), 16) % 100
    return 'val' if h < VAL_PERCENT else 'train'


def read_manifest(path):
    if not os.path.exists(path):
        return []
    with open(path, encoding='utf-8', newline='') as f:
        return list(csv.DictReader(f))


def read_all(root):
    """ROOT/manifests/*.csv 전부 합쳐서 읽기."""
    rows = []
    for p in sorted(glob.glob(os.path.join(root, 'manifests', '*.csv'))):
        rows += read_manifest(p)
    return rows


class ManifestWriter:
    """한 단계의 manifest에 줄을 추가한다. 이미 적힌 파일은 done에 들어 있어 건너뛸 수 있다(중단 후 이어서 하기)."""

    def __init__(self, root, step):
        os.makedirs(os.path.join(root, 'manifests'), exist_ok=True)
        self.path = os.path.join(root, 'manifests', f'{step}.csv')
        self.done = {r['file'] for r in read_manifest(self.path)}
        new = not os.path.exists(self.path)
        self.f = open(self.path, 'a', encoding='utf-8', newline='')
        self.w = csv.DictWriter(self.f, fieldnames=COLUMNS)
        if new:
            self.w.writeheader()

    def add(self, **row):
        if isinstance(row.get('params'), dict):
            row['params'] = json.dumps(row['params'], ensure_ascii=False)
        self.w.writerow({c: row.get(c, '') for c in COLUMNS})
        self.f.flush()   # Colab 세션이 끊겨도 여기까지는 남게
        self.done.add(row['file'])

    def close(self):
        self.f.close()


def center_box(w, h):
    """normalize.py가 4:3(또는 3:4)로 가운데를 잘라낼 때 **남는 영역** (left, top, right, bottom).
    부분 합성은 이 안에 넣어야 잘려 나가지 않는다."""
    W, H = (4, 3) if w >= h else (3, 4)
    if w * H > h * W:
        cw, ch = h * W // H, h
    else:
        cw, ch = w, w * H // W
    left, top = (w - cw) // 2, (h - ch) // 2
    return left, top, left + cw, top + ch


def real_rows(root, split=None):
    rows = [r for r in read_manifest(os.path.join(root, 'manifests', 'real.csv')) if r['label'] == '0']
    return [r for r in rows if split is None or r['split'] == split]


def fake_usage(root, exclude_step=None):
    """원본 사진(source_id)마다 '이걸 재료로 만든 가짜가 몇 장인지' 센다.
    manifests/*.csv 와 manifests_remote/*.csv(Drive에서 받아 온 표 — 사진은 로컬에 없고 표만 있음)를 함께 본다.
    exclude_step: 지금 돌리는 단계 자신의 표는 뺀다 → 이어서 하기 때 순서가 바뀌지 않게."""
    usage = {}
    for folder in ('manifests', 'manifests_remote'):
        for p in glob.glob(os.path.join(root, folder, '*.csv')):
            if exclude_step and os.path.basename(p) == f'{exclude_step}.csv':
                continue
            for r in read_manifest(p):
                if r['label'] == '1' and r['source_id']:
                    usage[r['source_id']] = usage.get(r['source_id'], 0) + 1
    return usage


def least_used_first(rows, usage, rng):
    """덜 쓰인 원본부터 앞에 오게 정렬한다. 같은 횟수끼리는 고정 시드로 섞는다(다시 돌려도 같은 순서)."""
    rows = list(rows)
    rng.shuffle(rows)
    rows.sort(key=lambda r: usage.get(r['source_id'], 0))    # 파이썬 정렬은 안정적 → 같은 횟수는 섞인 순서 유지
    return rows


class Syncer:
    """Colab처럼 Drive 연결이 끊길 수 있는 곳에서 쓴다: 결과는 빠른 로컬 디스크(work)에 먼저 쓰고,
    일정 장수마다 Drive(dest)로 복사한다. Drive가 끊겨도 생성은 멈추지 않고 다음 복사 때 다시 시도한다.

    - 복사 순서: PNG 먼저, 표(manifest) 나중 → 표에 적힌 파일은 항상 Drive에 이미 있다.
    - 시작할 때 Drive의 표를 받아 와서 '이미 만든 번호'를 이어받는다(중단 후 이어서 하기).
    - 로컬 파일은 지우지 않는다(Drive 복사 실패가 계속돼도 work 폴더에 남아 있다)."""

    def __init__(self, work, dest, name):
        self.work, self.dest, self.name = work, dest, name
        self.synced = set()
        self.local_man = os.path.join(work, 'manifests', f'{name}.csv')
        self.dest_man = os.path.join(dest, 'manifests', f'{name}.csv')

    def pull_manifest(self):
        """Drive의 표가 로컬보다 길면 로컬로 가져온다."""
        import shutil
        try:
            if os.path.exists(self.dest_man):
                n_dest = len(read_manifest(self.dest_man))
                n_local = len(read_manifest(self.local_man))
                if n_dest > n_local:
                    os.makedirs(os.path.dirname(self.local_man), exist_ok=True)
                    shutil.copy2(self.dest_man, self.local_man)
                self.synced = {r['file'] for r in read_manifest(self.dest_man)}
        except OSError as e:
            print(f'[sync] Drive 표를 못 읽음(무시하고 로컬에서 시작): {e}', flush=True)

    def push(self, retries=1, wait=30):
        """새 PNG → 표 순서로 Drive에 복사. 성공하면 True."""
        import shutil, time
        for attempt in range(retries):
            try:
                rows = read_manifest(self.local_man)
                for r in rows:
                    if r['file'] in self.synced:
                        continue
                    src = os.path.join(self.work, 'raw', r['file'])
                    dst = os.path.join(self.dest, 'raw', r['file'])
                    os.makedirs(os.path.dirname(dst), exist_ok=True)
                    shutil.copy2(src, dst)
                    self.synced.add(r['file'])
                os.makedirs(os.path.dirname(self.dest_man), exist_ok=True)
                tmp = self.dest_man + '.tmp'
                shutil.copy2(self.local_man, tmp)
                os.replace(tmp, self.dest_man)
                return True
            except OSError as e:
                print(f'[sync] Drive 복사 실패({attempt + 1}/{retries}): {e}', flush=True)
                if attempt + 1 < retries:
                    time.sleep(wait)
        return False
