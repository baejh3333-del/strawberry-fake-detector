"""12단계: 부분 스타일 변환(⑥) 가짜 만들기 — 게이팅 family E4(부분 편집). GPU 불필요, 장당 1초 이내.

11_sdxl_img2img.py가 만든 전체 스타일 변환 사진에서 **일부 영역만** 원본 사진 위에 붙인다. 나머지는 원본 그대로다.
영역은 10_sdxl_inpaint.py와 같은 방식(AI Hub 부위 사각형 1~2개, 면적 1~25%, 가장자리 15% 여백)으로 고르고, 경계는 살짝 흐린다.

- 필요한 것: manifests/sdxl_img2img.csv + raw/fake/sdxl_img2img/*.png (11단계 결과), raw/real/, raw/real_boxes.json, manifests/real.csv
- 11단계 사진은 원본을 규격(1024×768/768×1024)으로 맞춘 뒤 변환했으므로, 같은 규격 원본을 다시 만들어 크기를 맞춘다(같은 함수).
- 결과: raw/fake/partial_style/*.png, manifests/partial_style.csv (type=partial_style, family=E4)

--src / --name (10/4 추가): 재료로 쓸 전체 스타일 결과의 표 이름과, 만들 부분 스타일의 표 이름. 기본은 사실적판(sdxl_img2img → partial_style).
  화풍판은 재료마다 따로 돌린다: --src sdxl_paint --name partial_paint_sdxl / --src classic_style --name partial_paint_classic /
  --src klein_style --name partial_paint_klein. 재료 표의 모든 사진이 같은 원본 규격(1024×768/768×1024)이어야 한다(15·16·11단계가 그렇게 만든다).

Colab 실행 (11단계 결과가 개인 Drive에 있을 때):
  !python 12_partial_style.py /content/rr --styled_root /content/drive/MyDrive/work --work /content/work_ps --n 10 --sync_to /content/drive/MyDrive/work
  !python 12_partial_style.py /content/rr --styled_root /content/drive/MyDrive/work --src classic_style --name partial_paint_classic --work /content/work_ps_c --n 330 --sync_to /content/drive/MyDrive/work
"""

import argparse, importlib.util, json, os, random, time
from PIL import Image
from common import ManifestWriter, Syncer, read_manifest

NAME = 'partial_style'
SRC_NAME = 'sdxl_img2img'


def _load(name):
    """숫자로 시작하는 파일 이름은 import로 못 불러서, 경로로 읽는다."""
    spec = importlib.util.spec_from_file_location(name, os.path.join(os.path.dirname(os.path.abspath(__file__)), name + '.py'))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main(real_root, styled_root, work, n, seed, sync_to, sync_every, src_name=SRC_NAME, name=NAME):
    inp = _load('10_sdxl_inpaint')   # standardize, transform_boxes, make_mask, paste_masked를 그대로 재사용
    boxes_all = json.load(open(os.path.join(real_root, 'raw', 'real_boxes.json'), encoding='utf-8'))
    styled = read_manifest(os.path.join(styled_root, 'manifests', f'{src_name}.csv'))
    rng = random.Random(seed)
    rng.shuffle(styled)

    syncer = Syncer(work, sync_to, name) if sync_to else None
    if syncer:
        syncer.pull_manifest()
    mw = ManifestWriter(work, name)
    os.makedirs(os.path.join(work, 'raw', 'fake', name), exist_ok=True)
    made, skipped, t0 = 0, 0, time.time()
    for row in styled:
        if made >= n:
            break
        rel = f'fake/{name}/{os.path.splitext(os.path.basename(row["file"]))[0].replace("__i2i", "__ps")}.png'
        if rel in mw.done:
            made += 1
            continue
        rng_i = random.Random(f'{seed}-{row["file"]}')
        orig = Image.open(os.path.join(real_root, 'raw', 'real', f'{row["source_id"]}.jpg')).convert('RGB')
        std, crop, scale = inp.standardize(orig)
        cands = inp.transform_boxes(boxes_all.get(row['source_id'], []), crop, scale, std.size)
        if not cands:
            skipped += 1
            continue
        gen = Image.open(os.path.join(styled_root, 'raw', row['file'])).convert('RGB')
        if gen.size != std.size:
            skipped += 1
            continue
        rng_i.shuffle(cands)
        picked = cands[:rng_i.randint(1, 2)]
        mask = inp.make_mask(std.size, picked)
        out = inp.paste_masked(std, gen, mask)
        out.save(os.path.join(work, 'raw', rel))
        p = json.loads(row['params'])
        mw.add(file=rel, label=1, type='partial_style', family='E4', generator=row['generator'].replace('img2img', 'partial') if src_name == SRC_NAME else f'partial-{row["generator"]}',
               source_id=row['source_id'], group=row['group'], split=row['split'],
               params={'styled_file': row['file'], 'style': p.get('style'), 'strength': p.get('strength'),
                       'boxes': picked, 'size': list(std.size)})
        made += 1
        if syncer and made % sync_every == 0:
            syncer.push()
    mw.close()
    if syncer:
        ok = syncer.push(retries=5)
        print('Drive 복사 완료' if ok else f'!! Drive 복사 실패 — 결과는 {work} 에 남아 있으니 다시 실행하면 이어서 복사됩니다', flush=True)
    print(f'완료 {made}장(건너뜀 {skipped}) {time.time() - t0:.0f}초 → 표: {mw.path}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('real_root', help='raw/real, raw/real_boxes.json, manifests/real.csv 가 있는 폴더')
    ap.add_argument('--styled_root', required=True, help='11단계 결과(raw/fake/sdxl_img2img, manifests/sdxl_img2img.csv)가 있는 폴더')
    ap.add_argument('--work', required=True)
    ap.add_argument('--n', type=int, default=10)
    ap.add_argument('--seed', type=int, default=6)
    ap.add_argument('--sync_to', default=None)
    ap.add_argument('--sync_every', type=int, default=100)
    ap.add_argument('--src', default=SRC_NAME, help='재료가 될 전체 스타일 표 이름(기본 sdxl_img2img)')
    ap.add_argument('--name', default=NAME, help='만들 부분 스타일 표 이름(기본 partial_style)')
    a = ap.parse_args()
    main(a.real_root, a.styled_root, a.work, a.n, a.seed, a.sync_to, a.sync_every, a.src, a.name)
