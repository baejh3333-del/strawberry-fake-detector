"""전체 생성(유형 ①)용 프롬프트 재료 — 07(RealVisXL)·08(FLUX·Z-Image)이 함께 쓴다. 모델이 달라도 같은 분포로 뽑는다.

**채택: v2 (2026-09-29).** 공개샘플을 보고 쓴 버전 — 잎 위주(열매·꽃은 15%), 중간 거리 옆 시점, 흰 홈통·검은 재배 상자·번호표·끈끈이·온실.
시험 결과 v2는 열매와 넓은 온실 전경이 자주 나오는데, 그 모습을 팀이 원하는 방향으로 골랐다.

버전 기록 (발표·재현용)
- v1: 열매·꽃·클로즈업 위주, 'highly detailed·vivid' → 잡지 사진 같아 버림
- v2: 잎 위주 + 온실 배경, negative에 클로즈업·스튜디오 조명 → **채택**
- v3: 열매를 줄이려고 'strawberry'를 뺐더니 일반 관엽식물이 섞여 나옴
- v4: 잎 위주로 더 강하게(열매 10%, 위에서 본 구도 제외) → 열매·온실 전경이 적어 채택 안 함
"""

PROMPT_VERSION = 'v2'
FRUIT_RATIO = 0.15   # 열매·꽃이 조금 보이는 장면의 비율

SUBJECTS_LEAF = [
    'strawberry plants with dark green serrated leaves growing in a white hydroponic gutter',
    'a strawberry plant with large green leaves growing in a long black planter trough filled with soil',
    'young strawberry plants with green leaves in a black planter box with dark soil',
    'strawberry plants with dense green foliage in white pots on an elevated bed',
    'strawberry plants with many green leaves and thin runners in a white gutter with black drip tubes',
    'healthy strawberry plants with wide green leaves, plant fills the frame',
    'strawberry plants with green leaves growing in a white grow bag filled with substrate',
]
SUBJECTS_FRUIT = [
    'strawberry plants with green leaves and a few small white flowers',
    'strawberry plants with green leaves and one or two red strawberries partly hidden under the leaves',
    'strawberry plants with green leaves and a small unripe green strawberry',
]
SCENES = [
    'with black drip irrigation tubes and gravel on the floor below',
    'with a small numbered yellow sticker on the white gutter',
    'with a yellow sticky trap on a stick near the plants',
    'inside a greenhouse, rows of strawberry plants blurred in the background',
    'inside a greenhouse with a metal frame and plastic roof in the background',
    'with a blue container below the plants',
    'with white plastic gutters stacked in tiers',
    'with dark soil and white perlite visible around the plants',
]
SHOTS = [
    'medium shot taken at plant height from the side, plain snapshot with a smartphone',
    'unedited phone photo, flat natural light, plant fills the frame, medium distance',
    'plain documentary photo, overcast light, taken from the side at medium distance',
    'photo from slightly above the plants at medium distance, flat lighting',
]
NEGATIVE = ('close-up, macro, bokeh, shallow depth of field, studio lighting, glossy, perfect, painting, illustration, '
            'drawing, cartoon, anime, 3d render, cgi, text, watermark, logo, signature, deformed, blurry, low quality')


def make_prompt(rng):
    subjects = SUBJECTS_FRUIT if rng.random() < FRUIT_RATIO else SUBJECTS_LEAF
    return f'{rng.choice(subjects)}, {rng.choice(SCENES)}, {rng.choice(SHOTS)}, realistic photo'


def make_negative(prompt):
    return NEGATIVE   # v2는 장면과 상관없이 같은 negative (FLUX·Z-Image는 negative를 쓰지 않는다)


# ======================================================================================================
# v5: 성장 단계별 프롬프트 (2026-09-29) — 싹만 / 잎만(열매 없음) / 꽃 / 열매를 구분해서 넣는다.
# 어린 작물에는 열매도 꽃도 달지 않는다. 기존 v2 생성물(FLUX 800·Z-Image 700·RealVisXL 600·SDXL·Kandinsky)은 그대로 두고
# v5는 Infinity(09)부터 쓴다. 07·08도 --prompt v5 로 쓸 수 있다. manifest params에 stage가 기록된다.
# ======================================================================================================

STAGE_WEIGHTS = [('seedling', 0.15), ('leaf', 0.55), ('flower', 0.15), ('fruit', 0.15)]   # 합 1.0. 실제 사진은 잎이 대부분

STAGE_SUBJECTS = {
    # 싹만 — 어린 묘. 열매·꽃 없음, 작은 잎 몇 장
    'seedling': [
        'tiny young strawberry seedlings with a few small fresh green leaves growing in small pots with dark soil',
        'newly planted young strawberry seedlings with small leaves in a black planter box with dark soil',
        'young strawberry plants with only a few small green leaves growing in a white grow bag',
        'small strawberry sprouts with two or three little leaves growing in a black tray of soil',
    ],
    # 잎만 — 자란 작물, 열매·꽃 없음 (v4에서 열매를 억제한 문장)
    'leaf': [
        'a few large dark green serrated strawberry leaves filling the frame, plant growing in a white hydroponic gutter',
        'large matte dark green serrated strawberry leaves with thin stems, growing in a long black planter trough with dark soil',
        'dense dark green strawberry leaf foliage growing in a black planter box, leaves fill most of the frame',
        'wide dark green serrated strawberry leaves, growing in a white slotted plastic pot in a greenhouse gutter',
        'large dark green strawberry leaves and thin runners hanging over a white gutter with black drip tubes',
        'big matte dark green strawberry leaves growing in a white grow bag filled with substrate',
    ],
    # 꽃 — 하얀 꽃이 피어 있음, 열매는 아직 없음
    'flower': [
        'strawberry plants with dark green leaves and clusters of small white flowers',
        'a strawberry plant with green leaves and several white flowers on thin stems',
        'strawberry plants in bloom with white flowers with yellow centers among green leaves',
    ],
    # 열매 — 초록·흰 미숙과부터 붉게 익은 열매까지
    'fruit': [
        'strawberry plants with green leaves and a few red ripe strawberries hanging from the plant',
        'strawberry plants with green leaves, unripe small green strawberries and one or two red ones',
        'strawberry plants with green leaves and a small cluster of white and pink strawberries',
        'strawberry plants with dark green leaves and one red strawberry partly hidden under the leaves',
    ],
}
# SDXL 계열(negative를 쓰는 모델)에서만 쓰는 단계별 추가 negative — FLUX·Z-Image·Infinity는 negative를 쓰지 않아 위 주어 문장으로 통제한다
STAGE_NEG_EXTRA = {
    'seedling': ', strawberries, fruit, berries, flowers, large mature plant',
    'leaf': ', red strawberries, ripe fruit, berries, flowers',
    'flower': ', red strawberries, ripe fruit',
    'fruit': '',
}
# 씨앗 단계 장면에는 '재배 줄이 멀리 보인다'는 문장을 쓰지 않는다(다 자란 식물이 딸려 나옴)
SCENES_SEEDLING = [s for s in SCENES if 'strawberry plants' not in s]


def pick_stage(rng):
    x, acc = rng.random(), 0.0
    for name, w in STAGE_WEIGHTS:
        acc += w
        if x < acc:
            return name
    return STAGE_WEIGHTS[-1][0]


def make_prompt_v5(rng):
    """(프롬프트, 단계) — 단계: seedling / leaf / flower / fruit"""
    stage = pick_stage(rng)
    scenes = SCENES_SEEDLING if stage == 'seedling' else SCENES
    return f'{rng.choice(STAGE_SUBJECTS[stage])}, {rng.choice(scenes)}, {rng.choice(SHOTS)}, realistic photo', stage


def make_negative_v5(stage):
    return NEGATIVE + STAGE_NEG_EXTRA[stage]


def stage_of_prompt(prompt):
    """저장된 v5 프롬프트에서 단계를 되짚는다(검증용)."""
    for stage, subs in STAGE_SUBJECTS.items():
        if any(prompt.startswith(s) for s in subs):
            return stage
    return None


# ======================================================================================================
# v6 (2026-10-04): 대회 예시 이미지 중 전체 생성 사진을 다시 보고 쓴 버전 — 07·08·16이 --prompt v6 로 쓴다.
# v2~v5가 놓친 것: 예시 사진 일부는 **근접·낮은 시점·얕은 심도·뒤가 흐린 온실**의 스마트폰 사진인데,
# v2~v5의 NEGATIVE가 close-up·macro·bokeh·shallow depth of field를 전부 금지했다. 노란 끈끈이(트랩)는 세 장 모두에 있다.
# 그래서 v6는 근접 샷 55% / 중간 30% / 줄 전체 15%로 섞고, 노란 트랩이 55% 장면에 들어가며, 근접·보케를 금지하지 않는다.
# 근거는 공개된 문제 설명과 육안으로 본 인상뿐이다. 대회 제공 이미지는 어디에도 올리지 않는다.
# ======================================================================================================

V6_SHOTS = [
    ('closeup', 0.55, 'close-up low-angle smartphone photo, leaves in the foreground, shallow depth of field, blurred greenhouse background'),
    ('medium', 0.30, 'medium shot taken with an iPhone at plant height, soft background blur'),
    ('rowview', 0.15, 'wide view along a row of plants inside a greenhouse, deep focus, many plants visible'),
]
V6_STAGE_WEIGHTS = [('seedling', 0.10), ('leaf', 0.45), ('flower', 0.15), ('fruit', 0.30)]   # 샘플 3장 중 1장에 숙도가 다른 열매
V6_TRAP = 'with a yellow sticky insect trap on a stick, black specks on it'
V6_SCENES = [
    'in a black plastic grow tray with dark soil and white perlite, a blue container below',
    'growing in white hydroponic gutters stacked in tiers inside a greenhouse',
    'with rows of strawberry plants and a metal frame and plastic roof blurred in the background',
    'with black drip irrigation tubes and a small numbered yellow tag on a white gutter',
    'in white slotted pots on an elevated bed in a greenhouse',
]
V6_TRAP_RATIO = 0.55
V6_LIGHT = [
    'bright diffused daylight through greenhouse plastic',
    'slightly overexposed whites, natural phone camera processing',
    'soft overcast light, neutral colors',
]
V6_LOOK = ('unedited phone photo, fine leaf veins and tiny hairs on the stems, a little dust and soil specks on the leaves, '
           'matte not glossy, realistic')
# SDXL 계열(negative를 쓰는 모델)용. v2~v5와 달리 close-up·bokeh·shallow depth of field를 금지하지 않는다.
NEGATIVE_V6 = ('painting, illustration, drawing, cartoon, anime, 3d render, cgi, studio lighting, oversaturated, glossy, perfect, '
               'text, watermark, logo, deformed, low quality')


def _pick_weighted(rng, items):
    x, acc = rng.random(), 0.0
    for it in items:
        acc += it[1]
        if x < acc:
            return it
    return items[-1]


def make_prompt_v6(rng):
    """(프롬프트, 태그, negative) — 태그는 '단계/샷' (예: 'leaf/closeup'). manifest params의 stage에 저장된다."""
    stage = _pick_weighted(rng, V6_STAGE_WEIGHTS)[0]
    shot_tag, _, shot_text = _pick_weighted(rng, V6_SHOTS)
    subject = rng.choice(STAGE_SUBJECTS[stage])
    if rng.random() < V6_TRAP_RATIO and stage != 'seedling':
        scene = V6_TRAP
        if rng.random() < 0.5:                       # 트랩 + 재배 장면 두 가지를 함께 쓰기도 한다
            scene = f'{scene}, {rng.choice(V6_SCENES)}'
    else:
        scene = rng.choice(V6_SCENES)
    # 샷을 맨 앞에: SDXL(CLIP)은 앞 77토큰만 읽어서 뒤쪽 문구가 잘린다 — 가장 중요한 '근접 폰 사진'을 앞에 둔다
    prompt = f'{shot_text}, {subject}, {scene}, {rng.choice(V6_LIGHT)}, {V6_LOOK}'
    return prompt, f'{stage}/{shot_tag}', NEGATIVE_V6 + STAGE_NEG_EXTRA[stage]


# ======================================================================================================
# 적었다. 처음 만든 화풍 시험물은 9/29 방침 변경으로 폐기했었고, 10/4에 다시 살린다.
# 적었다. 처음 만든 화풍 시험물은 9/29 방침 변경으로 폐기했었고, 10/4에 다시 살린다.
# 11_sdxl_img2img.py --set paint (SDXL)와 16_klein.py --mode style (FLUX.2 klein)이 쓴다. 구조·글씨는 그대로 두는 것이 목표.
# ======================================================================================================

PAINT_STYLES = [
    # (이름, SDXL img2img용 문구, klein 지시문에 들어갈 설명)
    ('flat_illustration', 'flat digital illustration, clean dark outlines, cel shading, smooth flat color areas',
     'a flat digital illustration with clean dark outlines and smooth cel-shaded color areas'),
    ('anime_bg', 'hand-painted anime background art, soft painterly shading, clean line work, vibrant greens',
     'hand-painted anime background art with soft painterly shading and clean line work'),
    ('watercolor', 'watercolor painting, soft washes of color, visible paper texture, gentle edges',
     'a watercolor painting with soft washes of color and visible paper texture'),
    ('oil', 'oil painting, visible brush strokes, rich saturated colors, smooth blended shading',
     'an oil painting with visible brush strokes and smooth blended shading'),
    ('gouache', 'gouache painting, matte flat color shapes, simple smooth shading',
     'a gouache painting with matte flat color shapes and simple smooth shading'),
    ('digital_paint', 'smooth digital painting, polished concept art, smooth gradients, simplified textures',
     'a polished smooth digital painting with simplified textures and smooth gradients'),
]
PAINT_NEGATIVE = 'photo, photorealistic, realistic, 3d render, distorted text, watermark, deformed, blurry, low quality'


def make_paint_prompt(style_text):
    """SDXL img2img용 — 원본 구도를 지키라는 문구를 붙인다."""
    return f'strawberry plants in a greenhouse, {style_text}, same composition, same objects'


def make_style_instruction(klein_text):
    """FLUX.2 klein 지시형 편집용(영어). 구조·글씨·물체는 그대로 두고 화풍만 바꾸게 한다."""
    return (f'Redraw this photo as {klein_text}. Keep the exact composition, every object, plant, label and piece of text '
            'unchanged. Only change the rendering style.')


# ======================================================================================================
# 부분 생성(⑤) 지시형 편집 — 16_klein.py --mode edit. 지시형 편집기는 사진 내용을 모른 채 지시를 받으므로 대상이 없으면 아무 변화가 없다
# (스크립트가 변화량이 너무 작은 결과는 버리고 다른 지시로 다시 시도한다). 대회 부분 생성이 '무엇을 바꿨는지'는 모른다 —
# 물체 추가·삭제·교체를 고르게 섞는다.
# ======================================================================================================

EDIT_SUFFIX = ' Do not change anything else in the image.'
EDIT_INSTRUCTIONS = [
    # (종류, 지시문)
    ('add', 'Add two ripe red strawberries hanging among the leaves in the lower part of the image.'),
    ('add', 'Add a cluster of small white strawberry flowers on a thin stem near the center of the image.'),
    ('add', 'Add a few more dark green strawberry leaves growing from the plant on the right side.'),
    ('add', 'Add a small yellow sticky insect trap on a stick between the plants.'),
    ('add', 'Add a white label with a plant number on one of the pots.'),
    ('add', 'Add one small unripe green strawberry hanging below a leaf.'),
    ('remove', 'Remove the yellow sticker or tag from the image and fill the area naturally.'),
    ('remove', 'Remove one black drip tube from the image and fill the area naturally.'),
    ('remove', 'Remove the label on the pot and make the surface plain white.'),
    ('remove', 'Remove one leaf from the plant and show what is behind it.'),
    ('remove', 'Remove the flowers and keep everything else the same.'),
    ('remove', 'Remove the red strawberries and keep everything else the same.'),
    ('replace', 'Change the number on the tag to a different number.'),
    ('replace', 'Replace one leaf with a leaf that has dried brown edges.'),
    ('replace', 'Make the soil in one part of the image look wetter and darker.'),
    ('replace', 'Change one ripe red strawberry to a small unripe green one.'),
    ('replace', 'Replace the gravel on the floor with white perlite.'),
]


def make_edit_instruction(rng):
    """(종류, 지시문 전체)"""
    kind, text = rng.choice(EDIT_INSTRUCTIONS)
    return kind, text + EDIT_SUFFIX
