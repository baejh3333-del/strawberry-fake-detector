"""Infinity(자기회귀 생성 모델)가 Colab T4에서 도는지 10장으로 확인하는 시험 스크립트 (전략.md 11-4, 리스크 R9).

'대회에서 실제 쓰는 것처럼' = 딸기 시설재배 프롬프트 10개(가로 6 + 세로 4)로 만들고,
                            실제·가짜 모두 거치는 normalize.py까지 통과시켜 규격(1024×768 / 768×1024, q=92)으로 저장한다.

확인하는 것 (결과 요약이 마지막에 출력됨)
  1) 모델이 T4에서 로드되는가 (메모리 부족·bf16 오류 여부)
  2) 한 장에 몇 초 걸리는가 → 1000장이면 몇 시간인가
  3) 딸기 온실 사진처럼 보이는가 (눈으로 확인 — 결과 폴더의 preview.jpg 한 장으로 모아 둠)

사용법 (Colab 유료 T4 런타임, 셀 순서대로) — 아래 "Colab 셀" 참고. 이 파일은 `python infinity_test.py`로 실행.
가중치·이용 조건: FoundationVision/Infinity (MIT), 텍스트 인코더 google/flan-t5-xl (Apache-2.0)
  → 실제로 쓰기로 하면 README.md에 출처·이용 조건·프롬프트·시드를 적는다.

Colab 셀 (복사해서 사용)
  # 셀1: 코드 받기 + 설치 (Infinity는 pip 패키지가 아니라 저장소를 그대로 쓴다)
  !git clone --depth 1 https://github.com/FoundationVision/Infinity /content/Infinity
  !pip -q install omegaconf pytz easydict einops
  !pip -q install timm==0.9.16 || true      # Infinity 요구 버전. 충돌 나면 그대로 두고 셀2 오류 메시지를 알려 주세요
  # 셀2: 가중치 받기 (약 2B 모델 ≈ 8GB + VAE + flan-t5-xl ≈ 6GB → 디스크 여유 확인)
  from huggingface_hub import snapshot_download
  snapshot_download('FoundationVision/Infinity', local_dir='/content/infinity_w',
                    allow_patterns=['infinity_2b_reg.pth', 'infinity_vae_d32reg.pth', 'infinity_vae_d32_reg.pth'])
  snapshot_download('google/flan-t5-xl', local_dir='/content/flan-t5-xl', allow_patterns=['*.json', '*.model', '*.safetensors'])
  # 셀3: 실행 (tools 폴더를 Drive work/tools 에 올려 두었다고 가정)
  !python /content/drive/MyDrive/work/tools/infinity_test.py
"""

import os, sys, time, random, argparse, glob

INF_DIR = '/content/Infinity'                 # 셀1에서 받은 저장소
W_DIR = '/content/infinity_w'                 # 셀2에서 받은 Infinity 가중치
T5_DIR = '/content/flan-t5-xl'                # 셀2에서 받은 텍스트 인코더
OUT_DIR = '/content/drive/MyDrive/work/infinity_test'   # 결과 저장(Drive → 세션이 끊겨도 남는다)

# 딸기 시설재배 프롬프트. 공개샘플 3장의 '실제' 사진 분위기(온실 안, 자연광, 스마트폰/카메라 근접)를 말로 옮긴 것.
# (h_div_w = 세로/가로 : 0.75 → 가로 사진, 1.333 → 세로 사진)
PROMPTS = [
    (0.75,  'a photo of ripe red strawberries hanging from a raised bed in a greenhouse, natural daylight, shallow depth of field'),
    (0.75,  'close-up photo of strawberry plants with ripe and unripe fruits in a hydroponic greenhouse, water droplets on leaves'),
    (0.75,  'strawberry greenhouse interior, rows of elevated gutters with strawberry plants, white plastic film roof, photographed with a smartphone'),
    (0.75,  'a farmer hand holding a bunch of fresh red strawberries in a vertical strawberry farm, soft sunlight'),
    (0.75,  'strawberry flowers and green unripe fruits on the plant, greenhouse, macro photo, realistic'),
    (0.75,  'ripe strawberries with a few rotten ones on the plant, greenhouse cultivation, documentary photo'),
    (1.333, 'vertical photo of a single strawberry plant with red fruits in a greenhouse pot, natural light'),
    (1.333, 'strawberry seedlings in trays inside a greenhouse, top view, realistic photo'),
    (1.333, 'a cluster of red strawberries hanging over the edge of a growing bed, greenhouse, close up, 35mm photo'),
    (1.333, 'strawberry leaves with powdery mildew spots in a greenhouse, close-up, realistic agricultural photo'),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=len(PROMPTS))
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()

    sys.path.insert(0, INF_DIR)
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # normalize.py 를 같은 폴더에서 찾는다
    import torch, cv2, numpy as np
    from tools.run_infinity import load_tokenizer, load_visual_tokenizer, load_transformer, gen_one_img
    from infinity.utils.dynamic_resolution import dynamic_resolution_h_w, h_div_w_templates
    from normalize import normalize            # tools/normalize.py : 실제 파이프라인과 같은 규격 통일

    assert torch.cuda.is_available(), 'GPU가 없습니다. 런타임 → 런타임 유형 변경 → T4 GPU'
    print('GPU:', torch.cuda.get_device_name(0))

    # T4(sm75)는 bf16을 하드웨어로 지원하지 않는다 → 지원 안 되면 fp16 시도(숫자 깨짐 위험은 결과 그림으로 확인)
    use_bf16 = int(torch.cuda.is_bf16_supported())
    print('bf16 지원:', bool(use_bf16), '(T4는 보통 False)')

    vae_path = (glob.glob(f'{W_DIR}/infinity_vae_d32reg.pth') + glob.glob(f'{W_DIR}/infinity_vae_d32_reg.pth'))[0]
    args = argparse.Namespace(
        pn='1M', model_path=f'{W_DIR}/infinity_2b_reg.pth', cfg_insertion_layer=0, vae_type=32, vae_path=vae_path,
        add_lvl_embeding_only_first_block=1, use_bit_label=1, model_type='infinity_2b', rope2d_each_sa_layer=1,
        rope2d_normalized_by_hw=2, use_scale_schedule_embedding=0, sampling_per_bits=1, text_encoder_ckpt=T5_DIR,
        text_channels=2048, apply_spatial_patchify=0, h_div_w_template=1.000, use_flex_attn=0,
        cache_dir='/dev/shm', checkpoint_type='torch', seed=0, bf16=use_bf16, save_file='tmp.jpg')   # 공식 노트북과 같은 값

    t0 = time.time()
    text_tokenizer, text_encoder = load_tokenizer(t5_path=args.text_encoder_ckpt)
    vae = load_visual_tokenizer(args)
    infinity = load_transformer(vae, args)
    load_sec = time.time() - t0
    print(f'모델 로드 {load_sec:.0f}초, GPU 메모리 {torch.cuda.memory_allocated() / 2**30:.1f}GB 사용')

    raw_dir, norm_dir = f'{OUT_DIR}/raw', f'{OUT_DIR}/normalized'
    os.makedirs(raw_dir, exist_ok=True)
    random.seed(a.seed)
    times, fails, tmpl = [], 0, np.array(h_div_w_templates)
    for i, (h_div_w, prompt) in enumerate(PROMPTS[:a.n]):
        seed = a.seed * 1000 + i                  # 시드를 기록해 두면 같은 이미지를 다시 만들 수 있다(재현 검증)
        t = h_div_w_templates[int(np.argmin(np.abs(tmpl - h_div_w)))]
        sched = [(1, h, w) for (_, h, w) in dynamic_resolution_h_w[t][args.pn]['scales']]
        try:
            t1 = time.time()
            img = gen_one_img(infinity, vae, text_tokenizer, text_encoder, prompt, g_seed=seed, gt_leak=0, gt_ls_Bl=None,
                              cfg_list=3, tau_list=0.5, scale_schedule=sched, cfg_insertion_layer=[args.cfg_insertion_layer],
                              vae_type=args.vae_type, sampling_per_bits=args.sampling_per_bits, enable_positive_prompt=0)
            arr = img.cpu().numpy()
            if not np.isfinite(arr.astype(np.float32)).all():
                raise ValueError('결과에 NaN/inf')
            cv2.imwrite(f'{raw_dir}/inf_{i:02d}.png', arr)     # 원본은 무손실 PNG (JPEG 이중 압축 방지)
            normalize(f'{raw_dir}/inf_{i:02d}.png', f'{norm_dir}/inf_{i:02d}.jpg')   # 규격 통일 = 마지막에 한 번만 JPEG
            times.append(time.time() - t1)
            print(f'[{i + 1}/{a.n}] {times[-1]:.1f}초  seed={seed}  {arr.shape}  {prompt[:50]}')
        except Exception as e:                     # 한 장 실패가 전체를 멈추지 않게. 실패 내용은 결과에 그대로 보여 준다
            fails += 1
            print(f'[{i + 1}/{a.n}] 실패: {type(e).__name__}: {e}')
        torch.cuda.empty_cache()

    # 눈으로 확인하기 쉽게 한 장으로 모으기 (5열 격자, 작게)
    from PIL import Image
    files = sorted(glob.glob(f'{norm_dir}/*.jpg'))
    if files:
        th = [Image.open(f).convert('RGB').resize((256, 256)) for f in files]
        sheet = Image.new('RGB', (256 * 5, 256 * ((len(th) + 4) // 5)))
        for k, im in enumerate(th):
            sheet.paste(im, ((k % 5) * 256, (k // 5) * 256))
        sheet.save(f'{OUT_DIR}/preview.jpg')

    print('\n===== 결과 요약 =====')
    print(f'성공 {len(times)}장 / 실패 {fails}장, 최대 GPU 메모리 {torch.cuda.max_memory_allocated() / 2**30:.1f}GB / 15GB')
    if times:
        avg = sum(times) / len(times)
        print(f'한 장 평균 {avg:.1f}초 → 1000장이면 약 {avg * 1000 / 3600:.1f}시간 (모델 로드 {load_sec:.0f}초 별도)')
    print(f'결과: {OUT_DIR}/normalized/ (규격 통일본), preview.jpg(한눈에 보기)')
    print('다음 확인: ① tools/jpegscan.py 로 규격(1024x768/768x1024, q=6c3c42bf87) ② preview.jpg가 딸기 온실처럼 보이는지')
    print('판단: 전부 성공+그림 정상 → 본 생성 진행 / 로드·메모리 실패 → 포기하고 전략.md R9에 기록')


if __name__ == '__main__':
    main()
