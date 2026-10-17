<div align="center">

# Strawberry Fake Detector

**실제인가, AI 생성인가? 딸기 시설재배 사진을 판별하는 DINOv3 + Mixture-of-Experts detector**

[결과](#결과) · [모델 구조](#모델-구조) · [데이터 파이프라인](#데이터-파이프라인) · [시행착오](docs/lessons-learned.md) · [English](README.md)

![Python](https://img.shields.io/badge/Python-3.10+-blue) ![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c) ![Backbone](https://img.shields.io/badge/backbone-DINOv3%20ViT--L%2F16-green) ![Accuracy](https://img.shields.io/badge/leaderboard%20accuracy-0.85-brightgreen) ![License](https://img.shields.io/badge/code-MIT-lightgrey)

<img src="assets/architecture.png" width="92%" alt="모델 구조">

</div>

## 한눈에 보기

제1회 전국 농과계 대학 농업 AI 경진대회(이미지 미션) 프로젝트입니다. 딸기 사진이 **실제(0)** 인지 **생성·합성·변형(1)** 인지 분류합니다. 학습 데이터가 주어지지 않아서 가짜 이미지 생성 파이프라인부터 만들었습니다. 입문자 2명, 4주.

| | |
|---|---|
| **최종 점수** | 비공개 평가 524장(실제 262 / 가짜 262) **accuracy 0.85** |
| **모델** | 311M 파라미터, 0.64 GB, Colab T4에서 524장 **152초** (제한: 5분 · 오프라인 · 1GB 이하) |
| **핵심 아이디어** | 원본 해상도 조각 → 고정 DINOv3 → **생성 방식별 Expert**(attention pooling) + 전체 이미지 CNN |
| **가장 쓸모 있던 발견** | 내부 validation 0.874, 비공개 평가 0.77. leave-one-generator-out 평가로 원인을 규명 |

## 결과

<img src="assets/results.png" width="62%" alt="validation vs leaderboard">

| 제출 | 변경 | 내부 검증 | 리더보드 |
|---|---|---|---|
| exp02 | 첫 본 학습 | acc 0.874 / AUROC 0.958 | 0.77 |
| exp06 | 복제·국소 필터 라벨 재배정 | acc 0.875 | ≈ 0.75 |
| exp11b (v3) | 생성 방식별 Expert 24개, 반응 없던 11방식 제외 | acc 0.885 / AUROC 0.964 | 0.83 |
| **v5 (최종)** | 가중치 동일, 판정 문턱 p ≥ 0.05 (마지막 bias에 `+ln 19`) | acc 0.901 | **0.85** |

이후의 구조 아이디어 7개(LayerNorm, top-k pooling, 고주파 입력, dropout, FFT 등)는 전부 잡음 수준이었습니다. 병목은 모델 용량이 아니라 **모델이 본 생성기의 종류**였습니다.

<img src="assets/experiments.png" width="62%" alt="실험별 변화량">

> 실험당 seed 1개. 음영은 관측된 실행 간 잡음(≈ 0.005). 전체 표: [docs/experiments.md](docs/experiments.md)

## 모델 구조

| 설계 선택 | 이유 |
|---|---|
| **원본 해상도 조각 12장, 축소 없음** | 블러·VAE 8픽셀 격자·편집 경계 같은 픽셀 단위 흔적은 축소하면 사라진다. |
| **DINOv3 고정** | 생성기당 수백~천 장뿐이라 fine-tuning하면 *내가 만든* 생성기 지문을 외운다. |
| **attention pooling** | 부분 편집은 일부 조각에만 흔적이 있어 평균하면 묻힌다. |
| **전체 이미지 Common CNN** | 고주파 흔적(노이즈·JPEG)을 보완하고 입력 크기 제한이 없다. |
| **생성 방식별 Expert + Gate** | 생성기마다 지문이 다르다. |

학습: (1) Expert + Common CNN → (2) Gate → (3) 전체를 `BCE + 0.2·GateCE + 0.1·SupCon`으로 (SupCon은 같은 원본의 실제·가짜 쌍). 자세히: [docs/architecture.md](docs/architecture.md)

## 데이터 파이프라인

<img src="assets/pipeline.png" width="92%" alt="데이터 파이프라인">

15종 이상의 생성기로 약 1.7만 장. train/val은 **포기(plant) ID** 단위로 나눠 leakage를 막았고, leave-one-generator-out fold로 처음 본 방식에 대한 일반화를 쟀습니다. 출처·라이선스·프롬프트: [docs/data-and-generation.md](docs/data-and-generation.md)

## 빠른 시작

```bash
pip install -r requirements.txt

# 1) 무작위 가중치로 추론 패키지 smoke test (데이터 불필요)
python tools/smoke_test_submission.py

# 2) 학습 (data_pipeline/ 으로 만든 manifest와 이미지 필요)
python train/train.py --data <DATA_DIR> --pretrained <PRETRAINED_DIR> --out <RUN_DIR> --stage all --family_map gen24
```

> **포함하지 않은 것:** 학습 이미지, 대회 제공 데이터, 학습된 가중치. 실제 사진은 [AI Hub](https://aihub.or.kr/aihubdata/data/view.do?dataSetSn=71451)(본인 활용만, 재배포 금지)에서 받고, 가짜는 `data_pipeline/`으로 다시 생성합니다. DINOv3 백본은 [DINOv3 License](submission/model/LICENSE_DINOv3.md)에 따라 별도로 내려받습니다.

## 저장소 구조

```
submission/       제출한 추론 패키지 (detector.py, inference.py, model/config.json)
train/            3 stage trainer, 단위 테스트, 진단 도구 (LOGO, 문턱 탐색, Expert 사용도)
data_pipeline/    01–16: 실제 추출 → 필터 → 합성 → VAE 재구성 → inpaint → t2i → 스타일 → 국소 편집
tools/            JPEG 스캐너, 이미지 정규화, 제출물 smoke test
assets/           그림 (assets/make_figures.py로 재생성)
docs/             architecture · data-and-generation · experiments · lessons-learned
```

## 알려진 한계

처음 본 **AI 편집 모델**이 만든 부분 생성(unseen recall 0.32), FLUX-VAE 재구성(recall ≈ 0), 높은 품질의 JPEG 재압축 가짜, 학습에 없는 품종의 실제 사진("AI Hub 스타일 ⇒ 실제"라는 지름길). 분석: [docs/lessons-learned.md](docs/lessons-learned.md)

## 팀

[@baejh3333-del](https://github.com/baejh3333-del) · [@pjo789412-afk](https://github.com/pjo789412-afk)

## 감사의 말

[DINOv3](https://github.com/facebookresearch/dinov3) (Meta), [timm](https://github.com/huggingface/pytorch-image-models), [Hugging Face Transformers / Diffusers](https://github.com/huggingface), [LaMa](https://github.com/advimman/lama), FLUX, Stable Diffusion XL, Z-Image, Infinity, Kandinsky, AI Hub 딸기 데이터셋.

## License

코드: [MIT](LICENSE). 제3자 모델·데이터는 각자의 라이선스를 따릅니다 ([상세](docs/data-and-generation.md)).
