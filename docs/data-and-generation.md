# 데이터 구축


학습 데이터는 팀이 직접 만들어야 했다. 이 프로젝트에서 시간을 가장 많이 쓴 부분.

## 파이프라인
`data_pipeline/` 01~16단계: 실제 사진 추출 → 규격 통일(JPEG q92) → 필터 → 합성 → VAE 재구성 → 부분 생성 → 전체 생성 → 스타일 변환 → 국소 편집 → manifest.
- manifest 한 줄씩 `file, label, family, generator, source_id, split`, 시드 고정, 프롬프트 버전(v1~v6) 보존 → 재현 가능.

## 실제 사진
- 출처: AI Hub 딸기 데이터 (설향). 팀원 각자 신청 승인.
- 20장 → 500 → 1000 → **2500장** (포기 293개, train 2092 / val 408).
- AI Hub 사진은 카메라 원본이 아니라 1000~2100px **개체 크롭**. 같은 포기 반복 촬영이 많아 포기 ID 단위로 분할(누수 방지).
- 일부 사진에 노란/빨강 실험 개체표가 크게 찍힘 → "태그=실제" 오학습 위험으로 메모.
- 한계: 전부 AI Hub 설향 → 평가의 다른 종류 실제는 가짜로 보는 구조적 약점 (모음 §7).

## 가짜 (유형별)
| 유형 | 방법 | 물량 |
|---|---|---|
| ① 전체 생성 | RealVisXL V5.0 600, FLUX.1-schnell 800, Z-Image-Turbo 700, Infinity-2B 1000, Kandinsky 2.2 500(팀원), SDXL 600(팀원) | 약 4200 |
| ② 합성 | 배경 교체 521 + 객체 붙여넣기 479 | 1000 |
| ③ 지정 필터 | 블러·노이즈·소금후추·JPEG 각 250 | 1000 |
| ④ 전체 스타일 | SDXL img2img (사실적 1000 → 이후 화풍판 추가) | 1000+ |
| ⑤ 부분 생성 | LaMa 800, SDXL inpaint 600, SD1.5/Kandinsky inpaint, FLUX.2 klein 편집 | 2000+ |
| ⑥ 부분 스타일 | partial_style 954 + 화풍 부분 붙임 | 954+ |
| 기타 | FLUX VAE 재구성 500, 국소 편집 9종 2100 | |
- 최종 학습풀: 16987행 = out/C 11454 + 새 편집 1100 + 새 세트 4433. 학습 8577 / val 2681 (실제 408 / 가짜 2273), 생성 방식 24개(학습에서 11방식 제외).

## 겪은 문제와 해결
- 재료 고르기: 무작위로 뽑았더니 같은 원본에 가짜가 몰림 → `fake_usage` 집계 후 "덜 쓰인 원본 우선" (최종 0회 23 / 1회 1654 / 2회 823, 최대 2회).
- 프롬프트 v1→v5: v3는 열매를 억제하자 관엽식물이 섞임, v4는 열매·온실 전경이 줄어듦. v2 채택. FLUX/Z-Image는 negative prompt 불가라 'strawberry' 단어가 열매를 끌어옴. 이후 성장 단계(싹/잎/꽃/열매) 비율을 프롬프트에 반영(v5), 근접·얕은 심도(v6).
- 패키지 제약: `simple-lama-inpainting`이 `pillow<10`·`numpy<2` 요구 → 가중치만 받아 전처리·후처리 직접 구현.
- 생성 속도: RealVisXL L4 12초/A100 5초, FLUX는 A100 필수(L4 메모리 부족).
- 키 이름 불일치 408개(공식 `layer.N…` vs transformers `model.layer.N…`) → 변환 함수.

## 데이터 윤리/라이선스
- AI Hub: 본인 활용만, 재배포 금지 → 원본은 개인 PC에만 두고 공유하지 않았다.
- DINOv3 Meta 라이선스 사본 동봉, FLUX.1-dev 비상업 계열은 팀 방침상 제외.
- 가중치 공개 모델만 사용 (비공개 서비스에 사진을 올리지 않음).

## 미사용/미완
- 596 금실 500장(흰 베드 장면): 받았지만 학습에 미사용.
- 팀원 자료(실제 2416, 가짜 7006): 업로드 미완료로 가짜 0장으로 읽혀 일부(실제 439장)만 exp12r에 사용되고 폐기.

## 사용한 외부 모델과 라이선스

| 용도 | 모델 / 출처 | 라이선스 |
|---|---|---|
| 백본 (고정) | [`facebook/dinov3-vitl16-pretrain-lvd1689m`](https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m) | DINOv3 License (Meta) — 사본: [`submission/model/LICENSE_DINOv3.md`](../submission/model/LICENSE_DINOv3.md) |
| Common CNN | [`timm/efficientnet_b0.ra_in1k`](https://huggingface.co/timm/efficientnet_b0.ra_in1k) | Apache-2.0 |
| 전체 생성 | FLUX.1-schnell · Z-Image-Turbo · Kandinsky 2.2 · Infinity-2B | Apache-2.0 · Apache-2.0 · Apache-2.0 · MIT |
| 전체 생성 / 스타일 / inpaint | RealVisXL V5.0 · SDXL base 1.0 · SDXL inpainting | CreativeML Open RAIL++-M |
| 부분 생성 | LaMa (`saic-mdal/lama`) · SD 1.5 inpainting | Apache-2.0 · CreativeML Open RAIL-M |
| 지정 필터·합성·국소 편집 | PIL · numpy · OpenCV 자체 코드 | — |
| 실제 사진 | [AI Hub 시설작물(딸기) 개체 이미지 및 질병 이미지](https://aihub.or.kr/aihubdata/data/view.do?dataSetSn=71451) | AI Hub 이용조건 (본인 활용만, **재배포 금지** → 이 저장소에 포함하지 않음) |

FLUX.1-dev 등 비상업 계열 모델은 사용하지 않았다.

## 모든 학습 이미지의 규격 통일

실제·가짜 공통으로 4:3(또는 3:4) 가운데 크롭 → 1024×768(768×1024) LANCZOS 리사이즈 → JPEG quality 92, RGB, EXIF 없음. 가짜는 생성·편집을 모두 끝낸 뒤 마지막에 이 처리를 거친다. (`tools/normalize.py`)
