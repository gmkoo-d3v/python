# YOLO 특강 실습판

이 폴더는 YOLO 특강(하루 5차시) 실습 코드입니다. 1·2일차 내용은 6강 노트북 `06_object_detection.ipynb`에서 가져왔고, 3일차(추적)는 새로 만들었습니다.

| 파일 | 차시 | 내용 | 걸리는 시간(Mac, Apple Silicon) |
|---|---|---|---|
| lab1_iou.py | 1 | 박스 형식 바꾸기와 IoU | 약 10초(첫 실행은 설치 때문에 더 걸림) |
| lab2_nms.py | 2 | NMS 직접 구현 | 1초 |
| lab3_decode.py | 3 | 책임 칸 찾기와 박스 디코딩 | 1초 안 |
| lab4_yolov3.py | 4 | YOLO v3 뼈대와 출력 크기 | 약 1초 |
| lab5_yolo11.py | 5 | YOLO11 전이학습(Penn-Fudan 보행자) | 약 3분(50 epoch). `YOLO_EPOCHS=10`이면 약 40초 |
| lab6_video.py | 5(더 해 보기) | 학습한 모델로 동영상 탐지, 추론 크기 견주기 | 약 10초(lab5를 먼저 돌려야 함) |
| lab7_eval.py | 6 | IoU 기준별 AP, conf 고르기, pycocotools로 다시 재기 | 약 25초 |
| lab8_errors.py | 7 | 오류를 위치 어긋남, 중복, 배경 오탐, 놓침으로 세기 | 약 5초 |
| lab9_test_split.py | 8 | 학습·검증·시험으로 나눠 다시 학습, 검증 점수와 시험 점수 견주기 | 약 2분 40초 |
| lab10_onnx.py | 9 | ONNX로 내보내고 onnxruntime만으로 출력을 직접 풀기 | 약 25초 |
| lab11_speed.py | 10 | 입력 크기(320/416/640)와 int8 양자화, CPU 속도와 mAP | 약 40초 |
| lab12_count.py | 11 | 정답 ID가 있는 합성 영상 만들기, 검출만으로 사람 수 세기 | 약 30초(처음 한 번 검출), 그다음 몇 초 |
| lab13_iou_track.py | 12 | IoU와 헝가리안 매칭으로 잇는 추적기 | 몇 초 |
| lab14_sort.py | 13 | 칼만 필터로 예측하며 따라가기(SORT), max_age 바꿔 보기 | 몇 초 |
| lab15_bytetrack.py | 14 | ByteTrack(낮은 점수 검출로 2차 짝짓기), 세 추적기 견주기 | 몇 초 |
| lab16_line_count.py | 15 | 선 넘기 세기, 결과 영상 그리기, 내 영상에 돌리기 | 몇 초(내 영상은 길이에 따라) |
| lab17_pca.py | 16 | DINOv3 패치 특징 PCA, 사람 위 한 곳을 찍어 비슷한 패치 모으기 | 약 20초(처음엔 모델 받기 포함) |
| lab18_fewshot.py | 17 | EuroSAT 위성사진 10종을 라벨 1·5·20·50장으로 분류(1-NN, 선형) | 약 40초(처음엔 EuroSAT 90MB 받기 포함) |
| lab19_search.py | 18 | 사진 검색(precision@5), 두 사람 사이 머리·발 패치 짝짓기 | 약 15초 |
| lab20_reid.py | 19 | 겉모습으로 같은 사람 알아보기(그늘 포함), 끊긴 트랙 이어 붙이기 | 약 30초 |
| lab21_small.py | 20 | ViT-S와 ConvNeXt-T의 정확도·속도 견주기 | 약 1분(맥 CPU에서 ConvNeXt가 느림) |

lab6~lab16은 lab5가 만든 `runs/person/weights/best.pt`와 `data/PennFudan_yolo`를 씁니다(`common_yolo.py`).
lab12~16은 `common_track.py`가 만든 합성 영상과 검출 결과(`runs/track/`)를 함께 씁니다. 합성 영상은 Penn-Fudan 검증 사진의 사람을 오려 걷게 만든 16초짜리이고, 누가 누구인지 정답이 있어서 추적기가 ID를 몇 번 바꿨는지 셀 수 있습니다.
lab17~21은 `common_dino.py`로 DINOv3(timm의 `vit_small_patch16_dinov3.lvd1689m`, `convnext_tiny.dinov3_lvd1689m`)를 씁니다. 가중치는 처음 실행할 때 Hugging Face 로그인 없이 받아집니다. lab20은 3일차 합성 영상과 lab14·15의 추적기를 다시 씁니다. 특징 캐시는 `runs/dino/`에 남습니다.

## 실행

[uv](https://docs.astral.sh/uv/)가 있어야 합니다. 명령은 모두 이 폴더 안에서 실행합니다.

패키지는 이 폴더의 `pyproject.toml`과 `uv.lock`으로 관리합니다. lab1~21이 쓰는 패키지(torch, torchvision, ultralytics, timm, scikit-learn, scipy, pycocotools, onnx, onnxruntime, onnxslim 등)가 모두 들어 있어서 `--with`를 붙일 필요가 없습니다.

```bash
uv sync                        # 처음 한 번, 또는 pyproject.toml이 바뀌었을 때
uv run python lab1_iou.py      # lab1~4
uv run python lab5_yolo11.py   # lab5, lab6~21도 같은 방식
uv run python lab16_line_count.py 내영상.mp4   # 직접 찍은 영상
```

- 패키지를 허용된 최신 버전으로 올리려면 `uv lock --upgrade` 후 `uv sync`를 실행합니다.
- 프로젝트 환경을 만들지 않고 한 파일만 돌려 보려면 예전처럼 `uv run --no-project --with torch --with torchvision python lab1_iou.py` 꼴로 실행해도 됩니다. 필요한 패키지는 위 `pyproject.toml`의 dependencies와 같습니다.
- lab5는 처음 실행할 때 두 가지를 내려받습니다.
  - Penn-Fudan 데이터 약 50MB. `data/`에 저장합니다.
  - `yolo11s.pt` 18MB.
- 내려받다가 `CERTIFICATE_VERIFY_FAILED`(인증서 오류)가 나면 다음 순서로 해 보세요.
  1. `truststore`가 이미 의존성에 들어 있어 `common.ssl_context()`가 운영체제 인증서 저장소를 먼저 씁니다. 그래도 실패하면 브라우저로 https://www.cis.upenn.edu/~jshi/ped_html/PennFudanPed.zip 을 받아 `data/`에 풉니다.
  2. 학원·회사 네트워크라면 `yolo11s.pt` 내려받기에서도 같은 오류가 날 수 있습니다. 브라우저로 받아 이 폴더에 두면 됩니다.
- Penn-Fudan을 이미 갖고 있다면 `PENN_DATA_ROOT`에 `PennFudanPed` 폴더의 상위 폴더 경로를 넣어 주세요.
- 학습 결과는 `runs/person/`에 남습니다. 다시 학습하면 이 결과를 덮어씁니다. lab9는 `runs/person_3way/`, lab11은 `runs/speed/`에 따로 남깁니다.

## 라이선스

- lab1~lab4와 common.py는 교안 코드와 같은 조건을 따릅니다. common_yolo.py는 ultralytics를 쓰지 않습니다.
- lab12~16의 추적기(IoU, SORT, ByteTrack)와 지표(MOTA, IDF1)는 직접 짠 코드이고 scipy(BSD)만 씁니다. 다만 검출에 ultralytics를 쓰므로 lab12~16과 common_track.py 묶음도 아래 AGPL-3.0을 따릅니다. 지표 계산은 py-motmetrics와 같은 값이 나오는 것을 확인했습니다(py-motmetrics 1.4.0은 NumPy 2에서 돌지 않아 쓰지 않았습니다).
- **Built with DINOv3.** lab17~21이 쓰는 DINOv3 가중치는 Meta의 DINOv3 License를 따릅니다(https://ai.meta.com/resources/models-and-libraries/dinov3-license/). 상업적 사용을 막지 않지만, 가중치를 다시 나눠 줄 때는 이 라이선스 사본을 함께 주고 "Built with DINOv3"를 밝혀야 하며, 군사·무기 등 금지된 용도에는 쓸 수 없습니다. 이 묶음에는 가중치를 넣지 않았습니다(실행할 때 받음). lab17~19·21과 common_dino.py는 ultralytics를 쓰지 않습니다.
- EuroSAT은 MIT 라이선스입니다(https://github.com/phelber/EuroSAT).
- 추적 공개 데이터 MOT17·MOT20은 비상업 라이선스(CC BY-NC-SA)라 쓰지 않고 합성 영상을 만들었습니다.
- lab5~lab11은 ultralytics 패키지를 쓰므로 **AGPL-3.0**을 따릅니다. 약관(https://www.ultralytics.com/license)상 직접 학습한 모델에도 AGPL-3.0이 적용됩니다.
- Penn-Fudan 데이터는 원 배포처의 조건을 따릅니다.
