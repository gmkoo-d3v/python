# YOLO 특강 실습판

이 폴더는 YOLO 특강(하루 5차시) 실습 코드입니다. 내용은 6강 노트북 `06_object_detection.ipynb`에서 가져왔습니다.

| 파일 | 차시 | 내용 | 걸리는 시간(Mac, Apple Silicon) |
|---|---|---|---|
| lab1_iou.py | 1 | 박스 형식 바꾸기와 IoU | 약 10초(첫 실행은 설치 때문에 더 걸림) |
| lab2_nms.py | 2 | NMS 직접 구현 | 1초 |
| lab3_decode.py | 3 | 책임 칸 찾기와 박스 디코딩 | 1초 안 |
| lab4_yolov3.py | 4 | YOLO v3 뼈대와 출력 크기 | 약 1초 |
| lab5_yolo11.py | 5 | YOLO11 전이학습(Penn-Fudan 보행자) | 약 3분(50 epoch). `YOLO_EPOCHS=10`이면 약 40초 |
| lab6_video.py | 5(더 해 보기) | 학습한 모델로 동영상 탐지, 추론 크기 견주기 | 약 10초(lab5를 먼저 돌려야 함) |

## 실행

[uv](https://docs.astral.sh/uv/)가 있어야 합니다. 명령은 모두 이 폴더 안에서 실행합니다.

```bash
uv sync                        # 처음 한 번만: 이 폴더 전용 .venv를 만듭니다
uv run python lab1_iou.py      # lab1~4 모두 같은 방식
uv run python lab5_yolo11.py
uv run python lab6_video.py
```

- 이 폴더는 루트와 분리된 자체 `pyproject.toml`/`uv.lock`/`.venv`를 사용합니다.
- Windows에서는 CUDA 빌드(torch cu128)를, macOS/Linux에서는 PyPI 기본 빌드를 씁니다.
- 표의 시간은 Mac(Apple Silicon) 기준입니다. Windows에서는 GPU/전원 설정에 따라 더 오래 걸릴 수 있습니다.
- lab5는 처음 실행할 때 두 가지를 내려받습니다.
  - Penn-Fudan 데이터 약 50MB. `data/`에 저장합니다.
  - `yolo11s.pt` 18MB. GitHub 다운로드가 실패하면 [Hugging Face 미러](https://huggingface.co/Ultralytics/YOLO11/resolve/main/yolo11s.pt)에서 받아 이 폴더에 두면 됩니다.
- Penn-Fudan을 이미 갖고 있다면 `PENN_DATA_ROOT`에 `PennFudanPed` 폴더의 상위 폴더 경로를 넣어 주세요.
- 학습 결과는 `runs/person/`에 남습니다. 다시 학습하면 이 결과를 덮어씁니다.
- 빠르게 확인하려면 `YOLO_EPOCHS=10 uv run python lab5_yolo11.py`처럼 epoch 수를 줄일 수 있습니다.

## 라이선스

- lab1~lab4와 common.py는 교안 코드와 같은 조건을 따릅니다.
- lab5와 lab6은 ultralytics 패키지를 쓰므로 **AGPL-3.0**을 따릅니다. 약관(https://www.ultralytics.com/license)상 직접 학습한 모델에도 AGPL-3.0이 적용됩니다.
- Penn-Fudan 데이터는 원 배포처의 조건을 따릅니다.
