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
| lab7_eval.py | 6 | IoU 기준별 AP, conf 고르기, pycocotools로 다시 재기 | 약 25초 |
| lab8_errors.py | 7 | 오류를 위치 어긋남, 중복, 배경 오탐, 놓침으로 세기 | 약 5초 |
| lab9_test_split.py | 8 | 학습·검증·시험으로 나눠 다시 학습, 검증 점수와 시험 점수 견주기 | 약 2분 40초 |
| lab10_onnx.py | 9 | ONNX로 내보내고 onnxruntime만으로 출력을 직접 풀기 | 약 25초 |
| lab11_speed.py | 10 | 입력 크기(320/416/640)와 int8 양자화, CPU 속도와 mAP | 약 40초 |

lab6~lab11은 lab5가 만든 `runs/person/weights/best.pt`와 `data/PennFudan_yolo`를 씁니다(`common_yolo.py`).

## 실행

[uv](https://docs.astral.sh/uv/)가 있어야 합니다. 명령은 모두 이 폴더 안에서 실행합니다.

```bash
uv run --no-project --with torch --with torchvision python lab1_iou.py      # lab1~4 모두 같은 방식
uv run --no-project --with ultralytics --with pyyaml python lab5_yolo11.py
uv run --no-project --with ultralytics python lab6_video.py
uv run --no-project --with ultralytics --with pycocotools python lab7_eval.py
uv run --no-project --with ultralytics python lab8_errors.py
uv run --no-project --with ultralytics --with pyyaml python lab9_test_split.py
uv run --no-project --with ultralytics --with onnx --with onnxruntime --with onnxslim python lab10_onnx.py   # lab11도 같은 방식
```

- lab5는 처음 실행할 때 두 가지를 내려받습니다.
  - Penn-Fudan 데이터 약 50MB. `data/`에 저장합니다.
  - `yolo11s.pt` 18MB.
- 내려받다가 `CERTIFICATE_VERIFY_FAILED`(인증서 오류)가 나면 다음 순서로 해 보세요.
  1. 명령에 `--with truststore`를 더해 다시 실행합니다. 예: `uv run --no-project --with ultralytics --with pyyaml --with truststore python lab5_yolo11.py`
  2. 그래도 안 되면 브라우저로 https://www.cis.upenn.edu/~jshi/ped_html/PennFudanPed.zip 을 받아 `data/`에 풉니다.
  3. 학원·회사 네트워크라면 `yolo11s.pt` 내려받기에서도 같은 오류가 날 수 있습니다. 브라우저로 받아 이 폴더에 두면 됩니다.
- Penn-Fudan을 이미 갖고 있다면 `PENN_DATA_ROOT`에 `PennFudanPed` 폴더의 상위 폴더 경로를 넣어 주세요.
- 학습 결과는 `runs/person/`에 남습니다. 다시 학습하면 이 결과를 덮어씁니다. lab9는 `runs/person_3way/`, lab11은 `runs/speed/`에 따로 남깁니다.

## 라이선스

- lab1~lab4와 common.py는 교안 코드와 같은 조건을 따릅니다. common_yolo.py는 ultralytics를 쓰지 않습니다.
- lab5~lab11은 ultralytics 패키지를 쓰므로 **AGPL-3.0**을 따릅니다. 약관(https://www.ultralytics.com/license)상 직접 학습한 모델에도 AGPL-3.0이 적용됩니다.
- Penn-Fudan 데이터는 원 배포처의 조건을 따릅니다.
