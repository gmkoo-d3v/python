# lab - OpenCV + MediaPipe 0.10.5 (Python 3.11)

`pip install opencv-python`, `pip install mediapipe==0.10.5` 를 `uv` 로 설치된 격리 환경입니다.

## 환경
- Python: 3.11.9 (`C:\Users\user\AppData\Local\Programs\Python\Python311\python.exe`)
  - mediapipe 0.10.5는 cp38/cp39/cp310/cp311만 지원 → 3.12 이상 불가 → 3.11로 고정
  - `requires-python = ">=3.11,<3.12"` 로 제한
- uv Python pin: `C:\code\lab\.python-version` = `3.11`
- venv: `C:\code\lab\.venv` (Device Guard 우회 위해 허용 경로의 Python 3.11로 생성)
- 루트 `C:\code` 워크스페이스에서 `exclude = ["lab"]` 로 분리 → 루트(3.12)와 충돌 없음

## 설치된 패키지
- opencv-python 5.0.0.93 (opencv-contrib-python 포함)
- mediapipe 0.10.5
- numpy 2.4.6, matplotlib 등 의존성

## 사용법

```powershell
cd C:\code\lab

# uv로 실행 (권장)
uv run python test_opencv_mediapipe.py
uv run python -c "import cv2; import mediapipe as mp; print(cv2.__version__, mp.__version__)"

# venv 직접 활성화
.venv\Scripts\activate
python test_opencv_mediapipe.py

# pip 형태로 추가 설치 시
uv pip install <package>
# 또는
.venv\Scripts\pip install <package>

# 새 패키지 추가 (uv add)
uv add <package>
```

## pip install 대응
기존 요청이던:
```bash
pip install opencv-python
pip install mediapipe==0.10.5
```
는 아래로 수행됨:
```bash
cd C:\code\lab
uv add opencv-python mediapipe==0.10.5
# 내부적으로 uv pip install 과 동일, pyproject.toml에 기록됨
```

## 테스트
```
.venv\Scripts\python.exe test_opencv_mediapipe.py
# 출력: opencv 5.0.0, mediapipe 0.10.5, Hands 생성 성공
```
