"""10차시 실습: 빠르게 만들고, 무엇을 잃었는지 잰다 — 입력 크기와 int8 양자화.

모두 CPU에서 onnxruntime으로 잰다(현장의 작은 컴퓨터를 흉내). 숫자는 기계마다 다르니 비율로 읽는다.
"""
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
from onnxruntime.quantization import CalibrationDataReader, QuantFormat, QuantType, quantize_static
from ultralytics import YOLO

from common_yolo import BEST, HERE, YOLO_ROOT, require_lab5
from lab10_onnx import letterbox

SIZES = (320, 416, 640)
CALIB_IMAGES = 32          # 양자화 범위를 정하는 데 쓸 학습 사진 수(검증 사진은 쓰지 않는다)
REPEAT = 30
OUT = HERE / "runs" / "speed"
QUANTIZE_ALL = os.environ.get("QUANTIZE_ALL") == "1"


def export(size: int) -> Path:
    path = Path(YOLO(str(BEST)).export(format="onnx", imgsz=size, simplify=True, verbose=False))
    target = OUT / f"best_{size}.onnx"
    shutil.move(path, target)
    return target


def latency_ms(onnx_path: Path, size: int) -> float:
    import lab10_onnx
    lab10_onnx.IMAGE_SIZE = size   # 같은 전처리를 크기만 바꿔 쓴다
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    name = session.get_inputs()[0].name
    array, _, _ = letterbox(sorted((YOLO_ROOT / "valid" / "images").glob("*.png"))[0])
    for _ in range(3):
        session.run(None, {name: array})
    start = time.perf_counter()
    for _ in range(REPEAT):
        session.run(None, {name: array})
    lab10_onnx.IMAGE_SIZE = 416
    return (time.perf_counter() - start) / REPEAT * 1000


def accuracy(onnx_path: Path, size: int) -> tuple[float, float]:
    box = YOLO(str(onnx_path), task="detect").val(data=str(YOLO_ROOT / "data.yaml"), imgsz=size, device="cpu", plots=False, verbose=False).box
    return box.map50, box.map


class Calibration(CalibrationDataReader):
    """int8 범위를 정할 때 쓸 입력을 하나씩 내준다."""

    def __init__(self, input_name: str) -> None:
        paths = sorted((YOLO_ROOT / "train" / "images").glob("*.png"))[:CALIB_IMAGES]
        self.items = iter([{input_name: letterbox(p)[0]} for p in paths])

    def get_next(self) -> dict | None:
        return next(self.items, None)


def main() -> int:
    require_lab5()
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"{'모델':22s}{'크기(MiB)':>10s}{'CPU ms/장':>11s}{'mAP50':>8s}{'mAP50-95':>10s}")
    rows = {}
    for size in SIZES:
        path = export(size)
        rows[size] = (path, latency_ms(path, size), *accuracy(path, size))
        print(f"{'float32 imgsz ' + str(size):22s}{path.stat().st_size / 2**20:10.1f}{rows[size][1]:11.1f}{rows[size][2]:8.3f}{rows[size][3]:10.3f}")

    fp32 = rows[416][0]
    int8 = OUT / "best_416_int8.onnx"
    name = ort.InferenceSession(str(fp32), providers=["CPUExecutionProvider"]).get_inputs()[0].name
    quantize_static(str(fp32), str(int8), Calibration(name), quant_format=QuantFormat.QDQ,
                    activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8, per_channel=True,
                    # 합성곱만 int8로. 전부 누르면(QUANTIZE_ALL=1) 이 모델에서는 mAP가 0이 됐다.
                    # 출력 끝에서 좌표(0~416)와 점수(0~1)를 한 텐서로 잇기 때문으로 짐작한다(따로 가르지는 않았다)
                    op_types_to_quantize=None if QUANTIZE_ALL else ["Conv"])
    ms = latency_ms(int8, 416)
    m50, m = accuracy(int8, 416)
    label = "int8 전부 416" if QUANTIZE_ALL else "int8 합성곱만 416"
    print(f"{label:22s}{int8.stat().st_size / 2**20:10.1f}{ms:11.1f}{m50:8.3f}{m:10.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
