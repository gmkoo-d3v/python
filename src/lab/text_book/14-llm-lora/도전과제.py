# 14장 도전과제 — b01_load.py 응용
import torch
from transformers import AutoModelForCausalLM

이름 = "Qwen/Qwen2.5-0.5B-Instruct"
모델 = AutoModelForCausalLM.from_pretrained(이름, dtype=torch.float32)


def 가중치_바이트(모델: torch.nn.Module) -> int:
    """파라미터 하나하나의 (칸 수 × 칸 크기)를 더한다(4장 4.3절)."""
    return sum(p.numel() * p.element_size() for p in 모델.parameters())


print("파라미터:", f"{sum(p.numel() for p in 모델.parameters()):,}개")

fp32_바이트 = 가중치_바이트(모델)
print(f"float32: {fp32_바이트:,}바이트 = {fp32_바이트 / 1024**2:,.1f}MB (MiB)")

모델.half()                                    # 모든 실수 파라미터를 float16으로
fp16_바이트 = 가중치_바이트(모델)
print(f"float16: {fp16_바이트:,}바이트 = {fp16_바이트 / 1024**2:,.1f}MB (MiB)")
print(f"→ 절반: {fp32_바이트 / fp16_바이트:.0f}배, {fp32_바이트 - fp16_바이트:,}바이트 절약")
print("파라미터 dtype:", {p.dtype for p in 모델.parameters()})

# 494,032,768개 × 4바이트 = 1,976,131,072바이트 (1,884.6MiB)
# 494,032,768개 × 2바이트 =   988,065,536바이트 (  942.3MiB)
# element_size()가 '칸 하나의 크기'를 알려 주므로, 개수는 그대로 두고 크기만 바꾼다.
