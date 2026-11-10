# 10장 도전과제 — b03_structure.py · b08_freeze.py · b10_bn_trap.py · b11_preprocess.py · 전이도구.py 기반
import torch
from torch import nn
from torchvision import models
from torchvision.transforms import v2

from 전이도구 import 구간_시작

# ---------------------------------------------------------------
# 1번: 입력을 160×160으로 바꾸면 layer3 ~ fc의 모양은?
#   224 → 7이던 특징 맵이 160 → 5가 된다(160 / 32 = 5).
#   fc 바로 앞의 avgpool이 공간을 1×1로 접으므로 fc 입력은 512 그대로다.
# ---------------------------------------------------------------
모델 = models.resnet18(weights=None)
x = torch.zeros(1, 3, 160, 160)

보고_할_층 = {"layer3", "layer4", "avgpool", "fc"}
for 이름, 부품 in 모델.named_children():
    if 이름 == "fc":
        x = torch.flatten(x, 1)            # resnet18의 forward가 fc 바로 앞에서 하는 일
    x = 부품(x)
    if 이름 in 보고_할_층:
        print(f"{이름:8} {tuple(x.shape)}")

print("fc:", 모델.fc)
print("→ fc 입력 차원:", 모델.fc.in_features)

# ---------------------------------------------------------------
# 2번: 머리를 3클래스로 바꾸면 파라미터는?
#   몸통 11,176,512 + 새 머리(512×3 + 3) = 11,178,051
# ---------------------------------------------------------------
원래_전체 = sum(p.numel() for p in 모델.parameters())
원래_머리 = sum(p.numel() for p in 모델.fc.parameters())
몸통 = 원래_전체 - 원래_머리

모델.fc = nn.Linear(512, 3)
새_머리 = sum(p.numel() for p in 모델.fc.parameters())
새_전체 = sum(p.numel() for p in 모델.parameters())

print(f"몸통: {몸통:,}개 / 새 머리: {새_머리:,}개 / 새 전체: {새_전체:,}개")
with torch.no_grad():
    print("3클래스 출력:", tuple(모델.eval()(torch.zeros(1, 3, 160, 160)).shape))

# ---------------------------------------------------------------
# 4번: 학습 1,200장 + 검증 100장인데 검증 구간이 1000에서 시작하면?
#   클래스마다 학습은 0~1199, 검증은 1000~1099를 고른다.
#   → 검증 사진 100장이 전부 학습에도 들어간다(10클래스면 1,000장).
#   검증 정확도가 '처음 보는 사진' 성적이 아니라 '외운 사진' 성적이 되어
#   과적합을 못 잡고, 모델 선택·조기 종료도 낙관적으로 치우친다.
# ---------------------------------------------------------------
학습_수, 검증_수 = 1200, 100
학습_구간 = set(range(구간_시작["학습"], 구간_시작["학습"] + 학습_수))
검증_구간 = set(range(구간_시작["검증"], 구간_시작["검증"] + 검증_수))
겹침 = 학습_구간 & 검증_구간
print(f"클래스당 겹치는 사진: {len(겹침)}장 → 전체 {len(겹침) * 10:,}장")

# 고치는 방법 1: 검증 시작을 학습이 끝나는 1200으로 옮긴다
구간_시작["검증"] = 1200
검증_구간_고침 = set(range(구간_시작["검증"], 구간_시작["검증"] + 검증_수))
print("검증 시작을 1200으로 옮긴 뒤 겹침:", len(학습_구간 & 검증_구간_고침))

# 고치는 방법 2(구간을 그대로 둘 때): 학습을 1,000장 이하로 줄인다
print("학습을 1000장으로 줄이면 겹침:",
      len(set(range(0, 1000)) & set(range(1000, 1100))))

# ===============================================================
# 두 번째 도전과제 — b08_freeze.py · b10_bn_trap.py · b11_preprocess.py
# ===============================================================

# ---------------------------------------------------------------
# 5번(새 1번): 머리를 바꾼 뒤에 전부 얼리면?
#   얼리기 전에 만든 새 머리도 얼어 버려서 학습 대상이 0개가 된다.
#   기울기를 만들 텐서가 없으므로 backward()는 그래프를 못 만들고 오류를 낸다.
# ---------------------------------------------------------------
모델_거꾸로 = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
모델_거꾸로.fc = nn.Linear(모델_거꾸로.fc.in_features, 10)   # 먼저 머리 교체
for p in 모델_거꾸로.parameters():
    p.requires_grad_(False)                                  # 그다음 전부 얼림

print("학습 대상:", sum(p.numel() for p in 모델_거꾸로.parameters() if p.requires_grad), "개")
손실 = nn.functional.cross_entropy(
    모델_거꾸로(torch.randn(4, 3, 112, 112)), torch.tensor([0, 1, 2, 3]))
print("손실에 grad_fn이 있나:", 손실.grad_fn)
try:
    손실.backward()
except RuntimeError as 오류:
    print("backward() 오류:", 오류)

# ---------------------------------------------------------------
# 6번(새 2번): layer4만 다시 녹이면 학습 대상은?
#   layer4 8,393,728 + 새 머리 5,130 = 8,398,858개
#   전체 11,181,642개 중 75.1%
# ---------------------------------------------------------------
모델_layer4 = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
for p in 모델_layer4.parameters():
    p.requires_grad_(False)
모델_layer4.fc = nn.Linear(모델_layer4.fc.in_features, 10)
모델_layer4.layer4.requires_grad_(True)                      # 추가한 한 줄

전체 = sum(p.numel() for p in 모델_layer4.parameters())
학습_대상 = sum(p.numel() for p in 모델_layer4.parameters() if p.requires_grad)
print(f"전체 {전체:,}개 중 학습 대상 {학습_대상:,}개 ({학습_대상 / 전체:.1%})")
print("layer4:", f"{sum(p.numel() for p in 모델_layer4.layer4.parameters()):,}개",
      "/ 머리:", f"{sum(p.numel() for p in 모델_layer4.fc.parameters()):,}개")

# ---------------------------------------------------------------
# 7번(새 3번): ResNet18의 BatchNorm2d와 버퍼 수
#   BN 20개 × (running_mean, running_var, num_batches_tracked) = 버퍼 60개
#   버퍼는 parameters()에 안 들어가고 buffers()에만 들어간다.
# ---------------------------------------------------------------
모델_bn = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
bn_층 = [이름 for 이름, 층 in 모델_bn.named_modules() if isinstance(층, nn.BatchNorm2d)]
버퍼 = list(모델_bn.named_buffers())
print("BatchNorm2d:", len(bn_층), "개 / 버퍼:", len(버퍼), "개")
print("BN 한 층의 버퍼:", [이름 for 이름, _ in 모델_bn.bn1.named_buffers()])

# ---------------------------------------------------------------
# 8번(새 4번): scale을 빼먹으면?
#   scale=True  : 255 → 1.0 → (1.0 - 0.485) / 0.229 = 2.249
#   scale=False : 255 → 255.0 → (255.0 - 0.485) / 0.229 = 1,111.419
# ---------------------------------------------------------------
빨강_255 = torch.tensor([[[[255]], [[0]], [[0]]]], dtype=torch.uint8)   # R=255, G=B=0
정규화 = v2.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
스케일_있음 = v2.Compose([v2.ToDtype(torch.float32, scale=True), 정규화])
스케일_없음 = v2.Compose([v2.ToDtype(torch.float32), 정규화])

print("scale=True  →", f"{스케일_있음(빨강_255)[0, 0, 0, 0].item():.3f}")
print("scale 빼먹음 →", f"{스케일_없음(빨강_255)[0, 0, 0, 0].item():.1f}")
