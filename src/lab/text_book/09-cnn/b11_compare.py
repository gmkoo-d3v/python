# [2권 Chapter 9. CNN과 이미지 분류: 돋보기를 옮기며 무늬 찾기 · 9.5 첫 CNN으로 옷 사진 분류하기 · manuscript-vol2/chapters/09-cnn.md:590]
import torch

from 옷도구 import 옷CNN, 시험_데이터, 의류분류기

이름 = ["T셔츠", "바지", "풀오버", "드레스", "코트", "샌들", "셔츠", "운동화", "가방", "앵클부츠"]
X, y = 시험_데이터()

예측 = {}
for 별명, 모델, 파일 in [("MLP", 의류분류기(), "fashion_mlp.pt"), ("CNN", 옷CNN(), "fashion_cnn.pt")]:
    모델.load_state_dict(torch.load(파일, weights_only=True))
    모델.eval()
    with torch.no_grad():
        예측[별명] = 모델(X).argmax(dim=1)
    print(f"{별명} 전체 정확도 {(예측[별명] == y).float().mean():.2%}")

print("클래스     MLP     CNN    차이")
for k in range(10):
    a, b = [(예측[m][y == k] == k).float().mean().item() for m in ("MLP", "CNN")]
    print(f"{이름[k]:6} {a:6.1%}  {b:6.1%}  {(b - a) * 100:+5.1f}%p")

for m in ("MLP", "CNN"):                         # 모델마다 가장 많이 헷갈린 짝 3개
    혼동 = torch.zeros(10, 10, dtype=torch.int64)
    혼동.index_put_((y, 예측[m]), torch.ones(len(y), dtype=torch.int64), accumulate=True)
    혼동.fill_diagonal_(0)
    짝들 = [divmod(i, 10) for i in 혼동.flatten().topk(3).indices.tolist()]
    print(m, "헷갈림:", ", ".join(f"{이름[a]}→{이름[b]} {혼동[a, b]}번" for a, b in 짝들))


    셔츠 = 이름.index("셔츠")  # 6
    셔츠_예측 = 예측["CNN"][y == 셔츠]  # 셔츠 사진 1000장에 대한 CNN 예측
    개수 = torch.bincount(셔츠_예측, minlength=10)  # 클래스별 개수
    print(f"CNN이 셔츠 {len(셔츠_예측)}장을 예측한 결과:")
    for k in range(10):
        print(f"  {이름[k]:6} {개수[k].item():4}장")
"""
MLP 전체 정확도 88.25%
CNN 전체 정확도 91.26%
클래스     MLP     CNN    차이
T셔츠     81.9%   86.1%   +4.2%p
바지      96.9%   99.3%   +2.4%p
풀오버     75.1%   87.4%  +12.3%p
드레스     88.4%   89.7%   +1.3%p
코트      89.1%   81.7%   -7.4%p
샌들      95.9%   98.1%   +2.2%p
셔츠      68.9%   78.1%   +9.2%p
운동화     95.6%   98.3%   +2.7%p
가방      95.6%   97.9%   +2.3%p
앵클부츠    95.1%   96.0%   +0.9%p
MLP 헷갈림: 풀오버→코트 160번, T셔츠→셔츠 117번, 셔츠→코트 92번
CNN 헷갈림: T셔츠→셔츠 112번, 셔츠→T셔츠 104번, 코트→셔츠 87번
"""