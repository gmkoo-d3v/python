# 전이학습 해커톤 — Flowers-102

꽃 사진 102종을 분류한다. **학습 사진은 한 종에 10장뿐**이다. 부록 A에서 배운 과적합 막기로
시험 정확도를 최대한 끌어올린다.

## 시작

```bash
uv run baseline.py
```

ResNet18 사전학습 가중치(약 45MB)를 처음 한 번 내려받는다. 끝나면 `submission.csv`가 생긴다.

| 폴더 | 사진 | 라벨 |
|---|---|---|
| `data/train` | 1,020 | 폴더 이름 `000`~`101` |
| `data/val` | 1,020 | 폴더 이름 `000`~`101` |
| `data/test` | 6,149 | 없음 — 이것을 맞힌다 |

## 고칠 곳

`baseline.py`의 `[바꿀 곳 1]`~`[5]`: 변환 · 모델(무엇을 얼리나) · 손실 · 옵티마이저와 스케줄 · 학습 루프.
`predict()`에 테스트 시 증강을 넣어도 된다. `common.py`는 고치지 않는다.

## 규칙

- 모델은 `torchvision.models.resnet18`의 ImageNet 가중치(`IMAGENET1K_V1`)만. 다른 모델·가중치 금지.
- 외부 데이터 금지. torchvision 의 Flowers102 를 내려받는 것도 금지 — 받은 zip 만 쓴다.
- 입력 해상도 224px 이하. 앙상블은 모델 2개까지.
- 검증 사진(`data/val`)은 학습에 합쳐도 된다 — 라벨이 있는 우리 데이터다.
- `data/test` 사진은 예측에만 쓴다. 시험 사진으로 학습하거나 손으로 라벨을 붙이지 않는다.
- 상위 팀은 주최 측 컴퓨터에서 코드를 다시 돌려 확인한다. 비공개 점수가 1%p 넘게 다르면 순위를 인정하지 않는다.

## 제출과 순위

- 공개 보드: 시험 사진의 절반으로 매긴 점수. 대회 중에 보여 준다. 팀당 10번까지 제출.
- 비공개 보드: 나머지 절반. **최종 순위는 이것으로** 정한다. 끝날 때 최종 제출 1개를 고른다
  (고르지 않으면 공개 점수가 가장 높았던 것).
- 제출 파일 이름은 `팀이름_번호.csv`, 그 제출을 만든 `baseline.py`를 같이 낸다.
- 3,000장 기준 정확도 흩어짐(표준오차)은 약 0.5%p. 1%p 안쪽 차이는 같은 점수로 본다.

## 발표 (상위 팀 5분)

어떤 기법이 몇 %p를 올렸는지 표 하나. 각 기법은 **시드 2개 이상**으로 돌려 평균 ± 표준편차로 적는다.


```
- 기본 설정
epoch  1  검증 0.2755
epoch  2  검증 0.5618
epoch  3  검증 0.6559
epoch  4  검증 0.7216
epoch  5  검증 0.7382
epoch  6  검증 0.7500
epoch  7  검증 0.7716
epoch  8  검증 0.7775
epoch  9  검증 0.7794
epoch 10  검증 0.7794
최종 검증 정확도 0.7794  @val_acc=0.7794
제출 파일: C:\code\src\lab\transfer_learning\submission.csv
```
```
첫시도 IMAGE_SIZE=95
epoch  1  검증 0.1892
epoch  2  검증 0.4137
epoch  3  검증 0.5000
epoch  4  검증 0.5225
epoch  5  검증 0.5422
epoch  6  검증 0.5549
epoch  7  검증 0.5725
epoch  8  검증 0.5725
epoch  9  검증 0.5794
epoch 10  검증 0.5863
최종 검증 정확도 0.5863  @val_acc=0.5863
제출 파일: C:\code\src\lab\transfer_learning\submission.csv
```
```
두번째 image_size = 320 <- 224 제약 초과
p.requires_grad = True
epoch  1  검증 0.3324
epoch  2  검증 0.5902
epoch  3  검증 0.6833
epoch  4  검증 0.6892
epoch  5  검증 0.8284
epoch  6  검증 0.7676
epoch  7  검증 0.8676
epoch  8  검증 0.8853
epoch  9  검증 0.8922
epoch 10  검증 0.8971
최종 검증 정확도 0.8971  @val_acc=0.8971
제출 파일: C:\code\src\lab\transfer_learning\submission.csv
```
```
이미지 사이즈 220
p.requires_grad = true
epoch  1  검증 0.3343
epoch  2  검증 0.5902
epoch  3  검증 0.6588
epoch  4  검증 0.7843
epoch  5  검증 0.8010
epoch  6  검증 0.7971
epoch  7  검증 0.8196
epoch  8  검증 0.8412
epoch  9  검증 0.8480
epoch 10  검증 0.8451
최종 검증 정확도 0.8451  @val_acc=0.8451
제출 파일: C:\code\src\lab\transfer_learning\submission.csv
```

```

```