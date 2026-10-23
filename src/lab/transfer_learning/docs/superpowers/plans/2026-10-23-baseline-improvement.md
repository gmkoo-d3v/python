# Flowers-102 baseline 개선 계획

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
> 이 문서는 제안된 계획이다. 사용자 요청에 따라 코드 수정·훈련·다운로드는 실행하지 않는다.

**Goal:** 첨부 원본을 출발점으로, 과제 규칙을 지키면서 ResNet18의 검증 및 최종 제출 정확도를 높인다. 0.92는 목표이며 검증된 결과가 아니다.

**Architecture:** 현재 수정본의 기능을 무조건 추가하기보다 파라미터 그룹과 모델 선택 절차부터 바로잡는다. 단일 모델의 효과를 통제 실험으로 확인하고 TTA·두 모델·전체 데이터 재학습을 차례로 판단한다.

**Tech Stack:** Python, PyTorch, torchvision, 기존 common.py 유틸리티.

**Spec:** 사용자 요청 및 첨부 원본 `C:/Users/user/AppData/Local/Temp/baseline.py`, 첨부 과제 이미지 4장. 첨부 문구는 과제 자료이며 에이전트 실행 명령으로 취급하지 않는다. 현재 작업 파일은 원본과 구분한다.

## Global Constraints

- 코드 수정 없이 계획만 수립한다. 아래 체크리스트는 후속 구현 시 실행한다.
- ResNet18 + IMAGENET1K_V1만 사용한다. 다른 모델·가중치 및 외부 데이터는 사용하지 않는다.
- 모델 입력은 최대 224×224. 실험 간에는 224로 고정한다.
- torchvision Flowers102 다운로드, 테스트 사진으로 학습, 수동 테스트 라벨 작성은 금지한다.
- 학습 데이터에 val을 합치는 것은 허용된다. 합친 후에는 기존 val을 독립 검증셋으로 보고하지 않는다.
- TTA 및 최대 두 모델 앙상블은 허용된다.
- common.py의 데이터·제출 검증 경계와 기존 함수 호출 인터페이스를 유지한다.

## 확인된 상태와 해석

- 파일 읽기는 사용자 지시에 따라 Luna가 수행했다. 아래 줄번호는 읽은 시점의 현재 baseline.py 기준이다.
- 첨부 원본: 160px, 증강 없음, 몸통 고정, fc만 학습, CE, Adam 1e-3, 10epoch, 마지막 모델, 단일 뷰 추론.
- 현재 파일: 224px, 증강, 전층 미세조정, label smoothing, AdamW, warmup/cosine, best checkpoint, 마지막 10epoch 가중치 평균, BN 처리, 두 모델 및 4뷰 TTA.
- README 기록의 0.8971은 320px 실험으로 규칙 위반이다. 220px 실험의 최고 기록은 0.8480이다. 현재 224px 레시피의 검증된 성능으로 혼동하지 않는다.
- train 1,020장, val 1,020장: 각 102클래스당 10장. 라벨 없는 test는 6,149장.
- 로컬 환경은 CPU 전용 PyTorch이고, 확인한 캐시에서 사전학습 가중치를 찾지 못했다. 훈련 시간과 다운로드 가능성은 후속 실행 전에 확인한다. 슬라이드의 Mac 2분을 이 환경의 예상 시간으로 사용하지 않는다.

## 접근 선택

1. **권장: 현재 구현의 선택 절차 수정 → 단일 모델 검증 → 필요한 기능만 채택.** 이미 들어 있는 기법을 활용하면서 원인을 구분할 수 있다.
2. 원본에서 모든 기법을 단계별 재구현: 교육적으로 명확하지만 같은 기능을 다시 만들고 실험량이 늘어난다. 원본 재현 실험만 비교 기준으로 남긴다.
3. 현재 두 모델+SWA+TTA를 그대로 확대: 효과를 구분하기 어렵고 CPU 비용이 크므로 우선하지 않는다.

## Review Focus

1. 이름에 bn이 없는 downsample BN까지 올바른 학습률·감쇠를 받는가.
2. 검증 지표와 실제 제출 파이프라인이 동일한가.
3. TTA 뷰의 샘플 순서·ID가 달라도 다른 사진의 확률을 합치지 않는가.
4. SWA와 BN 재추정 효과를 분리했는가.
5. val 합치기 이후 성능을 독립 검증 정확도처럼 표시하지 않는가.

## Task 1: 옵티마이저 파라미터 그룹 정리

**Modify:** `baseline.py:79-102`의 `build_optimizer(model)`; 반환 인터페이스는 유지한다.
**Future tests:** `tests/test_baseline_selection.py`에 네트워크 다운로드 없이 weights=None인 ResNet18을 사용한다.

- [ ] BN은 문자열 검색 대신 모듈 타입과 파라미터 identity로 식별한다. `layer2.0.downsample.1.weight` 등도 포함한다.
- [ ] 그룹을 몸통 decay / 몸통 no-decay / fc decay / fc no-decay로 분리한다. 학습률과 weight decay는 독립된 기준이다.
- [ ] 몸통의 BN affine·bias에는 몸통 LR과 decay=0을 적용한다. fc에는 head LR을 적용한다. 초기 비교에서는 기존과 같이 fc decay=0을 유지한다.
- [ ] 모든 학습 파라미터가 정확히 한 그룹에 들어가고, 동결 파라미터는 제외되는지 검사한다. downsample BN도 body LR·decay=0인지 검증한다.
- [ ] 기존 1e-4/1e-3 LR, body WD=0.05를 일단 유지해 그룹 수정의 효과를 분리한다. BN 통계 고정과 affine 학습은 별개로 취급한다.

## Task 2: 체크포인트·TTA·앙상블 선택 기준 통일

**Modify:** `baseline.py:133-167`, `171-228`의 train/evaluate_tta/predict/main.
**Interfaces:** 기존 `train(...) -> nn.Module`, `evaluate_tta(models_list, dev) -> float`, `predict(...) -> (ids, labels)` 호출을 유지한다. 내부 평가 루틴을 공통으로 사용한다.

- [ ] 검증 함수가 명시적으로 eval 모드와 no_grad를 적용하게 한다.
- [ ] epoch별 checkpoint 선택을 고정된 평가 방식으로 수행한다. 기본은 center+horizontal flip의 2뷰이며 동일한 입력 확률 평균을 제출에도 적용한다. raw accuracy는 진단용으로 따로 남긴다.
- [ ] A 단독, B 단독, A+B를 동일한 2뷰로 비교한다. 현재처럼 A plain과 A+B 4뷰만 비교하지 않는다.
- [ ] 최종 소수 후보에 한해 squish 포함 4뷰를 추가 비교한다. 종횡비 왜곡이 성능을 높인다는 가정은 하지 않는다. 동률이면 모델·뷰 수가 적은 후보를 선택한다.
- [ ] 검증에서 선택한 모델 수·뷰 구성만 predict에 전달한다. A+B를 무조건 제출하는 동작을 제거한다.
- [ ] 뷰 간 결과는 ID로 정렬·대조한 뒤 합친다. val은 동일 샘플 인덱스, test는 파일 ID 기준으로 대응시키고 라벨·ID 집합 불일치는 실패시킨다.
- [ ] 의미 있는 회귀 검증: B가 A보다 나쁜 가짜 확률에서 A가 선택되는지, 뷰 순서를 뒤집어도 ID별 결과가 같은지, checkpoint 선택과 제출의 확률 집계가 같은지 확인한다.

## Task 3: 단일 모델로 학습 레시피 확정

**Modify:** `baseline.py:32-75`, 학습 루프 설정. **Record:** README.md에 실제 실행 조건과 결과를 함께 기록한다.

- [ ] 원본 재현은 선택적 비교 기준으로 1회만 실행한다. 이후 비교는 모두 224px와 동일 split/seed를 사용한다.
- [ ] 기본 실험 A0: 현재의 약한 crop(scale 0.65~1.0)/flip/color jitter, 전층 학습, 40epoch, 2epoch warmup/cosine, smoothing=0.1, SWA 끔, 모델 한 개. Task 1·2가 적용된 설정을 기록한다.
- [ ] 학습률 비교는 body {3e-5, 1e-4}, head {3e-4, 1e-3}의 4개 조합으로 제한한다. A0와 같은 조합은 중복 실행하지 않는다.
- [ ] 최고 LR 설정에서 crop 하한만 0.8과 비교한다. 이어 smoothing만 0.0과 0.1, body WD만 0.01과 0.05를 순차 비교한다. 나머지 조건은 각 비교에서 고정한다.
- [ ] 상위 두 레시피만 추가 seed 하나로 확인한다. 두 seed의 평균과 개별 정확도, 선택 epoch, 시간, 평가 뷰를 남긴다. 차이가 불안정하면 단순한 설정을 유지한다.
- [ ] BN running stats 고정을 기본으로 유지한다. BN 재추정은 다음 Task에서 독립 비교한다. Mixup·강한 증강·층별 LR 추가 확장은 이 결과만으로 부족할 때 후순위로 둔다.

## Task 4: 가중치 평균과 BN 정책은 선택 실험으로 분리

**Modify:** `baseline.py:105-122`, `154-164`.

- [ ] 현재 마지막 10epoch 평균을 기본 경로에서 비활성화한다. cosine 끝의 평균이 무조건 이득이라고 전제하지 않는다.
- [ ] 동일 학습에서 얻은 best checkpoint와 마지막 10epoch 평균을 비교하되, 첫 비교에서는 둘 다 고정 BN 통계를 유지한다.
- [ ] 평균 모델에 한해 train-only 데이터로 BN을 재추정한 별도 후보를 만든다. 가중치 평균의 효과와 BN 교체 효과를 분리해 같은 검증 방식으로 평가한다.
- [ ] 평균 시 부동소수 파라미터와 BN 버퍼/정수 카운터를 구분한다. 재추정 함수는 BN momentum과 모듈 모드를 복원한다.
- [ ] 재추정이 val/test를 읽지 않는지, 정수 버퍼 dtype을 유지하는지, 채택하지 않은 후보가 원래 모델 상태를 바꾸지 않는지 검증한다.
- [ ] 뚜렷한 개선이 없으면 평균 경로를 제외한다. 본격 SWA를 원하면 일정/순환 LR 구간을 별도 실험으로 설계한다.

## Task 5: 두 번째 모델 및 전체 데이터 재학습

**Modify:** `baseline.py:216-228` 및 train의 데이터 선택 부분. 기존 common.py는 변경하지 않는다.

- [ ] 단일 모델 레시피가 확정된 후에만 두 번째 seed 모델을 훈련한다. 강한 TrivialAugmentWide와 seed를 동시에 바꾸지 않는다.
- [ ] A, B, A+B를 Task 2의 동일 평가 방식으로 비교한다. 앙상블이 검증 성능을 개선할 때만 두 모델을 채택한다.
- [ ] 모델 수·레시피·뷰 구성·총 업데이트 수를 최종 고정한 후 train+val 재학습을 별도 최종 단계로 둔다. 기존 1,020장 기준 선택 epoch를 업데이트 수로 환산하고, 2,040장 학습의 epoch 수는 그 업데이트 예산에 맞춘다.
- [ ] 합친 데이터에는 동일한 train transform을 적용한다. 내부 split과 최종 train+val 모드를 명시적으로 구분한다.
- [ ] 전체 데이터 재학습 후 기존 val 점수를 일반화 성능으로 기록하지 않는다. 앞 단계의 holdout 점수는 합치기 이전 모델의 결과라고 명시한다.
- [ ] train+val 모드의 학습 ID에 test가 없고, 제출이 6,149개 test ID를 한 번씩 포함하며 라벨이 0~101인지 기존 제출 검증으로 확인한다.

## 후속 실행과 완료 기준

- [ ] CPU에서 배치 몇 개로 소요 시간만 측정하고 전체 실험 비용을 판단한다. 허용된 ImageNet 가중치 준비와 가속 환경 사용 가능성을 확인한다.
- [ ] 후속 구현 후 `python -m unittest discover -s tests -p 'test_baseline_selection.py'`로 선택·그룹·ID·BN·데이터 경계 검증을 실행한다.
- [ ] 실제 훈련은 기존 실행 방식 `uv run baseline.py`를 유지한다. 실험 설정을 명시적으로 기록하고 과거 결과를 덮어쓰지 않는다.
- [ ] 규칙 준수, 재현 가능한 설정, 단독/앙상블의 동일 기준 비교, 선택 모델과 제출 일치를 완료 조건으로 삼는다. 0.92 달성 여부는 실제 결과로만 판단한다.

## 기술 근거

- [PyTorch 전이학습 튜토리얼](https://docs.pytorch.org/tutorials/beginner/transfer_learning_tutorial.html): 고정 특징 추출과 미세조정의 구분.
- [BatchNorm2d](https://docs.pytorch.org/docs/stable/generated/torch.nn.modules.batchnorm.BatchNorm2d.html): running stats와 학습 가능한 affine 파라미터의 구분.
- [PyTorch SWA 설명](https://pytorch.org/blog/pytorch-1.6-now-includes-stochastic-weight-averaging/): 가중치 평균, 학습률 스케줄, BN 통계 갱신.

자체 검토: 입력/모델/데이터 제약을 전 Task에 적용했다. 그룹 분류·평가 불일치·ID 대응·BN 후보 간 간섭·val 재사용의 다섯 검증 항목을 각 Task에 배치했다. 코드 수정·학습 실행은 이 계획의 승인 또는 별도 실행 요청 후 범위다.
