# LLM 인계 문서 — Flowers-102 해커톤 (2026-10-23 기준)

## 0. 한 줄 요약
ResNet18(IG1K_V1) 전이학습으로 test 6,149장 기준 **0.9455 달성(제출물 백업 완료)**.
현재 **v7(120 epoch) 학습 중** — 완료되면 채점 → v5 대비 판정 → 교체/유지 결정.
이 문서는 그 이후 작업 전체를 인계받는 LLM을 위한 것.

## 1. 반드시 읽을 파일 (우선순위)
| 파일 | 용도 |
|---|---|
| `README.md` | 대회 규칙 원문 (27-32행: 모델/데이터/해상도/앙상블/합침/test) |
| `baseline.py` | 현재 코드 (v7: 120ep, SWA 40, BN 재추정 공정 비교, 가중치 저장) |
| `log.txt` | 전체 기법 변경 로그 + 성적표 + 부록 A 대조표 |
| `grade.py` | 공식 채점기 (전체 6,149장) — **신뢰성 캘리브레이션 완료** |
| `answers.csv` | (id, label, board) — 공개/비공개 반쪽 구분표 |
| `common.py` | 고치지 않음(규칙). SEED=0, NUM_WORKERS=2, loader 등 |

## 2. 현재까지 실측 (모두 grade.py 6,149장)
| 버전 | 구성 | 점수 |
|---|---|---|
| baseline(원본 120px/헤드만) | CE+Adam 1e-3, 증강 0, 10ep | 0.7794 (val 기준) |
| v1 (220px, 전체 해동) | 첫 제출 | 0.9094 (공개 0.9105 / 비공개 0.9083) |
| v4 (합침+SWA 강제) | 선택 신호 없음 | 0.9086 |
| **v5 (합침+holdout 선택, 60ep)** | A+B 앙상블+TTA | **0.9455** (공개 0.9450 / **비공개 0.9460**) ← 현재 최고 |
| **v7 (실행 중)** | 120ep + SWA40 + 후보 공정 비교 | 미측정 ← 완료되면 채점 필요 |

**백업 파일**:
- `submission_v5_9406.csv` = 실제 v5 산출물 (MD5 `1a3a6cfe2255c5cbe1f358ce607745d6`) — **0.9455 확정본**
- `submission_v1_9094.csv` = v1 산출물
- `submission.csv` = 마지막 실행 결과 (현재 v5와 동일)
- `checkpoints/v7_{A,B}.pt` — v7 실행 완료 시 생김 (v5 가중치는 소실됨, 복구 불가)

## 3. v7 실행 중 (내가 방금 시작)
- shell id: `sh_167b09e78001UWmM1UecCDF6vY` (opencode 백그라운드)
- GPU 82% 사용 확인됨 (CPU 높은 건 DataLoader 워커, 정상)
- 완료 알림 오면 → `uv run grade.py submission.csv`로 채점 → 아래 4절 판정

**만약 실행이 죽어 있거나 인계 후 재실행 필요하면**:
```bash
cd src/lab/transfer_learning && uv run baseline.py
# v7은 A 120ep + B 120ep = 총 240 에폭, 조기 중단 없음, 20~40분 예상
```

## 4. v7 완료 후 판정 규칙 (패널 봉합 — 그대로 쓸 것)
```
전체 6,149장 채점 후:
- ≥ 0.9545 (v5 +1.0pp 이상) → v7 제출물이 새 챔피언. backup: cp submission.csv submission_v7_xxxx.csv
- 0.9435~0.9545 (±1.0pp 이내) → 통계적 동점 (SE ±0.38pp) → v7에서 학습 곡선이 안정적으로
  상승했고 holdout 점수도 높으면 v7 유지, 애매하면 v5 유지. 제출 슬롯 소모 금지.
- ≤ 0.9435 → v5 유지. v7에서 뭘 바꿨는지 log.txt에 원인 기록 후 회귀.
참고: v5 비공개(0.9460) > 공개(0.9450) → 선택 편향 없음. v7도 반쪽 분해 계산할 것.
```
반쪽 분해 계산법 (answers.csv 이용):
```python
import csv
ans = {r["id"]: (int(r["label"]), r["board"]) for r in csv.DictReader(open("answers.csv", encoding="utf-8"))}
tot = {"public": [0,0], "private": [0,0]}
for r in csv.DictReader(open("submission.csv", encoding="utf-8")):
    lab, board = ans[r["id"]]
    tot[board][0] += int(r["label"]) == lab; tot[board][1] += 1
print({k: round(v[0]/v[1], 4) for k, v in tot.items()})
```

## 5. v7 이후 잔여 작업 (패널 봉합 순서 — 이거부터)
패널2(Oracle/DeepSeek/GPT석/Claude석/Big Pickle) 봉합안 — 전부 2시드 평균으로 검증해 채택:
1. **L2 (무료, 재학습 불필요)**: TTA 뷰 확장 — squish→**letterbox** 교체(종횡비 왜곡 제거),
   회전 ±8~10° TTA(PIL rotate, `fillcolor=(124,116,103)` — 검은 모서리 금지).
   뷰 단독 정확도를 holdout에서 먼저 측정, 평균 대비 -1% 이내인 뷰만 추가.
2. **5-crop**: 조건부. 코너 crop 단독 정확도가 center 대비 -1% 이내일 때만 2개 추가.
3. **L3 (재학습 2배)**: cutmix(p=0.3, α=1.0, **마지막 20% 에폭에서 끔**)를 **B 브랜치만**.
   mixup은 미채택(60ep 미만에서 수렴 불량, cutmix가 파인그레인 유리).
   2시드 평균 +0.3pp 이상일 때만 채택.
4. **증강 완화(재학습 2배)**: A scale 0.65→0.70~0.75, B 0.5→0.65 — 19장/클래스가 된 시점
   과증강이 오히려 손해일 수 있음(DeepSeek).
5. 금지: holdout 102→30장 압축(SE ±4.3pp, 클래스 커버리지 붕괴), mixup 단독 적용, 5-crop 무조건,
   logit 기하평균(온도 불일치), BN 동결 해제(GPT석: 스킵, 현재 동결+재추정 구조 유지).

## 6. 룰 준수 체크리스트 (제출 전 매번)
- [ ] resnet18 + IMAGENET1K_V1만 (`baseline.py` build_model)
- [ ] 외부 데이터 없음, torchvision Flowers102 다운로드 없음
- [ ] 입력 224px 이하 — 중간 Resize 256 금지(greys zone)
- [ ] test는 순전파만 — 학습·BN 통계·라벨링에 미사용
- [ ] 앙상블 모델 2개 이하 (A+B)
- [ ] 가중치·submission 백업 (v5 가중치 소실 교훈)

## 7. 알아둘 함정 (실제 겪은 것)
1. **EMA는 3회 실패** (초기화 오염 → 에폭당 갱신 → 0.04 고정). SWA가 대안, 지금 확정.
2. **BN 통계는 학습 중 동결**(freeze_bn_stats), raw 후보도 BN 재추정 후 비교해야 공정.
3. **merged 런에서 `evaluate_tta(val)`은 누수** — val이 학습 데이터. split 인자 주의.
4. **제출 8회 남음, 최종 1개는 직접 고름(README 37행)** — 미사용 제출은 비용 0.
5. **공개보드는 비교 도구가 아님** (v1 0.9094 vs v4 0.9086 = SE 이내 동점).
   모든 판정은 grade.py(전체) 2시드 + paired로.
6. **발표 요구**: 기법당 시드 2개 이상 평균±표준편차 (README 44행) — v3 A/B 로그가
   시드 2개 근거로 쓰임 (log.txt 발표 표 참고).

## 8. 실행 명령 요약
```bash
cd src/lab/transfer_learning
uv run baseline.py                    # v7 학습 → submission.csv 생성
uv run grade.py submission.csv       # 채점 (전체 6,149장)
uv run grade.py submission_v5_9406.csv  # v5 대조 (0.9455 확인용)
# 위 4절 python 스크립트 → 반쪽 분해
```
