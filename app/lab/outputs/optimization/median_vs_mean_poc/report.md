# mean vs median 줄세우기 비교 (재경기 없음, v2 top30 재채점)

챔피언 추적 대상: NSGA-t5938
- Spearman ≈ 1.00 이면 두 줄세우기가 사실상 같음 → median 으로 바꿔도 순위 안 흔들림
- max(최대수익)는 참고지표 (목적 아님)

## oos  (후보 120명 / 시장 11개)
- Spearman(mean vs median) = **0.7674**
- 1등: mean → `NSGA-t5938`  /  median → `CMA-ES-t700`  → **바뀜!**
- (참고) max 1등 → `NSGA-t5845`  → mean 1등과 다름
- 챔피언 NSGA-t5938 순위: mean 2등 / median 56등 / max 14등
- 순위 변동 큰 후보 top5 (median등 − mean등):
  - NSGA-t3640: mean 9등 → median 74등 (+65)
  - CMA-ES-t730: mean 86등 → median 27등 (-59)
  - NSGA-t4650: mean 3등 → median 61등 (+58)
  - NSGA-t3697: mean 14등 → median 71등 (+57)
  - NSGA-t5938: mean 2등 → median 56등 (+55)

## exam  (후보 120명 / 시장 6개)
- Spearman(mean vs median) = **0.0995**
- 1등: mean → `TPE-t1327`  /  median → `TPE-t936`  → **바뀜!**
- (참고) max 1등 → `GP-t81`  → mean 1등과 다름
- 챔피언 NSGA-t5938 순위: mean 68등 / median 64등 / max 32등
- 순위 변동 큰 후보 top5 (median등 − mean등):
  - CMA-ES-t730: mean 118등 → median 39등 (-79)
  - GP-t81: mean 38등 → median 106등 (+67)
  - GP-t83: mean 38등 → median 106등 (+67)
  - GP-t84: mean 38등 → median 106등 (+67)
  - GP-t85: mean 38등 → median 106등 (+67)

## holdout  (후보 120명 / 시장 7개)
- Spearman(mean vs median) = **0.9579**
- 1등: mean → `NSGA-t3640`  /  median → `NSGA-t5598`  → **바뀜!**
- (참고) max 1등 → `NSGA-t5938`  → mean 1등과 다름
- 챔피언 NSGA-t5938 순위: mean 8등 / median 20등 / max 2등
- 순위 변동 큰 후보 top5 (median등 − mean등):
  - CMA-ES-t736: mean 36등 → median 89등 (+53)
  - CMA-ES-t882: mean 59등 → median 88등 (+29)
  - CMA-ES-t742: mean 50등 → median 74등 (+24)
  - CMA-ES-t782: mean 67등 → median 90등 (+23)
  - CMA-ES-t814: mean 34등 → median 56등 (+22)
