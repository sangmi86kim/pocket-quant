# PocketQuant — 최적화 설계 노트

> PocketQuant의 "전략 탐색"을 최적화 문제로 정식화한 문서.
> 게임 컨셉(포켓퀀트/체육관) 뒤에 실제로 어떤 최적화가 돌고 있는지,
> 그리고 다음 단계(Optuna NSGA-III 다목적)를 어떻게 정식화할지 정리한다.
>
> v0.2(가짜 데이터·생존률) 시절 내용은 폐기.
> 2026-06-11 v0.5 → 2026-06-13 v1 마감 → 2026-06-13 v1.x 시즌 시작 (야생 7마리 + CMA-ES 추가).

---

## 1. 문제 정의 (v0.5 단일목적 — ⚠️ legacy, 코드 제거됨 2026-06-13)

> 이 절은 **역사 기록 차원**으로 남긴다 — 단일목적 GA(`engine/evolve.py`)는
> 2026-06-13 v1 마감 시점에 코드가 삭제됐다. NSGA-III만 운영 (§4).
> 적합도 = 0~100 클램프 스탯이라 "최적화 목적에 클램프 스탯 금지"
> (AGENTS.md 6번) 규칙에 걸려 어차피 선발 경로에는 못 썼다.

```
maximize   Y(X)
   over    X
```

| 기호 | 의미 | 현재 형태 |
|------|------|----------|
| **X** | 결정변수 = 전략 (어떤 시그널을 쓸지) | 6비트 on/off 벡터 `{0,1}⁶` |
| **Y** | 목적함수 = 적합도(fitness) | **평균 70% + 최약 체육관 30%** (스칼라, 0~100) |

### X — 6비트 on/off

```
X = [DD, VOL, MA, MOM, REV_RSI, REV_BB]   각 0 또는 1
탐색공간 = 2⁶ − 1 = 63가지 (완전탐색 가능 → GA는 메커니즘 검증용)
```

- 시그널 = 일별 포지션(0~1)을 내는 함수. REV_*는 이벤트형(발동일만 의견, 평소 기권 NaN).
- 결합 = **기권 제외 평균** (`signals.combine_positions`).

### Y — 단일 적합도 (스칼라, maximize)

```
체육관별 fitness = mean(ATK, DEF, SKILL)        ← 각 0~100 스탯 (HP 가중치 0)
Y = 0.7 × mean(체육관별 fitness) + 0.3 × min(체육관별 fitness)
```

- ATK = CAGR(0~25% 스케일), DEF = Calmar(CAGR/|MDD|), SKILL = 샤프.
- 평가 환경: **QQQ 실데이터 6체육관**(닷컴/금융위기/회복장/코로나/상승장/횡보장),
  거래비용 0.1%/편도, 포지션 하루 lag(룩어헤드 방지), 워밍업 400일.
- 현 챔피언: `VOL+REV_RSI+REV_BB` 42.6점 (시드 42/7 GA 수렴 = 전수조사 1위 일치).

---

## 2. 목적함수 설계에서 발견한 함정 3개 (전부 실측, 재발 방지용)

최적화기는 목적함수의 빈틈을 반드시 찾아낸다. 지금까지 "아무것도 안 하기(전부 현금)"가
최적해로 부활한 경로가 세 번 있었다:

| # | 경로 | 증상 (실측) | 봉인 방법 |
|---|------|------------|----------|
| 1 | **HP(현금 비중)를 적합도에 포함** | 전부 현금이 HP 100 + DEF 만점 = 69점 1위 | HP 가중치 0 (표시 전용), DEF=Calmar로 재설계 |
| 2 | **worst-case(min) 가중치 과다** | 현금은 전 체육관 균일 16.7점 = "최약 과목이 없음" → min 가중 0.4부터 현금 30~43/65위 부상 | min 가중치 0.3 (게이트를 지키는 실측 최대치) |
| 3 | **score_vs_dca를 평균으로 결합** | 하락 체육관에서 DCA가 마이너스 → 현금이 이김 → 평균 시 현금 24~53/65위 | 평균 결합 금지 — NSGA-III에서 **벡터 그대로** 사용 |

- 회귀 가드: `tools/test_baselines.py` (전부 현금 < 풀매수 & 하위 25% 룰).
- 공통 교훈: **0~100 클램프 스탯/BST를 최적화 목적으로 쓰지 말 것** (raw 지표 사용).
  HP가 아니라도, "낮은 노출 = 안정 점수 공짜"가 되는 모든 통로가 같은 퇴화를 만든다.

---

## 3. GA 4단계 — 원리 이해용 손코딩 (제거됨, 2026-06-13)

⚠️ `engine/evolve.py`는 2026-06-13 정리에서 제거됐다 — NSGA-III만 운영한다.
단일목적 GA는 v0.5까지 교보재로 유지됐으나, v1 마감(가중치 전용 NSGA-III) 시점에
의존 코드가 없어져 함께 정리. 이 절(아래 4단계 표·on_generation 콜백 설명)은
**역사 기록 차원**으로 남긴다 — 새 코드는 NSGA-III(§4)를 참고할 것.

| 단계 | 함수 | 하는 일 |
|------|------|---------|
| 평가 | `evaluate` | X를 전 체육관 도전시켜 Y 계산 (실데이터 = 결정론, 1회) |
| 선택 | `select` | Y 높은 순 상위 절반을 부모로 (절단 선택) |
| 교배 | `crossover` | 두 부모의 6비트를 균등 교배 |
| 돌연변이 | `mutate` | 낮은 확률로 비트 하나 토글 (다양성 유지) |

`evolve(... on_generation=콜백)` — 세대마다 호출되는 콜백 훅.
63짜리 공간이라 GA는 사실 장식이다 — **다목적·연속 공간으로 가면 필수**가 된다.

---

## 4. 다음 단계: Optuna NSGA-III 다목적 확장 (설계 확정안)

### 4-1. 왜 다목적인가

체육관들은 서로 충돌하는 목적이다 (닷컴 방어 ↔ 회복장 올라타기는 trade-off).
스칼라로 뭉개면 "한 국면에서 처참히 죽는데 평균은 그럭저럭"인 전략이 숨는다 (함정 2·3).
결과물은 최강 1마리가 아니라 **전략 Pareto front** = 국면별 특화 + 올라운더 라인업.

### 4-2. Y — 목적 벡터 (6목적)

> **내부(옵티마이저) vs 표시·판정 분리** (06-13 사용자 결정):
> - **내부**: `score_vs_dca` 6목적 (raw 지표 다목적) — NSGA-III 탐색·필터·게이트.
> - **표시·판정**: "100만원 시드 → 종료 잔고 vs 성실이 잔고" 단위 통일. 사람이
>   읽는 모든 층(NSGA-III 통과 후보 표, 챔피언로드 3관문)에서 동일 단위.
> - 회귀 테스트 골든 넘버는 내부(`score_vs_dca`)라 안 깨짐.
> - 시드 1,000,000원은 비율(`total_return`)의 표시 환산일 뿐 — 옵티마이저 무영향.

기준선 = **일별 DCA** (실엔진: 토스 매일 $20 QQQM 자동 모으기, **매수 수수료 0원**).
전략은 0.1% 과금 — 비용 비대칭이 현실이고 그대로 모델링한다 (`battle.fight_dca`).

> ⚠️ 구현은 정확히는 **총자본 1/N 분할 진입 근사**다 (코덱스 리뷰 06-11 반영).
> 진짜 적립(매일 새 돈)은 외부 현금흐름이라 IRR로 재야 하는데, 전략은 첫날부터
> 총자본을 들고 시작하므로 같은 자본 기준에 놓아야 위 "빼기 비교"가 성립한다.
> 1차 근사에서 동일, 복리 교차항에서만 갈림 — 기준선 용도로 충분. `battle._dca_position` 참고.

```
score_vs_dca(체육관) = 0.4×(전략 총수익 − DCA 총수익)
                     + 0.4×(|DCA MDD| − |전략 MDD|)
                     + 0.2×(전략 샤프 − DCA 샤프)          # raw 지표만, 양수 = DCA 개선
```

6체육관을 그대로 6목적으로 쓰면 턴오버까지 7목적 = front가 너무 넓어진다.
→ **하락 2개를 min으로 압축** (코덱스 제안 채택):

```
Y = [ bear     = min(닷컴, 금융위기) score_vs_dca,   # maximize
      rebound  = 회복장 score_vs_dca,                # maximize  ← "살을 내줘도 뼈를 취한다"
      crash_v  = 코로나 score_vs_dca,                # maximize
      bull     = 상승장 score_vs_dca,                # maximize
      chop     = 횡보장 score_vs_dca,                # maximize
      turnover = 일평균 |포지션 변화|                 # minimize  ← 비용·운영 위험 대리
    ]
```

- 부호 뒤집기 불필요 — Optuna는 `directions=["maximize"]*5 + ["minimize"]`를 그대로 받는다.
- 턴오버를 점수 안에 숨기지 않고 별도 목적으로 두는 이유: 실거래엔 슬리피지·환전·
  운영 리스크가 수수료 밖에 더 있다 (비용 민감도 실측: 0.2%에서 효율 FAIL = 마진 얇음).

### 4-3. X — 결정변수 (연속 가중치 + 시그널 파라미터)

```python
# ① 시그널 가중치 — 기권 제외 '가중' 평균으로 결합 확장
w = [trial.suggest_float(f"w_{g}", 0.0, 1.0) for g in ALL_GENES]
# 결합: position = Σ wᵢ·posᵢ / Σ wᵢ   (그날 의견 낸 시그널만, NaN 기권 제외)
# → 정규화가 결합식 안에 있으므로 Σw=1 제약과 동치 (예산 제약 내장)

# ② 시그널 파라미터 — 첫 버전은 보수적으로 (코덱스 후보 공간)
DD_LIMIT      0.05 ~ 0.25      MA_WINDOW    50 ~ 250
MOM_LOOKBACK  20 ~ 120         RSI_OVERSOLD 20 ~ 40
BB_K          1.5 ~ 2.5        VOL 임계      분위수 기반
```

⚠️ 함정 (v0.2 시절부터 알던 것): 연속 가중치를 제약 없이 풀면 "전부 최대"로 수렴한다.
기권 제외 가중평균은 분모에 Σw가 있어 자동으로 비율만 의미를 가진다 = 퇴화 경사 제거.

⚠️ 첫 스터디는 탐색공간을 작게 — 목적함수 거동 검증이 우선, 전략 언어 확장은 그다음.

### 4-4. Optuna 구현 골격

```python
import optuna

study = optuna.create_study(
    directions=["maximize"] * 5 + ["minimize"],
    sampler=optuna.samplers.NSGAIIISampler(seed=42),
    storage="sqlite:///optuna_v1x.db",   # 시즌 임시 영역 — hall_of_fame.md 흡수 후 폐기
    study_name="nsga3_v1x_kis_s42",      # 매 시즌·시드마다 새 이름 (AGENTS.md §11)
    load_if_exists=False,                # 같은 이름 충돌 시 DuplicatedStudyError로 즉시 차단
)

def objective(trial):
    candidate = build_candidate(trial)              # X → 가중치+파라미터
    r = evaluate_on_gyms(candidate)                 # 6체육관 score_vs_dca + 턴오버
    return (min(r["닷컴"], r["금융위기"]),
            r["회복장"], r["코로나"], r["상승장"], r["횡보장"],
            r["turnover"])

study.optimize(objective, n_trials=2000, callbacks=[진행로그_콜백])
front = study.best_trials                           # Pareto front
```

### 4-5. Pareto 후처리 — 필터와 라벨

하드 필터 (front에서 배포 후보 거르기):
```
모든 체육관 score_vs_dca ≥ −tolerance     # 실측: 전 체육관 양수 조합 0개 → tolerance 필수
턴오버 ≤ 실거래 임계 (비용 민감도 근거)
```

> MDD 하드필터는 06-13 사용자 결정으로 제거: "어차피 깨져도 안 팔면 그만이야".
> 낙폭 페널티는 `score_vs_dca`의 0.4×낙폭개선 항에 이미 들어가 있고, 6목적
> 다목적이 위험을 체육관별로 분담한다. 600 trials로도 통과 0이 될 만큼 빡빡했음.

라벨 (Regime Scanner 30% 오버레이와 연결):
```
Defensive    : bear 최고 + MDD 최소        ┐ bear/stressed → 70% Balanced + 30% Defensive
Balanced     : 평균 최고 + 턴오버 허용     ├ bull/calm     → 70% Balanced + 30% Aggressive
Aggressive   : rebound/bull 최고          ┘ uncertain     → 100% Balanced
Low-turnover : 점수 허용선 내 턴오버 최소
```

### 4-5b. 국면별 1등 추적 — Regime Scanner 입력원 (`reports/regime_picks.json`)

훈련장(NSGA-III) + 챔피언로드 3관문 각각에서 시험단위(체육관/연도/세계)마다
잔고 1등 후보 + 국면 라벨을 누적 저장. 추후 Regime Scanner가 같은 라벨 체계로
"지금 어떤 국면" 추론하면 → 그 국면 1등 후보를 30% 오버레이.

```
gate0_training : 6체육관 × 잔고 1등 (인샘플)            ← service.run_nsga3
gate1_oos      : OOS 11년 × 잔고 1등 + 국면 라벨        ← app/league/victory_road.py
gate2_worlds   : arena 3개(전천후/bear/rebound) × 세계 1등 카운트  ← app/league/battle_frontier.py
gate3_holdout  : 봉인 6년 × 챔피언 잔고 + 국면 라벨     ← app/league/elite_four.py
```

국면 라벨 정의는 [Regime_Scanner](../Regime_Scanner) 프로젝트(`backend/signals.py`)와
동기 — 4종(`bull`/`bear`/`sideways`/`volatile`). 50/200 이동평균 · 60일 수익률 ·
20일 실현변동성 백분위(룩어헤드 방지). PocketQuant 쪽 단일 소스는
`app/backend/market/regime.py` — Regime_Scanner config 변경 시 양쪽 같이 손볼 것.

⚠️ 평행세계(gate2)는 합성이라 일별 판정 불가 → 풀명(전천후/bear/rebound) 그대로 사용.
인샘플(gate0)은 체육관 이름(닷컴/금융위기/...) 그대로.

### 4-6. 검증 프로토콜 (학습 = 실데이터, 검증 = 3단)

```
① OOS 리그 본선    : 졸업생을 연도별 실데이터에 재출전 (app/league/victory_road.py)
② 합성 스트레스     : 블록 부트스트랩 ~100세계 생존률 (훈련 목적 아님, Pareto 후 필터)
③ 봉인 hold-out     : post-COVID(2020-07~), NSGA-III 최종판정 때 딱 1회
```

엔진 안전망 (NSGA-III가 파라미터를 휘젓기 전 필수, 이미 구축):
- `tools/test_engine_regression.py` — 골든 넘버 16개 (REL_TOL 1e-4 = 4번째 소수점,
  yfinance 자동 보정 노이즈 흡수. 그보다 큰 차이면 의도된 변경 → 골든 갱신+worklog).
- `tools/test_no_lookahead.py` — 미래 절단 불변식 (시그널 파라미터가 바뀌어도 돌려서 확인).
- `tools/e2e.py` — 전 파이프라인 스모크 (10초). 폴더/data.py 같은 큰 변경 후 한 번.

---

## 5. 일반 다목적 최적화 파이프라인과의 매핑

이 랩의 구조는 산업 현장의 전형적인 다목적 파라미터 튜닝 파이프라인 그대로다:

| 일반 최적화 파이프라인 | PocketQuant |
|---|---|
| 품질 지표 N개 (목적함수) | 국면별 score_vs_dca 5개 + 턴오버 |
| 파라미터 탐색공간 | 시그널 가중치 len(ALL_GENES) + 파라미터 ~6 (작게 시작) |
| 더미 입력으로 루프 검증 → 실계측 교체 | 가짜 점수(v0.1) → yfinance 실계측(v0.3) ✅ 완료 |
| Pareto front → 운영 세팅 라인업 | front → Defensive/Balanced/Aggressive 라인업 |
| 학습/검증 분리 | 훈련 체육관 / OOS 리그 본선·부트스트랩·hold-out |
| 콜백 훅 | `on_generation` / Optuna callbacks |

구조는 같고 도메인만 다르다 — **"목적함수의 빈틈을 옵티마이저가 먼저 찾는다"**는
교훈(2절의 함정 3개)도 동일하게 적용된다.

---

## 6. 한 줄 요약

```
이기려는 대상 : 사용자의 실제 DCA 머신 (무비용 매일 적립)
목적          : 국면별 DCA 개선을 동시에 (Pareto), 턴오버는 최소로
수단          : Optuna NSGA-III — 시그널 가중치 공간, 시그널 파라미터 동결 (v1 과적합 회피)
검증          : OOS 리그 본선 → 합성 스트레스 → 봉인 hold-out (이 순서, 역류 금지)
자동화        : tools/e2e.py 한 번 = compileall + 게이트 + 진단 + nsga3 smoke
```

---

## 7. v1 마감 (2026-06-13) — 누적 기록은 `reports/hall_of_fame.md`

- 시즌: 2026-06-12 ~ 2026-06-13. 5 시드 × NSGA-III 2000 trials, 인구 100, HV-MA(5)
  얼리스탑, 적응형 mutation. 시드 간 잔고 합 ±0.4% 수렴.
- 챔피언로드 ② 평행세계 400 본판정 1위: **TOP06 (VOL 8% + REV_RSI 57% + REV_BB 30%)
  토탈 4.92억 / +23.0%** — 어플삭제맨(11위) · 현 챔피언(12위) 둘 다 누름.
- 한계 (다음 시즌 화두): 유전자 풀 6개 제약(Top10이 같은 종 변주) → **알 깨기**로
  새 시그널 풀 확장 (외부: KIS API 호가/체결, 매크로 VIX/DXY, 가격 내부 RSI 다이버전스 등).
  KIS 데이터 파이프라인 자리는 `app/backend/data_io/`에 미리 잡아뒀다.

---

## 8. 시즌 v1.x 시작 (2026-06-13) — 야생 7마리 + CMA-ES

### 시그널 풀 확장: 6 → 13

v1의 자산-횡단 한계 = 가격 6마리만 보던 게 원인. 야생 7마리 추가 (외부 정보원):

| 그룹 | 시그널 | 정보원 |
|---|---|---|
| 자산 내부 | VOL_SPIKE | 거래량 (yfinance OHLCV — `get_volume` 신설) |
| 글로벌 | FEAR · US10Y · DXY | `^VIX` · `^TNX` · `UUP` |
| 자산-횡단 | SPY_TLT · QQQ_SPY · QQQ_DIA | 비율 시계열 |

모두 이벤트형. 데이터 없는 시기(UUP 2007년 이전)는 자동 NaN 기권.
진단(`check_signals`): 5마리 PASS(상관 < 0.3), **2마리 WARN**(QQQ_SPY/QQQ_DIA가 MOM과
+0.47/+0.48, 자기들끼리 +0.81 = 중복 클러스터). GA/TPE/CMA-ES가 가중치로 자동 조정.

### 옵티마이저 3종 (사용자 안: TPE에 CMA-ES도 추가)

| 엔진 | 모듈 | 성격 |
|---|---|---|
| NSGA-III | `app/academy/study/nsga3.py` | 다목적 (6목적 score_vs_dca + turnover) → Pareto front |
| TPE | `app/academy/study/tpe.py` | 단일목적 Bayesian (잔고 합 max). v1 챔피언 TPE-s11 출신 |
| **CMA-ES** | `app/academy/study/cma_es.py` (v1.x 신설) | 단일목적 진화전략. 연속 공간 강함 |

셋 다 같은 인터페이스(`run_study(trials, seed, storage, study_name, on_progress,
loaded_gyms, dca)`) → service에서 sampler만 갈아끼울 수 있다. `cmaes>=0.10` 의존성 추가.

### 운영 규칙 (AGENTS.md §11 재확인)

- 모든 엔진 `load_if_exists=False` 고정. `study_name`은 매 시즌·시드마다 새로.
- sqlite db는 시즌 임시 작업 영역 → `hall_of_fame_<버전>.md` 신설 + 인덱스 갱신 후 폐기 (`rm *.db`).
- 다른 탐색공간(옛 풀 6 vs 새 풀 13) 절대 같은 db에 안 섞임 — `DuplicatedStudyError`로 즉시 차단.

### 다음 단계 (v1.x 본탐색)

- NSGA-III · TPE · CMA-ES 셋 다 13마리 풀로 5시드 sweep
- 비교 기준: 잔고 합 안정성 + 챔피언 가중치 분포 (어느 sampler가 더 잘 수렴하나)
- 자산-횡단 검증 (SPY 6체육관 + SPY OOS + SPY 사천왕)
- 합격선: **SPY에서 어플삭제맨 이기는 챔피언** ([Issue #1](../../issues/1) Phase 3)
- 워밍업: TPE-s11 옛 6마리 가중치 → 새 13차원 변환 후 `enqueue_trial` 주입 (옛 6마리만 살리고 새 7마리는 0)
