# PocketQuant — 최적화 설계 노트

> PocketQuant의 "전략 탐색"을 최적화 문제로 정식화한 문서.
> 게임 컨셉(포켓몬/체육관) 뒤에 실제로 어떤 최적화가 돌고 있는지,
> 그리고 다음 단계(Optuna NSGA-III 다목적)를 어떻게 정식화할지 정리한다.
>
> v0.2(가짜 데이터·생존률) 시절 내용은 폐기하고 v0.5 기준으로 갱신 (2026-06-11).

---

## 1. 문제 정의 (현재 — v0.5 단일목적)

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

- 회귀 가드: `tests/test_baselines.py` (전부 현금 < 풀매수 & 하위 25% 룰).
- 공통 교훈: **0~100 클램프 스탯/BST를 최적화 목적으로 쓰지 말 것** (raw 지표 사용).
  HP가 아니라도, "낮은 노출 = 안정 점수 공짜"가 되는 모든 통로가 같은 퇴화를 만든다.

---

## 3. GA 4단계 (`engine/evolve.py`) — 원리 이해용 손코딩

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

기준선 = **일별 DCA** (실엔진: 토스 매일 $20 QQQM 자동 모으기, **매수 수수료 0원**).
전략은 0.1% 과금 — 비용 비대칭이 현실이고 그대로 모델링한다 (`battle.fight_dca`).

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
    storage="sqlite:///optuna_pocketquant.db",     # 중단/재개
    study_name="nsga3_v1",
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
최악 MDD ≤ DCA 최악 MDD
턴오버 ≤ 실거래 임계 (비용 민감도 근거)
```

라벨 (Regime Scanner 30% 오버레이와 연결):
```
Defensive    : bear 최고 + MDD 최소        ┐ bear/stressed → 70% Balanced + 30% Defensive
Balanced     : 평균 최고 + 턴오버 허용     ├ bull/calm     → 70% Balanced + 30% Aggressive
Aggressive   : rebound/bull 최고          ┘ uncertain     → 100% Balanced
Low-turnover : 점수 허용선 내 턴오버 최소
```

### 4-6. 검증 프로토콜 (학습 = 실데이터, 검증 = 3단)

```
① QQQ 워크 포워드   : 과거 선발 → 다음 1년 OOS (tests/walk_forward.py, 심판 — 결과 보고 목적함수 고치지 않기)
② 합성 스트레스     : 블록 부트스트랩 ~100세계 생존률 (훈련 목적 아님, Pareto 후 필터)
③ 봉인 hold-out     : post-COVID(2020-07~), NSGA-III 최종판정 때 딱 1회
```

엔진 안전망 (NSGA-III가 파라미터를 휘젓기 전 필수, 이미 구축):
- `tests/test_engine_regression.py` — 골든 넘버 16개 (어긋나면: 의도된 변경 → 골든 갱신+워크로그 / 아니면 버그)
- `tests/test_no_lookahead.py` — 미래 절단 불변식 (시그널 파라미터가 바뀌어도 돌려서 확인)

---

## 5. 본업(화질 다목적 최적화)과의 매핑

| 화질 최적화 (회사) | PocketQuant |
|---|---|
| 화질 지표 5개 (목적함수) | 국면별 score_vs_dca 5개 + 턴오버 |
| 이미지 파라미터 LUT (~9⁴⁰) | 시그널 가중치 6 + 파라미터 ~6 (작게 시작) |
| 랜덤 검증 → 실계측 교체 | 가짜 점수(v0.1) → yfinance 실계측(v0.3) ✅ 완료 |
| Pareto front → 세팅 라인업 | front → Defensive/Balanced/Aggressive 라인업 |
| 학습/검증 분리 | 훈련 체육관 / 워크포워드·부트스트랩·hold-out |
| 콜백 훅 | `on_generation` / Optuna callbacks |

구조는 같고 도메인만 다르다 — **"목적함수의 빈틈을 옵티마이저가 먼저 찾는다"**는
교훈(2절의 함정 3개)도 동일하게 적용된다.

---

## 6. 한 줄 요약

```
이기려는 대상 : 사용자의 실제 DCA 머신 (무비용 매일 적립)
목적          : 국면별 DCA 개선을 동시에 (Pareto), 턴오버는 최소로
수단          : Optuna NSGA-III — 손코딩 GA에서 익힌 구조를 프레임워크로 확대
검증          : 워크포워드 → 합성 스트레스 → 봉인 hold-out (이 순서, 역류 금지)
```
