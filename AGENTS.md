# PocketQuant CLI Backend Harness v0.3

> 이 문서는 **현재 코드 기준**으로 작성된 사양서다. (코드가 source of truth)
> "향후 확장 로드맵" 항목만 아직 미구현이며, 그 외는 실제 코드와 일치한다.

## 목표

PocketQuant의 CLI-only 백엔드 하네스.

전략 포켓몬을 생성하고, 여러 시장 국면에서 **실데이터 백테스트**를 수행해
HP/ATK/DEF/SKILL 4스탯을 뽑은 뒤 결과를 출력한다.

## 범위

- GUI 없음
- DB 없음
- **실데이터 사용**: yfinance로 SPY 가격을 받아 백테스트 (디스크 캐시 → 이후 오프라인)
- LLM 없음
- 김박사/오박사 없음
- CLI 실행만 지원

> ⚠️ v0.3에서 **생존/사망 이진판정을 폐기**했다. "상폐(0원)만 아니면 진 게 아니다" →
> 모든 결과를 연속 스탯(0~100)으로 잰다. 페널티(0점 처리) 없음.

---

## 프로젝트 구조

```text
pocket_quant/
├─ main.py                # CLI 입력만 받음 (argparse → service 호출)
├─ requirements.txt       # yfinance · pandas · numpy
├─ data_cache/            # (gitignore) 다운로드한 SPY 가격 CSV 캐시
├─ app/
│  ├─ __init__.py
│  ├─ service.py          # 실행 흐름 조립 (단판/진화 순서 + 시드 + 출력)
│  └─ backend/
│     ├─ __init__.py
│     ├─ models.py        # 데이터 모델(Stats/Strategy/Gym/Report) + 스탯 가중치/등급
│     ├─ data.py          # yfinance 다운로드 + 디스크 캐시
│     ├─ signals.py       # 유전자 = 진짜 지표 로직 → 일별 포지션(0~1)
│     ├─ strategy.py      # 전략 생성 + 이름 자동 생성
│     ├─ gym.py           # 체육관 정의 (실제 시장 국면 기간)
│     ├─ battle.py        # 실데이터 백테스트 → 스탯 산출
│     └─ evolve.py        # 단일목적 GA (적합도=스탯 가중합)
├─ README.md
├─ README.html
└─ AGENTS.md
```

> 참고: 3층 분리 — `main.py`(입력) → `app/service.py`(흐름 조립·출력) → `app/backend/*`(계산).
> 출력 포맷은 별도 `report.py` 없이 `service.py` 안에서 처리한다. `tests/`는 아직 없음.

---

## models.py

데이터의 '모양'만 정의한다. 로직은 최소.

### 상수 (모듈 레벨)

```python
STAT_WEIGHTS = {"HP": 1.0, "ATK": 1.0, "DEF": 1.0, "SKILL": 1.0}   # GA 적합도 가중치 = 진화 방향
GRADES       = [(0.9, "S"), (0.7, "A"), (0.5, "B"), (0.3, "C"), (0.0, "D")]  # 적합도/100 기준
```

> 유전자 명단(`ALL_GENES`)은 더 이상 models에 없다. 진짜 출처는 **signals.py의
> `GENE_SIGNALS`/`ALL_GENES`** — 실제 시그널을 가진 유전자만이 진짜 유전자다.
> (옛 `GENE_SCORES` 가짜 점수표와 `Strategy.base_score()`는 제거됨.)

### Stats dataclass  (신규)

```python
hp: float; atk: float; def_: float; skill: float   # 각 0~100 정규화 (def는 예약어라 def_)
# @property bst     -> 종족치 = 네 스탯 합 (0~400)
# @property fitness -> 스탯 가중평균 (0~100, STAT_WEIGHTS 적용)
```

### Strategy dataclass

```python
genes: list[str]      # 예) ["DD", "RSI"]
name: str = ""
```

### Gym dataclass

```python
name: str
difficulty: int       # 연출용 메타데이터 (판정 미사용)
volatility: int       # 연출용 메타데이터 (판정 미사용)
ticker: str = "SPY"   # 그 시기를 재현할 자산
start: str            # 평가 시작일 "YYYY-MM-DD"
end: str              # 평가 종료일
```

### BattleResult dataclass

```python
gym_name: str
stats: Stats              # 그 시장에서 뽑힌 HP/ATK/DEF/SKILL
cagr: float              # 연율수익 (표시용)
total_return: float      # 기간 총수익 (표시용)
max_drawdown: float      # 내 전략 MDD (음수)
market_drawdown: float   # 시장(단순보유) MDD (음수, DEF 계산 기준)
```

### Report dataclass

```python
strategy: Strategy
results: list[BattleResult]
# @property: stats(체육관별 평균 스탯), fitness, bst, grade
```

---

## strategy.py

### 역할

전략 생성 및 이름 자동 생성.

### 함수

```python
create_strategy(gene_count: int | None = None) -> Strategy
make_name(genes: list[str]) -> str
```

### 규칙

* gene_count가 없으면 1~5개 랜덤
* 유전자 중복 없음 (`random.sample`)
* 유전자 점수 합산 (`Strategy.base_score()`)
* 이름 자동 생성

### 이름 규칙

```text
RSI몬
DD-RSI몬
DD-RSI-MA몬
```

* 약 20% 확률로 특수 이름 부여 (예: "타이탄 드래곤") — **이미 구현됨**
  * TITLES = ["ATH", "디아블로", "헤르메스", "타이탄"]
  * SUFFIXES = ["몬", "드래곤", "킹", "마스터"]

---

## gym.py

### 기본 체육관 (4개) — 실제 시장 국면 기간

각 체육관은 SPY 가격으로 그 기간을 백테스트한다. difficulty/volatility는 연출용 메타데이터.

```text
FINANCIAL_CRISIS   2007-10-01 ~ 2009-06-30   # S&P -57%, 시스템 붕괴 → 최대 낙폭
DOTCOM             2000-01-01 ~ 2002-12-31   # -49%(나스닥-78%), 2.5년 느린 약세장
RATE_SHOCK         2022-01-01 ~ 2022-12-31   # -25%, 채권 동반하락이나 질서있음
COVID              2020-01-01 ~ 2020-08-31   # -34%지만 V자 즉시회복(버티면 생존)
```

> SPY(1993~ 상장)가 4개 국면을 모두 커버. boss / TRUMP 체육관은 향후 확장 시 고려.

### 함수

```python
all_gyms() -> list[Gym]
```

---

## data.py

파이프라인 1단계 = **모든 데이터 로딩(I/O)을 전담**한다. (battle은 계산만)
yfinance로 가격을 받고 디스크에 캐시한다 (`data_cache/{ticker}_{start}_{end}.csv`).

```python
get_prices(ticker, start, end) -> pd.Series   # 수정종가. 캐시 우선 → 없으면 다운로드 후 저장
load_gym(gym) -> LoadedGym                     # 체육관 1곳 가격 로딩(앞쪽 WARMUP_DAYS=400 버퍼 포함)
load_gyms(gyms) -> list[LoadedGym]             # 전 체육관 한 번에 로딩

@dataclass LoadedGym: gym: Gym; prices: pd.Series   # 체육관 + 미리 당겨둔 가격
```

* 캐시 우선이라 첫 실행만 네트워크, 이후 오프라인. 역사 기간은 값이 안 변하니 안전.
* 미리 로딩해 두면 진화 모드에서 가격을 전략마다 다시 안 읽음 → **국면당 1회 로딩**.

---

## signals.py

유전자 = 진짜 지표 로직. 가격을 받아 일별 포지션(0~1)을 만든다.

```text
DD  : 리스크      - 고점 대비 -10% 넘게 빠지면 0 (드로다운 스탑)
RSI : 심리        - RSI(14) < 70이면 1, 과열이면 0 (과매수/과매도)
MA  : 추세        - 가격 > 200일 이평이면 1 (이평)
BB  : 변동성      - 볼린저 상단밴드 위면 0, 그 외 1 (볼린저)
VOL : 시장 상태   - 변동성 레짐: 평온 1.0 / 중간 0.5 / 격동 0.0
MOM : 추세 강화   - 최근 ~63일(3개월) 수익률 양수면 1 (모멘텀)
```

```python
GENE_SIGNALS = {"DD":..., "RSI":..., "MA":..., "BB":..., "VOL":..., "MOM":...}  # 유전자 명단의 단일 출처
ALL_GENES    = list(GENE_SIGNALS.keys())        # strategy.py / evolve.py가 여기서 import
combined_position(genes, prices) -> pd.Series   # 유전자별 포지션의 평균(0~1)
```

---

## battle.py

### 전투 로직 (실데이터 백테스트)

가격은 LoadedGym으로 **이미 받아둔 상태**로 들어온다. battle은 I/O 없이 계산만.

```text
1) signals.combined_position → 일별 포지션, shift(1)로 하루 lag (룩어헤드 방지)
2) strat_ret = position.shift(1) * market_ret  → 자산곡선
3) 워밍업 버퍼 잘라내고 평가 구간만 슬라이스 (window_start = gym.start)
4) CAGR / MDD / 샤프 / 평균현금 측정
5) 0~100 스탯 정규화:
     HP    = 평균현금 * 100
     ATK   = scale(CAGR,  -0.25 ~ +0.25)
     DEF   = scale(1 - 내MDD/시장MDD, 0 ~ 1)   # 시장이 안 빠졌으면 100
     SKILL = scale(샤프,  -1 ~ 3)
```

> 정규화 구간(ATK_CAGR_*, SKILL_SHARPE_*)은 battle.py 상단 상수 = 튜닝 포인트.

### 함수

```python
fight(strategy: Strategy, loaded: LoadedGym) -> BattleResult        # 한 시장 → 스탯블록
challenge(strategy: Strategy, loaded_gyms: list[LoadedGym]) -> Report
```

---

## evolve.py (단일목적 GA)

### 역할

전략 개체군을 여러 세대 진화시켜 "종합 적합도(스탯 가중합)"가 가장 높은 전략을 찾는다.

### 적합도 (단일목적)

```text
fitness = 종합 스탯 가중평균 (0~100, 숫자 하나) = report.fitness
실데이터는 결정론적 → 평가는 1회만 돈다 (trials 개념 제거).
```

### GA 4단계 + 함수

```python
evaluate(strategy, loaded_gyms) -> {"fitness", "per_gym"(BST), "stats"}   # 1. 평가(1회)
select(scored, keep) -> list[Strategy]                       # 2. 선택(절단)
crossover(genes_a, genes_b) -> list[str]                     # 3. 교배(균등)
mutate(genes, rate) -> list[str]                             # 4. 돌연변이(추가/제거)
evolve(loaded_gyms, pop_size, generations, on_generation) -> (best, stats)
```

* `on_generation(gen, best, stats)` : 세대별 콜백 훅 (로깅/시각화/향후 early stop)

### 관찰 (의도된 것)

* 실데이터엔 진짜 트레이드오프(현금↑→방어↑ 공격↓)가 있어 **전 유전자 조합으로 안 수렴**한다.
  (가짜 점수 시절과 달라진 핵심 — 예: `MA` 단일이 챔피언이 되기도.)
* 스탯 4개가 서로 당기는 이 충돌이 다목적(NSGA-III) 필요성의 근거.
* `STAT_WEIGHTS`를 바꾸면 진화 방향(공격형/방어형)이 바뀐다.

---

## main.py

### 역할

CLI 입력만 받는다. argparse로 인자를 파싱해 `app/service.py`의 `run_single`/`run_evolve`를
호출만 한다. 계산도 흐름 조립도 출력도 하지 않는다 (전부 service/backend 담당).

```python
run_single(gene_count, seed)        # service.py — 단판 흐름
run_evolve(pop, generations, seed)  # service.py — 진화 흐름
```

### 파이프라인 순서 (service.py가 조립)

```text
① 데이터 로딩   load_gyms(all_gyms())       [data.py]   ← 가장 먼저, 한 번만
② 전략 만들기   create_strategy() / 개체군   [strategy.py / evolve.py]
③ 전투(백테스트) challenge / fight            [battle.py] ← 가격 받아 계산만
④ 결과          Report → 스탯블록 출력        [service.py]
⑤ 진화(선택)    evolve가 ②③을 세대 반복      [evolve.py]
```

> 데이터·전략은 서로 독립이고 ③ battle에서 처음 만난다.
> 데이터 로딩(I/O)은 data.py가 전담, battle은 순수 계산 → 책임 분리.

### argparse 옵션

```bash
# [단판 모드]
-g, --genes        # 전략 유전자 개수 지정 (생략 시 랜덤)

# [진화 모드 — 단일목적 GA]
--evolve           # 진화 모드 실행
--pop              # 개체군 크기 (기본 20)
--generations      # 세대 수 (기본 10)

# [공통]
--seed             # 랜덤 시드 고정 (재현 가능 → GA 검증용)
```

### 실행 예시

```bash
python main.py                                              # 단판
python main.py -g 3
python main.py --evolve                                     # 진화(GA)
python main.py --evolve --pop 20 --generations 8 --seed 42  # 재현 가능한 진화
```

> 반드시 프로젝트 루트에서 실행. (`app.backend....` 절대 경로 import 구조)

### 출력 예시 (실제 출력)

```text
=== PocketQuant ===

데이터 로딩 (SPY · 4개 국면)

전략 생성
VOL + MOM
이름: 타이탄 마스터

시장별 백테스트 (실데이터)
  FINANCIAL_CRISIS   CAGR   -6.7%  MDD  -19.5% (시장  -55.2%)  BST 174.5
  COVID              CAGR   13.2%  MDD  -10.9% (시장  -33.7%)  BST 236.0

스탯블록 (종합)
    HP   (자본력)   56.4  ###########
    ATK  (공격력)   38.1  ########
    DEF  (방어력)   45.8  #########
    SKILL(솜씨)     14.9  ###

종족치(BST) 155.1 / 400   적합도 38.8   등급 C
```

### 등급 규칙 (적합도 = 스탯 가중평균 0~100)

```text
S: 적합도 >= 90
A: 적합도 >= 70
B: 적합도 >= 50
C: 적합도 >= 30
D: 그 외
```

---

## 개발 원칙

### 유지

* 단순한 CLI MVP
* 실행 가능한 코드 우선
* 200~300줄 내외
* 구조는 확장 가능하게

### 금지

* GUI
* DB
* 웹서버
* 실시간 시세 / 투자 API (백테스트용 과거 데이터 yfinance만 허용)
* LLM
* 김박사/오박사
* 과도한 추상화

---

## 구현 완료

### v0.2 단일목적 GA  ✅
* 개체군 + 선택(절단) + 교배(균등) + 돌연변이 + 세대 반복 (`evolve.py`)
* `--seed`로 재현 가능, 세대별 콜백 훅

### v0.3 실데이터 + 스탯블록  ✅
* `data.py`(yfinance+캐시), `signals.py`(유전자=진짜 지표) 추가
* `battle.fight()` 랜덤 → 진짜 백테스트, 생존판정 폐기 → HP/ATK/DEF/SKILL 4스탯
* GA 적합도 = 스탯 가중합(BST 기반), trials 평균 제거(결정론적)

---

## 향후 확장 로드맵 (미구현)

### 다음: 다목적 최적화 (NSGA-III)

* 단일 적합도(스탯 가중합) → **스탯 4개 = 각각의 목적**
* 결과 = 최강 전략 1개가 아니라 **전략 Pareto front** (국면별 특화 + 올라운더)
* v0.3에서 관찰한 스탯 충돌(공격 vs 방어/현금)이 목적함수 정의의 근거.

### 정비

* `tests/` 추가 (test_strategy / test_gym / test_battle / test_evolve)
  * 전략 생성·중복 없음·gene_count 일치 / 체육관 목록·difficulty 존재
  * fight 결과 타입·survived bool·seed 재현성 / GA 수렴·교배·돌연변이

### 전략 도감

* 전략 저장 / 전적 기록 / 생존률 누적

### v1.0 김박사 NPC

* 룰베이스 대사
* 이후 LLM 연결
