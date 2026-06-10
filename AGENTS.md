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

## 모듈 책임 (한 줄 정의 — 이 역할 경계를 지킨다)

> **main은 받고, service는 흐름 짜고, dex는 설명하고, signals는 판단하고, battle은 싸우고, evolve는 진화시킨다.**

| 모듈 | 한 줄 책임 |
|------|-----------|
| `main.py`           | **받고** — config.json을 읽어 service에 넘긴다 (CLI 플래그/argparse 없음) |
| `service.py`        | **흐름 짜고** — 단판/진화/도감 실행 순서를 조립하고 출력한다 |
| `genes/dex.py`      | **설명하고** — 유전자(포켓몬) 설명 카드를 제공한다 |
| `genes/signals.py`  | **판단하고** — 유전자가 가격을 보고 포지션(0~1) 또는 기권(NaN)을 정한다 |
| `engine/battle.py`  | **싸우고** — 포지션으로 백테스트해 스탯을 뽑는다 |
| `engine/evolve.py`  | **진화시킨다** — GA로 전략 개체군을 세대 진화시킨다 |
| (보조) `market/data.py` 땡겨오고 · `market/gym.py` 무대 · `engine/strategy.py` 만들고 · `core/models.py` 모양 정의 |

---

## 프로젝트 구조 (백엔드 = 기능별 서브패키지)

```text
pocket_quant/
├─ main.py                   # 받고 — config.json 읽어 service 호출 (argparse 없음)
├─ config.json               # 실행 옵션 (mode/genes/pop/generations/seed/md/capital)
├─ requirements.txt          # yfinance · pandas · numpy
├─ data_cache/               # (gitignore) 다운로드한 가격 CSV 캐시
├─ app/
│  ├─ __init__.py
│  ├─ service.py             # 흐름 짜고 — 단판/진화/도감 순서 + 시드 + 출력
│  └─ backend/
│     ├─ core/               # 토대
│     │  └─ models.py        #   모양 정의 — Stats/Strategy/Gym/Report + 스탯가중치/등급
│     ├─ market/             # 시장(데이터·무대)
│     │  ├─ data.py          #   땡겨오고 — yfinance 다운로드 + 디스크 캐시
│     │  └─ gym.py           #   무대 — 체육관 정의 (실제 시장 국면 기간)
│     ├─ genes/              # 유전자(시그널·도감)
│     │  ├─ signals.py       #   판단하고 — 지표 → 포지션(0~1)/기권(NaN), 결합 규칙
│     │  └─ dex.py           #   설명하고 — 포켓몬 도감(SIGNAL_CARDS)
│     └─ engine/             # 엔진(전투·생성·진화)
│        ├─ battle.py        #   싸우고 — 실데이터 백테스트 → 스탯 산출
│        ├─ strategy.py      #   만들고 — 전략 생성 + 이름 자동 생성
│        └─ evolve.py        #   진화시킨다 — 단일목적 GA (적합도=스탯 가중합)
├─ tests/
│  ├─ test_baselines.py      # 적합도 퇴화 검증 ('전부 현금'/'항상 풀매수' 기준선)
│  └─ check_signals.py       # 시그널 진단 (노출/발동률/상관 — 풀 교체 시마다 실행)
├─ worklog/                  # 실험실 노트 (리뷰·계획서·검증 결과)
├─ README.md
└─ AGENTS.md
```

> 참고: 3층 분리 — `main.py`(입력) → `app/service.py`(흐름 조립·출력) → `app/backend/*`(계산).
> backend 의존 방향: `core ← market/genes ← engine` (engine이 최상위 소비자, 순환 없음).
> 출력 포맷은 별도 `report.py` 없이 `service.py` 안에서 처리한다.
> tests 실행: `python tests/test_baselines.py` · `python tests/check_signals.py` (pytest도 호환).

---

## core/models.py

데이터의 '모양'만 정의한다. 로직은 최소.

### 상수 (모듈 레벨)

```python
STAT_WEIGHTS = {"HP": 0.0, "ATK": 1.0, "DEF": 1.0, "SKILL": 1.0}   # GA 적합도 가중치 = 진화 방향
GRADES       = [(0.9, "S"), (0.7, "A"), (0.5, "B"), (0.3, "C"), (0.0, "D")]  # 적합도/100 기준
```

> **HP 가중치가 0인 이유 (퇴화 방지):** HP(현금 비중)를 적합도에 넣으면 '전부 현금'이
> HP 100 + (구)DEF 100을 받아 "아무것도 안 하기"가 최적해가 된다(구 설계 실측: 전부
> 현금 ~69점 vs 풀매수 ~29점). 현금은 수단이지 목표가 아니므로 HP는 **표시 전용**이고,
> 적합도는 ATK/DEF/SKILL 3개 성과 스탯의 가중평균이다. 검증은 `tests/test_baselines.py`.

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
total_return: float      # 기간 총수익 (실투자 시뮬용: 시작자본 × (1+이값))
market_return: float     # 단순보유 기간 총수익 (비교용)
max_drawdown: float      # 내 전략 MDD (음수)
market_drawdown: float   # 시장(단순보유) MDD (음수, 비교 표시용 — DEF 계산엔 미사용)
```

### Report dataclass

```python
strategy: Strategy
results: list[BattleResult]
# @property: stats(체육관별 평균 스탯), fitness, bst, grade
```

---

## engine/strategy.py

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

## market/gym.py

### 기본 체육관 (4개) — 실제 시장 국면 기간

각 체육관은 SPY 가격으로 그 기간을 백테스트한다. difficulty/volatility는 연출용 메타데이터(1~10).

```text
2008 금융위기 체육관     2008-01-01 ~ 2009-06-30   # 시스템 붕괴 대폭락 → 방어형 천국
2020 코로나 급락 체육관   2020-02-01 ~ 2020-06-30   # V자 급락→즉시회복 (버티면 생존)
2017 불타입 상승장 체육관 2017-01-01 ~ 2017-12-31   # 저변동성 우상향 (추세/모멘텀 강세)
2015-2016 횡보장 체육관   2015-01-01 ~ 2016-12-31   # 방향 없는 출렁임 (추세형 헛신호)
```

> 폭락·급락·상승·횡보 4국면을 일부러 섞음(국면 다양성=과적합 방지). SPY(1993~)가 전부 커버.

### 함수

```python
all_gyms() -> list[Gym]
```

---

## market/data.py

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

## genes/signals.py

유전자 = 진짜 지표 로직. 가격을 받아 일별 포지션(0~1) 또는 **기권(NaN)** 을 만든다.

**[2026-06-10 재배치] 3타입 × 2마리** (근거·검증: `worklog/2026-06-10_signal_rework_plan.md` + `_results.md`):

```text
💧 위험회피(상시형):
  DD      : 리스크     - 고점 대비 -10% 넘게 빠지면 0 (드로다운 스탑)
  VOL     : 시장 상태  - 변동성 레짐: 평온 1.0 / 중간 0.5 / 격동 0.0
🔥 추세순응(상시형):
  MA      : 추세       - 가격 > 200일 이평이면 1 (이평)
  MOM     : 추세 강화  - 최근 ~63일(3개월) 수익률 양수면 1 (모멘텀)
🌿 역발상(이벤트형 — 발동일만 의견, 평소 기권 NaN):
  REV_RSI : 심리       - RSI(14) < 30 과매도 투매면 매수 의견 1.0
  REV_BB  : 변동성     - 볼린저 하단밴드 아래 과대 낙폭이면 매수 의견 1.0
```

> 구 RSI(과열 감산)·BB(상단 감산)는 실측상 거의 상수 1(죽은 시그널)이라 **명단 제외**
> (함수는 비교용으로 보존). 과열 회피는 양의 기대수익을 버리는 규칙이었다.

```python
GENE_SIGNALS = {"DD","VOL","MA","MOM","REV_RSI","REV_BB" -> 함수}  # 유전자 명단의 단일 출처
ALL_GENES    = list(GENE_SIGNALS.keys())          # strategy.py / evolve.py가 여기서 import
combine_positions(positions) -> pd.Series         # '기권 제외 평균' (전원 기권한 날 = 0.0)
combined_position(genes, prices) -> pd.Series     # 유전자 이름 → 시그널 → combine_positions
```

* **기권(NaN) 규칙**: 이벤트형 시그널은 발동일에만 의견을 낸다. 기권을 0으로 집계하면
  "의견 없음"="현금 가라"가 돼 상시 현금 앵커가 되므로(잉어킹 강제 출전 문제),
  그날 의견 낸 시그널끼리만 평균한다. 전원 기권 = 포지션 0.0.
* 튜닝 파라미터(MA_WINDOW, RSI_*, BB_*, DD_LIMIT, VOL_*, MOM_LOOKBACK)는 파일 상단 상수로 모음.
  REV 발동 유지기간·과매도선·가중치는 의도적으로 미구현 — Optuna(NSGA-III) 결정변수 후보.

---

## genes/dex.py (포켓몬 도감)

판정엔 안 쓰는 '설명' 데이터. `mode: "dex"`로 출력하며 service가 읽어 포맷한다.

```python
from .signals import GENE_SIGNALS   # 명단 일치 검증용
SIGNAL_CARDS = {유전자: {name, type, role, personality, effect, strength, weakness}}
# 모듈 로드 시 assert set(SIGNAL_CARDS) == set(GENE_SIGNALS) — 도감/명단 불일치 즉시 차단
```

> 의존: `dex → signals` 단방향 (순환 없음).

---

## engine/battle.py

### 전투 로직 (실데이터 백테스트)

가격은 LoadedGym으로 **이미 받아둔 상태**로 들어온다. battle은 I/O 없이 계산만.

```text
1) signals.combined_position → 일별 포지션, shift(1)로 하루 lag (룩어헤드 방지)
2) strat_ret = position.shift(1) * market_ret − |Δposition| × TRADE_COST  → 자산곡선
   TRADE_COST = 0.001 (토스증권 미국주식 위탁수수료 0.1%, 2025-12 상시화. 편도 과금)
3) 워밍업 버퍼 잘라내고 평가 구간만 슬라이스 (window_start = gym.start)
4) CAGR / MDD / 샤프 / 평균현금 측정
5) 0~100 스탯 정규화:
     HP    = 평균현금 * 100                    # 표시 전용 (적합도 가중치 0)
     ATK   = scale(CAGR,  0 ~ +0.25)           # 안 벌면 0점 (구: -25%~+25%는 현금이 50점)
     DEF   = scale(Calmar, -1 ~ 3)             # Calmar = CAGR/|내MDD|. 낙폭 0이면: 벌었으면 100, 아니면 25
     SKILL = scale(샤프,  -1 ~ 3)
```

> 정규화 구간(ATK_CAGR_*, DEF_CALMAR_*, SKILL_SHARPE_*)은 battle.py 상단 상수 = 튜닝 포인트.
>
> **DEF가 Calmar인 이유:** 구 설계 `1 - 내MDD/시장MDD`는 비중을 줄일수록 거의 1:1로
> 점수가 올라 '전부 현금'이 방어 만점이었다. Calmar는 비중 일괄 축소에 거의 불변
> (분자 CAGR·분모 MDD가 같이 줄어듦)이라 그 퇴화 경사가 없고, '낙폭을 적게 겪으며
> 번 전략'만 방어 점수를 받는다.

### 함수

```python
fight(strategy: Strategy, loaded: LoadedGym) -> BattleResult        # 한 시장 → 스탯블록
challenge(strategy: Strategy, loaded_gyms: list[LoadedGym]) -> Report
```

---

## engine/evolve.py (단일목적 GA)

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

* 실데이터엔 진짜 트레이드오프(공격 vs 위험조정 방어)가 있어 **전 유전자 조합으로 안 수렴**한다.
  (시그널 재배치 후 실측: 시드 42/7 모두 `VOL+REV_BB`(54.8점)로 수렴 — 전수조사 63조합 중 1위와 일치.
  방어형(VOL) + 역발상(REV_BB)의 타입 조합 = 재배치가 의도한 상성 구조가 실제로 선택됨.)
* 스탯들이 서로 당기는 이 충돌이 다목적(NSGA-III) 필요성의 근거.
* `STAT_WEIGHTS`를 바꾸면 진화 방향(공격형/방어형)이 바뀐다. 단 HP는 0 고정이 기본
  (올리면 '전부 현금' 퇴화가 되살아남 — `tests/test_baselines.py`로 확인 가능).

---

## main.py + config.json

### 역할

**config.json을 읽어** `app/service.py`의 함수를 호출만 한다. argparse/CLI 플래그는
쓰지 않는다(사용자 결정 — 옵션은 파일로 관리). 계산도 흐름 조립도 출력도 하지 않는다.
config.json이 없으면 main.py의 DEFAULTS로 동작하고, 윈도우 편집기의 BOM도 처리한다(utf-8-sig).

```python
run_single(gene_count, seed, md_path, capital)        # service.py — 단판 흐름
run_evolve(pop, generations, seed, md_path, capital)  # service.py — 진화 흐름
run_pokedex()                                         # service.py — 도감(mode: "dex")
```

> `md`: Markdown 리포트 저장. `""`이면 `reports/pocketquant_{single|evolve}_report.md`,
> 경로 문자열이면 그 경로, `null`이면 저장 안 함.
>
> `capital`: 실전 시뮬. 국면마다 `시작자본 × (1 + total_return)` = 최종 잔고를
> 단순보유와 함께 표시. 체육관은 서로 다른 시대라 **국면별 독립** 시뮬(연속 복리 아님).

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

### config.json 옵션

```jsonc
{
    "mode": "single",     // single(단판) | evolve(진화 GA) | dex(도감)
    "genes": null,        // [단판] 유전자 개수 (null = 랜덤)
    "pop": 20,            // [진화] 개체군 크기 (내부 최소 2)
    "generations": 10,    // [진화] 세대 수 (내부 최소 1)
    "seed": null,         // 랜덤 시드 (null = 매번 다름, 숫자 = 재현 가능)
    "md": null,           // Markdown 리포트: null=안 씀, ""=기본 경로, "경로"=지정
    "capital": null       // 실전 시뮬 시작 자본(원). 예) 10000000
}
```

### 실행 예시

```bash
python main.py    # config.json의 mode대로 실행 — 옵션을 바꾸려면 config.json 수정
```

```jsonc
// 재현 가능한 진화:   { "mode": "evolve", "seed": 42 }
// 유전자 3개 단판:    { "mode": "single", "genes": 3 }
// 도감 출력:          { "mode": "dex" }
// 진화 + 리포트 저장: { "mode": "evolve", "md": "" }
```

> 반드시 프로젝트 루트에서 실행. (`app.backend....` 절대 경로 import 구조)

### 출력 예시 (실제 출력)

```text
=== PocketQuant 단판 백테스트 ===

1. 데이터 로딩: SPY 실데이터 4개 국면

2. 전략 생성
  유전자: REV_BB + DD
  이름: 헤르메스 드래곤

3. 시장별 백테스트 결과
  2008 금융위기 체육관      연수익   -5.9%  전략 최대낙폭  -16.5%  시장 최대낙폭  -51.9%  종족치 115.7점
  2020 코로나 급락 체육관    연수익  -10.6%  전략 최대낙폭  -16.2%  시장 최대낙폭  -33.7%  종족치  79.7점
  2017 불타입 상승장 체육관   연수익   21.8%  전략 최대낙폭   -2.6%  시장 최대낙폭   -2.6%  종족치 286.2점
  2015-2016 횡보장 체육관  연수익    0.9%  전략 최대낙폭  -18.9%  시장 최대낙폭  -13.0%  종족치  61.0점

4. 종합 스탯
  체력   HP    현금 비중    37.3점  #######
  공격력 ATK   연수익       22.7점  #####
  방어력 DEF   낙폭대비수익    37.7점  ########
  솜씨   SKILL 샤프       38.0점  ########

종족치 합계 135.7 / 400
최종 적합도 32.8점   등급 C
```

> 출력 포맷은 service.py 기준(수시 변경 가능). 위는 예시 스냅샷.

### 등급 규칙 (적합도 = ATK/DEF/SKILL 가중평균 0~100, HP 제외)

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

### v0.3.1 적합도 퇴화 수정  ✅
* 문제: 동일가중 적합도에서 '전부 현금'이 ~69점으로 거의 최고점 (HP 100 + 구DEF 100)
* 수정: HP 가중치 0(표시 전용) · ATK 스케일 0%~+25% · DEF = Calmar(CAGR/|MDD|)
* 검증: `tests/test_baselines.py` — 전 조합 63 + 기준선 2 줄세우기에서
  '전부 현금' 65위/65(16.7점) < '항상 풀매수' 5위(38.2점) 확인
* GA 챔피언: VOL 단일(현금 의존) → `RSI+VOL`(42.1점, 시드 42/7 동일, 전수조사 1위와 일치)

### v0.3.2 거래비용  ✅
* `TRADE_COST = 0.001` — 토스증권 미국주식 위탁수수료 0.1%(2025-12 상시화) 기준.
  `strat_ret -= |Δposition| × TRADE_COST` (편도 과금, 평가 구간 내 매매만)
* 효과: 고회전 조합 일괄 하락(RSI+VOL -2.6점), 무매매인 풀매수는 3위로 상승.
  챔피언 `RSI+VOL` → `VOL` 단일(40.0점, 전수조사 1위와 일치)
* 관찰: 비용 반영 후 최고 전략(40.0) vs 풀매수(38.2) 격차가 1.8점뿐 —
  **현 시그널 풀은 현실적 비용 아래서 단순보유를 거의 못 이긴다** = 시그널 정비의 근거

### v0.4 스타팅 6마리 재배치 + 기권 결합 + config + 백엔드 재구성  ✅
* **시그널 풀 = 3타입 × 2마리**: DD/VOL(위험회피) + MA/MOM(추세) + REV_RSI/REV_BB(역발상, 신규).
  죽은 시그널이던 구 RSI(과열)·BB(상단)는 명단 제외 (함수는 보존).
* **기권(NaN) 결합**: 이벤트형 시그널은 발동일만 의견, 평소엔 평균에서 제외.
  전원 기권 = 0.0. 게이트: 기존 조합 252개 회귀 오차 0 확인.
* 검증(상세: `worklog/2026-06-10_signal_rework_results.md`): REV 페어는 상시형과 전부
  음의 상관(새 정보) · 신챔피언 `VOL+REV_BB` 54.8점(구 40.0)으로 4체육관 전부 개선
  (코로나 +2.8%→+12.2%) · SPY 4시대 모두 MDD 개선 + 샤프 동급 이상(4/4 PASS).
* main.py argparse 제거 → **config.json** (사용자 결정: CLI 플래그 금지).
* backend를 기능별 서브패키지(core/market/genes/engine)로 재구성 — 동작 동일 확인.

---

## 향후 확장 로드맵 (미구현)

### 다음: 다목적 최적화 (NSGA-III)

* 단일 적합도(스탯 가중합) → **스탯 4개 = 각각의 목적**
* 결과 = 최강 전략 1개가 아니라 **전략 Pareto front** (국면별 특화 + 올라운더)
* v0.3에서 관찰한 스탯 충돌(공격 vs 방어/현금)이 목적함수 정의의 근거.

### 정비

* `tests/` 확장 (test_strategy / test_gym / test_battle / test_evolve — test_baselines는 완료)
  * 전략 생성·중복 없음·gene_count 일치 / 체육관 목록·difficulty 존재
  * fight 결과 스탯 범위·seed 재현성 / GA 수렴·교배·돌연변이

### 전략 도감

* 전략 저장 / 전적 기록 / 생존률 누적

### v1.0 김박사 NPC

* 룰베이스 대사
* 이후 LLM 연결
