# PocketQuant — 에이전트 작업 사양서 (v1.x, 2026-06-13 기준)

> 이 문서는 **이 레포에서 작업하는 코딩 에이전트를 위한 온보딩 문서**다.
> 코드가 source of truth — 문서와 코드가 다르면 코드가 맞고, 문서를 고친다.
> 사람용 소개는 [README.md](README.md), 최적화 정식화는 [OPTIMIZATION.md](OPTIMIZATION.md).
> 기본 코드 리뷰·수정 담당 에이전트는 **Codex 5.5 연구원**으로 기록한다.

---

## ⚠️ 절대 규칙 (작업 전 숙지)

1. **퍼블릭 레포다.** 사용자의 개인 정보·직장/업무 맥락을 코드·문서·**커밋 메시지** 어디에도
   쓰지 않는다. 개인 맥락 기록은 gitignore 영역(`worklog/`)에만. (히스토리 재작성까지 간 전례 있음)
2. **argparse/CLI 플래그 금지.** 실행 옵션은 어댑터 모듈 상수(`SEED`, `TRIALS` 등)나
   `service.run_nsga3(...)` 호출 인자로 직접 표현한다 (사용자 확정 결정).
3. **골든 넘버 규약.** 엔진 계산을 건드리면 `python tools/test_engine_regression.py`가 어긋난다 —
   ① 의도한 설계 변경이면 골든 갱신 + worklog에 사유 기록 ② 아니면 버그, 커밋 금지.
4. **룩어헤드(컨닝) 금지.** 시그널은 과거만 본다. 시그널 추가/수정 시 `tools/test_no_lookahead.py` 필수.
5. **hold-out(2020-07~)은 소진됐다** (2026-06-11 사천왕전 1회 사용). 그 구간을 보고 적합도·
   가중치·파라미터를 고치는 것 = 오염 = 반칙. 이후 최종 판정은 미래 데이터로만.
6. **최적화 목적에 HP(현금 비중)·BST·0~100 클램프 스탯 금지** — raw 지표만. ('돼지저금통 퇴화' 3회 봉인 전례)
   *2026-06-13: 단일목적 GA(`evolve.py`) 제거 — NSGA-III만 운영. 이 규칙은 새 시그널/지표 추가 시도 유효.*
7. **사망 판정 금지.** 슬로건: "상폐가 아니면 뒤진 게 아니다". 판정 언어는 생존/탈락이 아니라
   **도전권/벤치** (벤치 = 명단 보존, 재도전 가능).
8. **오박사(LLM)는 해설 전용** — 판정·합불·매매 권유 금지. LLM을 적합도 루프에 넣지 않는다. (위치: `app/lab/oak.py`)
9. **용어**: 시그널 = 포켓퀀트(세는 단위 '마리') · 전략 = 트레이더(세는 단위 '명').
   사용자/운용자/전략 주체를 **절대 '트레이너'라고 부르지 않는다.** 이 레포의 공식 호칭은
   항상 **트레이더**다. 오박사 대사·labnotes·README·코드 주석도 동일하게 맞춘다.
   '포켓몬'은 README 도입부 비유·법적 고지 두 곳에만 존재한다.
10. **코드 스타일**: 한국어 왜-주석(설계 이유·실측 근거) 중심, `from __future__` 금지,
    타입 힌트는 시그니처에만 절제, 튜닝 상수는 모듈 상단에 모음. 과한 추상화 금지.
11. **Optuna storage 누적 금지** — `load_if_exists=False` 고정 (nsga3·tpe 엔진).
    - `study_name`은 매 시즌/실험마다 새로 (예: `nsga3_v1`, `tpe_v1`, `nsga3_v1x_kis_seed42`)
    - sqlite db는 시즌 임시 작업 영역 — 리그 결과를 `hall_of_fame_<버전>.md` 신설 + 인덱스 갱신한 뒤 폐기
    - 같은 이름으로 다른 탐색공간(tune_params·시그널 풀·목적 수) 재개 = "도감 오염" — Optuna `DuplicatedStudyError`로 즉시 차단됨

---

## 세계관 용어 (코드·출력에 그대로 쓰임)

| 용어 | 뜻 |
|---|---|
| 포켓퀀트 (13마리) | **스타팅 6** (DD/VOL/MA/MOM/REV_RSI/REV_BB) + **야생 7** (VOL_SPIKE/FEAR/US10Y/DXY/SPY_TLT/QQQ_SPY/QQQ_DIA, v1.x 2026-06-13 잡힘) |
| 트레이더 | 전략 = 포켓퀀트들을 어떤 가중치로 데려가는가 (Optuna 트라이얼 1개 = 트레이더 1명) |
| 체육관 (6개) | QQQ 실데이터 시장 국면. 관장: 버블/리먼/불사조/브이/황소/미로 |
| 성실이 | 라이벌 DCA 봇 — 매일 1/N 적립, **수수료 0원**(토스 자동 모으기). 이길 대상 |
| 돼지저금통 | '전부 현금' 기준선(0%) — 투자자 현타용(저축왕 3%와 대비). 퇴화 게이트(tools/test_baselines.py)가 상시 감시 |
| 챔피언로드 | 검증 관문: ①리그 본선(OOS 연도) ②배틀 프론티어(평행세계) ③사천왕(hold-out, 소진) |
| 오박사 | 로컬 LLM 해설 NPC (LM Studio, `app/lab/oak.py`) — 한강 둔치 쌉고인물. 단판 5~7문장 브리핑 + labnotes 모드(`app/lab/oak_labnotes.py`) |
| 실험실(lab) | 오박사 + 도감 + labnotes 모음 (`app/lab/`) — 메인 게임 루프 밖, 부재해도 본 게임 영향 X |
| 에그랩 | 새 포켓퀀트 부화 연구소 (`app/lab/egglab/README.md`) — 다음 알파 |

---

## 구조 (2026-06-13 재구성: app/league + app/lab + tools 분리)

```text
pocket_quant/
├─ OPTIMIZATION.md            # 최적화 정식화 + NSGA-III 설계 + 함정 3개 기록
├─ app/lab/egglab/README.md           # 새 시그널 부화 절차·알 후보 (다음 알파)
├─ character/                 # 캐릭터 이미지 (dr_oh.png, monsieur.png — 커밋됨)
├─ app/
│  ├─ service.py              # NSGA-III 실행 흐름 조립 + Regime Scanner 입력 갱신
│  ├─ lab/                    # 실험실 — LLM NPC + 도감 + labnotes
│  │  ├─ oak.py               #   오박사 LM Studio 해설 (단판 5~7문장 브리핑)
│  │  ├─ oak_labnotes.py      #   labnotes 작성 어댑터 (긴 연구 일지 모드)
│  │  └─ dex.py               #   도감 데이터 + print_pokedex()  (`python -m app.lab.dex`)
│  ├─ league/                 # 리그 — NSGA-III 탐색 + 챔피언로드 ①②③ + 라인업/분석
│  │  ├─ victory_road.py      #   관문 ①: 졸업생 OOS 연도 시험
│  │  ├─ battle_frontier.py   #   관문 ②: 블록 부트스트랩 평행세계 운빨 검사
│  │  ├─ elite_four.py        #   관문 ③: hold-out (소진. 재실행 = 참고용)
│  │  ├─ champion_road_lineup.py · baselines_comparison.py    # ① 풀라인업·기준선 비교
│  │  ├─ battle_frontier_lineup.py · battle_frontier_total.py # ② 풀라인업·토탈 분석
│  │  ├─ elite_four_lineup.py                                  # ③ 풀라인업
│  │  └─ sweep_seeds.py · top10_champions.py                   # 5 시드 탐색 + Top10 추출
│  └─ backend/
│     ├─ core/models.py       # Stats/Strategy/Gym/BattleResult/Report + 적합도
│     ├─ market/gym.py        # 체육관 6개 (전부 QQQ) — post-COVID 추가 금지 주석 참고
│     ├─ market/regime.py     # Regime Scanner (50/200 MA + 60일 수익률 + 20일 RV 백분위)
│     ├─ genes/signals.py     # 시그널 → 포지션(0~1)/기권(NaN), 기권 제외 (가중)평균 결합
│     ├─ engine/
│     │  └─ battle.py         # _score_position(공용 채점기) · fight · fight_dca(성실이)
│     │                       #   · score_vs_dca · 비용 0.1%/편도 (성실이만 무비용)
│     └─ data_io/             # 입력/리포팅 파이프라인 (워크플로우 아님)
│        ├─ data.py           #   yfinance + get_prices + get_volume + data_cache/<ticker>/
│        ├─ inspect_front.py · report_nsga3.py · battle_frontier_total.py  # 스터디 → CSV/HTML/MD
│        └─ (예정) kis_client.py  # KIS API 외인기관/호가/체결 — 새 시그널 원천
│  └─ academy/
│     ├─ academy.py           # 블록 부트스트랩 합성 체육관(내신) 생성
│     └─ study/               # 옵티마이저 4종: nsga3/tpe/cma_es/gp
│
├─ data_cache/                # 캐시 (gitignore) — 티커별 서브폴더: QQQ/, (예정) KIS/...
├─ tools/                     # validator + 진단 (코드/로직 검증만 — 워크플로우 X)
│  ├─ test_baselines.py · test_engine_regression.py · test_no_lookahead.py
│  ├─ test_weighted_combine.py · check_signals.py · check_dca.py
│  └─ walk_forward.py         # 선발 과정 OOS (자산/기간/비용 민감도)
├─ worklog/                   # (gitignore) 실험 노트 — 개인 맥락은 여기만
├─ labnotes/                  # 오박사 연구 일지 (커밋 가능 — 개인 맥락 금지)
└─ reports/                   # 리그 영구 기록 — hall_of_fame.md (시즌 인덱스) + hall_of_fame_<버전>.md 시즌별 파일, regime_picks.json
                              #   단계별 산출물(.json, sweep/lineup .md)은 시즌 마감 시 폐기 — 코드가
                              #   재실행하면 다시 생성됨. 영구 기록은 hall_of_fame_<버전>.md 신설 + 인덱스 갱신하고 임시 폴더 삭제.
```

의존 방향: `backend(core ← market/genes ← engine) ← lab/league ← service ← main` (순환 없음).
**폴더 규칙:** `tools/` = 검증만, 워크플로우(리그 실행) 절대 금지. 리그는 `app/league/`.
오박사·도감·labnotes는 `app/lab/`. 산출물은 `reports/` (버전 폴더 권장 — 예: `league_v1/`).

---

## 핵심 설계 (현재 값)

- **체육관 6개, 전부 QQQ** (훈련 자산 = 실투자 자산): 닷컴(2000-03~02-12) · 금융위기(2008-01~09-06) ·
  회복장(2009-03~10-12) · 코로나V(2020-02~06) · 상승장(2017) · 횡보장(2015~16).
  **post-COVID(2020-07~)는 훈련 체육관 추가 금지** (사천왕 — 소진됐어도 훈련 오염 금지는 유지).
- **스탯(0~100, 표시용)**: HP=평균현금(적합도 가중 0) · ATK=CAGR(0~25%) · DEF=Calmar(-1~3) ·
  SKILL=샤프(-1~3). 정규화 구간은 battle.py 상단 상수.
- **적합도(단일목적) = 체육관별 fitness의 [평균 70% + 최약 30%]** (models.Report).
  50/50은 돼지저금통 부활로 기각 — min 가중 0.3이 퇴화 게이트를 지키는 실측 최대치.
- **거래비용**: 트레이더 0.1%/편도(턴오버 과금) vs 성실이 0원 — 비대칭이 현실 모델.
- **결합**: `combine_positions(positions, weights=None)` — 기권(NaN) 제외 (가중)평균.
  분모에 Σw = 예산제약 내장. weights=None은 동일가중과 비트 동일(골든 보호).
- **시그널 풀 (2026-06-13 v1.x)**: 13마리. 스타팅 6(가격 기반) + 야생 7(외부 정보원).
  외부 정보원은 yfinance로 받음 (`^VIX`, `^TNX`, `UUP`, `SPY`, `TLT`, `QQQ`, `DIA`).
  외부 데이터 없는 시기는 자동 NaN 기권 (UUP는 2007년~).
- **옵티마이저 4종** (`app/academy/study/`):
  - `nsga3.py` — **다목적**. 6목적 score_vs_dca (5국면 + turnover). Pareto 라인업.
  - `tpe.py` — **단일목적 Bayesian**. 잔고 합 max. v1 챔피언(TPE-s11)이 여기서 나옴.
  - `cma_es.py` — **단일목적 CMA-ES**. 연속 공간 강함. NSGA 계열 친숙 (사용자 본업).
  - `gp.py` — **단일목적 GP Bayesian**. v1.x 본탐색에서 가장 빠른 saturation을 보였으나 OOS 음의 상관으로 벤치.
  셋 다 같은 인터페이스 (`run_study(trials, seed, storage, study_name, load_if_exists=False, ...)`).
  채점(`battle.py`)은 `engine/`에 유지 — 탐색(search)과 평가(engine) 분리.
- **NSGA-III 목적**: [bear=min(닷컴,GFC), rebound, crash_v, bull, chop] score_vs_dca
  maximize + turnover minimize. `tune_params=False`가 기본 (v1 과적합 전멸 사례).
- **score_vs_dca** = 0.4×수익차 + 0.4×낙폭개선 + 0.2×샤프차 (성실이 대비, raw만).
  ⚠️ 평균으로 뭉개 단일 적합도로 쓰지 말 것(돼지저금통 부활 실측) — 벡터 그대로.

## 심판단 (tools/) — `python tools/<파일>.py`

| 파일 | 역할 | 성격 |
|---|---|---|
| test_baselines.py | 돼지저금통 감시 — 전부현금<풀매수 & 하위25% | 게이트 (커밋 전) |
| test_engine_regression.py | 골든 넘버 — 엔진 계산 불변 확인 | 게이트 (커밋 전) |
| test_no_lookahead.py | 컨닝 검사 — 미래 절단 후 과거 포지션 불변 | 게이트 (시그널 변경 시) |
| test_weighted_combine.py | 가중 결합 불변식 5종 | 게이트 (결합 변경 시) |
| check_signals.py | 노출/발동률/상관 — 새 시그널의 '새 정보' 검사 | 진단 |
| check_dca.py | 성실이 기준선 + score_vs_dca 전수조사 | 진단 |
| walk_forward.py | 선발 과정 OOS (legacy 클램프 스탯 GA 기반 — 새 풀에선 별도 사용) | 참고 |
| e2e.py | 전 파이프라인 스모크 (게이트 4 + 진단 2 + NSGA-III). walk_forward 제외 | 큰 변경 후 |

## 리그 (app/league/) — `python app/league/<파일>.py`

| 파일 | 역할 |
|---|---|
| victory_road.py · battle_frontier.py · elite_four.py | 챔피언로드 ① ② ③ 본 어댑터 |
| champion_road_lineup.py · baselines_comparison.py | ① 풀라인업 입장 · 기준선 4인방 비교 |
| battle_frontier_lineup.py · battle_frontier_total.py | ② 풀라인업 · 토탈 잔고 분석 |
| elite_four_lineup.py | ③ 풀라인업 |
| sweep_seeds.py · top10_champions.py | 5 시드 NSGA-III 탐색 + Top10 추출 |

## 리포팅·데이터 IO (app/backend/data_io/) — `python -m app.backend.data_io.<name>`

| 파일 | 역할 |
|---|---|
| inspect_front.py · report_nsga3.py | 저장된 스터디 분석 / HTML 리포트 생성 |
| battle_frontier_total.py | 관문 ② lineup JSON 재처리 — 토탈 수익 .md 출력 |
| (예정) kis_client.py | KIS API 입력 — 외인기관/호가 잔량/체결 강도/거래량 급증 |

## `service.run_nsga3(...)` 인자 (NSGA-III 진입점)

```python
run_nsga3(
  trials=600,                  # 총 목표 trial 수 (재개 시 모자란 만큼만)
  seed=None,                   # 랜덤 시드 (None=매번 다름)
  storage=None,                # 예: "sqlite:///optuna_pocketquant.db" (중단/재개)
  study_name="nsga3_v2_weights",
  tune_params=False,           # True=파라미터도 탐색 (v1 과적합 — 고도화용)
  population_size=50,          # NSGA-III 한 세대 크기
  early_stop_window=None,      # HV MA(N) 정체 시 self stop. None=끔
  adaptive_mutation=False,     # True=HV 신호로 mutation_prob 자동 조정
)
```

진입점은 어댑터/도구가 옵션을 인자로 직접 주입한다(예: `tools/e2e.py`의 NSGA-III smoke).
별도 config 파일·CLI 플래그는 없다.

**도감 보기**: `python -m app.lab.dex` (모드 분기 없음 — 직접 호출).
**오박사 단판 브리핑**: 단판 모드가 없어졌으므로 `professor_briefing(report)`은 챔피언로드
관문 ③(`elite_four.py`)에서만 호출됨. labnotes 작성: `python -m app.lab.oak_labnotes`.

---

## 현재 상태 스냅샷 (2026-06-11)

- **챔피언**: 동일가중 `VOL+REV_RSI+REV_BB` 42.6점 — 시드 42/7 수렴 = 전수조사 1위.
  챔피언로드 관문 ①(유일 도전권)·②(승률 69%/방어 98%) 통과.
- **사천왕전 결과: 벽** — 연 단위 라이벌전 평균 -2.7(FAIL) / 방어 PASS(MDD -17.4% vs -35.1%).
  단 6년 누적은 성실이 전 지표 우위(CAGR +13.5% vs +11.1%, 샤프 1.02 vs 0.92) —
  "성실이를 매년 이기는 알파"는 아니고 "DCA 코어 위 방어 오버레이"로 가치 입증.
- **가중치 천장 확인**: 리그 v2에서 가중치 조정만으론 챔피언을 못 넘음 — 새 알파는 새 시그널에서.
- **스페셜리스트**: #1918(bear 1위) — 하락 평행세계 승률 86%/방어 100%. Regime Scanner
  Defensive 틸트 1호 후보 (스터디 DB `nsga3_v2_weights`에 보존).
- **버전 히스토리(압축)**: v0.2 GA → v0.3 실데이터+스탯 → v0.3.1 돼지저금통 1차 봉인 →
  v0.3.2 거래비용 → v0.4 시그널 재배치+기권 결합+config → v0.4.1 worst-case 적합도(70/30) →
  v0.5 QQQ 6체육관+성실이+심판단+NSGA-III+챔피언로드+오박사. 상세는 worklog/(로컬).

## 다음 작업 (우선순위)

1. **에그랩 부화 1호** — `app/lab/egglab/README.md`의 알 후보(VIX/TNX/QQQ-SPY/SOX-QQQ/DXY)에서 선택,
   부화 절차 6단계(컨닝→새 정보 상관→미시→예선→리그→챔피언로드)
2. **웜스타트 시드 제너레이터** — 다음 리그 전 필수 (`study.enqueue_trial`, v2에서 필요성 실증)
3. **Regime Scanner + 70/30 오버레이** — #1918 틸트 후보 대기
4. **아카데미(cGAN)** — 장기. 나스닥 학습, 국면 라벨 조건 합성. 학습 재료도 hold-out 규칙 적용
