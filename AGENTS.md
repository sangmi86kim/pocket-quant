# PocketQuant — 에이전트 작업 헌장

> 코딩 에이전트의 **최소 운영 규칙**. 코드가 source of truth다. 문서와 코드가 다르면 코드를 확인하고 문서를 고친다.

사람용 소개는 [README.md](README.md), 최적화 정식화는 [OPTIMIZATION.md](OPTIMIZATION.md), 상세 실행 절차는 [pocketquant-engineer](.codex/skills/pocketquant-engineer/SKILL.md).

---

## 역할

Codex가 수석. 코드·수치·검증 정합성이 걸리면 Codex가 맡는다.

- **Codex**: 코드 리뷰, 리팩터, 비용 모델, 리그 판정, 수치 검산, 최종 운영 판단.
- **Opus**: 오박사 페르소나 문체와 최종 해설. Codex 산출물을 먼저 확인한 뒤 쓴다.
- **Sonnet**: Markdown, 표, 링크, 그래프 경로, 중복 문장 정리.
- **Fable**: README 세계관 인물이지 실무 실행 기준이 아니다.

---

## 절대 규칙

1. **퍼블릭 레포다.** 개인 정보·직장/업무 맥락을 코드, 문서, 커밋 메시지에 남기지 않는다.
2. **실행은 `.venv/Scripts/python.exe`로 한다.** 글로벌 Python은 torch/scipy/sklearn 누락으로 학습·검증을 속일 수 있다.
3. **argparse/CLI 플래그 금지.** 실행 옵션은 모듈 상수, config/payload, 또는 `run_study(...)` 같은 직접 호출 인자로 표현한다.
4. **LLM은 판정 루프에 들어가지 않는다.** 적합도, 가중치 결정, 합불 판단, 매매 권유에 LLM 답을 쓰지 않는다.
5. **룩어헤드 금지.** 시그널은 과거만 본다. 미래 데이터로 과거 포지션이 바뀌면 버그다.
6. **hold-out 오염 금지.** post-COVID 2020-07 이후 구간은 이미 일부 소진됐다. 그 결과를 보고 가중치·파라미터·설계를 고치지 않는다.
7. **미래 봉인 유지.** `FUTURE_SEAL_DATE = 2026-06-19` 이후 데이터는 학습·튜닝·검증 설계에 쓰지 않는다.
8. **에그랩 봉인.** `app/pocket/eggs/`의 합성 시그널 부화는 사용자 명시 지시 전까지 금지다.
9. **사망 판정 금지.** 전략은 죽는 게 아니라 도전권/벤치로 관리한다. 벤치는 명단 보존과 재도전 가능 상태다.
10. **용어 고정.** 시그널 = 포켓퀀트, 전략 = 트레이더. 사용자/운용자/전략 주체를 트레이너라고 부르지 않는다.
11. **코드 스타일.** 한국어 왜-주석 중심, `from __future__` 금지, 타입 힌트는 시그니처에 절제해서 쓴다.
12. **`__init__.py`에 업무 로직 금지.** 패키지 표식과 짧은 설명만 둔다.

---

## 검증 게이트

코드를 고쳤으면 커밋 전 기본 게이트를 통과시킨다.

```powershell
.venv\Scripts\python.exe -m ruff check .
.venv\Scripts\python.exe -m mypy <고친파일.py> --ignore-missing-imports --follow-imports=silent
.venv\Scripts\python.exe -m pytest --ignore=tools/test_baselines.py
```

- mypy는 이번에 고친 `.py` 파일만 본다. `mypy app tools` 금지.
- 문서·worklog만 고친 경우 기본 코드 게이트는 면제다.
- `tools/test_baselines.py`, `tools/e2e.py`는 느리다. 사용자가 "전체 validate"를 명시했을 때만 돌린다.

작업별 추가 게이트:

| 변경 영역 | 추가 확인 |
|---|---|
| 엔진 계산, 비용, `battle.py` | `tools/test_engine_regression.py`. 골든이 깨지면 의도 변경인지 먼저 판정한다. |
| 시그널, 외부 데이터 | `tools/test_signals_fuzz.py`, `tools/test_no_lookahead.py`. 새 외부 티커는 `EXTERNAL_TICKERS`에 등록한다. |
| 결합 로직 | `tools/test_weighted_combine.py`. 기권은 `NaN`이고 분모에서 제외한다. |
| 아카데미 | 필요 시 `tools/smoke_academy.py`. |
| 리그·시험 통합 | 필요 시 `tools/smoke_league.py`. |

---

## 핵심 계약

### 모델·엔진

- `app/pocket/models.py`는 데이터 모양 모듈이다. `Stats`, `Strategy`, `Gym`, `BattleResult`, `Report` 같은 컨테이너 중심으로 유지한다.
- 엔진 계산은 `app/pocket/battle.py`, 시그널 계산은 `app/pocket/signals.py`가 맡는다.
- `Stats.fitness`와 `Report.fitness`는 레거시/표시 호환 경로다. 시즌3 이후 선발 화폐는 raw 종료 잔고와 리그 결과다.
- HP, BST, 0~100 클램프 스탯을 최적화 목적에 넣지 않는다.

### 비용 모델

현재 시즌3 기본 비용 모델은 `season3_flat_1bp_band5`다.

- 트레이더 수수료: 0.1%/편도
- 성실이(DCA) 수수료: 0원
- 슬리피지: 1bp, 성실이 포함 전원 공통
- No-trade band: 5%, 목표 포지션이 아니라 실제 체결 포지션/turnover에 적용
- `cost_model.complete != True`인 학습 산출물은 졸업시험·리그·보고서 입력으로 쓰지 않는다.

### Optuna·학습

- 탐색공간이 바뀌면 새 `study_name`을 쓴다.
- sqlite DB와 top30 JSON 같은 학습 중간 산출물은 `app/academy/training/db/`에 둔다.
- 학습은 에이전트가 오래 지켜보는 일이 아니다. 스크립트가 재개 가능하고, 완주 시 harvest와 리포트를 자동 산출해야 한다.
- 임시 chunk/reap 우회 스크립트는 표준 경로로 흡수한 뒤 남기지 않는다.

---

## 폴더 경계

| 폴더 | 역할 |
|---|---|
| `app/world/` | 데이터 로딩, 캐시, 국면 라벨 |
| `app/pocket/` | 시그널, 전투 엔진, 도감, 데이터 모델 |
| `app/academy/` | 학습과 시험 |
| `app/league/` | 리그 관문과 시즌 무관 운영 |
| `app/lab/` | 진단·분석 스크립트. 메인 게임 루프 밖 |
| `tools/` | 게이트와 스모크만. 연구 스크립트나 리그 실행 워크플로우 금지 |
| `reports/` | 정식 보고서. 개인 맥락 금지 |
| `worklog/` | 로컬 전용, git 비추적 |

아카데미 산출물 경계:

- 학습 DB/중간 산출물: `app/academy/training/db/`
- 학습 리포트/그래프: `app/academy/training/results/<season>/graph/`
- 시험 리포트/그래프: `app/academy/exam/results/<season>/graph/`

---

## 리포트 그래프

정식 리포트에는 핵심 결과 그래프를 둔다.

- 연구소보고서: `reports/연구소보고서/graph/`
- 리그 결과: `reports/포켓퀀트리그/graph/<version>/`
- lab 그래프를 정식 리포트에서 쓸 때는 정식 리포트의 `graph/`로 복사한다.
- gitignore 폴더의 이미지를 정식 리포트에서 직접 참조하지 않는다.
- matplotlib PNG 권장. 한글 그래프는 `Malgun Gothic`, `axes.unicode_minus=False`.
- 그림만 남기지 말고 재생성 가능한 스크립트를 같이 둔다.

---

## 현재 상태

- 시즌3 리그 완료. 기준 산출물은 `classroom_top30_20260622_195214_v2.json`.
- 비용 모델은 `season3_flat_1bp_band5`.
- 포켓퀀트 풀은 14마리: 스타팅 6 + 야생 8(`FEAR_NQ` 포함).
- 시즌3 결론: 상승장 알파를 억지 방어 보충으로 깎지 말고, 시즌4는 Regime Scanner 오버레이로 국면별 출전 판단을 분리한다.

---

## 작업 시작 체크

1. 관련 코드와 README/OPTIMIZATION 중 필요한 부분만 읽는다.
2. 상세 절차가 필요한 코드 작업이면 [pocketquant-engineer](.codex/skills/pocketquant-engineer/SKILL.md)를 쓴다.
3. 수정 후 실제 사용 경로와 테스트 수집·실행 여부를 확인한다.
