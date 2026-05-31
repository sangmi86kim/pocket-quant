# 🎮 PocketQuant

> 전략을 "포켓몬"처럼 키우고, 과거 위기 상황을 "체육관"으로 돌파시키는 장난감 퀀트 프로젝트

**PocketQuant MVP v0.1** — 복잡한 금융 로직보다 **게임 컨셉**을 우선한 CLI 프로젝트입니다.
전략 포켓몬을 랜덤 생성하고, 가짜 데이터 기반 체육관에서 생존 테스트를 합니다.

---

## 🧩 컨셉

| 게임 용어 | 실제 의미 |
|-----------|-----------|
| 전략 포켓몬 | 트레이딩 전략 |
| 유전자 | 전략 속성 (지표) |
| 체육관 도전 | 백테스트 |
| 생존 / 사망 | 백테스트 통과 / 실패 |

---

## 🧬 유전자

전략은 아래 유전자를 랜덤 조합해 만들어집니다.

| 유전자 | 점수 | 의미 |
|--------|------|------|
| `DD`  | +20 | Drawdown |
| `RSI` | +15 | Relative Strength Index |
| `MA`  | +25 | Moving Average |
| `BB`  | +10 | Bollinger Bands |
| `FX`  | +5  | FX 노출 |

예시: `DD몬`, `DD-RSI몬`, `DD-RSI-MA몬`, `타이탄 드래곤`

---

## 🏟️ 체육관

각 체육관은 `difficulty`, `volatility` 두 변수만 가집니다. (실제 주가 데이터 미사용 — 가짜 데이터)

| 체육관 | difficulty | volatility |
|--------|-----------|-----------|
| `DOTCOM`           | 90 | 80 |
| `FINANCIAL_CRISIS` | 85 | 70 |
| `COVID`            | 40 | 90 |
| `RATE_SHOCK`       | 60 | 50 |

---

## ⚔️ 전투 로직

```
최종 점수 = (유전자 점수 합) + random(-20 ~ +20)

최종 점수 >= 체육관 difficulty  →  생존
그렇지 않으면                    →  사망
```

생존률에 따라 등급이 매겨집니다.

| 등급 | 생존률 |
|------|--------|
| `S` | ≥ 0.9 |
| `A` | ≥ 0.7 |
| `B` | ≥ 0.5 |
| `C` | ≥ 0.3 |
| `D` | 그 외 |

---

## 🚀 실행

> ⚠️ 반드시 **프로젝트 루트**에서 실행하세요. `app/backend/` 내부 파일을 직접 실행하면 상대 import가 깨집니다.

```bash
python main.py          # 유전자 개수 랜덤
python main.py -g 3     # 유전자 3개 고정
```

한글이 깨지면 (Windows):

```powershell
$env:PYTHONIOENCODING="utf-8"
python main.py
```

### 출력 예시

```
=== PocketQuant ===

전략 생성
DD + RSI + MA
이름: DD-RSI-MA몬

체육관 도전

DOTCOM
생존

COVID
생존

RATE_SHOCK
사망

결과
생존 2
사망 1
등급 B
```

---

## 📁 구조

```
pocket_quant/
├─ main.py              # CLI 진입점 (argparse)
└─ app/
   └─ backend/
      ├─ models.py      # dataclass + 유전자 점수 / 등급 테이블
      ├─ strategy.py    # 전략 포켓몬 생성 + 이름 자동 생성
      ├─ gym.py         # 체육관 4종 정의
      └─ battle.py      # 점수 계산 → 생존 / 사망 판정
```

---

## 🔮 확장 여지 (의도적으로 단순 유지)

- **실제 데이터** → 가짜 데이터를 실제 가격 시계열로 교체 시 `battle.fight()`만 수정
- **체육관 추가** → `gym.py`의 `GYMS` 리스트에 한 줄
- **유전자 추가** → `models.py`의 `GENE_SCORES`에 한 줄

---

## 📜 라이선스

장난감 프로젝트입니다. 자유롭게 사용하세요. 🎈
