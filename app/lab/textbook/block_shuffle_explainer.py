"""교육용 그림 — 합성 교과서가 가격형/레벨형 신호를 어떻게 만드나 (현재 vs 고침).

[왜] 시즌 3 GP 점검 중 발견: 교과서(`curriculum/textbook.py`)가 외부 정보원을 타입별로 다르게
복원한다. **가격형**(SPY/TLT/QQQ비율)은 수익률로 잘라 누적 복원 → 연속(멀쩡). **레벨형**
(^VIX/^TNX)은 raw 절대값을 그대로 이어붙임 → 21일 토막 경계서 점프(평온 VIX 13 → 위기 52).
US10Y/FEAR 같은 '평균 대비 급락/급등' 신호가 이 가짜 경계 점프를 진짜 이벤트로 착각해 발동 →
합성장 오학습(GP가 그 가짜에 맞춰 US10Y=1을 정답으로 외운 게 'GP 무죄'의 증거).

[출력] app/lab/outputs/textbook/block_shuffle_explainer/ 에 그림 2장.
  - textbook_current.png : 현재 교과서의 현실 — 가격형(연속 OK) vs 레벨형(경계 점프, 짝퉁)
  - textbook_fixed.png   : 고친 교과서 — 가격형(그대로) vs 레벨형(변화량+평균회귀 → 연속 & 밴드 유지)

[주의] 실데이터 아님 — 개념 설명용 토이 시계열. 결론은 `season3_textbook_diagnosis.md` 참조.
실행: .venv/Scripts/python.exe -m app.lab.textbook.block_shuffle_explainer
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

BLOCK = 21
SEAMS = [BLOCK, 2 * BLOCK]
RDIR = os.path.join(os.path.dirname(__file__), "..", "outputs", "textbook",
                    "block_shuffle_explainer")
OUT_CURRENT = os.path.join(RDIR, "textbook_current.png")
OUT_FIXED = os.path.join(RDIR, "textbook_fixed.png")

GREEN, RED = "#2e7d32", "#c62828"


def _make_series(rng):
    """3토막(평온/위기/보통)으로 가격형 1개·레벨형 raw·레벨형 fix 만들기."""
    # 가격형: 토막별 수익률을 누적(cumprod) → 연속. (현재·고침 동일 = 손댈 것 없음)
    rets = np.concatenate([
        rng.normal(0.0009, 0.006, BLOCK),    # 평온: 완만 상승·저변동
        rng.normal(-0.004, 0.030, BLOCK),    # 위기: 하락·고변동
        rng.normal(0.0012, 0.010, BLOCK),    # 보통
    ])
    price = 100 * np.exp(np.cumsum(rets))

    # 레벨형 토막(절대값): 평온~13 / 위기~52 / 보통~18, 토막 안은 일별 출렁임 보존
    calm = 13 + np.cumsum(rng.normal(0, 0.7, BLOCK))
    crisis = 52 + np.cumsum(rng.normal(0, 2.4, BLOCK))
    normal = 18 + np.cumsum(rng.normal(0, 0.9, BLOCK))
    blocks = [calm, crisis, normal]
    level_raw = np.concatenate(blocks)       # 현재: raw 이어붙임 → 경계 점프

    # 고침: 변화량(delta)만 직전 레벨에서 이어 + 평균회귀 앵커(μ=22) → 연속 & 밴드 유지
    mu, theta, cur, fix = 22.0, 0.06, 14.0, []
    for blk in blocks:
        for dd in np.diff(blk, prepend=blk[0]):   # 첫날 변화량 0 → 이음매 연속
            cur = float(np.clip(cur + dd + theta * (mu - cur), 9, 70))
            fix.append(cur)
    return price, level_raw, np.array(fix)


def _seams(ax):
    for s in SEAMS:
        ax.axvline(s - 0.5, ls="--", color="gray", alpha=0.55)


def _fig(price, level, *, level_color, level_title, suptitle, out, annotate_jump):
    x = np.arange(len(price))
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(10.5, 7.2), sharex=True)

    a1.plot(x, price, color=GREEN, lw=1.8)
    _seams(a1)
    a1.set_title("가격형(예: QQQ·SPY 비율) — 수익률로 이어 → 이음매 연속 (OK, 손댈 것 없음)")
    a1.set_ylabel("합성 가격")

    a2.plot(x, level, color=level_color, lw=1.8)
    _seams(a2)
    if annotate_jump:
        a2.annotate("가짜 급등\n(실제엔 없음)", xy=(BLOCK - 0.5, 34), xytext=(BLOCK + 1, 30),
                    color=RED, fontsize=9, arrowprops=dict(arrowstyle="->", color=RED))
        a2.annotate("가짜 급락 → US10Y 오발동", xy=(2 * BLOCK - 0.5, 37),
                    xytext=(2 * BLOCK - 19, 44), color=RED, fontsize=9,
                    arrowprops=dict(arrowstyle="->", color=RED))
    else:
        a2.axhline(22, ls=":", color="gray", alpha=0.7)
        a2.annotate("이음매 연속 + 22 근처로 평균회귀", xy=(BLOCK - 0.5, level[BLOCK]),
                    xytext=(BLOCK + 1, level[BLOCK] + 6), color=GREEN, fontsize=9,
                    arrowprops=dict(arrowstyle="->", color=GREEN))
    a2.set_title(level_title)
    a2.set_ylabel("VIX 레벨(레벨형)")
    a2.set_xlabel("거래일 (21일 블록 × 3: 평온 → 위기 → 보통)")

    fig.suptitle(suptitle, fontsize=12, y=1.0)
    fig.tight_layout()
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    os.makedirs(RDIR, exist_ok=True)
    rng = np.random.default_rng(7)
    price, level_raw, level_fix = _make_series(rng)

    _fig(price, level_raw, level_color=RED,
         level_title="레벨형(VIX·금리) — 절대값 raw 이어붙임 → 경계서 확 튐 (13→52→18, 가짜 이벤트)",
         suptitle="현재 교과서의 현실 — 가격형은 멀쩡, 레벨형은 짝퉁(경계 점프)",
         out=OUT_CURRENT, annotate_jump=True)

    _fig(price, level_fix, level_color=GREEN,
         level_title="레벨형 고침 — 변화량(delta) + 평균회귀 앵커 → 이음매 연속 & 밴드 유지",
         suptitle="고친 교과서 — 레벨형도 가격형처럼 연속, 평균회귀로 절대 밴드 보존",
         out=OUT_FIXED, annotate_jump=False)

    print(f"[저장] {OUT_CURRENT}\n[저장] {OUT_FIXED}")


if __name__ == "__main__":
    main()
