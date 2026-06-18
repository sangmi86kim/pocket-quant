"""교육용 그림 — 블록 셔플에서 '레벨형 raw 이어붙이기'가 왜 가짜 이벤트를 만드나.

[왜] 시즌 3 GP 점검 중 발견: 합성 교과서(`curriculum/textbook.py`)가 외부 정보원을
타입별로 다르게 복원한다 — price형(SPY/TLT/QQQ비율)은 수익률로 잘라 연속 복원,
level형(^VIX/^TNX/거래량)은 raw 값 그대로 이어붙임. raw는 21일 블록 경계마다 절대값이
점프(평온 VIX 12 → 위기 55)해서, US10Y/FEAR 같은 '평균 대비 급락/급등' 신호가 실제로는
없는 가짜 이벤트에 발동 → 합성장에서 오학습. 이 메커니즘을 토이 데이터로 시각화한다.

[주의] 실데이터 아님 — 개념 설명용 토이 시계열. 결론은 `app/lab/reports/
gp_external_synth_diagnosis.md` 참조.

실행: .venv/Scripts/python.exe -m app.lab.block_shuffle_explainer
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

BLOCK = 21
OUT_PNG = os.path.join(os.path.dirname(__file__), "reports", "block_shuffle_explainer.png")


def main() -> None:
    os.makedirs(os.path.dirname(OUT_PNG), exist_ok=True)
    rng = np.random.default_rng(7)
    # 서로 다른 국면의 VIX 토막 3개 (평온/위기/보통).
    # 토막 안은 실제 일별 출렁임이 살아있음(랜덤워크) — 랜덤인 건 토막 순서지 하루하루가 아님.
    calm = 13 + np.cumsum(rng.normal(0, 0.7, BLOCK))
    crisis = 52 + np.cumsum(rng.normal(0, 2.4, BLOCK))
    normal = 18 + np.cumsum(rng.normal(0, 0.9, BLOCK))
    blocks = [calm, crisis, normal]

    raw = np.concatenate(blocks)
    x = np.arange(len(raw))
    seams = [BLOCK, 2 * BLOCK]

    # delta 복원: 블록 안 '변화량'만 직전 레벨에서 이어 누적 + 바닥 clip
    cur, cont = 13.0, []
    for blk in blocks:
        for dd in np.diff(blk, prepend=blk[0]):   # 첫날 변화량 0 → 점프 없이 이어짐
            cur = float(np.clip(cur + dd, 8, None))
            cont.append(cur)
    cont = np.array(cont)

    fig, (a1, a2) = plt.subplots(2, 1, figsize=(10.5, 7.2), sharex=True)

    a1.plot(x, raw, color="#c62828", lw=1.8)
    for s in seams:
        a1.axvline(s - 0.5, ls="--", color="gray", alpha=0.6)
    a1.annotate("가짜 급등\n(실제엔 없음)", xy=(BLOCK - 0.5, 34), xytext=(BLOCK + 1, 30),
                color="#c62828", fontsize=9,
                arrowprops=dict(arrowstyle="->", color="#c62828"))
    a1.annotate("가짜 급락 → US10Y 오발동", xy=(2 * BLOCK - 0.5, 37), xytext=(2 * BLOCK - 19, 44),
                color="#c62828", fontsize=9,
                arrowprops=dict(arrowstyle="->", color="#c62828"))
    a1.set_title("❌ 레벨형 raw 토막 이어붙이기 — 이음매서 확 튐 (12→55→18)")
    a1.set_ylabel("VIX 레벨")

    a2.plot(x, cont, color="#2e7d32", lw=1.8)
    for s in seams:
        a2.axvline(s - 0.5, ls="--", color="gray", alpha=0.6)
    a2.annotate("이음매 연속", xy=(BLOCK - 0.5, cont[BLOCK]), xytext=(BLOCK + 1, cont[BLOCK] + 4),
                color="#2e7d32", fontsize=9,
                arrowprops=dict(arrowstyle="->", color="#2e7d32"))
    a2.set_title("✅ 변화량으로 이어 누적(delta+clip) — 이음매 안 튐 / 단 절대레벨은 표류(평균회귀 미보존)")
    a2.set_ylabel("VIX 레벨")
    a2.set_xlabel("거래일 (21일 블록 × 3)")

    fig.suptitle("블록 셔플: 레벨형(raw) vs 변화량 복원 — 왜 경계 점프가 가짜 이벤트를 만드나",
                 fontsize=12, y=1.0)
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"[저장] {OUT_PNG}")


if __name__ == "__main__":
    main()
