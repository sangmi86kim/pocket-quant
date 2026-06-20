"""새 교과서(RS 하이브리드) 생성기 + 옛 교과서와 생성 그래프 비교 (lab).

[새 교과서 = 국면조건부 블록 부트스트랩]
옛 교과서(textbook.make_world)는 21일 진짜 토막을 *무작위로* 뽑아 잇는다 → 분위기
비율을 통제 못 함. 새 교과서는 RS(마르코프 국면전환)가 *블록 단위 국면 순서*를 먼저
정하고, 각 칸마다 그 국면에 해당하는 *진짜 토막*을 뽑아 잇는다.
  → 분위기 순서는 통제, 재료는 진짜 토막(수익률·꼬리 그대로) = 어제 RS 정규분포의
    극단 급락 문제 없음.

[이 파일이 답하는 것]
  1. 옛/새 교과서로 만든 가짜 시장 가격 곡선 — 실제 대비 이상한 급락·급등 없나?
  2. 일별 수익률 꼬리(급락급등) 분포 — 실제 vs 옛 vs 새, 최대 일변동.
  3. 분위기 비율(상승/하락/횡보/변동) — 실제 vs 옛 vs 새 (새가 통제로 가까워지나).

[주의] 여기 RS는 P·π를 실제에서 추정(균일 비교용). 약점 국면 가중(적대적 시드)은 다음 단계.

실행: .venv/Scripts/python.exe -m app.lab.rs_hybrid_textbook
"""
import collections
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from app.academy.curriculum import textbook as tb
from app.academy.curriculum.textbook import (
    BLOCK_DAYS, DATA_END, DATA_START, DEFAULT_EXTERNAL_STREAMS, WARMUP_TDAYS,
    _from_block_unit, _load_external,
)
from app.pocket.models import Gym
from app.world.data_loader import LoadedGym, get_prices
from app.world.regime import REGIME_LABELS, classify_daily

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

STATES = ["bull", "bear", "sideways", "volatile"]
SIDX = {s: i for i, s in enumerate(STATES)}
LABEL_COLOR = {"bull": "#2e7d32", "bear": "#c62828",
               "sideways": "#9e9e9e", "volatile": "#f9a825"}
N_WORLDS = 30
OUT_DIR = os.path.join(os.path.dirname(__file__), "outputs", "rs_hybrid_textbook")


def _block_pools_and_chain(qqq_raw: pd.Series, common_idx: pd.Index):
    """진짜 토막을 국면별로 모으고(pool), 블록 단위 전이행렬 P·초기분포 π를 추정.

    블록 라벨 = 그 21일 블록의 *마지막 날* 국면(분류기는 지연 함수라 끝 라벨이 대표적).
    """
    labels = classify_daily(qqq_raw).reindex(common_idx)
    lab = labels.to_numpy()
    total = len(common_idx)

    pools: dict[str, list[int]] = {s: [] for s in STATES}
    for i in range(total - BLOCK_DAYS + 1):
        end = lab[i + BLOCK_DAYS - 1]
        if isinstance(end, str):
            pools[end].append(i)

    seq = np.array([SIDX[lab[s + BLOCK_DAYS - 1]]
                    for s in range(0, total - BLOCK_DAYS + 1, BLOCK_DAYS)
                    if isinstance(lab[s + BLOCK_DAYS - 1], str)])

    P = np.zeros((4, 4))
    for a, b in zip(seq[:-1], seq[1:]):
        P[a, b] += 1
    for r in range(4):           # 관측 안 된 국면 행은 균일로(폴백)
        if P[r].sum() == 0:
            P[r] = 1.0
    P = P / P.sum(axis=1, keepdims=True)
    pi = np.array([np.mean(seq == i) for i in range(4)]) if len(seq) else np.ones(4) / 4
    pi = pi / pi.sum()
    return pools, P, pi


def make_world_rs(seed: int = 42, external_streams=DEFAULT_EXTERNAL_STREAMS,
                  start: str = DATA_START, end: str = DATA_END,
                  ticker: str = "QQQ", n_days: int | None = None,
                  skew: dict | None = None) -> LoadedGym:
    """새 교과서 1권 — RS가 국면 순서를 정하고, 각 칸에 그 국면의 진짜 토막을 끼운다.

    옛 make_world와 같은 LoadedGym(합성 QQQ + external_streams attrs)을 반환 →
    학습 파이프라인에 그대로 주입 가능. 차이는 '블록 선택을 RS 국면열로 통제'한 것뿐.

    skew: {국면: 배수} — 그 국면의 출제 확률을 비틈(적대적 시드용). 예 {"bear":3}=
      하락을 3배 더 출제. None이면 실제 비율 그대로(균일 블록부트와 사실상 동일).
    """
    if n_days is None:
        n_days = WARMUP_TDAYS + tb.EVAL_TDAYS

    qqq_raw = get_prices(ticker, start, end)
    qqq_returns = (qqq_raw / qqq_raw.shift(1)).apply(np.log).dropna()
    common_idx = qqq_returns.index
    aligned = _load_external(external_streams, start, end, common_idx)
    pools, P, pi = _block_pools_and_chain(qqq_raw, common_idx)
    if skew:                              # 적대적: 특정 국면 출제 확률을 비틀고 재정규화
        m = np.array([skew.get(s, 1.0) for s in STATES])
        pi = pi * m
        pi = pi / pi.sum()
        P = P * m[None, :]
        P = P / P.sum(axis=1, keepdims=True)

    rng = np.random.default_rng(seed)
    n_blocks = (n_days + BLOCK_DAYS - 1) // BLOCK_DAYS
    s = int(rng.choice(4, p=pi))
    indices: list[int] = []
    for _ in range(n_blocks):
        pool = pools[STATES[s]]
        i0 = (int(pool[rng.integers(0, len(pool))]) if pool
              else int(rng.integers(0, len(common_idx) - BLOCK_DAYS + 1)))
        indices.extend(range(i0, i0 + BLOCK_DAYS))
        s = int(rng.choice(4, p=P[s]))
    sample_indices = np.array(indices[:n_days])

    qqq_synth = _from_block_unit(np.asarray(qqq_returns.values)[sample_indices],
                                 "price", init=100.0)
    dates = pd.bdate_range("2001-01-01", periods=n_days)
    prices = pd.Series(qqq_synth, index=dates, name=ticker)

    external_synth = {ticker: prices.copy()}
    for ext_ticker, (block_unit, stream_type) in aligned.items():
        values = np.asarray(block_unit.values)[sample_indices]
        external_synth[ext_ticker] = pd.Series(_from_block_unit(values, stream_type),
                                               index=dates)
    prices.attrs["synthetic"] = True
    prices.attrs["external_streams"] = external_synth

    eval_start = prices.index[WARMUP_TDAYS] if n_days > WARMUP_TDAYS else prices.index[0]
    gym = Gym("세계공장#RS합성", difficulty=0, volatility=0, ticker="SYNTH",
              start=eval_start.strftime("%Y-%m-%d"),
              end=prices.index[-1].strftime("%Y-%m-%d"))
    return LoadedGym(gym=gym, prices=prices)


# 레벨/금리 지수 — 시즌3에서 경계 절벽(가짜 이벤트)이 문제됐던 스트림들
EXT_CHECK = [("^VIX", "공포지수 VIX (level/raw)"),
             ("^VXN", "공포지수 VXN (level/raw, 2001~)"),
             ("^TNX", "10년 금리 ^TNX (yield/offset)")]


def _boundary_jump(s: pd.Series) -> tuple[float, float]:
    """21일 블록 경계의 일변화 vs 블록 내부 일변화 (절벽이면 경계가 훨씬 큼)."""
    v = s.to_numpy(dtype=float)
    d = np.abs(np.diff(v))
    bidx = np.array([b - 1 for b in range(BLOCK_DAYS, len(v), BLOCK_DAYS)], dtype=int)
    is_b = np.zeros(len(d), dtype=bool)
    is_b[bidx] = True
    return float(np.nanmean(d[is_b])), float(np.nanmean(d[~is_b]))


def _dist(counter) -> dict:
    tot = sum(counter.values()) or 1
    return {k: 100.0 * counter.get(k, 0) / tot for k in STATES}


def _regime_dist(price_list) -> dict:
    c: collections.Counter = collections.Counter()
    for p in price_list:
        lab = classify_daily(p)
        c.update(lab[lab.index >= p.index[WARMUP_TDAYS]].values)
    return _dist(c)


def _pooled_logrets(price_list) -> np.ndarray:
    return np.concatenate([np.log(p / p.shift(1)).dropna().to_numpy()
                           for p in price_list])


def run():
    qqq_raw = get_prices("QQQ", DATA_START, DATA_END)
    real_lab = classify_daily(qqq_raw)
    real_d = _dist(collections.Counter(real_lab.values))
    real_rets = np.log(qqq_raw / qqq_raw.shift(1)).dropna().to_numpy()

    old_lg = [tb.make_world(seed=s) for s in range(N_WORLDS)]
    new_lg = [make_world_rs(seed=s) for s in range(N_WORLDS)]
    old = [lg.prices for lg in old_lg]
    new = [lg.prices for lg in new_lg]
    old_d, new_d = _regime_dist(old), _regime_dist(new)
    old_rets, new_rets = _pooled_logrets(old), _pooled_logrets(new)

    def dist_to_real(d):
        return sum(abs(d[k] - real_d[k]) for k in STATES)

    def maxmove(r):
        return 100.0 * np.max(np.abs(r))

    print("분위기 비율 (%)         " + "  ".join(REGIME_LABELS[k] for k in STATES) + "  | 실제거리")
    for name, d in [("실제", real_d), ("옛 교과서", old_d), ("새 교과서", new_d)]:
        print(f"  {name:10s} " + "  ".join(f"{d[k]:4.1f}" for k in STATES)
              + f"  | {dist_to_real(d):4.1f}")
    print(f"\n최대 일변동(%): 실제 {maxmove(real_rets):.1f} / "
          f"옛 {maxmove(old_rets):.1f} / 새 {maxmove(new_rets):.1f}")

    # 레벨/금리 지수 경계 절벽 점검 (경계 일변화 vs 내부 일변화 vs 실제)
    bound = {}
    print("\n레벨/금리 경계 절벽 (경계 일변화 / 내부 일변화):")
    for tk, label in EXT_CHECK:
        real_s = get_prices(tk, DATA_START, DATA_END)
        real_typ = float(np.nanmean(np.abs(np.diff(real_s.to_numpy(dtype=float)))))
        ob, ow = _bstats(old_lg, tk)
        nb, nw = _bstats(new_lg, tk)
        bound[tk] = (label, real_typ, ob, ow, nb, nw)
        print(f"  {tk:6s} 실제내부 {real_typ:.3f} | 옛 {ob:.3f}/{ow:.3f} | 새 {nb:.3f}/{nw:.3f}")

    os.makedirs(OUT_DIR, exist_ok=True)
    _fig(old[:3], new[:3], real_d, old_d, new_d, real_rets, old_rets, new_rets, maxmove)
    _fig_external(old_lg[0], new_lg[0])
    _write_md(real_d, old_d, new_d, dist_to_real, maxmove,
              real_rets, old_rets, new_rets, bound)
    print(f"[저장] {OUT_DIR}")


def _bstats(lg_list, ticker: str) -> tuple[float, float]:
    bs, ws = [], []
    for lg in lg_list:
        b, w = _boundary_jump(lg.prices.attrs["external_streams"][ticker])
        bs.append(b)
        ws.append(w)
    return float(np.nanmean(bs)), float(np.nanmean(ws))


def _fig_external(old_lg0, new_lg0):
    fig, ax = plt.subplots(len(EXT_CHECK), 2, figsize=(13.5, 9), sharex=True)
    for row, (tk, label) in enumerate(EXT_CHECK):
        for col, (lg, name) in enumerate([(old_lg0, "옛"), (new_lg0, "새")]):
            s = lg.prices.attrs["external_streams"][tk].to_numpy(dtype=float)
            a = ax[row, col]
            a.plot(np.arange(len(s)), s, lw=0.9, color="#37474f")
            for b in range(BLOCK_DAYS, len(s), BLOCK_DAYS):
                a.axvline(b - 0.5, color="#f9a825", alpha=0.13, lw=0.6)
            if row == 0:
                a.set_title(f"{name} 교과서", fontsize=11)
            if col == 0:
                a.set_ylabel(label, fontsize=9)
    ax[-1, 0].set_xlabel("거래일")
    ax[-1, 1].set_xlabel("거래일")
    fig.suptitle("레벨/금리 지수 — 토막 경계(주황선)에서 절벽 생기나", fontsize=13, y=0.997)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "rs_hybrid_external.png"), dpi=130, bbox_inches="tight")
    plt.close(fig)


def _fig(old3, new3, real_d, old_d, new_d, real_rets, old_rets, new_rets, maxmove):
    fig, ax = plt.subplots(2, 2, figsize=(13.5, 9))

    for p in old3:
        ax[0, 0].plot(np.arange(len(p)), p.to_numpy(), lw=1.0, alpha=0.85)
    ax[0, 0].set_title("① 옛 교과서 — 가짜 시장 3권 (무작위 토막)", fontsize=11)
    for p in new3:
        ax[0, 1].plot(np.arange(len(p)), p.to_numpy(), lw=1.0, alpha=0.85)
    ax[0, 1].set_title("① 새 교과서 — 가짜 시장 3권 (국면 골라 붙임)", fontsize=11)
    ylim = (min(ax[0, 0].get_ylim()[0], ax[0, 1].get_ylim()[0]),
            max(ax[0, 0].get_ylim()[1], ax[0, 1].get_ylim()[1]))
    for a in (ax[0, 0], ax[0, 1]):
        a.set_ylim(ylim)
        a.set_xlabel("거래일")
        a.set_ylabel("합성 지수(100 시작)")

    xs = np.arange(len(STATES))
    groups = [("실제", real_d, 1.0, None), ("옛 교과서", old_d, 0.55, "//"),
              ("새 교과서", new_d, 0.55, "xx")]
    w = 0.8 / len(groups)
    for i, (g, d, alpha, hatch) in enumerate(groups):
        off = (i - (len(groups) - 1) / 2) * w
        ax[1, 0].bar(xs + off, [d[k] for k in STATES], w,
                     color=[LABEL_COLOR[k] for k in STATES], alpha=alpha,
                     edgecolor="black", linewidth=0.4, hatch=hatch)
        ax[1, 0].bar([-9], [0], w, label=g, color="gray", alpha=alpha, hatch=hatch)
    ax[1, 0].set_xlim(-0.6, len(STATES) - 0.4)
    ax[1, 0].set_xticks(xs)
    ax[1, 0].set_xticklabels([REGIME_LABELS[k] for k in STATES])
    ax[1, 0].set_ylabel("비율 (%)")
    ax[1, 0].set_title("② 분위기 비율 — 실제 vs 옛 vs 새", fontsize=11)
    ax[1, 0].legend(fontsize=8)

    bins = np.linspace(-0.12, 0.12, 80)
    for r, lbl, c in [(real_rets, "실제", "#37474f"),
                      (old_rets, "옛 교과서", "#1565c0"),
                      (new_rets, "새 교과서", "#c62828")]:
        ax[1, 1].hist(r, bins=bins, histtype="step", density=True, lw=1.4, label=lbl, color=c)
    ax[1, 1].set_yscale("log")
    ax[1, 1].set_title("③ 일별 수익률 꼬리 (급락·급등) — 겹치면 안전", fontsize=11)
    ax[1, 1].set_xlabel("일별 로그수익률")
    ax[1, 1].set_ylabel("밀도(로그)")
    ax[1, 1].legend(fontsize=8)
    ax[1, 1].text(0.02, 0.97,
                  f"최대 일변동\n실제 {maxmove(real_rets):.1f}% / 옛 {maxmove(old_rets):.1f}% / 새 {maxmove(new_rets):.1f}%",
                  transform=ax[1, 1].transAxes, va="top", fontsize=8.5,
                  bbox=dict(boxstyle="round", fc="white", ec="gray", alpha=0.85))

    fig.suptitle("교과서 생성 비교 — 옛(무작위) vs 새(국면조건부 블록)", fontsize=13, y=0.995)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "rs_hybrid_textbook.png"), dpi=130, bbox_inches="tight")
    plt.close(fig)


def _write_md(real_d, old_d, new_d, dist_to_real, maxmove,
              real_rets, old_rets, new_rets, bound):
    L = [
        "# 교과서 생성 비교 — 옛(무작위 블록) vs 새(국면조건부 블록)\n",
        f"> 실제 QQQ({DATA_START}~{DATA_END}) 재료. 옛=textbook.make_world, "
        f"새=make_world_rs. 각 {N_WORLDS}권 생성 후 재분류·꼬리 비교.\n",
        "\n![생성 비교](rs_hybrid_textbook.png)\n",
        "\n## 분위기 비율 (재분류, %)\n\n",
        "| 출처 | " + " | ".join(REGIME_LABELS[k] for k in STATES) + " | 실제거리 |\n",
        "|---|" + "--:|" * (len(STATES) + 1) + "\n",
    ]
    for name, d in [("**실제**", real_d), ("옛 교과서", old_d), ("새 교과서", new_d)]:
        L.append(f"| {name} | " + " | ".join(f"{d[k]:.1f}" for k in STATES)
                 + f" | {dist_to_real(d):.1f} |\n")
    L.append(
        f"\n## 급락·급등 안전 점검 (최대 일변동)\n\n"
        f"- 실제 {maxmove(real_rets):.1f}% / 옛 {maxmove(old_rets):.1f}% / "
        f"새 {maxmove(new_rets):.1f}%\n"
        "- 옛·새 둘 다 **진짜 토막의 진짜 수익률**을 쓰므로 일별 꼬리가 실제와 일치 "
        "(그래프 ③ 겹침) = 인위적 급락·급등 없음. 어제 RS 정규분포 방식(극단 μ로 하락 "
        "과대생성)과 결정적으로 다른 점.\n"
        "\n## 읽는 법\n"
        "- ① 곡선: 옛·새 모두 그럴듯한 지수 경로, 튀는 점프 없음(이음매 연속).\n"
        "- ② 비율: 새 교과서가 RS로 분위기 순서를 통제. (P·π를 실제서 추정한 균일 비교라 "
        "차이가 크지 않을 수 있음 — 진짜 레버는 약점 국면 가중=적대적 시드, 다음 단계.)\n"
        "- ③ 꼬리: 셋이 겹치면 합성이 실제 변동성을 왜곡 안 한다는 뜻.\n"
    )
    L.append(
        "\n## 레벨/금리 지수 경계 절벽 점검 (시즌3 문제 재발 여부)\n\n"
        "![외부 스트림](rs_hybrid_external.png)\n\n"
        "단위 = 하루 평균 변화 절대값. '경계'가 '내부'·'실제'보다 훨씬 크면 절벽(가짜 이벤트).\n\n"
        "| 지수 | 실제 내부 | 옛 경계 | 옛 내부 | 새 경계 | 새 내부 |\n"
        "|---|--:|--:|--:|--:|--:|\n"
    )
    for tk, (label, real_typ, ob, ow, nb, nw) in bound.items():
        L.append(f"| {tk} | {real_typ:.3f} | {ob:.3f} | {ow:.3f} | {nb:.3f} | {nw:.3f} |\n")
    L.append(
        "\n- **^TNX(offset)**: 경계≈내부≈실제면 절벽 없음 = 시즌3 수리 유지(새 방식도 안전).\n"
        "- **VIX/VXN(raw)**: 경계가 내부보다 커도 무해 — FEAR는 절대 임계(>30/>47)만 보지 "
        "변화를 안 본다(시즌3 결정). 새 방식이 옛 방식과 같은 수준이면 OK.\n"
        "\n재현: `.venv/Scripts/python.exe -m app.lab.rs_hybrid_textbook`\n"
    )
    with open(os.path.join(OUT_DIR, "report.md"), "w", encoding="utf-8") as f:
        f.write("".join(L))


if __name__ == "__main__":
    run()
