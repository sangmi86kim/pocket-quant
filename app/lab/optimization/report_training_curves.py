"""학습 과정 검증 레포트 — 각 학습 단계가 제대로 수렴했는지 그래프로 확인한다.

[무엇] ① NSGA 1차·2차의 목적별 best-so-far 수렴곡선(+ 탐색 구름) ② NSGA 최종 파레토
프론트 ③ 교실별 선발 후보(1차 vs 보충)의 학습 종료잔고 분포.

[정직한 한계] 시즌3부터 단일목적 교실도 sqlite storage를 남긴다. GP는 seed리그라 표본 수가 작다.

실행: .venv/Scripts/python.exe -m app.lab.optimization.report_training_curves
산출: app/academy/reports/training_curves_{stamp}_*.png + training_verification_{stamp}.md
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import optuna

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False
optuna.logging.set_verbosity(optuna.logging.WARNING)

ROOT = Path(__file__).resolve().parents[3]
RESULTS = ROOT / "app" / "academy" / "training" / "results"
REPORTS = ROOT / "app" / "academy" / "reports"

OBJS = [("median_balance", "중앙 종료잔고 (↑좋음)", "max"),
        ("worst_balance", "최악 종료잔고 (↑좋음)", "max")]


def _load(db: Path, name: str):
    return optuna.load_study(study_name=name, storage=f"sqlite:///{db.as_posix()}")


def _series(study):
    """COMPLETE trial을 번호순으로 → (번호, [median, worst], turnover) 목록."""
    rows = [(t.number, list(t.values)) for t in study.trials
            if t.values is not None and len(t.values) >= 2]
    rows.sort(key=lambda r: r[0])
    return rows


def _turnovers(study):
    rows = []
    for t in study.trials:
        if t.values is None:
            continue
        metrics = t.user_attrs.get("academy_metrics")
        if isinstance(metrics, dict) and "turnover" in metrics:
            rows.append((t.number, float(metrics["turnover"])))
        elif len(t.values) >= 3:
            rows.append((t.number, float(t.values[2])))
    rows.sort(key=lambda r: r[0])
    return rows


def _best_so_far(vals, mode):
    out, cur = [], None
    for v in vals:
        cur = v if cur is None else (max(cur, v) if mode == "max" else min(cur, v))
        out.append(cur)
    return out


def _nsga_dbs(stamp):
    """이 stamp의 NSGA phase1/phase2 DB만 정확히 고른다(옛 stamp·redo 혼입 금지)."""
    def ref(db):
        if not db.exists():
            return None
        names = optuna.study.get_all_study_names(f"sqlite:///{db.as_posix()}")
        return (db, names[0]) if names else None
    phase1 = ref(RESULTS / f"classroom_nsga3_{stamp}_phase1.db")
    phase2 = ref(RESULTS / f"classroom_nsga3_{stamp}_phase2.db")
    # 옛 복구분(stamp별 phase2가 없을 때만) redo DB 폴백 — 정상 run엔 안 탐.
    if phase2 is None:
        phase2 = ref(RESULTS / "classroom_nsga3_redo_phase2.db")
    return phase1, phase2


def fig_convergence(stamp, phase1, phase2, academy_seed):
    """2행(1차/2차) × 2열(목적) best-so-far 수렴곡선 + 탐색 구름."""
    stages = [("1차 (RS 교과서)", phase1), ("2차 (약점 보충)", phase2)]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for r, (label, ref) in enumerate(stages):
        rows = _series(_load(*ref))
        xs = [n for n, _ in rows]
        for c, (key, title, mode) in enumerate(OBJS):
            ax = axes[r][c]
            raw = [v[c] for _, v in rows]
            ax.scatter(xs, raw, s=4, alpha=0.12, color="#888", label="시도")
            ax.plot(xs, _best_so_far(raw, mode), color="#d1495b", lw=2, label="best-so-far")
            ax.set_title(f"{label} · {title}", fontsize=10)
            ax.set_xlabel("trial 번호")
            if c == 0:
                ax.set_ylabel("값")
            ax.grid(alpha=0.25)
            if r == 0 and c == 0:
                ax.legend(fontsize=8, loc="lower right")
    fig.suptitle(f"NSGA-III 학습 수렴 (academy_seed={academy_seed}, stamp={stamp})\n"
                 f"빨강=지금까지 최고치, 회색=각 시도 — 우상향/평탄화면 수렴 성공", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = REPORTS / f"training_curves_{stamp}_nsga_convergence.png"
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return out


def fig_nsga_hv(stamp, phase1, phase2):
    """NSGA 하이퍼볼륨(HV) 학습 트렌드 — 다목적의 진짜 수렴곡선. DB의 hv_trend에서.

    목적별 best-so-far는 극단 1점만 본다. HV는 파레토 프론트 전체의 부피라, 프론트가
    촘촘·넓어지는 것까지 잡는 다목적 수렴의 정식 잣대다(조기종료 판정도 이걸로 한다).
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, (label, ref) in zip(axes, [("1차 (RS)", phase1), ("2차 보충", phase2)]):
        if ref is None:
            ax.axis("off")
            continue
        tr = _load(*ref).user_attrs.get("hv_trend")
        if not tr:
            ax.set_title(f"{label} — hv_trend 없음")
            ax.axis("off")
            continue
        xs = [r[0] for r in tr]
        ax.plot(xs, [r[1] for r in tr], color="#bbb", lw=1, alpha=0.7, label="raw HV")
        ax.plot(xs, [r[2] for r in tr], color="#d1495b", lw=2, label="HV 이동평균(5세대)")
        ax.set_title(f"{label} 하이퍼볼륨 ({len(tr)}세대)")
        ax.set_xlabel("trial 번호")
        ax.set_ylabel("정규화 HV (↑좋음)")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8, loc="lower right")
    fig.suptitle(f"NSGA-III 하이퍼볼륨(HV) 학습 트렌드 (stamp={stamp})\n"
                 f"다목적 수렴의 정식 잣대 — 프론트 전체 부피가 커질수록 HV↑ (조기종료 기준)",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    out = REPORTS / f"training_curves_{stamp}_nsga_hv.png"
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return out


def fig_front(stamp, phase1, phase2):
    """최종 파레토 프론트: 중앙 vs 최악 잔고, 색=회전율 진단값."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    for ax, (label, ref) in zip(axes, [("1차", phase1), ("2차 보충", phase2)]):
        study = _load(*ref)
        rows = _series(study)
        turns = dict(_turnovers(study))
        med = [v[0] / 1e6 for _, v in rows]
        wor = [v[1] / 1e6 for _, v in rows]
        tn = [turns.get(n, 0.0) for n, _ in rows]
        sc = ax.scatter(med, wor, c=tn, cmap="viridis", s=10, alpha=0.6)
        ax.set_title(f"{label} 프론트 (n={len(rows)})")
        ax.set_xlabel("중앙 종료잔고 (백만)")
        ax.set_ylabel("최악 종료잔고 (백만)")
        ax.grid(alpha=0.25)
        fig.colorbar(sc, ax=ax, label="회전율(↓좋음)")
    fig.suptitle(f"NSGA-III 최종 파레토 프론트 (stamp={stamp}) — 우상단일수록 강함", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out = REPORTS / f"training_curves_{stamp}_nsga_front.png"
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return out


def _single_series(db: Path, study_name: str):
    """단일목적 study → (trial번호, value) 번호순. COMPLETE만."""
    s = _load(db, study_name)
    rows = [(t.number, t.value) for t in s.trials if t.value is not None]
    rows.sort(key=lambda r: r[0])
    return rows


def fig_single_convergence(stamp):
    """단일목적·GP 수렴곡선 (storage 보존 후 생기는 DB에서). 없으면 None."""
    singles = [("TPE", "tpe"), ("CMA-ES", "cma_es")]
    present = []
    for nm, slug in singles:
        p1 = RESULTS / f"classroom_{slug}_{stamp}_phase1.db"
        if p1.exists():
            present.append((nm, slug, "single"))
    gp1 = RESULTS / f"classroom_gp_{stamp}_phase1.db"
    if gp1.exists():
        present.append(("GP", "gp", "seedleague"))
    if not present:
        return None

    fig, axes = plt.subplots(len(present), 2, figsize=(12, 3.2 * len(present)),
                             squeeze=False)
    for r, (nm, slug, kind) in enumerate(present):
        for c, phase in enumerate(["phase1", "phase2"]):
            ax = axes[r][c]
            db = RESULTS / f"classroom_{slug}_{stamp}_{phase}.db"
            label = "1차" if phase == "phase1" else "2차 보충"
            if not db.exists():
                ax.axis("off")
                continue
            st = f"sqlite:///{db.as_posix()}"
            names = optuna.study.get_all_study_names(st)
            if kind == "seedleague":
                for i, snm in enumerate(sorted(names)):
                    rows = _single_series(db, snm)
                    xs = [n for n, _ in rows]
                    ax.plot(xs, _best_so_far([v for _, v in rows], "max"),
                            lw=1.5, alpha=0.8, label=f"seed{i}")
                ax.legend(fontsize=7, ncol=2)
            else:
                rows = _single_series(db, names[0])
                xs = [n for n, _ in rows]
                ax.scatter(xs, [v for _, v in rows], s=4, alpha=0.12, color="#888")
                ax.plot(xs, _best_so_far([v for _, v in rows], "max"),
                        color="#d1495b", lw=2)
            ax.set_title(f"{nm} · {label}", fontsize=10)
            ax.set_xlabel("trial 번호")
            if c == 0:
                ax.set_ylabel("종료잔고 (↑좋음)")
            ax.grid(alpha=0.25)
    fig.suptitle(f"단일목적·GP 학습 수렴 (stamp={stamp}) — best-so-far 우상향/평탄=수렴\n"
                 f"GP는 seed별 대표(seed리그)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = REPORTS / f"training_curves_{stamp}_single_convergence.png"
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return out


def _median_val(item):
    if "values" in item:
        return item["values"][0]
    return item.get("value")


def fig_selection(stamp, classrooms):
    """교실별 선발 후보(1차 vs 보충)의 학습 종료잔고 분포."""
    labels, data, colors = [], [], []
    cmap = {"TPE": "#4e79a7", "CMA-ES": "#59a14f", "NSGA-III": "#e15759", "GP": "#b07aa1"}
    for c in classrooms:
        nm = c["name"]
        for phase, key in [("1차", c.get("phase1", {}).get("topk")), ("보충", c.get("topk"))]:
            if not key:
                continue
            vals = [_median_val(it) / 1e6 for it in key if _median_val(it) is not None]
            if vals:
                labels.append(f"{nm}\n{phase}")
                data.append(vals)
                colors.append(cmap.get(nm, "#888"))
    fig, ax = plt.subplots(figsize=(13, 5.5))
    bp = ax.boxplot(data, patch_artist=True, showmeans=True, widths=0.6)
    for patch, col in zip(bp["boxes"], colors):
        patch.set_facecolor(col)
        patch.set_alpha(0.55)
    for i, vals in enumerate(data, 1):
        ax.scatter([i] * len(vals), vals, s=14, color="#333", alpha=0.5, zorder=3)
    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("선발 후보 중앙 종료잔고 (백만)")
    ax.set_title(f"교실별 선발 후보 분포 (stamp={stamp}) — 학습 합성장 내부 점수\n"
                 f"※ 시즌3은 3교실(CMA-ES·GP·NSGA-III), GP는 seed리그라 n=5",
                 fontsize=11)
    ax.grid(alpha=0.25, axis="y")
    fig.tight_layout()
    out = REPORTS / f"training_curves_{stamp}_selection.png"
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return out


def main():
    REPORTS.mkdir(parents=True, exist_ok=True)
    top30 = sorted(RESULTS.glob("classroom_top30_*_v2.json"))[-1]
    data = json.loads(top30.read_text(encoding="utf-8"))
    stamp = data["stamp"]
    academy_seed = data.get("academy_seed")
    classrooms = data["classrooms"]
    phase1, phase2 = _nsga_dbs(stamp)

    conv = fig_convergence(stamp, phase1, phase2, academy_seed)
    hv = fig_nsga_hv(stamp, phase1, phase2)
    front = fig_front(stamp, phase1, phase2)
    sel = fig_selection(stamp, classrooms)
    single = fig_single_convergence(stamp)

    n1 = len(_series(_load(*phase1)))
    n2 = len(_series(_load(*phase2)))
    classroom_names = [c["name"] for c in classrooms]
    single_names = [name for name in classroom_names if name != "NSGA-III"]
    single_label = "·".join(single_names) if single_names else "단일목적"
    single_md = (f"""![단일목적 수렴](training_curves_{stamp}_single_convergence.png)

{single_label}도 storage 보존 후 학습이라 수렴곡선 확인 가능."""
                 if single else
                 "_이 학기는 storage 보존 전 학습이라 단일목적 수렴곡선 없음 "
                 "(다음 학습부터 생성)._")
    limit_md = (f"- **{len(classroom_names)}교실 전부 학습 이력 DB 보존** — "
                f"NSGA(목적별 best-so-far + HV) · {single_label}(best-so-far) "
                "모두 수렴곡선 검증 가능. GP는 seed리그라 n=5(표본 작음)."
                if single else
                f"- **학습 이력 보존 = NSGA뿐.** {single_label}는 storage 없이 메모리 study라 trial별\n"
                "  수렴곡선이 **유실**됨 → 최종 선발 후보 분포만 표시. GP는 seed리그라 n=5(표본 작음).\n"
                "- 재발방지: AGENTS #20 '단일목적도 storage 붙여 학습 이력 보존' — 다음 학습부터 해소.")
    md = REPORTS / f"training_verification_{stamp}.md"
    md.write_text(f"""# 학습 과정 검증 레포트 — {stamp}

> 생성: `app/lab/optimization/report_training_curves.py` · academy_seed={academy_seed}

## 1. NSGA-III 수렴 (학습 이력 보존됨 — DB)

![수렴](training_curves_{stamp}_nsga_convergence.png)

- 1차 {n1} trial · 2차(보충) {n2} trial. 빨강 best-so-far가 우상향 후 평탄 = 수렴 성공.
- 목적 2개: 중앙 종료잔고(↑)·최악 종료잔고(↑). 회전율은 turnover cap 스펙으로 별도 필터.

### 하이퍼볼륨(HV) 트렌드 — 다목적 수렴의 정식 잣대

![HV](training_curves_{stamp}_nsga_hv.png)

- best-so-far가 극단 1점만 본다면, HV는 **파레토 프론트 전체 부피**라 프론트가 촘촘·넓어지는
  것까지 잡는다. 빨강(이동평균)이 우상향 후 평탄 = 진짜 수렴. 조기종료도 이 HV로 판정한다.

![파레토](training_curves_{stamp}_nsga_front.png)

## 2. 단일목적·GP 수렴 ({single_label})

{single_md}

## 3. {len(classroom_names)}교실 선발 분포

![선발](training_curves_{stamp}_selection.png)

## 4. 정직한 한계

{limit_md}
""", encoding="utf-8")

    print(f"[저장] {conv.name}\n[저장] {hv.name}\n[저장] {front.name}\n"
          f"[저장] {sel.name}\n[저장] {md.name}")
    if single:
        print(f"[저장] {single.name}")
    else:
        print("[정보] 단일목적 DB 없음(storage 보존 전 학기) - 단일목적 곡선 생략")
    print(f"NSGA phase1={n1} phase2={n2} · 교실={[c['name'] for c in classrooms]}")


if __name__ == "__main__":
    main()
