"""Select Top-K candidates from classroom study results."""
import json
from pathlib import Path

from app.academy.training.candidate import decode_params
from app.pocket.signals import SIGNAL_NAMES


ROOT = Path(__file__).resolve().parents[3]
TRAINING_RESULTS_DIR = ROOT / "app" / "academy" / "training" / "results"
K = 20


def weights_label(params: dict, threshold: float = 0.08) -> str:
    weights, _ = decode_params(params)
    total = sum(weights) or 1.0
    parts = sorted(
        [(g, w / total) for g, w in zip(SIGNAL_NAMES, weights)
         if w / total >= threshold],
        key=lambda x: x[1],
        reverse=True,
    )
    return ", ".join(f"{g} {p * 100:.0f}%" for g, p in parts) or "spread"


def latest_study_file() -> Path:
    files = sorted(TRAINING_RESULTS_DIR.glob("classroom_studies_*.json"))
    if not files:
        raise FileNotFoundError(
            f"no classroom study results in {TRAINING_RESULTS_DIR}")
    return files[-1]


def select_single(classroom: dict, k: int) -> list[dict]:
    rows = sorted(classroom["items"], key=lambda r: r["value"], reverse=True)
    out = []
    for rank, row in enumerate(rows[:k], start=1):
        out.append({
            "rank": rank,
            "trial": row["trial"],
            "balance_sum": row["value"],
            "weights": weights_label(row["params"]),
            "params": row["params"],
        })
    return out


def select_multi(classroom: dict, k: int) -> list[dict]:
    rows = classroom["items"]
    graduates = [r for r in rows if r.get("graduated")]
    pool = graduates or rows
    pool = sorted(pool, key=lambda r: r["academy"]["score"], reverse=True)
    out = []
    for rank, row in enumerate(pool[:k], start=1):
        out.append({
            "rank": rank,
            "trial": row["trial"],
            "score": row["academy"]["score"],
            "mean_balance": row["academy"]["mean_balance"],
            "worst_balance": row["academy"]["worst_balance"],
            "turnover": row["academy"]["turnover"],
            "graduated": row["graduated"],
            "weights": weights_label(row["params"]),
            "params": row["params"],
        })
    return out


def select_topk(source: Path | None = None, k: int = K) -> dict:
    source = source or latest_study_file()
    data = json.loads(source.read_text(encoding="utf-8"))
    selected = []
    for classroom in data["classrooms"]:
        if classroom["kind"] == "single":
            topk = select_single(classroom, k)
        else:
            topk = select_multi(classroom, k)
        row = dict(classroom)
        row.pop("items", None)
        row["topk"] = topk
        selected.append(row)
    return {
        "source": str(source),
        "stamp": data["stamp"],
        "k": k,
        "classrooms": selected,
    }


def write_markdown(selection: dict, path: Path) -> None:
    lines = [f"# Classroom Top{selection['k']}", ""]
    lines.append(f"- source: `{selection['source']}`")
    lines.append("")
    for classroom in selection["classrooms"]:
        lines.append(f"## {classroom['name']}")
        lines.append("")
        meta = [f"seed={classroom['seed']}", f"trials={classroom['trials']}"]
        if classroom["kind"] == "multi":
            meta.extend([
                f"academy_seed={classroom['academy_seed']}",
                f"front={classroom['front_size']}",
                f"passed={classroom['passed']}",
                f"hv_stopped={classroom['hv_stopped']}",
            ])
        else:
            meta.append(f"early_stop={classroom['early_stop']}")
        lines.append("- " + " / ".join(meta))
        lines.append("")
        lines.append("| rank | trial | score/balance | worst | turnover | graduated | weights |")
        lines.append("|---:|---:|---:|---:|---:|:---:|---|")
        for row in classroom["topk"]:
            if classroom["kind"] == "multi":
                score = f"{row['score']:.4f}"
                worst = f"{row['worst_balance']:.0f}"
                turnover = f"{row['turnover']:.4f}"
                graduated = "Y" if row["graduated"] else "N"
            else:
                score = f"{row['balance_sum']:.0f}"
                worst = ""
                turnover = ""
                graduated = ""
            lines.append(
                f"| {row['rank']} | {row['trial']} | {score} | {worst} | "
                f"{turnover} | {graduated} | {row['weights']} |"
            )
        lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_outputs(selection: dict) -> tuple[Path, Path]:
    TRAINING_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = selection["stamp"]
    k = selection["k"]
    json_path = TRAINING_RESULTS_DIR / f"classroom_top{k}_{stamp}.json"
    md_path = TRAINING_RESULTS_DIR / f"classroom_top{k}_{stamp}.md"
    json_path.write_text(json.dumps(selection, ensure_ascii=False, indent=2),
                         encoding="utf-8")
    write_markdown(selection, md_path)
    return json_path, md_path


def main() -> None:
    selection = select_topk(k=K)
    json_path, md_path = write_outputs(selection)
    print(f"json={json_path}", flush=True)
    print(f"md={md_path}", flush=True)


if __name__ == "__main__":
    main()
