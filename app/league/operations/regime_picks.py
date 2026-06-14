import json
from pathlib import Path


REGIME_PICKS_PATH = Path("reports/regime_picks.json")


def update_regime_picks(section: str, data) -> Path:
    REGIME_PICKS_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {}
    if REGIME_PICKS_PATH.exists():
        payload = json.loads(REGIME_PICKS_PATH.read_text(encoding="utf-8"))
    payload[section] = data
    REGIME_PICKS_PATH.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return REGIME_PICKS_PATH
