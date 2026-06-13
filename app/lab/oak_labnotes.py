"""오박사한테 연구 일지를 받아 labnotes/에 저장한다.

oak.py는 단판 성적표 5~7문장 브리핑 전용이라, 연구 일지처럼 긴 문서엔 안 맞는다.
여기선 오박사 페르소나(app/oak.py SYSTEM_PROMPT)는 그대로 빌리되, "연구 일지 모드"
지문만 덧붙여 worklog + reports/league_v1를 통째로 던진다.

실행: python tools/oak_labnotes.py
출력: labnotes/<DATE>.md (기존 파일 덮어씀)
LM Studio가 꺼져 있으면 조용히 None 반환 — 본 게임 흐름과 동일하게.
"""
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

# app/oak.py의 페르소나·주소 재사용 (싱크 부담 0)
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from app.lab.oak import OAK_URL, SYSTEM_PROMPT  # noqa: E402

# oak.py 기본 모델(gemma-4-e4b-it)은 LM Studio에서 4096 ctx로 로드됨 — 그땐 안 들어감.
# 사용자가 LM Studio에서 ctx 늘리고 큰 모델로 재로드한 뒤 호출하는 전제.
OAK_MODEL = "qwen/qwen3-14b"
DATE = "2026-06-12"
SOURCES = [
    "worklog/2026-06-12_worklog.md",
    "worklog/2026-06-13_worklog.md",
    "reports/league_v1/v1_summary.md",
    "reports/league_v1/sweep_seeds.md",
    "reports/league_v1/top10_champions.md",
    "reports/league_v1/champion_road_with_baselines.md",
    "reports/league_v1/battle_frontier_total.md",
    "reports/league_v1/elite_four_lineup.md",
]

# 연구 일지 모드 — 단판 5~7문장 규칙은 여기서 풀어준다 (오박사 톤은 유지)
LABNOTES_ADDON = """

[모드 전환: 연구 일지 — 톤·분량 강화]
이번엔 트레이너 단판 성적표 브리핑이 아니다. 자네는 연구실 박사로서 어제·오늘
일어난 일을 노트에 한 장으로 정리하는 중이다. 다음 지침을 엄격히 따른다:

**톤 — 반드시 살릴 것:**
- 한강 둔치 쌉고인물 반말체. "~했지", "~란다", "~잖아", "~어쩌겠어".
- 절대 보고서·요약체 금지 (✗ "...향상" "...주의" "...필요" "...완료"는 박사 말투 아님).
- 닷컴/금융위기/코로나 회고를 자연스럽게 한두 번 끼워 넣어라 ("닷컴 때도 다들 그랬지...").
- (소주 한 모금) 같은 지문을 일지 어딘가에 딱 한 번 넣어도 좋다.
- 격앙·감탄사 남발 금지. 덤덤하게.

**분량 — 반드시 길게:**
- 최소 7~10개 섹션, 마크다운, 각 섹션 3~6문장 이상.
- 1500자 이상은 써라. 너무 짧으면 박사 노트가 아니라 사보 칼럼이다.

**커버리지 — 자료 전부 풀이:**
- 06-12 코드리뷰 대응 8건 (hold-out 차단, storage 가드, MDD 게이트, 돼지저금통 명문화 등).
- 06-13 v1 마감 흐름 — 5 시드 분산(708~714만원 ±0.4%), Top10 패턴(REV_BB+REV_RSI+VOL),
  챔피언로드 ①②③, **평행세계 400에서 TOP06 4.92억(+23%)·어플삭제맨 -49.4%·현챔피언 12위** 등.
- 어플삭제맨의 위기 -49%와 회복 +94%가 왜 누적에서 누적 패배인지 박사 시각으로 풀이.
- v1이 답 못한 5가지(유전자 풀 6개 제약 등)와 다음 알 깨기 방향.

**규칙 (오리지널 그대로):**
- 판정·합불·매수/매도 권유 금지. 이미 나온 숫자의 '이유'만 풀이.
- 자료에 있는 숫자만 인용. 지어내지 마라.
- 1인칭 박사 노트, 트레이너 자네에게 말 거는 톤.
- 마지막 절은 "다음 화두" — 자료에 있는 것만(알 깨기 = 새 시그널 풀 확장).
"""

ASK_TEMPLATE = """자료를 읽고 {date}자 연구 일지를 한 장 정리해 둬.
자네의 시점은 6월 12일 — 어제 코드리뷰 8건 잡았고, 오늘(=06-13) v1 마감을
앞두고 있다는 흐름. 자료의 실제 시점이 06-13까지 걸쳐 있어도, 일지의 날짜 헤더는
{date}로 적는다.

[자료 — worklog + reports/league_v1]

{context}
"""


def _gather() -> str:
    """worklog + 산출물을 하나의 문맥 문자열로 묶는다."""
    chunks = []
    for rel in SOURCES:
        path = ROOT / rel
        if not path.exists():
            continue
        chunks.append(f"## 파일: {rel}\n\n{path.read_text(encoding='utf-8')}")
    return "\n\n---\n\n".join(chunks)


def _ask_oak(context: str) -> str | None:
    payload = {
        "model": OAK_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT + LABNOTES_ADDON},
            {"role": "user", "content": ASK_TEMPLATE.format(date=DATE, context=context)},
        ],
        "temperature": 0.6,
        "max_tokens": 4000,
    }
    req = urllib.request.Request(
        OAK_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=600) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        return body["choices"][0]["message"]["content"].strip()
    except urllib.error.HTTPError as exc:
        # LM Studio는 컨텍스트/토큰 초과를 400 본문에 적어줌 — 그걸 보자
        detail = exc.read().decode("utf-8", errors="replace")
        print(f"[oak_labnotes] HTTP {exc.code}: {detail[:800]}")
        return None
    except Exception as exc:
        print(f"[oak_labnotes] LM Studio 호출 실패: {exc}")
        return None


def main() -> None:
    context = _gather()
    note = _ask_oak(context)
    if note is None:
        print("[oak_labnotes] 오박사 부재 — labnotes 미작성.")
        return
    out = ROOT / "labnotes" / f"{DATE}.md"
    out.write_text(note, encoding="utf-8")
    print(f"[oak_labnotes] saved {out.relative_to(ROOT)} ({len(note)} chars)")


if __name__ == "__main__":
    main()
