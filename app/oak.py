"""
oak.py - 오박사 (LLM 해설 NPC)

백테스트가 끝난 성적표를 받아 "왜 이런 스탯이 나왔는지"를 해설한다.

[철칙 — 오박사는 판정하지 않는다]
승패·합불·점수는 전부 백테스트/심판(tests/)이 정한다. 오박사는 이미 나온
결과를 읽고 풀이만 하는 리포팅 전용 NPC다. LLM을 적합도 루프에 넣지 않는다.
연결 실패 시에도 본 게임 출력은 멀쩡해야 하므로 조용히 부재 처리한다.

[연결] LM Studio 로컬 서버 (OpenAI 호환 API). 모델/주소는 아래 상수에서 변경.
사용: config.json에 "oak": true → 단판/진화 리포트 끝에 브리핑이 붙는다.
"""
import json
import urllib.request

# ── 연구소 주소 (LM Studio 기본값. 모델 이름은 LM Studio에 로드된 그대로) ──
OAK_URL = "http://localhost:1234/v1/chat/completions"
OAK_MODEL = "gemma-4-e4b-it"
TIMEOUT_SEC = 120
MAX_TOKENS = 500

# 오박사 페르소나 + 해설 규칙. 사용자는 퀀트 입문자 — 쉬운 말, 용어는 괄호 병기.
SYSTEM_PROMPT = """너는 PocketQuant 연구소의 '오박사'다. 포켓몬 박사처럼 친근하고 간결한 한국어로 말한다.

세계관: 포켓몬 1마리 = 시그널(매수/매도 규칙), 트레이더 = 시그널들을 데리고 다니는 전략,
체육관 = 과거 시장 국면(QQQ 실데이터), 성실이 = 매일 같은 금액을 사 모으는 라이벌 DCA 봇.
스탯: 공격력=연수익(CAGR), 최대 데미지=최대낙폭(MDD), 컨트롤=샤프(출렁임 대비 수익),
HP=현금 비중(표시 전용).

규칙:
1. 절대 판정하지 마라 — 합격/불합격/추천/매수/매도 같은 말 금지. 이미 나온 숫자의 '이유'만 풀이한다.
2. 5~7문장. 금융 용어를 쓰면 괄호로 쉬운 말을 붙인다.
3. 구성: ① 어느 체육관에서 강했고 어디서 약했나 ② 왜 그런가(데려간 포켓몬들의 성격으로 귀인)
   ③ 마지막 한 문장은 트레이너에게 건네는 따뜻한 한마디.
4. 숫자를 지어내지 마라 — 성적표에 있는 숫자만 인용한다."""


def _report_context(report) -> str:
    """Report 객체를 오박사가 읽을 성적표 텍스트로 변환한다."""
    s = report.strategy
    lines = [f"트레이더: {s.name} (데려간 포켓몬: {', '.join(s.genes)})", "", "체육관별 성적:"]
    for r in report.results:
        lines.append(
            f"- {r.gym_name}: 수익 {r.total_return:+.1%} (그냥 들고 있기 {r.market_return:+.1%})"
            f" · 최대 데미지 {r.max_drawdown:.1%} (시장 {r.market_drawdown:.1%})"
            f" · 종족치 {r.stats.bst:.0f}/400")
    st = report.stats
    weak_name, weak_fit = report.weakest_gym
    lines += [
        "",
        f"종합 스탯: HP(현금 비중) {st.hp:.0f} · 공격력(연수익) {st.atk:.0f}"
        f" · 방어력(낙폭 대비 수익) {st.def_:.0f} · 컨트롤(샤프) {st.skill:.0f}  (각 0~100)",
        f"적합도: {report.fitness:.1f}/100 (평균 70% + 최약 체육관 30%) · 등급 {report.grade}",
        f"최약 체육관: {weak_name} ({weak_fit:.1f}점)",
    ]
    return "\n".join(lines)


def professor_briefing(report) -> str | None:
    """성적표를 오박사에게 보내 브리핑을 받는다. 연구소 부재(연결 실패) 시 None."""
    payload = {
        "model": OAK_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": _report_context(report)
                + "\n\n이 성적표를 브리핑해 주세요, 박사님."},
        ],
        "temperature": 0.7,
        "max_tokens": MAX_TOKENS,
    }
    req = urllib.request.Request(
        OAK_URL, data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT_SEC) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        return body["choices"][0]["message"]["content"].strip()
    except Exception:
        return None       # 오박사 부재 — 본 게임 출력은 영향받지 않는다
