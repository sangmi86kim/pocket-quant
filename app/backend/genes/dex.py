"""
dex.py - 포켓몬 도감 (각 유전자/시그널의 사람 친화적 설명 카드)

판정·계산엔 안 쓰는 '플레이버' 데이터다. 단, 카드 키는 반드시 실제 시그널
명단(signals.GENE_SIGNALS)과 일치해야 한다 (모듈 로드 시 assert로 검증).
"""
from .signals import GENE_SIGNALS

SIGNAL_CARDS = {
    "DD": {
        "name": "DD몬",
        "type": "탱커",
        "role": "손실 방어",
        "personality": "체력이 깎이면 바로 숨는 생존형",
        "effect": "최근 고점 대비 하락폭이 커지면 현금 비중을 높인다.",
        "strength": "폭락장 방어",
        "weakness": "강한 상승장에서 수익을 덜 먹을 수 있음",
    },
    "REV_RSI": {
        "name": "REV-RSI몬",
        "type": "역발상",
        "role": "과매도 투매 매수 (이벤트형)",
        "personality": "평소엔 벤치에서 졸다가 시장이 공포에 질리면 출전하는 저격수",
        "effect": "RSI가 과매도선(30) 아래로 투매되면 매수 의견을 낸다. 평소엔 기권.",
        "strength": "V자 급반등 (2020 코로나류)",
        "weakness": "갈아내리는 긴 하락장에선 일찍 잡을 수 있음",
    },
    "MA": {
        "name": "MA몬",
        "type": "정석",
        "role": "추세 확인",
        "personality": "평균선 위에서만 싸우는 FM형",
        "effect": "가격이 이동평균선 위면 공격, 아래면 방어한다.",
        "strength": "중장기 추세장",
        "weakness": "횡보장에서 잦은 헛신호",
    },
    "REV_BB": {
        "name": "REV-BB몬",
        "type": "역발상",
        "role": "과대 낙폭 매수 (이벤트형)",
        "personality": "통계적으로 과하게 빠진 날만 조용히 줍는 통계학자",
        "effect": "가격이 볼린저 하단밴드 아래로 떨어지면 매수 의견을 낸다. 평소엔 기권.",
        "strength": "급락 후 되돌림, 횡보장 바닥 줍기",
        "weakness": "변동성 폭발 구간에선 하단이 같이 내려가 신호가 늦을 수 있음",
    },
    "VOL": {
        "name": "VOL몬",
        "type": "정찰",
        "role": "위험도 감지",
        "personality": "시장 소음이 커지면 먼저 숨는 겁 많은 정찰병",
        "effect": "최근 변동성이 커지면 위험 신호로 판단한다.",
        "strength": "급락장 초입 감지",
        "weakness": "변동성 큰 상승장에서 기회를 놓칠 수 있음",
    },
    "MOM": {
        "name": "MOM몬",
        "type": "돌격",
        "role": "모멘텀 추종",
        "personality": "강한 놈 편에 붙는 추세 추종형",
        "effect": "최근 수익률이 좋으면 공격, 나쁘면 방어한다.",
        "strength": "강한 상승장",
        "weakness": "반전장에서 늦게 얻어맞을 수 있음",
    },
}

# 도감과 실제 시그널 명단이 어긋나면 즉시 알려준다(유전자 추가/삭제 시 깜빡 방지).
assert set(SIGNAL_CARDS) == set(GENE_SIGNALS), \
    f"도감/시그널 불일치: {set(SIGNAL_CARDS) ^ set(GENE_SIGNALS)}"
