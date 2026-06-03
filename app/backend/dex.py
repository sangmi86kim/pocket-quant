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
    "RSI": {
        "name": "RSI몬",
        "type": "심리",
        "role": "과매수/과매도 감지",
        "personality": "남들이 공포에 팔 때 슬쩍 줍는 역발상형",
        "effect": "RSI가 낮으면 매수, 높으면 방어한다.",
        "strength": "과매도 반등장",
        "weakness": "강한 추세장에서는 일찍 빠질 수 있음",
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
    "BB": {
        "name": "BB몬",
        "type": "방어",
        "role": "밴드/변동성 감지",
        "personality": "너무 튀면 경계하는 안정형",
        "effect": "가격이 밴드 하단/상단 근처에 있는지 보고 포지션을 조절한다.",
        "strength": "횡보장/반등장",
        "weakness": "추세가 강하면 역방향으로 맞을 수 있음",
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
