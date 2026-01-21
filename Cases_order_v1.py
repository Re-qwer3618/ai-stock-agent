# Strategy_HogaKing.py
import pandas as pd
import os
import numpy as np

# [설정] 호가 데이터가 저장된 경로 (사장님 컴퓨터 경로에 맞게 수정됨)
BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ORDERBOOK_PATH = os.path.join(BASE_PATH, "Raw_Orderbook")

def prepare_data(df, config):
    """
    [전략 명] 호가킹 (Orderbook King)
    [논리] 총 매수 잔량이 총 매도 잔량보다 압도적으로 많으면 '지지선'이 강하다고 판단.
    """
    
    # 1. 기본 데이터 준비
    code = df['Symbol'].iloc[0]
    result = df.copy()
    
    # 2. 호가 데이터 파일 찾기
    hoga_file = os.path.join(ORDERBOOK_PATH, f"{code}.csv")
    
    buy_signal = False
    reason = "데이터 없음"
    
    if os.path.exists(hoga_file):
        try:
            # 3. 호가 데이터 읽기 (최근 10건만 빠르게 읽음)
            # CSV 헤더: Symbol,Date,Val1..Val40,TotalSell,TotalBuy
            h_df = pd.read_csv(hoga_file).tail(10)
            
            if len(h_df) > 0:
                last_row = h_df.iloc[-1]
                
                # 수치 데이터로 변환 (문자열일 수 있으므로 안전하게)
                total_sell_vol = float(last_row['TotalSell']) # 총 매도 잔량
                total_buy_vol = float(last_row['TotalBuy'])   # 총 매수 잔량
                
                # [핵심 로직] 줄다리기 비율 계산
                # 매수 잔량이 0이면 에러나니까 1로 처리
                if total_sell_vol == 0: total_sell_vol = 1 
                
                ratio = total_buy_vol / total_sell_vol
                
                # 4. 판단 기준 (매수벽이 매도벽보다 1.5배 이상 두꺼울 때)
                if ratio >= 1.5:
                    buy_signal = True
                    reason = f"매수벽 강력 (매수{int(total_buy_vol)} vs 매도{int(total_sell_vol)} / 비율 {ratio:.1f}배)"
                elif ratio <= 0.5:
                    reason = f"매도세 우위 (매도 물량이 2배 많음)"
                else:
                    reason = f"힘겨루기 중 (비율 {ratio:.1f}배)"
                    
        except Exception as e:
            reason = f"호가 분석 중 에러: {e}"
    else:
        reason = "실시간 호가 데이터 없음 (장중이 아니거나 수집 안됨)"

    # 5. 결과 반환
    # (주의: 호가가 좋아도 차트가 너무 역배열이면 거르는 안전장치 추가)
    last_candle = df.iloc[-1]
    is_chart_bad = last_candle['Close'] < last_candle['MA20'] # 20일선 아래면 조심
    
    if buy_signal and is_chart_bad:
        buy_signal = False
        reason += " -> 하락 추세라 진입 보류"
        
    result['Buy_Signal'] = False
    result.loc[result.index[-1], 'Buy_Signal'] = buy_signal
    result['Reason_Msg'] = reason
    
    return result