import pandas as pd
import os

# [전략 설정]
MARKET_INDEX_PATH = r"E:\AI_DATA_CENTER\Stock_Data_V5\Market_Index"
EXECUTION_PATH = r"E:\AI_DATA_CENTER\Stock_Data_V5\Raw_Execution"

def prepare_data(df, config):
    """
    1. 시장 지수 확인 (코스닥이 5일선 위에 있는가?)
    2. 체결 강도 확인 (최근 체결 데이터에서 매수세가 강한가?)
    """
    
    # --- 1. 시장 지수(코스닥 101) 필터링 ---
    is_market_good = False
    try:
        kosdaq_path = os.path.join(MARKET_INDEX_PATH, "101_daily.csv")
        if os.path.exists(kosdaq_path):
            mk_df = pd.read_csv(kosdaq_path)
            last_mk = mk_df.iloc[-1]
            # 코스닥 종가가 5일 이동평균보다 높으면 '장 좋음' 판단
            if last_mk['Close'] > last_mk['MA5']:
                is_market_good = True
    except:
        pass # 파일 없으면 일단 패스

    # --- 2. 체결 강도(Buying Power) 계산 ---
    # 수집기가 저장하고 있는 실시간 체결 파일(Raw_Execution)을 읽음
    code = df['Symbol'].iloc[0] # 현재 분석 중인 종목 코드
    power_score = 0
    
    try:
        exec_file = os.path.join(EXECUTION_PATH, f"{code}.csv")
        if os.path.exists(exec_file):
            # 최근 100건의 체결 내역만 읽음
            # (주의: 파일이 계속 커지므로 tail만 읽는 최적화가 필요할 수 있음)
            exec_df = pd.read_csv(exec_file).tail(50)
            
            # 체결강도(Power) 평균이 120 이상인가?
            avg_power = exec_df['Power'].astype(float).mean()
            if avg_power > 120:
                power_score = avg_power
    except:
        pass

    # --- 3. 최종 매수 신호 결정 ---
    last_row = df.iloc[-1]
    
    # 기본 기술적 분석 (이평선 정배열)
    is_up_trend = last_row['MA5'] > last_row['MA20']
    
    buy_signal = False
    reason = "관망"

    if is_market_good and is_up_trend and (power_score > 120):
        buy_signal = True
        reason = f"시장양호 + 수급폭발(체결강도 {power_score:.1f}%)"
    elif not is_market_good:
        reason = "시장 하락세(코스닥 약세)"
    elif power_score <= 120:
        reason = "수급 약함"

    # 결과 반환
    result = df.copy()
    result['Buy_Signal'] = False
    result.loc[result.index[-1], 'Buy_Signal'] = buy_signal
    result['Reason_Msg'] = reason
    
    return result