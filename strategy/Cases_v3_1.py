import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import mplfinance as mpf
import streamlit as st

# =========================================================
# [Part 1] UI 설정
# =========================================================
def strategy_ui():
    st.sidebar.markdown("### 🤫 Case 1: 매도세 실종 (Volume Dry-up)")
    st.sidebar.info("신저가 근처에서 거래량이 급감하며 주가가 횡보하는 '바닥 다지기' 패턴을 찾습니다.")
    
    # 전략 파라미터
    with st.expander("⚙️ 전략 파라미터", expanded=True):
        vol_drop = st.slider("거래량 감소율 (%)", 30, 80, 50, help="20일 평균 대비 거래량이 몇 % 이하로 줄어야 할까요?")
        window = st.slider("신저가 관찰 기간 (일)", 20, 120, 60, help="최근 며칠 내 최저가 근처여야 하나요?")
    
    # 공통 필터
    with st.expander("🛡️ 안전장치 (필터)", expanded=False):
        use_ma_filter = st.checkbox("20일 이평선 지지 확인", value=False)

    # 매매 설정
    st.sidebar.markdown("---")
    tp = st.sidebar.number_input("목표 수익률(%)", value=15.0)
    sl = st.sidebar.number_input("손절 제한(%)", value=-5.0)

    return {
        "vol_drop": vol_drop,
        "window": window,
        "use_ma_filter": use_ma_filter,
        "target_profit": tp,
        "stop_loss": sl
    }

# =========================================================
# [Part 2] 데이터 처리 (Logic)
# =========================================================
def prepare_data(df, config):
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date').reset_index(drop=True)

    # 지표 계산
    df['Vol_MA_20'] = df['Volume'].rolling(20).mean()
    df['Day_Chg'] = df['Close'].pct_change() * 100
    df['MA_20'] = df['Close'].rolling(20).mean()

    # --- 로직: Case 1 ---
    vol_ratio = config['vol_drop'] / 100.0
    window = config['window']
    
    # 1. 신저가 근처 (최근 N일 최저가 대비 5% 이내)
    recent_low = df['Low'].rolling(window=window).min()
    is_low_area = df['Close'] <= recent_low * 1.05
    
    # 2. 거래량 급감
    is_vol_dry = df['Volume'] < (df['Vol_MA_20'] * vol_ratio)
    
    # 3. 주가 안정 (폭락 아님)
    is_stable = df['Day_Chg'] > -3.0
    
    signal = is_low_area & is_vol_dry & is_stable

    # 필터
    if config['use_ma_filter']:
        signal = signal & (df['Close'] > df['MA_20'])

    df['Buy_Signal'] = signal
    df['Reason_Msg'] = np.where(signal, "Case1(DryUp)", "")
    
    return df

# =========================================================
# [Part 3] 매매 실행 (Fixed)
# =========================================================
def execute_trade(df, config):
    initial_capital = config['account']['initial_capital']
    fee_rate = config['account']['fee_rate']
    tp_rate = config['target_profit'] / 100.0
    sl_rate = config['stop_loss'] / 100.0
    
    balance = initial_capital
    shares = 0
    avg_price = 0
    logs = []
    
    start_idx = 60
    if len(df) < start_idx: return initial_capital, logs

    for i in range(start_idx, len(df)):
        row = df.iloc[i]
        
        # 1. 매도 (Sell)
        if shares > 0:
            tp_price = avg_price * (1 + tp_rate)
            sl_price = avg_price * (1 + sl_rate)
            
            sell_price = 0
            reason = ""
            
            if row['High'] >= tp_price: 
                sell_price = max(row['Open'], tp_price)
                reason = "TP(익절)"
            elif row['Low'] <= sl_price: 
                sell_price = min(row['Open'], sl_price)
                reason = "SL(손절)"
            
            if sell_price > 0:
                revenue = shares * sell_price * (1 - fee_rate)
                profit = revenue - (shares * avg_price)
                
                # [수정] 수익률 계산 및 로그 추가
                profit_rate = (sell_price - avg_price) / avg_price * 100
                
                logs.append({
                    "Date": row['Date'].strftime('%Y-%m-%d'), 
                    "Type": "Sell", 
                    "Price": int(sell_price), 
                    "Shares": shares, 
                    "Profit": int(profit),
                    "Profit_Rate": round(profit_rate, 2), # <--- 여기가 누락되었었습니다!
                    "Reason": reason, 
                    "Day_Chg(%)": round(row['Day_Chg'], 2)
                })
                
                balance += revenue
                shares = 0
                avg_price = 0
                continue
                
        # 2. 매수 (Buy)
        if shares == 0 and row['Buy_Signal']:
            buy_shares = int((balance * 0.99) / row['Open'])
            if buy_shares > 0:
                shares = buy_shares
                avg_price = row['Open']
                balance -= shares * avg_price
                
                logs.append({
                    "Date": row['Date'].strftime('%Y-%m-%d'), 
                    "Type": "Buy", 
                    "Price": int(row['Open']), 
                    "Shares": shares, 
                    "Profit": 0, 
                    "Profit_Rate": 0, 
                    "Reason": row['Reason_Msg'], 
                    "Day_Chg(%)": round(row['Day_Chg'], 2)
                })

    final = balance + (shares * df.iloc[-1]['Close']) if shares > 0 else balance
    return final, logs

# =========================================================
# [Part 4] 차트 생성
# =========================================================
def create_chart_image(df, logs, save_dir, code, config=None):
    if len(df) == 0: return
    
    # 차트용 데이터 복사 (원본 보존)
    chart_df = df.copy()
    chart_df.set_index('Date', inplace=True)
    
    save_path = os.path.join(save_dir, f"{code}_chart.png")
    
    try:
        mpf.plot(chart_df, type='candle', volume=True, 
                 title=f"Case 1: {code}", style='yahoo', 
                 savefig=save_path, figsize=(12,6))
    except Exception as e:
        print(f"Chart Error: {e}")