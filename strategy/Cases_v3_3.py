import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import mplfinance as mpf
import streamlit as st

def strategy_ui():
    st.sidebar.markdown("### 🧱 Case 3: 지지선 반등 (MA Support)")
    st.sidebar.info("주요 이평선까지 눌렸을 때 지지를 받고 양봉이 뜨는 순간을 노립니다.")
    
    with st.expander("⚙️ 전략 파라미터", expanded=True):
        ma_period = st.selectbox("지지 이평선 선택", [20, 60, 120], index=0)
        tolerance = st.slider("지지선 근접 오차 (%)", 1.0, 5.0, 2.0, help="이평선과 얼마나 가까워야 지지로 인정할까요?")

    st.sidebar.markdown("---")
    tp = st.sidebar.number_input("목표 수익률(%)", value=15.0)
    sl = st.sidebar.number_input("손절 제한(%)", value=-5.0)

    return {"ma_period": ma_period, "tolerance": tolerance, "target_profit": tp, "stop_loss": sl}

def prepare_data(df, config):
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date').reset_index(drop=True)

    df['Day_Chg'] = df['Close'].pct_change() * 100
    
    # --- 로직: Case 3 ---
    ma_pd = config['ma_period']
    ma_col = f'MA_{ma_pd}'
    tolerance = config['tolerance'] / 100.0
    
    df[ma_col] = df['Close'].rolling(ma_pd).mean()
    
    # 1. 지지선 근접
    dist_to_ma = abs(df['Low'] - df[ma_col]) / df[ma_col]
    near_support = dist_to_ma <= tolerance
    
    # 2. 양봉 발생 (지지 확인)
    is_bullish = df['Close'] > df['Open']
    
    # 3. 추세 필터 (주가가 MA 위에 있거나 살짝 걸쳐야 함, 완전 이탈은 제외)
    above_support = df['Close'] > (df[ma_col] * 0.98)
    
    df['Buy_Signal'] = near_support & is_bullish & above_support
    df['Reason_Msg'] = np.where(df['Buy_Signal'], f"Case3(MA{ma_pd})", "")
    
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

def create_chart_image(df, logs, save_dir, code, config=None):
    if len(df) == 0: return
    df.set_index('Date', inplace=True)
    save_path = os.path.join(save_dir, f"{code}_chart.png")
    try: mpf.plot(df, type='candle', volume=True, title=f"Case 3: {code}", style='yahoo', savefig=save_path, figsize=(12,6))
    except: pass