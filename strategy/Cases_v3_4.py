import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import mplfinance as mpf
import streamlit as st

def strategy_ui():
    st.sidebar.markdown("### 🚀 Case 4: 선발대 확인 (Scout)")
    st.sidebar.info("저점(바닥)을 찍고 나서, 며칠 내에 '의미 있는 상승'이 나올 때 매수합니다. (확인 매매)")
    
    with st.expander("⚙️ 전략 파라미터", expanded=True):
        scout_pct = st.slider("선발대 상승률 (%)", 1.0, 10.0, 3.0, help="확실한 반등으로 인정할 당일 상승률")
        wait_days = st.slider("저점 후 유효기간 (일)", 1, 20, 5, help="저점을 찍고 며칠 내에 선발대가 나와야 진입할까요?")
        gap_allow = st.checkbox("갭상승 시 진입 허용", value=True)

    st.sidebar.markdown("---")
    tp = st.sidebar.number_input("목표 수익률(%)", value=15.0)
    sl = st.sidebar.number_input("손절 제한(%)", value=-5.0)

    return {"scout_pct": scout_pct, "wait_days": wait_days, "gap_allow": gap_allow, "target_profit": tp, "stop_loss": sl}

def prepare_data(df, config):
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date').reset_index(drop=True)

    df['Day_Chg'] = df['Close'].pct_change() * 100
    
    # --- 로직: Case 4 ---
    scout_pct = config['scout_pct']
    wait_days = config['wait_days']
    
    # 1. 저점(Local Min) 인식
    prev_low = df['Low'].shift(1)
    prev2_low = df['Low'].shift(2)
    is_local_min = (prev_low < prev2_low) & (prev_low < df['Low'])
    
    # 2. 과거 N일 내 저점이 있었는가? (유효기간 체크)
    was_low_recently = is_local_min.shift(1).rolling(window=wait_days, min_periods=1).max().fillna(0).astype(bool)
    
    # 3. 오늘 선발대(강한 상승) 출현
    is_scout = df['Day_Chg'] >= scout_pct
    
    # 4. 갭상승 필터
    is_gap_up = df['Open'] > df['Close'].shift(1)
    if not config['gap_allow']:
        is_scout = is_scout & (~is_gap_up)
        
    df['Buy_Signal'] = was_low_recently & is_scout
    df['Reason_Msg'] = np.where(df['Buy_Signal'], "Case4(Scout)", "")
    
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
    try: mpf.plot(df, type='candle', volume=True, title=f"Case 4: {code}", style='yahoo', savefig=save_path, figsize=(12,6))
    except: pass