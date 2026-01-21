import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import mplfinance as mpf
import streamlit as st

def strategy_ui():
    st.sidebar.markdown("### 😱 Case 2: 투매 후 아래꼬리 (Panic Reversal)")
    st.sidebar.info("급락 후 거래량이 터지며 말아 올리는 '공포 매수' 패턴입니다.")
    
    with st.expander("⚙️ 전략 파라미터", expanded=True):
        drop_pct = st.slider("장중 하락폭 (%)", 3, 15, 5, help="장중 저가가 시가 대비 몇 % 이상 빠졌었나요?")
        tail_ratio = st.slider("아래꼬리 비율 (%)", 30, 80, 50, help="전체 캔들 길이 중 아래꼬리가 차지하는 비중")
        vol_mult = st.slider("거래량 폭증 배수", 1.0, 5.0, 2.0, help="평소(20일 평균)보다 거래량이 몇 배 터져야 하나요?")

    st.sidebar.markdown("---")
    tp = st.sidebar.number_input("목표 수익률(%)", value=10.0)
    sl = st.sidebar.number_input("손절 제한(%)", value=-5.0)

    return {"drop_pct": drop_pct, "tail_ratio": tail_ratio, "vol_mult": vol_mult, "target_profit": tp, "stop_loss": sl}

def prepare_data(df, config):
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date').reset_index(drop=True)

    df['Vol_MA_20'] = df['Volume'].rolling(20).mean()
    df['Day_Chg'] = df['Close'].pct_change() * 100

    # --- 로직: Case 2 ---
    drop_pct = config['drop_pct'] / 100.0
    tail_ratio = config['tail_ratio'] / 100.0
    vol_mult = config['vol_mult']
    
    # 1. 꼬리 계산
    body_bottom = df[['Open', 'Close']].min(axis=1)
    lower_wick = body_bottom - df['Low']
    total_range = df['High'] - df['Low']
    
    # 2. 조건 확인
    is_deep_dip = (df['Low'] < df['Open'] * (1 - drop_pct))
    is_long_tail = (lower_wick / total_range.replace(0, 1)) >= tail_ratio
    is_vol_spike = df['Volume'] > (df['Vol_MA_20'] * vol_mult)
    
    df['Buy_Signal'] = is_deep_dip & is_long_tail & is_vol_spike
    df['Reason_Msg'] = np.where(df['Buy_Signal'], "Case2(Panic)", "")
    
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
    try: mpf.plot(df, type='candle', volume=True, title=f"Case 2: {code}", style='yahoo', savefig=save_path, figsize=(12,6))
    except: pass