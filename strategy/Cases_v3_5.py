import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import mplfinance as mpf
import streamlit as st

# =========================================================
# [Part 1] UI 설정 (Case 5 전용)
# =========================================================
def strategy_ui():
    st.sidebar.markdown("### 🏹 Case 5: 매집 & 추세 (Fine-Tuned)")
    st.sidebar.info("기존 Case 5의 단점(하락 횡보 매수)을 보완하기 위해 '추세 필터'와 '최소 거래대금' 조건을 추가한 버전입니다.")
    
    # 1. 핵심 파라미터 (매집 판단)
    with st.expander("🔍 매집(횡보) 기준 설정", expanded=True):
        std_limit = st.slider("가격 변동성 제한 (%)", 1.0, 5.0, 2.0, help="변동성이 이 값 이하로 낮아야 '횡보'로 인정")
        vol_ratio = st.slider("거래량 위축 기준 (%)", 30, 90, 60, help="20일 평균 대비 거래량이 이 % 수준으로 줄어야 함")

    # 2. [필수] 추세 필터 (하락장 회피)
    with st.expander("📈 추세 필터 (Safety Guard)", expanded=True):
        use_trend = st.checkbox("이평선 정배열 조건", value=True, help="주가가 장기 이평선 위에 있을 때만 매수")
        ma_trend_period = st.selectbox("기준 장기 이평선", [60, 120], index=0)
    
    # 3. [옵션] 최소 거래대금 (소외주 회피)
    with st.expander("💰 거래대금 필터", expanded=False):
        min_money = st.number_input("최소 거래대금 (억원)", value=10, step=5)
    
    # 4. 익절/손절 설정
    st.sidebar.markdown("---")
    tp = st.sidebar.number_input("목표 수익률(%)", value=15.0)
    sl = st.sidebar.number_input("손절 제한(%)", value=-5.0)

    return {
        "std_limit": std_limit,
        "vol_ratio": vol_ratio,
        "use_trend": use_trend,
        "ma_trend_period": ma_trend_period,
        "min_money": min_money * 100000000, # 억 단위 변환
        "target_profit": tp,
        "stop_loss": sl
    }

# =========================================================
# [Part 2] 전략 로직 (Logic)
# =========================================================
def prepare_data(df, config):
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date').reset_index(drop=True)

    # 1. 보조지표 계산
    df['MA_20'] = df['Close'].rolling(20).mean()
    df['MA_60'] = df['Close'].rolling(60).mean()
    df['MA_120'] = df['Close'].rolling(120).mean()
    df['Vol_MA_20'] = df['Volume'].rolling(20).mean()
    
    # 변동성 (표준편차/평균)
    df['Roll_Std'] = df['Close'].rolling(10).std()
    df['Roll_Mean'] = df['Close'].rolling(10).mean()
    df['Volatility'] = (df['Roll_Std'] / df['Roll_Mean']) * 100
    
    df['Day_Chg'] = df['Close'].pct_change() * 100
    df['Money'] = df['Close'] * df['Volume'] # 거래대금

    # 2. 로직 적용
    # (A) 횡보 조건 (변동성이 낮음)
    cond_tight = df['Volatility'] <= config['std_limit']
    
    # (B) 거래량 급감 (매도세 실종)
    cond_no_vol = df['Volume'] < (df['Vol_MA_20'] * (config['vol_ratio'] / 100.0))
    
    # (C) 추세 필터 (Trend Filter) - 핵심 보완점
    if config['use_trend']:
        ma_col = f"MA_{config['ma_trend_period']}"
        # 주가가 장기 이평선보다 위에 있어야 함 (정배열 초입 or 눌림목)
        cond_trend = df['Close'] > df[ma_col]
    else:
        cond_trend = True
        
    # (D) 최소 거래대금 (잡주 제외)
    cond_money = df['Money'] >= config['min_money']

    # 최종 신호
    df['Buy_Signal'] = cond_tight & cond_no_vol & cond_trend & cond_money
    
    # 이유 메시지 생성
    msg = "Case5(Accum)"
    if config['use_trend']: msg += "+Trend"
    df['Reason_Msg'] = np.where(df['Buy_Signal'], msg, "")

    return df

# =========================================================
# [Part 3] 매매 실행 (Execution)
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

    # 데이터가 충분히 쌓인 120일 이후부터 시작
    start_idx = 120
    if len(df) < start_idx: return initial_capital, logs
    
    for i in range(start_idx, len(df)):
        row = df.iloc[i]
        
        # 1. 매도 (Sell)
        if shares > 0:
            sell_price = 0
            sell_reason = ""
            
            tp_price = avg_price * (1 + tp_rate)
            sl_price = avg_price * (1 + sl_rate)

            if row['High'] >= tp_price:
                sell_price = max(row['Open'], tp_price)
                sell_reason = "TP(익절)"
            elif row['Low'] <= sl_price:
                sell_price = min(row['Open'], sl_price)
                sell_reason = "SL(손절)"
            
            if sell_price > 0:
                revenue = shares * sell_price * (1 - fee_rate)
                profit = revenue - (shares * avg_price)
                profit_rate = (sell_price - avg_price) / avg_price * 100
                balance += revenue
                
                logs.append({
                    "Date": row['Date'].strftime('%Y-%m-%d %H:%M'),
                    "Type": "Sell",
                    "Price": int(sell_price),
                    "Shares": shares,
                    "Balance": int(balance),
                    "Profit": int(profit),
                    "Profit_Rate": round(profit_rate, 2),
                    "Reason": sell_reason,
                    "Day_Chg(%)": round(row['Day_Chg'], 2),
                })
                shares = 0
                avg_price = 0
                continue

        # 2. 매수 (Buy)
        if shares == 0 and row['Buy_Signal']:
            can_buy_amt = balance * 0.99
            buy_shares = int(can_buy_amt / row['Open'])
            
            if buy_shares > 0:
                shares = buy_shares
                avg_price = row['Open']
                balance -= shares * avg_price
                
                logs.append({
                    "Date": row['Date'].strftime('%Y-%m-%d %H:%M'),
                    "Type": "Buy",
                    "Price": int(row['Open']),
                    "Shares": shares,
                    "Balance": int(balance),
                    "Profit": 0,
                    "Profit_Rate": 0,
                    "Reason": row['Reason_Msg'],
                    "Day_Chg(%)": round(row['Day_Chg'], 2),
                })

    final_asset = balance
    if shares > 0:
        final_asset += shares * df.iloc[-1]['Close']

    return final_asset, logs

# =========================================================
# [Part 4] 차트 (Visualization)
# =========================================================
def create_chart_image(df, logs, save_dir, code, config=None):
    if len(df) == 0: return

    chart_df = df.copy()
    chart_df.set_index('Date', inplace=True)
    
    add_plots = []
    
    # 60일, 120일 이평선 표시 (추세 확인용)
    if 'MA_60' in chart_df.columns:
        add_plots.append(mpf.make_addplot(chart_df['MA_60'], color='green', width=1.0))
    if 'MA_120' in chart_df.columns:
        add_plots.append(mpf.make_addplot(chart_df['MA_120'], color='gray', width=1.0, linestyle='--'))

    # 매매 마커
    buy_markers = [np.nan] * len(chart_df)
    sell_markers = [np.nan] * len(chart_df)
    
    for log in logs:
        date_ts = pd.to_datetime(log['Date'])
        if date_ts in chart_df.index:
            idx = chart_df.index.get_loc(date_ts)
            if isinstance(idx, (slice, np.ndarray)): idx = idx.start if isinstance(idx, slice) else idx[0]
                
            if log['Type'] == 'Buy':
                buy_markers[idx] = chart_df.iloc[idx]['Low'] * 0.98
            elif log['Type'] == 'Sell':
                sell_markers[idx] = chart_df.iloc[idx]['High'] * 1.02

    if any(~np.isnan(buy_markers)):
        add_plots.append(mpf.make_addplot(buy_markers, type='scatter', markersize=100, marker='^', color='red'))
    if any(~np.isnan(sell_markers)):
        add_plots.append(mpf.make_addplot(sell_markers, type='scatter', markersize=100, marker='v', color='blue'))

    save_path = os.path.join(save_dir, f"{code}_chart.png")
    
    try:
        mpf.plot(
            chart_df, type='candle', volume=True, addplot=add_plots, 
            title=f"Case 5 Trend: {code}", figsize=(14, 8), 
            savefig=save_path, tight_layout=True, warn_too_much_data=20000,
            style='yahoo'
        )
    except Exception:
        pass