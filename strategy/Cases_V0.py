import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import mplfinance as mpf

# =========================================================
# [Part 1] UI 및 설정 (Configuration)
# =========================================================
def strategy_ui():
    import streamlit as st
    st.sidebar.markdown("### 🛠️ 교과서 패턴 (Modular Strategy)")
    
    # 1. 데이터 및 필터 설정
    col1, col2 = st.sidebar.columns(2)
    use_minute = col1.checkbox("분차트 사용", value=False)
    use_ma = col2.checkbox("이동평균선", value=False)
    use_orderbook = st.sidebar.checkbox("호가창(OrderBook)", value=False)

    # 2. [NEW] 선발대(Scout) 설정 - 저점 확인 후 진입 전략
    with st.expander("🚀 선발대(Scout) 진입 설정", expanded=True):
        use_scout = st.checkbox("선발대 확인 후 진입 (T+1)", value=True, help="저점 신호가 뜬 다음날, 상승세(선발대)를 확인할 때만 매수합니다.")
        
        if use_scout:
            scout_pct = st.slider("선발대 기준 등락률 (%)", 1.0, 10.0, 3.0, help="당일 몇 % 이상 상승해야 선발대로 인정할까요?")
            
            st.caption("👇 시가(Open) 위치에 따른 진입 허용")
            c_gap1, c_gap2 = st.columns(2)
            enable_gap_up = c_gap1.checkbox("갭상승 시 진입", value=True, help="시가가 전일 종가보다 높게 시작")
            enable_gap_down = c_gap2.checkbox("갭하락/보합 진입", value=True, help="시가가 전일 종가보다 낮거나 같게 시작")
        else:
            # 변수 초기화 (사용 안 함)
            scout_pct = 0.0
            enable_gap_up = True
            enable_gap_down = True

    # 3. 저점/거래량 파라미터
    with st.expander("📉 저점 및 거래량 설정", expanded=False):
        vol_req = st.checkbox("거래량 감소 필수", value=True)
        vol_ma_pd = st.selectbox("거래량 이평(MA)", [5, 10, 20, 60], index=2)
        vol_drop = st.slider("거래량 감소 기준(%)", 30, 100, 60)
        window = st.slider("신저가 기간", 60, 360, 120)
        threshold = st.slider("지지선 오차(%)", 1.0, 5.0, 3.0)

    # 4. 기타 파라미터
    ma_short, ma_long = 5, 20
    if use_ma:
        with st.expander("〰️ 이동평균선 설정", expanded=False):
            ma_short = st.number_input("단기 이평선", value=5)
            ma_long = st.number_input("장기 이평선", value=20)

    ob_ratio = 1.0
    if use_orderbook:
        with st.expander("📊 호가창 설정", expanded=False):
            ob_ratio = st.slider("매도/매수 잔량비", 0.5, 3.0, 1.5)

    return {
        "use_minute": use_minute,
        "use_ma": use_ma,
        "use_orderbook": use_orderbook,
        
        # 선발대 파라미터
        "use_scout": use_scout,
        "scout_pct": scout_pct,
        "enable_gap_up": enable_gap_up,
        "enable_gap_down": enable_gap_down,
        
        # 기존 파라미터
        "lp_window": window,
        "lp_threshold": threshold / 100.0,
        "lp_vol_drop": vol_drop / 100.0,
        "lp_vol_ma": vol_ma_pd,
        "lp_vol_req": vol_req,
        "ma_short": ma_short,
        "ma_long": ma_long,
        "ob_ratio": ob_ratio
    }

# =========================================================
# [Part 2] 핵심 전략 로직
# =========================================================

def logic_low_point(df, config):
    """ [전략 1] 저점 및 거래량 분석 """
    threshold = config.get('lp_threshold', 0.03)
    vol_drop = config.get('lp_vol_drop', 0.6)
    window = config.get('lp_window', 120)
    vol_ma_days = config.get('lp_vol_ma', 20)
    
    prev_low = df['Low'].shift(1)
    prev2_low = df['Low'].shift(2)
    recent_low_60 = df['Low'].shift(2).rolling(window=60).min()
    vol_ma = df['Volume'].shift(1).rolling(window=vol_ma_days).mean()
    prev_vol = df['Volume'].shift(1)

    local_min = (prev_low < prev2_low) & (prev_low < df['Low'])
    near_support = abs(prev_low - recent_low_60) / recent_low_60 <= threshold
    is_vol_drop = prev_vol < (vol_ma * vol_drop)
    
    if config.get('lp_vol_req', True):
        signal = local_min & near_support & is_vol_drop
    else:
        signal = local_min & near_support
        
    return signal

def logic_scout_entry(df, low_point_signal, config):
    """
    [전략 2] 선발대(Scout) 확인 로직 (T+1)
    어제 저점 신호가 떴고(low_point_signal shifted),
    오늘 주가가 선발대 기준(scout_pct)만큼 올랐는가?
    + 시가 갭상승/하락 조건 체크
    """
    # 1. 어제 저점 신호가 있었는가?
    prev_was_low = low_point_signal.shift(1).fillna(False)
    
    # 2. 오늘 선발대 조건 (등락률 >= N%)
    # Day_Chg는 prepare_data에서 미리 계산됨 ((종가-시가)/시가 * 100 or 전일대비)
    # 여기서는 '전일 종가 대비 당일 종가 등락률'을 기준으로 함
    day_change_pct = df['Close'].pct_change() * 100 
    is_scout_candle = day_change_pct >= config.get('scout_pct', 3.0)
    
    # 3. 시가 갭(Gap) 위치 조건
    prev_close = df['Close'].shift(1)
    is_gap_up = df['Open'] > prev_close
    
    gap_allowed = (is_gap_up & config.get('enable_gap_up', True)) | \
                  (~is_gap_up & config.get('enable_gap_down', True))

    # 최종 진입 신호
    final_signal = prev_was_low & is_scout_candle & gap_allowed
    
    # 메시지 생성
    msg = np.where(final_signal, 
                   f"Scout(DayChg>{config.get('scout_pct')}%, Gap={'Up' if True else 'Down'})", 
                   "")
                   
    # Gap Up/Down 문자열을 벡터화 처리하기 까다로우므로 단순화, 실제 로그엔 상세히 남음
    
    return final_signal, msg

def logic_moving_average(df, config):
    """ [전략 3] 이동평균선 """
    if not config.get('use_ma', False): return True, ""
    
    s_period, l_period = config.get('ma_short', 5), config.get('ma_long', 20)
    ma_long = df['Close'].rolling(window=l_period).mean()
    trend_ok = df['Close'] > ma_long
    return trend_ok, np.where(trend_ok, f"MA(>MA{l_period})", "")

def logic_order_book(df, config):
    """ [전략 4] 호가창 """
    if not config.get('use_orderbook', False): return True, ""
    
    required = ['Total_Ask_Size', 'Total_Bid_Size']
    if not set(required).issubset(df.columns): return True, "" 

    ratio = config.get('ob_ratio', 1.5)
    bid = df['Total_Bid_Size'].replace(0, 1)
    cond = (df['Total_Ask_Size'] / bid) >= ratio
    return cond, np.where(cond, f"OB(Ask/Bid>{ratio})", "")


# =========================================================
# [Part 3] 데이터 전처리 및 통합
# =========================================================
def prepare_data(df, config):
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date').reset_index(drop=True)

    # 1. 기초 지표 계산
    df['Day_Chg'] = df['Close'].pct_change() * 100
    df['Vol_Chg'] = df['Volume'].pct_change() * 100
    
    # 2. 각 전략 실행
    
    # (A) 저점 포착 (이건 매수 후보군 탐색용)
    sig_low_candidate = logic_low_point(df, config)
    
    # (B) 최종 매수 신호 결정
    use_scout = config.get('use_scout', True)
    
    if use_scout:
        # 선발대 모드: 어제 저점 + 오늘 급등 확인
        sig_final, msg_scout = logic_scout_entry(df, sig_low_candidate, config)
        msg_base = "LowPoint(T-1) + "  # 메시지 접두어
    else:
        # 기존 모드: 저점 찍으면 바로 매수
        sig_final = sig_low_candidate
        msg_scout = np.where(sig_final, "LowPoint(Direct)", "")
        msg_base = ""

    # (C) 필터링 (이평선, 호가창)
    sig_ma, msg_ma = logic_moving_average(df, config)
    sig_ob, msg_ob = logic_order_book(df, config)
    
    # 3. 최종 결합
    df['Buy_Signal'] = sig_final & sig_ma & sig_ob
    
    # 4. 메시지 통합
    # 벡터화 연산을 위해 numpy 활용 (속도 최적화)
    # 메시지가 있는 경우에만 합침
    
    # 기본 메시지 (LowPoint or Scout)
    full_msg = np.char.add(msg_base, msg_scout.astype(str))
    
    # MA 메시지 추가
    if config.get('use_ma'):
        # MA 메시지가 있으면 " + MA..." 붙임
        ma_add = np.where(msg_ma != "", np.char.add(" + ", msg_ma.astype(str)), "")
        full_msg = np.char.add(full_msg, ma_add)

    # OB 메시지 추가
    if config.get('use_orderbook'):
        ob_add = np.where(msg_ob != "", np.char.add(" + ", msg_ob.astype(str)), "")
        full_msg = np.char.add(full_msg, ob_add)
        
    df['Reason_Msg'] = np.where(df['Buy_Signal'], full_msg, "")

    return df

# =========================================================
# [Part 4] 매매 실행 (Execution)
# =========================================================
def execute_trade(df, config):
    initial_capital = config['account']['initial_capital']
    fee_rate = config['account']['fee_rate']
    tp_rate = config.get('target_profit', 15) / 100.0
    sl_rate = config.get('stop_loss', -5) / 100.0

    balance = initial_capital
    shares = 0
    avg_price = 0
    logs = []

    # 윈도우 기간 고려하여 시작 인덱스 설정
    ma_max = config.get('ma_long', 20) if config.get('use_ma') else 0
    start_idx = max(config.get('lp_window', 120), ma_max) + 2
    if start_idx >= len(df): start_idx = 60 
    
    for i in range(start_idx, len(df)):
        row = df.iloc[i]
        
        # 1. 매도
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
                    "Vol_Chg(%)": round(row['Vol_Chg'], 2),
                    "Market_Chg(%)": round(row.get('Market_Chg', 0), 2)
                })
                shares = 0
                avg_price = 0
                continue

        # 2. 매수
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
                    "Vol_Chg(%)": round(row['Vol_Chg'], 2),
                    "Market_Chg(%)": round(row.get('Market_Chg', 0), 2)
                })

    final_asset = balance
    if shares > 0:
        final_asset += shares * df.iloc[-1]['Close']

    return final_asset, logs

# =========================================================
# [Part 5] 차트 시각화 (Visualization)
# =========================================================
def create_chart_image(df, logs, save_dir, code, config=None):
    if len(df) == 0: return

    chart_df = df.copy()
    chart_df.set_index('Date', inplace=True)
    chart_df.index.name = 'Date'
    
    add_plots = []
    
    # 1. 시장 지수
    if 'Market_Close' in chart_df.columns and chart_df['Market_Close'].sum() > 0:
        mkt_plot = mpf.make_addplot(
            chart_df['Market_Close'], color='orange', secondary_y=True, width=1.0, linestyle='--'
        )
        add_plots.append(mkt_plot)

    # 2. 이동평균선
    if config and config.get('use_ma'):
        ma_s = config.get('ma_short', 5)
        ma_l = config.get('ma_long', 20)
        ma_s_line = chart_df['Close'].rolling(window=ma_s).mean()
        ma_l_line = chart_df['Close'].rolling(window=ma_l).mean()
        add_plots.append(mpf.make_addplot(ma_s_line, color='fuchsia', width=1.0)) 
        add_plots.append(mpf.make_addplot(ma_l_line, color='gold', width=1.2))

    # 3. 매매 마커
    buy_markers = [np.nan] * len(chart_df)
    sell_markers = [np.nan] * len(chart_df)
    
    for log in logs:
        date_ts = pd.to_datetime(log['Date'])
        if date_ts in chart_df.index:
            idx = chart_df.index.get_loc(date_ts)
            if isinstance(idx, (slice, np.ndarray)):
                idx = idx.start if isinstance(idx, slice) else idx[0]
                
            if log['Type'] == 'Buy':
                buy_markers[idx] = chart_df.iloc[idx]['Low'] * 0.98
            elif log['Type'] == 'Sell':
                sell_markers[idx] = chart_df.iloc[idx]['High'] * 1.02

    if any(~np.isnan(buy_markers)):
        add_plots.append(mpf.make_addplot(buy_markers, type='scatter', markersize=100, marker='^', color='red'))
    if any(~np.isnan(sell_markers)):
        add_plots.append(mpf.make_addplot(sell_markers, type='scatter', markersize=100, marker='v', color='blue'))

    mc = mpf.make_marketcolors(up='red', down='blue', edge='inherit', wick='inherit', volume='inherit')
    s = mpf.make_mpf_style(marketcolors=mc, figcolor='white', facecolor='white', edgecolor='black', gridstyle=':')
    
    save_path = os.path.join(save_dir, f"{code}_chart.png")
    
    title = f"Sim: {code}"
    if config and config.get('use_scout'):
        title += f" | Scout(+{config.get('scout_pct')}%)"
    
    try:
        mpf.plot(
            chart_df, type='candle', volume=True, addplot=add_plots, style=s,
            title=title, figsize=(14, 8), savefig=save_path, tight_layout=True, warn_too_much_data=20000
        )
    except Exception as e:
        print(f"Chart Error {code}: {e}")