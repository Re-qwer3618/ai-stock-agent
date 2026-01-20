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
    st.sidebar.markdown("### 🛠️ 시나리오별 매수 전략 (Scenario Strategy)")
    
    # [NEW] 전략 모드 선택 (가장 중요한 스위치)
    strategy_mode = st.sidebar.selectbox(
        "📌 매수 시나리오 선택",
        [
            "Case 1: 매도세 실종형 (Volume Dry-up)",
            "Case 2: 투매 후 아래꼬리형 (Panic Reversal)",
            "Case 3: 지지선 반등형 (MA Support)",
            "Case 4: 선발대 확인형 (Scout Strategy)",
            "Case 5: 호재/매집형 (Accumulation)"
        ],
        index=3, # 기본값은 Case 4 (기존 전략)
        help="시뮬레이션할 주가 패턴 시나리오를 선택하세요."
    )

    st.sidebar.markdown("---")
    
    # ---------------------------------------------------------
    # 전략별 전용 파라미터 (Dynamic UI)
    # ---------------------------------------------------------
    params = {"strategy_mode": strategy_mode}
    
    if "Case 1" in strategy_mode:
        st.sidebar.caption("📉 **[Case 1] 바닥권 거래량 급감**")
        params['c1_vol_drop'] = st.sidebar.slider("거래량 감소율 (%)", 30, 80, 50, help="20일 평균 대비 거래량이 몇 % 이하로 줄어야 할까요?")
        params['c1_window'] = st.sidebar.slider("신저가 관찰 기간 (일)", 20, 120, 60, help="최근 며칠 내 최저가 근처여야 하나요?")
        
    elif "Case 2" in strategy_mode:
        st.sidebar.caption("😱 **[Case 2] 투매 후 아래꼬리 반등**")
        params['c2_drop_pct'] = st.sidebar.slider("장중 하락폭 (%)", 3, 15, 5, help="장중 저가가 시가 대비 몇 % 이상 빠졌었나요?")
        params['c2_tail_ratio'] = st.sidebar.slider("아래꼬리 비율 (%)", 30, 80, 50, help="전체 캔들 길이 중 아래꼬리가 차지하는 비중")
        params['c2_vol_mult'] = st.sidebar.slider("거래량 폭증 배수", 1.0, 5.0, 2.0, help="평소(20일 평균)보다 거래량이 몇 배 터져야 하나요?")
        
    elif "Case 3" in strategy_mode:
        st.sidebar.caption("support **[Case 3] 이평선 지지 반등**")
        params['c3_ma_period'] = st.sidebar.selectbox("지지 이평선 선택", [20, 60, 120], index=0)
        params['c3_tolerance'] = st.sidebar.slider("지지선 근접 오차 (%)", 1.0, 5.0, 2.0, help="이평선과 얼마나 가까워야 지지로 인정할까요?")
        
    elif "Case 4" in strategy_mode:
        st.sidebar.caption("🚀 **[Case 4] 저점 후 선발대(반등) 확인**")
        params['c4_scout_pct'] = st.sidebar.slider("선발대 상승률 (%)", 1.0, 10.0, 3.0, help="확실한 반등(선발대)으로 인정할 당일 상승률")
        params['c4_wait_days'] = st.sidebar.slider("저점 후 유효기간 (일)", 1, 20, 5, help="저점을 찍고 며칠 내에 선발대가 나와야 진입할까요?")
        params['c4_gap_allow'] = st.sidebar.checkbox("갭상승 시 진입 허용", value=True)
        
    elif "Case 5" in strategy_mode:
        st.sidebar.caption("🤫 **[Case 5] 가격/거래량 괴리 (매집)**")
        params['c5_std_dev'] = st.sidebar.slider("가격 변동성 제한 (%)", 1.0, 5.0, 2.0, help="주가가 얼마나 횡보(안정)해야 하나요?")
        params['c5_vol_ratio'] = st.sidebar.slider("거래량 위축 기준 (%)", 30, 90, 60, help="평소 대비 거래량이 얼마나 적어야 매집으로 볼까요?")

    st.sidebar.markdown("---")

    # ---------------------------------------------------------
    # 공통 필터 (Common Filters) - 기존 UI 유지
    # ---------------------------------------------------------
    st.sidebar.markdown("#### 🔧 공통 보조 지표")
    
    col1, col2 = st.sidebar.columns(2)
    use_ma_filter = col1.checkbox("정배열/이평 필터", value=False, help="주가가 장기 이평선 위에 있을 때만 매수")
    use_ob_filter = col2.checkbox("호가창 필터", value=False, help="매도 물량이 매수 물량보다 많을 때만 매수")

    ma_short, ma_long = 5, 20
    if use_ma_filter:
        with st.expander("〰️ 이동평균선 설정"):
            ma_short = st.number_input("단기 이평", value=5)
            ma_long = st.number_input("장기 이평", value=20)
            
    ob_ratio = 1.5
    if use_ob_filter:
        with st.expander("📊 호가창 비율 설정"):
            ob_ratio = st.slider("매도/매수 잔량비", 0.5, 3.0, 1.5)

    # 파라미터 병합
    params.update({
        "use_ma_filter": use_ma_filter,
        "ma_short": ma_short,
        "ma_long": ma_long,
        "use_ob_filter": use_ob_filter,
        "ob_ratio": ob_ratio
    })
    
    return params


# =========================================================
# [Part 2] 핵심 전략 로직 구현 (Logic Implementation)
# =========================================================

def calc_basics(df):
    """기초 데이터 계산"""
    df['MA_5'] = df['Close'].rolling(5).mean()
    df['MA_20'] = df['Close'].rolling(20).mean()
    df['MA_60'] = df['Close'].rolling(60).mean()
    df['MA_120'] = df['Close'].rolling(120).mean()
    df['Vol_MA_20'] = df['Volume'].rolling(20).mean()
    df['Day_Chg'] = df['Close'].pct_change() * 100
    df['Vol_Chg'] = df['Volume'].pct_change() * 100
    return df

# --- [Case 1] 매도세 실종형 ---
def logic_case1_dryup(df, config):
    vol_ratio = config.get('c1_vol_drop', 50) / 100.0
    window = config.get('c1_window', 60)
    
    # 1. 신저가 근처인가? (최근 N일 최저가 대비 5% 이내)
    recent_low = df['Low'].rolling(window=window).min()
    is_low_area = df['Close'] <= recent_low * 1.05
    
    # 2. 거래량이 말랐는가?
    is_vol_dry = df['Volume'] < (df['Vol_MA_20'] * vol_ratio)
    
    # 3. 주가가 폭락 중은 아닌가? (소폭 하락 or 보합)
    is_stable = df['Day_Chg'] > -3.0
    
    signal = is_low_area & is_vol_dry & is_stable
    return signal, "Case1(VolDry)"

# --- [Case 2] 투매 후 아래꼬리형 ---
def logic_case2_panic(df, config):
    drop_pct = config.get('c2_drop_pct', 5) / 100.0
    tail_ratio = config.get('c2_tail_ratio', 50) / 100.0
    vol_mult = config.get('c2_vol_mult', 2.0)
    
    # 1. 장중 급락 발생 (Low가 Open 대비 많이 빠짐)
    # 아래꼬리 길이 = min(Open, Close) - Low
    # 전체 길이 = High - Low
    body_bottom = df[['Open', 'Close']].min(axis=1)
    lower_wick = body_bottom - df['Low']
    total_range = df['High'] - df['Low']
    
    is_deep_dip = (df['Low'] < df['Open'] * (1 - drop_pct))
    is_long_tail = (lower_wick / total_range.replace(0, 1)) >= tail_ratio
    
    # 2. 거래량 폭발 (투매 받아내기)
    is_vol_spike = df['Volume'] > (df['Vol_MA_20'] * vol_mult)
    
    signal = is_deep_dip & is_long_tail & is_vol_spike
    return signal, "Case2(PanicWick)"

# --- [Case 3] 지지선 반등형 ---
def logic_case3_support(df, config):
    ma_pd = config.get('c3_ma_period', 20)
    tolerance = config.get('c3_tolerance', 2.0) / 100.0
    
    ma_col = f'MA_{ma_pd}'
    if ma_col not in df.columns: return pd.Series(False, index=df.index), ""
    
    # 1. 지지선 근접 (MA와 Low의 차이가 작음)
    dist_to_ma = abs(df['Low'] - df[ma_col]) / df[ma_col]
    near_support = dist_to_ma <= tolerance
    
    # 2. 양봉 발생 (지지를 확인)
    is_bullish = df['Close'] > df['Open']
    
    # 3. 주가가 MA 위에 있거나 살짝 아래 (완전 이탈은 제외)
    # is_above_ma = df['Close'] > df[ma_col] * 0.98
    
    signal = near_support & is_bullish
    return signal, f"Case3(Sup{ma_pd})"

# --- [Case 4] 선발대 확인형 (기존 로직) ---
def logic_case4_scout(df, config):
    scout_pct = config.get('c4_scout_pct', 3.0)
    wait_days = config.get('c4_wait_days', 5)
    
    # 1. 저점(Local Min) 인식
    prev_low = df['Low'].shift(1)
    prev2_low = df['Low'].shift(2)
    is_local_min = (prev_low < prev2_low) & (prev_low < df['Low'])
    
    # 2. 과거 N일 내 저점이 있었는가?
    was_low_recently = is_local_min.shift(1).rolling(window=wait_days, min_periods=1).max().fillna(0).astype(bool)
    
    # 3. 오늘 선발대(강한 상승) 출현
    is_scout = df['Day_Chg'] >= scout_pct
    
    # 4. 갭상승 필터
    is_gap_up = df['Open'] > df['Close'].shift(1)
    if not config.get('c4_gap_allow', True):
        is_scout = is_scout & (~is_gap_up)
        
    signal = was_low_recently & is_scout
    return signal, "Case4(Scout)"

# --- [Case 5] 호재/매집형 (가격 괴리) ---
def logic_case5_accum(df, config):
    std_limit = config.get('c5_std_dev', 2.0)
    vol_limit = config.get('c5_vol_ratio', 60) / 100.0
    
    # 1. 가격 변동성이 극도로 낮음 (횡보)
    # 최근 10일간 Close의 표준편차 / 평균
    rolling_std = df['Close'].rolling(10).std()
    rolling_mean = df['Close'].rolling(10).mean()
    cv = (rolling_std / rolling_mean) * 100
    is_tight = cv <= std_limit
    
    # 2. 거래량 실종 (매도세 없음)
    is_no_vol = df['Volume'] < (df['Vol_MA_20'] * vol_limit)
    
    # 3. 주가 수준이 너무 낮지 않음 (완전 역배열 폭락은 제외)
    # 60일 이평선 대비 90% 이상은 유지
    is_holding = df['Close'] > (df['MA_60'] * 0.9)
    
    signal = is_tight & is_no_vol & is_holding
    return signal, "Case5(Accum)"


# =========================================================
# [Part 3] 데이터 처리 및 신호 결합 (Data Processing)
# =========================================================
def prepare_data(df, config):
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date').reset_index(drop=True)

    # 1. 기본 지표 계산
    df = calc_basics(df)
    
    # 2. 전략 모드에 따른 신호 생성
    mode = config.get('strategy_mode', '')
    
    if "Case 1" in mode:
        raw_signal, msg = logic_case1_dryup(df, config)
    elif "Case 2" in mode:
        raw_signal, msg = logic_case2_panic(df, config)
    elif "Case 3" in mode:
        raw_signal, msg = logic_case3_support(df, config)
    elif "Case 4" in mode:
        raw_signal, msg = logic_case4_scout(df, config)
    elif "Case 5" in mode:
        raw_signal, msg = logic_case5_accum(df, config)
    else:
        raw_signal = pd.Series(False, index=df.index)
        msg = ""

    # 3. 공통 필터 적용 (이평선, 호가창)
    final_signal = raw_signal.copy()
    reason_msg = pd.Series([msg] * len(df), index=df.index)
    
    # (A) MA Filter
    if config.get('use_ma_filter'):
        ma_l = config.get('ma_long', 20)
        ma_ok = df['Close'] > df['MA_20'] # 단순화: 20일선 위에 있어야 함
        final_signal = final_signal & ma_ok
        # 필터 탈락 시 메시지 제거 혹은 유지 (여기선 제거)
        reason_msg[~ma_ok] = ""

    # (B) Orderbook Filter (데이터가 있을 경우만)
    if config.get('use_ob_filter') and 'Total_Ask_Size' in df.columns:
        ratio = config.get('ob_ratio', 1.5)
        ob_ok = (df['Total_Ask_Size'] / df['Total_Bid_Size'].replace(0, 1)) >= ratio
        final_signal = final_signal & ob_ok
        reason_msg[~ob_ok] = ""

    df['Buy_Signal'] = final_signal
    df['Reason_Msg'] = np.where(final_signal, reason_msg, "")
    
    return df


# =========================================================
# [Part 4] 매매 실행 (Execution) - 기존 유지
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

    # 데이터가 충분히 쌓인 뒤부터 시작 (최대 120일)
    start_idx = 120
    if len(df) <= start_idx: start_idx = 0
    
    for i in range(start_idx, len(df)):
        row = df.iloc[i]
        
        # 1. 매도 (Sell)
        if shares > 0:
            sell_price = 0
            sell_reason = ""
            
            # 고가/저가 기준으로 익절/손절 체크
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
# [Part 5] 차트 생성 (Visualization) - 기존 유지
# =========================================================
def create_chart_image(df, logs, save_dir, code, config=None):
    if len(df) == 0: return

    chart_df = df.copy()
    chart_df.set_index('Date', inplace=True)
    
    add_plots = []
    
    # MA Plot
    if 'MA_20' in chart_df.columns:
        add_plots.append(mpf.make_addplot(chart_df['MA_20'], color='gold', width=1.2))
    if 'MA_60' in chart_df.columns:
        add_plots.append(mpf.make_addplot(chart_df['MA_60'], color='green', width=1.0))

    # Buy/Sell Markers
    buy_markers = [np.nan] * len(chart_df)
    sell_markers = [np.nan] * len(chart_df)
    
    for log in logs:
        date_ts = pd.to_datetime(log['Date'])
        if date_ts in chart_df.index:
            idx = chart_df.index.get_loc(date_ts)
            # 인덱스가 중복될 경우 첫번째 사용
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
    
    # 타이틀에 전략 모드 표시
    title_text = f"Sim: {code}"
    if config:
        # 긴 이름 줄이기
        mode_short = config.get('strategy_mode', '').split(':')[0]
        title_text += f" | {mode_short}"

    try:
        mpf.plot(
            chart_df, type='candle', volume=True, addplot=add_plots, style=s,
            title=title_text, figsize=(14, 8), savefig=save_path, tight_layout=True, warn_too_much_data=20000
        )
    except Exception as e:
        print(f"Chart Error {code}: {e}")