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
    st.sidebar.markdown("#### 1. 적용할 데이터/전략 선택")
    col1, col2 = st.sidebar.columns(2)
    use_minute = col1.checkbox("분차트 사용", value=False, help="분봉 데이터 파일도 분석에 포함합니다.")
    use_ma = col2.checkbox("이동평균선(MA)", value=False, help="이동평균선 조건을 추가로 검사합니다.")
    use_orderbook = st.sidebar.checkbox("호가창(OrderBook)", value=False, help="호가 데이터(매수/매도 잔량)를 분석합니다. (데이터 있을 시)")

    # 2. 세부 파라미터 설정
    st.sidebar.markdown("#### 2. 세부 파라미터")
    
    # (A) 저점/거래량 파라미터
    with st.expander("📉 저점 및 거래량 설정", expanded=True):
        vol_req = st.checkbox("거래량 감소 필수", value=True)
        vol_ma_pd = st.selectbox("거래량 이평(MA)", [5, 10, 20, 60, 120], index=2)
        vol_drop = st.slider("거래량 감소 기준(%)", 30, 100, 60)
        window = st.slider("신저가 기간", 20, 720, 120)
        threshold = st.slider("지지선 오차(%)", 1.0, 5.0, 3.0)

    # (B) 이동평균선 파라미터 (사용 시 활성화)
    ma_short, ma_long = 5, 20
    if use_ma:
        with st.expander("〰️ 이동평균선 설정", expanded=True):
            ma_short = st.number_input("단기 이평선", value=5)
            ma_long = st.number_input("장기 이평선", value=20)
            st.caption("기본 전략: 현재가가 장기 이평선 위에 있어야 함 (추세 필터)")

    # (C) 호가창 파라미터 (사용 시 활성화)
    ob_ratio = 1.0
    if use_orderbook:
        with st.expander("📊 호가창 설정", expanded=True):
            ob_ratio = st.slider("매도/매수 잔량비", 0.5, 3.0, 1.5, help="매도잔량이 매수잔량보다 몇 배 많아야 하는가?")

    return {
        # 활성화 여부 플래그
        "use_minute": use_minute,
        "use_ma": use_ma,
        "use_orderbook": use_orderbook,
        
        # 파라미터
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
# [Part 2] 핵심 전략 로직 (Modular Logic)
# -> 추후 업데이트 시 이 부분의 함수들만 수정하면 됩니다.
# =========================================================

def logic_low_point(df, config):
    """
    [전략 1] 교과서적 저점 및 거래량 분석
    """
    threshold = config.get('lp_threshold', 0.03)
    vol_drop = config.get('lp_vol_drop', 0.6)
    window = config.get('lp_window', 120)
    vol_ma_days = config.get('lp_vol_ma', 20)
    
    # 지표 계산
    prev_low = df['Low'].shift(1)
    prev2_low = df['Low'].shift(2)
    recent_low_60 = df['Low'].shift(2).rolling(window=60).min()
    vol_ma = df['Volume'].shift(1).rolling(window=vol_ma_days).mean()
    prev_vol = df['Volume'].shift(1)

    # 조건: 단기 바닥 + 지지선 + 거래량감소
    local_min = (prev_low < prev2_low) & (prev_low < df['Low'])
    near_support = abs(prev_low - recent_low_60) / recent_low_60 <= threshold
    is_vol_drop = prev_vol < (vol_ma * vol_drop)
    
    # 필수 여부에 따른 신호 결합
    if config.get('lp_vol_req', True):
        signal = local_min & near_support & is_vol_drop
        msg = f"LowPoint(Vol < {int(vol_drop*100)}%)"
    else:
        signal = local_min & near_support
        # 거래량 조건 만족 여부에 따라 메시지 분기
        msg = np.where(is_vol_drop, f"LowPoint(Vol < {int(vol_drop*100)}%)", "LowPoint(Support Only)")
        
    return signal, msg

def logic_moving_average(df, config):
    """
    [전략 2] 이동평균선 전략
    -> 추후 '골든크로스', '정배열' 등 상세 로직 업데이트는 여기서 진행
    """
    if not config.get('use_ma', False):
        return True, "" # 미사용 시 항상 통과

    s_period = config.get('ma_short', 5)
    l_period = config.get('ma_long', 20)
    
    # 이평선 계산 (이미 데이터에 있을 수 있지만 안전하게 재계산)
    ma_short = df['Close'].rolling(window=s_period).mean()
    ma_long = df['Close'].rolling(window=l_period).mean()
    
    # [현재 전략] 주가가 장기 이평선 위에 있고, 단기 이평선이 상승 중일 때
    # (단순 예시입니다. 나중에 복잡한 로직으로 교체하세요)
    trend_ok = df['Close'] > ma_long
    
    msg = np.where(trend_ok, f"MA(Price > MA{l_period})", "")
    return trend_ok, msg

def logic_order_book(df, config):
    """
    [전략 3] 호가창 분석 전략
    -> 호가 데이터 컬럼이 있는지 확인 후 로직 수행
    """
    if not config.get('use_orderbook', False):
        return True, ""

    # 데이터 컬럼 확인 (예: 'Ask_Rem'(매도잔량), 'Bid_Rem'(매수잔량) 컬럼이 있다고 가정)
    # 실제 데이터 파일의 컬럼명에 맞춰 수정 필요
    required_cols = ['Total_Ask_Size', 'Total_Bid_Size'] # 예시 컬럼명
    
    # 컬럼이 하나라도 없으면 분석 불가 -> Pass (True 반환)
    if not set(required_cols).issubset(df.columns):
        return True, "" 

    ratio = config.get('ob_ratio', 1.5)
    
    # [현재 전략] 총매도잔량이 총매수잔량보다 N배 많아야 함 (바닥권 매집 신호)
    # 0으로 나누기 방지
    bid_size = df['Total_Bid_Size'].replace(0, 1)
    ask_size = df['Total_Ask_Size']
    
    condition = (ask_size / bid_size) >= ratio
    
    msg = np.where(condition, f"OrderBook(Ask/Bid > {ratio})", "")
    return condition, msg


# =========================================================
# [Part 3] 데이터 전처리 및 통합 (Integration)
# =========================================================
def prepare_data(df, config):
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date').reset_index(drop=True)

    # 1. 보조 지표 계산 (상세 로그용)
    df['Day_Chg'] = df['Close'].pct_change() * 100
    df['Vol_Chg'] = df['Volume'].pct_change() * 100
    
    # 2. 각 모듈별 로직 실행
    # (A) 저점 로직
    sig_low, msg_low = logic_low_point(df, config)
    
    # (B) 이평선 로직
    sig_ma, msg_ma = logic_moving_average(df, config)
    
    # (C) 호가창 로직
    sig_ob, msg_ob = logic_order_book(df, config)
    
    # 3. 최종 신호 결합 (AND 조건)
    # 모든 활성화된 전략이 True여야 최종 매수
    df['Buy_Signal'] = sig_low & sig_ma & sig_ob
    
    # 4. 이유(Reason) 메시지 통합
    # 각 로직에서 나온 메시지를 합침. 예: "LowPoint(...) + MA(...) + OB(...)"
    # 벡터화 연산을 위해 list comprehension 대신 numpy 활용 권장하나 문자열 합치기는 apply가 편함
    
    def combine_msgs(row):
        reasons = []
        if row['Msg_Low']: reasons.append(row['Msg_Low'])
        if config.get('use_ma') and row['Msg_MA']: reasons.append(row['Msg_MA'])
        if config.get('use_orderbook') and row['Msg_OB']: reasons.append(row['Msg_OB'])
        return " + ".join(reasons) if reasons else ""

    # 임시 컬럼 생성
    df['Msg_Low'] = msg_low
    df['Msg_MA'] = msg_ma
    df['Msg_OB'] = msg_ob
    
    # 메시지 통합 (신호가 있는 날만 계산하여 속도 최적화)
    df['Reason_Msg'] = ""
    mask = df['Buy_Signal']
    if mask.any():
        df.loc[mask, 'Reason_Msg'] = df[mask].apply(combine_msgs, axis=1)
        
    # 임시 컬럼 삭제
    df.drop(columns=['Msg_Low', 'Msg_MA', 'Msg_OB'], inplace=True)
    
    return df

# =========================================================
# [Part 4] 매매 실행 (Execution) - 기존과 동일
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

    # 지표 계산 기간만큼 스킵 (가장 긴 윈도우 기준)
    # MA 기간 고려
    ma_max = config.get('ma_long', 20) if config.get('use_ma') else 0
    start_idx = max(config.get('lp_window', 120), ma_max) + 2
    if start_idx >= len(df): start_idx = 60 
    
    for i in range(start_idx, len(df)):
        row = df.iloc[i]
        
        # 1. 매도 처리
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

        # 2. 매수 처리
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
    
    # 1. 시장 지수 (기존)
    if 'Market_Close' in chart_df.columns and chart_df['Market_Close'].sum() > 0:
        mkt_plot = mpf.make_addplot(
            chart_df['Market_Close'], color='orange', secondary_y=True, width=1.0, linestyle='--'
        )
        add_plots.append(mkt_plot)

    # 2. [NEW] 이동평균선 시각화 (활성화 시)
    if config and config.get('use_ma'):
        ma_s = config.get('ma_short', 5)
        ma_l = config.get('ma_long', 20)
        # 차트 데이터에 MA 계산 (mplfinance mav 옵션 대신 addplot으로 제어)
        ma_s_line = chart_df['Close'].rolling(window=ma_s).mean()
        ma_l_line = chart_df['Close'].rolling(window=ma_l).mean()
        
        add_plots.append(mpf.make_addplot(ma_s_line, color='fuchsia', width=1.0)) # 단기: 핑크
        add_plots.append(mpf.make_addplot(ma_l_line, color='gold', width=1.2))    # 장기: 골드

    # 3. 매매 마커 (기존)
    buy_markers = [np.nan] * len(chart_df)
    sell_markers = [np.nan] * len(chart_df)
    
    for log in logs:
        # Date 문자열 파싱 (분봉 포맷 대응)
        # 로그의 날짜 문자열(YYYY-MM-DD HH:MM)을 Timestamp로 변환
        date_ts = pd.to_datetime(log['Date'])
        
        # 정확한 인덱스 찾기 (분봉 데이터 시간 매칭)
        if date_ts in chart_df.index:
            idx = chart_df.index.get_loc(date_ts)
            # 중복 인덱스 처리
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

    # 스타일 및 출력
    mc = mpf.make_marketcolors(up='red', down='blue', edge='inherit', wick='inherit', volume='inherit')
    s = mpf.make_mpf_style(marketcolors=mc, figcolor='white', facecolor='white', edgecolor='black', gridstyle=':')
    
    save_path = os.path.join(save_dir, f"{code}_chart.png")
    
    # 타이틀 구성
    title = f"Sim: {code}"
    if config and config.get('use_ma'):
        title += f" | MA({config.get('ma_short')}/{config.get('ma_long')})"
    
    try:
        mpf.plot(
            chart_df, type='candle', volume=True, addplot=add_plots, style=s,
            title=title, figsize=(14, 8), savefig=save_path, tight_layout=True, warn_too_much_data=20000
        )
    except Exception as e:
        print(f"Chart Error {code}: {e}")