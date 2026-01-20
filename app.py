import streamlit as st
import os
import datetime
import FinanceDataReader as fdr  # 주식 데이터 라이브러리
import plotly.graph_objects as go  # 멋진 차트 그리는 도구

# Pinecone & Gemini 관련 라이브러리
from pinecone import Pinecone
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai

# ==========================================
# 1. 페이지 설정 및 초기화
# ==========================================
st.set_page_config(page_title="AI 주식 분석 에이전트", layout="wide")

st.title("📈 실시간 AI 주식 분석기 (Pro Ver.)")
st.caption("당신의 Pinecone DB(투자 전략)와 실시간 시장 데이터를 결합해 분석합니다.")

# ---------------------------------------------------------
# [핵심] 1. 사용자 로그인 (Google API Key 입력) - 비용 절감용
# ---------------------------------------------------------
with st.sidebar:
    st.header("🔐 로그인")
    user_api_key = st.text_input(
        "Google API Key를 입력하세요", 
        type="password", 
        help="https://aistudio.google.com/ 에서 무료로 발급 가능합니다."
    )
    st.markdown("---")
    st.info("💡 Pinecone DB는 개발자가 제공합니다.")

# 키가 없으면 여기서 멈춤 (앱 보호)
if not user_api_key:
    st.warning("👈 왼쪽 사이드바에 Google API Key를 입력해주세요.")
    st.stop()

# ---------------------------------------------------------
# [핵심] 2. 환경 설정 (Google은 사용자 키, Pinecone은 개발자 키)
# ---------------------------------------------------------
# 1) Google 설정
os.environ["GOOGLE_API_KEY"] = user_api_key # 랭체인을 위해 환경변수 설정
genai.configure(api_key=user_api_key)       # 제미나이 설정

# 2) Pinecone 설정 (secrets.toml에서 가져옴)
if "PINECONE_API_KEY" not in st.secrets:
    st.error("설정 오류: Pinecone API 키가 secrets.toml에 없습니다.")
    st.stop()

# ==========================================
# 3. 모델 및 DB 연결 준비
# ==========================================
# (1) Gemini 모델 준비
gemini_model = genai.GenerativeModel("gemini-2.0-flash")

# (2) Pinecone 연결
index_name = "ai-stock-agent"
pc = Pinecone(api_key=st.secrets["PINECONE_API_KEY"])
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

pinecone_index = None
try:
    pinecone_index = pc.Index(index_name)
    # 연결 확인용 (사이드바에 표시)
    stats = pinecone_index.describe_index_stats()
    st.sidebar.success(f"✅ DB 연결됨 ({stats.get('total_vector_count', 0)}개 데이터)")
except Exception as e:
    st.sidebar.error(f"❌ DB 연결 실패: {e}")

# ==========================================
# 4. 기능 함수 정의 (캐싱 & 데이터 수집)
# ==========================================

# [중요] 똑똑한 비서 함수 (캐싱 적용: 10분간 기억)
@st.cache_data(ttl=600)
def ask_gemini(prompt_text):
    """Gemini에게 질문을 던지고 답변을 받아오는 함수 (비용 절감)"""
    try:
        response = gemini_model.generate_content(prompt_text)
        return response.text
    except Exception as e:
        return f"AI 분석 중 오류 발생: {e}"

@st.cache_data
def get_stock_dict():
    """KRX 종목 리스트 가져오기"""
    df = fdr.StockListing('KRX')
    stock_dict = dict(zip(df['Name'], df['Code']))
    return stock_dict

def get_stock_data(code):
    """특정 종목의 최신 주가 정보 가져오기"""
    try:
        df = fdr.DataReader(code, '2024') 
        if df.empty: return None
        
        last_row = df.iloc[-1]
        prev_row = df.iloc[-2] if len(df) > 1 else last_row
        
        return {
            "current_price": int(last_row['Close']),
            "change_rate": round(((last_row['Close'] - prev_row['Close']) / prev_row['Close']) * 100, 2),
            "volume": int(last_row['Volume']),
            "date": last_row.name.strftime("%Y-%m-%d")
        }
    except:
        return None

def plot_chart(code, name):
    """캔들 차트 그리기"""
    try:
        start_date = (datetime.datetime.now() - datetime.timedelta(days=365)).strftime("%Y-%m-%d")
        df = fdr.DataReader(code, start_date)
        if df.empty: return

        fig = go.Figure(data=[go.Candlestick(
            x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'],
            increasing_line_color='red', decreasing_line_color='blue'
        )])
        fig.update_layout(title=f"{name} ({code}) 일봉 차트", height=400, xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)
    except:
        st.error("차트 로딩 실패")

# ==========================================
# 5. 메인 화면 구성
# ==========================================
st.divider()
col1, col2 = st.columns([1, 2])

# (1) 종목 선택 영역
stock_dict = get_stock_dict()
with col1:
    st.subheader("1. 종목 선택")
    stock_name = st.selectbox("분석할 종목", options=stock_dict.keys(), index=list(stock_dict.keys()).index("삼성전자") if "삼성전자" in stock_dict else 0)
    stock_code = stock_dict[stock_name]

    realtime_data = get_stock_data(stock_code)
    if realtime_data:
        st.metric(label="현재가", value=f"{realtime_data['current_price']:,}원", delta=f"{realtime_data['change_rate']}%")
        plot_chart(stock_code, stock_name)
    else:
        st.error("데이터 수신 실패")

# (2) AI 분석 영역
with col2:
    st.subheader("2. AI 전략 분석")
    query = st.text_input("궁금한 점을 물어보세요", "현재 차트 흐름과 보유한 전략을 기반으로 매매 의견 줘")

    if st.button("🚀 AI 분석 실행"):
        if not realtime_data:
            st.warning("종목 데이터가 없습니다.")
            st.stop()
            
        with st.spinner(f"Pinecone DB에서 전략을 검색하고 분석 중입니다..."):
            
            # 1️⃣ RAG: Pinecone에서 관련 전략/지식 검색
            rag_context = "관련된 저장된 전략 없음."
            if pinecone_index:
                try:
                    # 질문을 벡터로 변환
                    query_embedding = embeddings.embed_query(query)
                    # Pinecone 검색 (Top 3)
                    results = pinecone_index.query(
                        vector=query_embedding,
                        top_k=3,
                        include_metadata=True
                    )
                    # 검색된 내용 합치기
                    texts = [match['metadata']['text'] for match in results.get('matches', []) if 'text' in match['metadata']]
                    if texts:
                        rag_context = "\n\n".join(texts)
                except Exception as e:
                    st.warning(f"DB 검색 중 오류: {e}")

            # 2️⃣ 프롬프트 작성
            prompt = f"""당신은 퀀트 투자 전문가입니다. 아래 데이터를 바탕으로 분석하세요.

[분석 대상] {stock_name} ({stock_code}), 기준일: {realtime_data['date']}
[시장 데이터] 현재가: {realtime_data['current_price']}원, 등락률: {realtime_data['change_rate']}%, 거래량: {realtime_data['volume']}

[참고 전략 및 지식 (DB 검색 결과)]
{rag_context}

[사용자 질문]
"{query}"

[답변 가이드]
1. 시장 현황을 간단히 요약할 것.
2. 위 [참고 전략]에 나온 내용과 현재 차트 상황을 연결해서 분석할 것. (전략이 없으면 일반적인 기술적 분석 수행)
3. 구체적인 매매 근거를 댈 것.
4. 초등학생도 이해할 수 있게 쉽고 명확하게 설명할 것.
"""
            # 3️⃣ AI 호출 (캐싱된 함수 사용)
            result_text = ask_gemini(prompt)
            st.markdown(result_text)
            
            # (선택) 참고한 자료 보여주기
            with st.expander("📚 참고한 DB 전략 보기"):
                st.write(rag_context)
