import streamlit as st
import os
import FinanceDataReader as fdr  # 주식 데이터 라이브러리
import datetime

# Pinecone & Gemini
from pinecone import Pinecone
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore
import google.generativeai as genai
import plotly.graph_objects as go  # 멋진 차트 그리는 도구
from langchain.schema import Document  # Document 클래스 임포트 추가

# =========================
# 1. 설정 및 초기화
# =========================
st.set_page_config(page_title="AI 주식 분석 에이전트", layout="wide")

st.title("📈 실시간 AI 주식 분석기 (Hybrid Ver.)")
st.caption("Pinecone의 '투자 이론'과 실시간 '시장 데이터'를 결합해 분석합니다.")

# API 키 확인
if "GOOGLE_API_KEY" not in st.secrets or "PINECONE_API_KEY" not in st.secrets:
    st.error("API 키가 설정되지 않았습니다. Streamlit Secrets를 확인하세요.")
    st.stop()

os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
gemini_model = genai.GenerativeModel("gemini-1.5-flash")
pc = Pinecone(api_key=st.secrets["PINECONE_API_KEY"])

# =========================
# 2. 함수: 실시간 주식 데이터 가져오기 (Naver 증권 기반)
# =========================
@st.cache_data
def get_stock_dict():
    """
    한국거래소(KRX)의 모든 종목 이름과 코드를 가져와서
    '이름': '코드' 형태의 전화번호부를 만듭니다.
    """
    # KRX 전체 리스트 가져오기 (시간이 조금 걸릴 수 있음)
    df = fdr.StockListing('KRX')
    # 이름과 코드를 짝지어서 딕셔너리로 변환 (예: {'삼성전자': '005930', ...})
    stock_dict = dict(zip(df['Name'], df['Code']))
    return stock_dict

def get_stock_data(code):
    """
    Finance-DataReader를 이용해 특정 종목의 최신 주가 정보를 가져옵니다.
    """
    try:
        # 최근 5일치 데이터만 가져옴 (오늘 날짜 확인용)
        df = fdr.DataReader(code, '2024') 
        if df.empty:
            return None
        
        last_row = df.iloc[-1]
        prev_row = df.iloc[-2] if len(df) > 1 else last_row
        
        # 데이터 정리
        data = {
            "current_price": int(last_row['Close']),
            "change_rate": round(((last_row['Close'] - prev_row['Close']) / prev_row['Close']) * 100, 2),
            "volume": int(last_row['Volume']),
            "date": last_row.name.strftime("%Y-%m-%d")
        }
        return data
    except Exception as e:
        return None

def plot_chart(code, name):
    """
    1년치 주가 데이터를 가져와서 캔들 차트(봉차트)를 그립니다.
    """
    try:
        # 1년치 데이터 가져오기 (오늘부터 365일 전까지)
        start_date = (datetime.datetime.now() - datetime.timedelta(days=365)).strftime("%Y-%m-%d")
        df = fdr.DataReader(code, start_date)
        
        if df.empty:
            st.error("차트 데이터를 불러올 수 없습니다.")
            return

        # 차트 그리기 (Candlestick)
        fig = go.Figure(data=[go.Candlestick(
            x=df.index,
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            increasing_line_color='red',  # 상승은 빨강
            decreasing_line_color='blue'  # 하락은 파랑
        )])

        # 차트 꾸미기 (제목, 크기 등)
        fig.update_layout(
            title=f"{name} ({code}) 일봉 차트",
            xaxis_title="날짜",
            yaxis_title="주가",
            height=400,
            margin=dict(l=20, r=20, t=50, b=20)
        )
        
        # Streamlit에 차트 출력
        st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.error(f"차트 생성 중 오류 발생: {e}")

# =========================
# 3. Pinecone 인덱스 연결
# =========================
index_name = "ai-stock-agent"

# API 키를 환경변수에 등록
os.environ["PINECONE_API_KEY"] = st.secrets["PINECONE_API_KEY"]

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001"
)

# Pinecone 연결 (디버깅 정보 추가)
try:
    vectorstore = PineconeVectorStore.from_existing_index(
        index_name=index_name,
        embedding=embeddings
    )
    
    # 디버깅: 인덱스 상태 확인
    index = pc.Index(index_name)
    stats = index.describe_index_stats()
    st.sidebar.success(f"✅ Pinecone 연결 성공: {stats.get('total_vector_count', 0)}개 벡터")
except Exception as e:
    st.sidebar.error(f"❌ Pinecone 연결 실패: {e}")
    vectorstore = None

# =========================
# 4. 메인: 실시간 분석 파트
# =========================
st.divider()
col1, col2 = st.columns([1, 2])

# 전화번호부(종목 리스트) 불러오기
stock_dict = get_stock_dict()

with col1:
    st.subheader("1. 종목 설정")
    
    # 텍스트 입력 대신 '선택 상자(Selectbox)' 사용
    stock_name = st.selectbox(
        "종목을 선택하세요", 
        options=stock_dict.keys(),
        index=list(stock_dict.keys()).index("삼성전자") if "삼성전자" in stock_dict else 0
    )
    
    # 선택한 이름으로 코드 찾기 (자동 변환)
    stock_code = stock_dict[stock_name] 

    st.write(f"📌 종목코드: {stock_code}")

    # 실시간 데이터 가져오기
    realtime_data = None
    if stock_code:
        realtime_data = get_stock_data(stock_code)
        if realtime_data:
            st.success(f"✅ {stock_name} 데이터 수신 성공")
            st.metric(
                label="현재가", 
                value=f"{realtime_data['current_price']:,}원", 
                delta=f"{realtime_data['change_rate']}%"
            )

            # 차트 그리기 추가
            with st.expander("📊 1년 주가 차트 보기 (클릭)", expanded=True):
                plot_chart(stock_code, stock_name)
        else:
            st.error("데이터를 가져올 수 없습니다.")

with col2:
    st.subheader("2. AI 심층 분석 요청")
    query = st.text_input("구체적으로 무엇이 궁금한가요?", "현재 차트 흐름과 매매 전략을 분석해줘")

    if st.button("🚀 분석 시작"):
        if not realtime_data:
            st.warning("먼저 유효한 종목코드를 입력해주세요.")
            st.stop()
            
        with st.spinner(f"'{stock_name}'의 데이터를 분석하고 교과서를 뒤적이는 중..."):
            
            # 1️⃣ RAG: 질문과 관련된 투자 이론(Textbook) 검색 (수정된 부분)
            try:
                # 쿼리를 임베딩으로 변환
                query_embedding = embeddings.embed_query(query)
                
                # Pinecone에 직접 쿼리
                index = pc.Index(index_name)
                results = index.query(
                    vector=query_embedding,
                    top_k=3,
                    include_metadata=True,
                    namespace=""  # 기본 namespace 사용
                )
                
                # 결과를 LangChain Document 형식으로 변환
                docs = [
                    Document(
                        page_content=match.get('metadata', {}).get('text', ''),
                        metadata=match.get('metadata', {})
                    )
                    for match in results.get('matches', [])
                ]
                
                textbook_context = "\n".join([doc.page_content for doc in docs]) if docs else "특별한 저장된 이론 없음."
                
            except Exception as e:
                st.warning(f"투자 이론 검색 중 오류 발생: {e}")
                textbook_context = "투자 이론을 불러올 수 없습니다. 실시간 데이터만으로 분석합니다."

            # 2️⃣ Prompt Engineering: [실시간 데이터] + [투자 이론] 결합
            prompt = f"""
당신은 '수석 주식 애널리스트'입니다. 아래 제공된 [실시간 시장 데이터]와 [투자 이론(Textbook)]을 종합하여 분석 보고서를 작성하세요.

### 1. 분석 대상
- 종목명: {stock_name} ({stock_code})
- 기준일: {realtime_data['date']}

### 2. 실시간 시장 데이터 (Fact)
- 현재가: {realtime_data['current_price']:,}원
- 전일 대비 등락률: {realtime_data['change_rate']}%
- 거래량: {realtime_data['volume']:,}주

### 3. 참고할 투자 이론 (Knowledge Base)
{textbook_context}

### 4. 사용자 질문
"{query}"

### 5. 답변 작성 가이드
- **구조:** [시장 현황 요약] -> [이론적 분석] -> [리스크 요인] -> [최종 결론] 순으로 작성.
- **톤앤매너:** 전문적이지만 이해하기 쉽게(초등학생도 이해 가능하게).
- **필수:** 투자의견(매수/매도/관망)을 낼 때는 반드시 위 [투자 이론]이나 [시장 데이터]를 근거로 들 것.
"""
            # 3️⃣ Gemini 호출
            response = gemini_model.generate_content(prompt)
            
            # 4️⃣ 결과 출력
            st.markdown(response.text)
            
            # (옵션) 참고한 이론 보여주기
            with st.expander("📚 분석에 참고한 '투자 교과서' 내용 보기"):
                st.write(textbook_context)