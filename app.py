import streamlit as st
import os

# Pinecone
from pinecone import Pinecone
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore

# Gemini (직접 REST 사용)
import google.generativeai as genai


# =========================
# 1. 제목 및 기본 설정
# =========================
st.title("🧠 나만의 AI-agent (Pinecone Ver.)")
st.caption("분석이 필요한 종목에 대해서 AI가 분석해줍니다.")


# =========================
# 2. API 키 설정
# =========================
if "GOOGLE_API_KEY" not in st.secrets or "PINECONE_API_KEY" not in st.secrets:
    st.error("API 키가 설정되지 않았습니다. Streamlit Secrets를 확인하세요.")
    st.stop()

os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]

# Gemini 설정 (REST, 동기)
genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
gemini_model = genai.GenerativeModel("gemini-1.5-flash")

# Pinecone 설정
pc = Pinecone(api_key=st.secrets["PINECONE_API_KEY"])


# =========================
# 3. Pinecone 인덱스 연결
# =========================
index_name = "ai-stock-agent"  # 파인콘 콘솔에 실제 존재해야 함

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001"
)

# ⚠️ from_existing_index 사용하지 않음 (오류 원인)
index = pc.Index(index_name)

vectorstore = PineconeVectorStore(
    index=index,
    embedding=embeddings
)


# =========================
# 4. 사이드바: 종목 추가
# =========================
with st.sidebar:
    st.header("📝 종목 추가하기")
    txt_input = st.text_area("분석할 종목 또는 메모를 입력하세요", height=150)

    if st.button("종목 분석 데이터 저장"):
        if txt_input.strip():
            vectorstore.add_texts([txt_input])
            st.success("Pinecone에 성공적으로 저장되었습니다 💾")
        else:
            st.warning("내용을 입력해주세요.")


# =========================
# 5. 질문하기 (RAG)
# =========================
st.header("🔍 질문하기")
query = st.text_input("무엇이 궁금한가요?")

if st.button("질문 보내기"):
    if not query.strip():
        st.warning("질문을 입력해주세요.")
        st.stop()

    with st.spinner("기억을 검색하고 분석 중입니다..."):
        # 1️⃣ Pinecone에서 관련 문서 검색
        docs = vectorstore.similarity_search(query, k=4)

        if not docs:
            st.warning("참고할 데이터가 없습니다. 먼저 종목을 추가해주세요.")
            st.stop()

        # 2️⃣ Context 구성
        context = "\n\n".join([doc.page_content for doc in docs])

        prompt = f"""
너는 주식 분석 AI 에이전트다.
아래의 정보를 참고해서 질문에 대해 명확하고 간결하게 답변해라.

[참고 정보]
{context}

[질문]
{query}
"""

        # 3️⃣ Gemini 호출 (동기 / REST)
        response = gemini_model.generate_content(prompt)

        # =========================
        # 6. 결과 출력
        # =========================
        st.subheader("🤖 AI의 답변")
        st.write(response.text)

        with st.expander("📚 참고한 소스 보기"):
            for i, doc in enumerate(docs, start=1):
                st.write(f"{i}. {doc.page_content}")
