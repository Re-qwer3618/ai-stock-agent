import streamlit as st
import os
from pinecone import Pinecone
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.retrieval_qa.base import RetrievalQA


# 1. 제목 및 설정
st.title("🧠 나만의 세컨드 브레인 (Pinecone Ver.)")
st.caption("기억하고 싶은 내용을 입력하면, AI가 기억했다가 대답해줍니다.")

# 2. API 키 설정 (스트림릿 클라우드 비밀보관소에서 가져옴)
# 로컬에서 테스트할 땐 에러가 날 수 있으니 배포 후 작동을 권장합니다.
if "GOOGLE_API_KEY" in st.secrets and "PINECONE_API_KEY" in st.secrets:
    os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
    pc = Pinecone(api_key=st.secrets["PINECONE_API_KEY"])
else:
    st.error("API 키가 설정되지 않았습니다. Streamlit Secrets를 확인하세요.")
    st.stop()

# 3. Pinecone 인덱스 연결
index_name = "second-brain" # 파인콘 홈페이지에서 만든 이름과 같아야 함!
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# 벡터 저장소 연결
vectorstore = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)


# 4. 사이드바: 기억 입력하기
with st.sidebar:
    st.header("📝 기억 추가하기")
    txt_input = st.text_area("기억할 내용을 입력하세요", height=150)
    if st.button("기억하기"):
        if txt_input:
            # 텍스트를 벡터로 변환해서 Pinecone에 저장 (Upsert)
            vectorstore.add_texts([txt_input])
            st.success("성공적으로 기억했습니다! 💾")
        else:
            st.warning("내용을 입력해주세요.")

# 5. 메인 화면: 질문하기
st.header("🔍 질문하기")
query = st.text_input("무엇이 궁금한가요?")

if st.button("질문 보내기"):
    if query:
        with st.spinner("기억을 뒤지는 중..."):
            llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)
            
            # RAG 체인 생성 (검색 -> 답변)
            qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=vectorstore.as_retriever(),
                return_source_documents=True
            )
            
            result = qa_chain.invoke({"query": query})
            st.write("🤖 **AI의 답변:**")
            st.write(result["result"])
            
            # 근거 자료 보여주기 (옵션)
            with st.expander("참고한 기억 보기"):
                for doc in result["source_documents"]:
                    st.write(f"- {doc.page_content}")