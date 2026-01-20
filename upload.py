import os
import time
from pinecone import Pinecone
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import streamlit as st # secrets를 가져오기 위해 사용 (또는 직접 키 입력)

# ==========================================
# 1. 설정 (API 키 및 모델 준비)
# ==========================================
# 주의: Streamlit Cloud가 아닌 로컬에서 돌릴 때는 secrets.toml 파일이 있어야 합니다.
# 만약 에러가 나면 아래에 직접 키를 입력하세요. os.environ["..."] = "sk-..."
if "GOOGLE_API_KEY" in st.secrets:
    os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
    os.environ["PINECONE_API_KEY"] = st.secrets["PINECONE_API_KEY"]

# 임베딩 모델 준비 (app.py와 똑같은 모델을 써야 찾을 수 있어요!)
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# Pinecone 연결
pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
index_name = "ai-stock-agent"
index = pc.Index(index_name)

# ==========================================
# 2. 저장할 데이터 준비 (여기에 내용을 적으세요)
# ==========================================
# 예시: 워렌 버핏의 투자 명언과 주식 기초 지식
knowledge_list = [
    "주식 시장은 인내심 없는 사람의 돈을 인내심 있는 사람에게 옮기는 도구다. (워렌 버핏)",
    "공포에 사서 환희에 팔아라. 남들이 욕심을 낼 때 두려워하고, 남들이 두려워할 때 욕심을 내야 한다.",
    "RSI(상대강도지수)가 30 이하이면 과매도 구간으로 간주하여 매수 관점으로 볼 수 있다.",
    "골든크로스는 단기 이동평균선이 장기 이동평균선을 뚫고 올라가는 것으로, 강력한 매수 신호다.",
    "PER(주가수익비율)이 낮으면 기업 가치 대비 주가가 저평가되어 있다는 뜻일 수 있다."
]

print(f"📚 총 {len(knowledge_list)}개의 지식을 저장할 준비를 합니다...")

# ==========================================
# 3. 데이터 변환 및 저장 (업로드)
# ==========================================
vectors_to_upsert = []

for i, text in enumerate(knowledge_list):
    try:
        # 1) 글자를 숫자로 변환 (임베딩)
        vector = embeddings.embed_query(text)
        
        # 2) Pinecone에 넣을 데이터 포맷 만들기
        # id: 데이터의 주민등록번호 (유니크해야 함)
        # values: 숫자로 변환된 벡터
        # metadata: 원래 글자 (나중에 다시 꺼내볼 때 필요)
        data = {
            "id": f"knowledge-{i}",  # ID는 겹치지 않게 설정
            "values": vector,
            "metadata": {"text": text}
        }
        vectors_to_upsert.append(data)
        print(f"✅ 변환 완료: {text[:20]}...")
        
    except Exception as e:
        print(f"❌ 변환 실패: {e}")

# 3) Pinecone에 한 번에 저장 (Upsert)
if vectors_to_upsert:
    index.upsert(vectors=vectors_to_upsert)
    print("\n🎉 모든 데이터가 Pinecone 도서관에 성공적으로 저장되었습니다!")
else:
    print("\n⚠️ 저장할 데이터가 없습니다.")