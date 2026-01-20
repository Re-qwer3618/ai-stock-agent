import os
import streamlit as st
from pinecone import Pinecone

# ==========================================
# 1. 설정 (API 키)
# ==========================================
# secrets.toml이나 환경변수에서 키를 가져옵니다.
if hasattr(st, "secrets"):
    if "PINECONE_API_KEY" in st.secrets:
        os.environ["PINECONE_API_KEY"] = st.secrets["PINECONE_API_KEY"]

# 직접 입력이 필요한 경우 아래 주석을 풀고 입력하세요.
# os.environ["PINECONE_API_KEY"] = "여기에_파인콘_키_입력"

# ==========================================
# 2. Pinecone 연결
# ==========================================
INDEX_NAME = "ai-stock-agent"

print(f"🔌 Pinecone 인덱스 '{INDEX_NAME}'에 연결 중...")
try:
    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    index = pc.Index(INDEX_NAME)
    
    # 현재 데이터 개수 확인
    stats = index.describe_index_stats()
    print(f"📊 현재 저장된 데이터 개수: {stats['total_vector_count']}개")

except Exception as e:
    print(f"❌ 연결 실패: {e}")
    exit()

# ==========================================
# 3. 삭제 명령 (주의하세요!)
# ==========================================
confirm = input("💥 정말로 모든 데이터를 삭제하시겠습니까? (yes/no): ")

if confirm.lower() == "yes":
    try:
        # [핵심] namespace를 지정하지 않았다면 기본 공간의 모든 데이터를 지웁니다.
        index.delete(delete_all=True)
        print("\n🧹 싹~ 다 지웠습니다! (초기화 완료)")
        
        # 확인 사살
        time.sleep(2)
        stats = index.describe_index_stats()
        print(f"📊 삭제 후 데이터 개수: {stats['total_vector_count']}개")
        
    except Exception as e:
        print(f"⚠️ 삭제 중 오류 발생: {e}")
else:
    print("휴~ 삭제를 취소했습니다.")