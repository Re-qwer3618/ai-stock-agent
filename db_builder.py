import os
import json
import pandas as pd
import time
from tqdm import tqdm  # 진행률 표시바
from pinecone import Pinecone, ServerlessSpec
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import streamlit as st

# ==========================================
# 1. 설정 (API 키 및 환경)
# ==========================================
# Streamlit Cloud 배포용 (Secrets 사용)
if hasattr(st, "secrets"):
    if "GOOGLE_API_KEY" in st.secrets:
        os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
    if "PINECONE_API_KEY" in st.secrets:
        os.environ["PINECONE_API_KEY"] = st.secrets["PINECONE_API_KEY"]

# 로컬 테스트용 (직접 입력 필요시 주석 해제 후 입력)
# os.environ["GOOGLE_API_KEY"] = "여기에_구글_API키"
# os.environ["PINECONE_API_KEY"] = "여기에_파인콘_API키"

# 인덱스 이름 (app.py와 동일해야 함)
INDEX_NAME = "ai-stock-agent"

# ==========================================
# 2. 초기화 (모델 & DB 연결)
# ==========================================
print("🔌 Pinecone 및 Gemini 모델 연결 중...")
try:
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    index = pc.Index(INDEX_NAME)
    print("✅ 연결 성공!")
except Exception as e:
    print(f"❌ 연결 실패: {e}")
    print("API 키가 올바르게 설정되었는지 확인해주세요.")
    exit()

# ==========================================
# 3. 데이터 로딩 및 처리 함수
# ==========================================
def process_csv(file_path):
    """CSV 파일을 읽어 텍스트 리스트로 변환"""
    data = []
    try:
        df = pd.read_csv(file_path)
        print(f"📂 CSV 로딩: {len(df)}행 발견 ({file_path})")
        
        for idx, row in df.iterrows():
            # 각 컬럼의 이름과 값을 합쳐서 하나의 문장으로 만듦
            # 예: "Date: 2024-01-01, Name: 삼성전자, Close: 70000"
            text_chunks = [f"{col}: {val}" for col, val in row.items()]
            text = ", ".join(text_chunks)
            data.append({"id": f"csv-{idx}", "text": text})
            
    except Exception as e:
        print(f"⚠️ CSV 처리 중 오류: {e}")
    return data

def process_jsonl(file_path):
    """JSONL 파일을 읽어 텍스트 리스트로 변환"""
    data = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            print(f"📂 JSONL 로딩: {len(lines)}행 발견 ({file_path})")
            
            for idx, line in enumerate(lines):
                if not line.strip(): continue
                json_obj = json.loads(line)
                # JSON 객체를 문자열로 변환
                text = json.dumps(json_obj, ensure_ascii=False)
                data.append({"id": f"jsonl-{idx}", "text": text})
                
    except Exception as e:
        print(f"⚠️ JSONL 처리 중 오류: {e}")
    return data

# ==========================================
# 4. 메인 실행 (업로드 로직)
# ==========================================
def main():
    all_data = []
    
    # 1) 파일 읽기
    if os.path.exists("Etc_V1.csv"):
        all_data.extend(process_csv("Etc_V1.csv"))
    else:
        print("⚠️ Etc_V1.csv 파일이 없습니다.")

    if os.path.exists("Etc_V1.jsonl"):
        all_data.extend(process_jsonl("Etc_V1.jsonl"))
    else:
        print("⚠️ Etc_V1.jsonl 파일이 없습니다.")

    if not all_data:
        print("❌ 저장할 데이터가 없습니다. 파일 위치를 확인하세요.")
        return

    print(f"\n🚀 총 {len(all_data)}개의 데이터를 Pinecone에 업로드합니다...")
    
    # 2) 배치 업로드 (100개씩 끊어서 전송 - 안정성 확보)
    batch_size = 100
    
    for i in tqdm(range(0, len(all_data), batch_size), desc="업로드 진행률"):
        batch = all_data[i : i + batch_size]
        vectors = []
        
        for item in batch:
            try:
                # 텍스트 -> 벡터 변환 (Embedding)
                vector_values = embeddings.embed_query(item['text'])
                
                vectors.append({
                    "id": item['id'],
                    "values": vector_values,
                    "metadata": {"text": item['text']}
                })
            except Exception as e:
                print(f"⚠️ 변환 실패 (ID: {item['id']}): {e}")
                continue
        
        # Pinecone에 저장 (Upsert)
        if vectors:
            index.upsert(vectors=vectors)
            
    print("\n🎉 모든 데이터 업로드 완료! 이제 app.py에서 검색할 수 있습니다.")

if __name__ == "__main__":
    main()