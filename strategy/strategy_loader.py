import os
import glob
import time
from tqdm import tqdm
from pinecone import Pinecone
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
import streamlit as st

# ==========================================
# 1. 설정 (API 키)
# ==========================================
# 로컬에서 실행 시 secrets.toml을 자동으로 찾습니다.
# 만약 에러가 나면 아래 주석 풀고 직접 키 입력하세요.
# os.environ["GOOGLE_API_KEY"] = "내_구글_키"
# os.environ["PINECONE_API_KEY"] = "내_파인콘_키"

if "GOOGLE_API_KEY" in st.secrets:
    os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
    os.environ["PINECONE_API_KEY"] = st.secrets["PINECONE_API_KEY"]

genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

# ==========================================
# 2. 준비 (모델 연결)
# ==========================================
INDEX_NAME = "ai-stock-agent"

embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
gemini = genai.GenerativeModel("gemini-2.0-flash")
pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
index = pc.Index(INDEX_NAME)

# ==========================================
# 3. 핵심 기능: 코드를 자연어로 번역하기
# ==========================================
def summarize_code(filename, code_content):
    """
    Gemini에게 파이썬 코드를 주면, 어떤 전략인지 한글로 설명해달라고 시킴
    """
    prompt = f"""
    당신은 퀀트 트레이딩 전문가입니다. 아래 파이썬 코드를 분석해서 투자 전략을 일반인도 알기 쉽게 요약해주세요.
    
    파일명: {filename}
    코드 내용:
    {code_content}
    
    [작성 양식]
    1. 전략 이름: (파일명 기반으로 작성)
    2. 매수 조건: (언제 사는지 구체적으로)
    3. 매도 조건: (언제 파는지 구체적으로)
    4. 특징 및 리스크: (이 전략의 장단점)
    """
    
    response = gemini.generate_content(prompt)
    return response.text

# ==========================================
# 4. 메인 실행 (파일 읽어서 업로드)
# ==========================================
def main():
    # 현재 폴더에 있는 모든 .py 파일 찾기 (전략 파일들)
    strategy_files = glob.glob("Cases_*.py") + glob.glob("Strategy_*.py")
    
    if not strategy_files:
        print("❌ 전략 파일(.py)을 찾을 수 없습니다.")
        return

    print(f"🚀 총 {len(strategy_files)}개의 전략 파일을 분석하고 DB에 저장합니다...")

    vectors_to_upsert = []

    for file_path in tqdm(strategy_files, desc="분석 중"):
        try:
            # 1) 파일 읽기
            with open(file_path, "r", encoding="utf-8") as f:
                code_content = f.read()

            # 2) Gemini가 코드를 '해설서'로 번역 (여기가 핵심!)
            summary_text = summarize_code(file_path, code_content)
            
            # 3) 해설서를 벡터로 변환 (Embedding)
            # 검색을 위해 '요약된 내용'을 벡터화합니다.
            vector = embeddings.embed_query(summary_text)

            # 4) 저장할 데이터 패키징
            vectors_to_upsert.append({
                "id": f"strategy-{os.path.basename(file_path)}", # ID는 파일명으로
                "values": vector,
                "metadata": {
                    "text": summary_text,       # 검색되면 보여줄 해설
                    "source_code": code_content # 원본 코드도 같이 저장 (참고용)
                }
            })
            
            # 너무 빨리 요청하면 에러 날 수 있으니 살짝 쉬기
            time.sleep(1)

        except Exception as e:
            print(f"⚠️ {file_path} 처리 실패: {e}")

    # 5) Pinecone에 저장
    if vectors_to_upsert:
        index.upsert(vectors=vectors_to_upsert)
        print(f"\n🎉 성공! {len(vectors_to_upsert)}개의 전략이 '자연어'로 DB에 저장되었습니다.")
    else:
        print("저장할 데이터가 없습니다.")

if __name__ == "__main__":
    main()