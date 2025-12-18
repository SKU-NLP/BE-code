import pandas as pd
import numpy as np
import ast
import re
import os

from sentence_transformers import SentenceTransformer
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI
from dotenv import load_dotenv

# ======================================================
# 0. 기본 설정
# ======================================================
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

app = FastAPI(title="맥락 기반 대학 추천 API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ======================================================
# 1. 텍스트 정규화
# ======================================================
def normalize_text(text: str) -> str:
    text = str(text)
    text = re.sub(r"[,\u00b7・]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

# ======================================================
# 2. 텍스트 + 임베딩 DB 로딩 (핵심 수정 부분)
# ======================================================
print("📂 텍스트 + 임베딩 DB 로딩 중...")

# ① 원본 텍스트 데이터
text_df = pd.read_csv("language.csv").fillna("")
text_df.columns = text_df.columns.str.strip()

# ② 임베딩 데이터
emb_df = pd.read_csv("test_language.csv").fillna("")
emb_df["embedding"] = emb_df["embedding"].apply(
    lambda x: np.array(ast.literal_eval(x))
)

# ③ 행 수 검증
if len(text_df) != len(emb_df):
    raise ValueError("❌ language.csv 와 test_language.csv 행 개수가 다릅니다.")

# ④ 결합 (행 순서 기준)
df = text_df.copy()
df["embedding"] = emb_df["embedding"].values

# ⑤ 지역 컬럼 통합
if "소재지(상세)" in df.columns:
    df["지역"] = df["소재지(상세)"]
elif "소재지" in df.columns:
    df["지역"] = df["소재지"]
else:
    df["지역"] = ""

# ⑥ 임베딩 행렬
corpus_embeddings = np.vstack(df["embedding"].values)

# ⑦ 모델 로딩
model = SentenceTransformer("intfloat/multilingual-e5-base")

print("✅ DB 로딩 완료")

# ======================================================
# 3. 맥락 기반 검색 (임베딩 유사도)
# ======================================================
def search_major_contextual(user_query: str, top_k: int = 5):
    query = "query: " + normalize_text(user_query)

    query_embedding = model.encode(
        query,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    df_work = df.copy()
    df_work["score"] = np.dot(corpus_embeddings, query_embedding)

    results = df_work.sort_values("score", ascending=False).head(top_k)

    # 완전 무관 질문 방어
    if results.iloc[0]["score"] < 0.65:
        return pd.DataFrame()

    return results

# ======================================================
# 4. GPT 하이브리드 프롬프트
# ======================================================
def build_hybrid_prompt(user_query: str, results_df: pd.DataFrame):
    context = ""

    if not results_df.empty:
        for _, row in results_df.iterrows():
            context += (
                f"대학: {row['대학명']} / "
                f"단과대학: {row['단과대학']} / "
                f"학과: {row['학과명']} / "
                f"지역: {row['지역']} / "
                f"특징: {row['학과특성']} / "
                f"계열: {row['표준분류계열(중)']}\n"
            )

    return f"""
너는 한국의 대학 입시 및 진로 전문 상담 챗봇이다.

[사용자 질문]
{user_query}

[내부 참고 데이터]
{context if context else "직접적인 학과 매칭 데이터는 없지만, 유사 계열을 기준으로 추천해라."}

[답변 지침]
1. 내부 데이터가 있다면 반드시 근거로 활용해라.
2. 학과 성격, 진로 방향, 취업 분야를 함께 설명해라.
3. 수험생의 상황을 고려한 현실적인 조언을 포함해라.
4. 자연스럽고 상담하듯 답변해라.
"""

# ======================================================
# 5. API 엔드포인트
# ======================================================
class ChatRequest(BaseModel):
    question: str

@app.post("/chat")
def chat(req: ChatRequest):
    results = search_major_contextual(req.question)
    prompt = build_hybrid_prompt(req.question, results)

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "너는 입시 데이터와 상식을 결합해 상담하는 전문가다."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.5
    )

    return {
        "answer": response.choices[0].message.content,
        "matched_count": len(results)
    }
