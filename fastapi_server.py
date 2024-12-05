from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import base64
from io import BytesIO
import torch
import openai
from dotenv import load_dotenv
import os
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# .env 파일 로드
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")  # .env에 저장된 API 키 사용
# CLIP 모델 로드
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

app = FastAPI()

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 텍스트 프롬프트 목록
text_prompts = [
    "Assault incident", "Fight incident", "Burglary in progress",
    "Vandalism observed", "Person fainted", "Person wandering",
    "Trespassing detected", "Illegal dumping", "Robbery happening",
    "Dating violence or harassment", "Kidnapping", "Drunken behavior"
]

# 최근 업로드된 이미지를 저장할 변수
last_image_data = None
last_description = ""
last_confidence = 0.0
last_summary = ""  # GPT-3.5 Turbo로 생성된 요약 저장

# LangChain을 사용하여 GPT-3.5 Turbo 모델로 요약 생성
llm = OpenAI(temperature=0.7)

# 상황 설명을 요약할 프롬프트 템플릿 생성
summary_template = """
상황: {description}에 대한 신뢰도 {confidence:.2f}.
이미지 속의 사람들의 인상착의나 무슨행동을 하고 있는지 요약해주세요
"""
prompt_template = PromptTemplate(input_variables=["description", "confidence"], template=summary_template)

# LLMChain 사용
llm_chain = LLMChain(llm=llm, prompt=prompt_template)

# 루트 엔드포인트: 최근 업로드된 이미지와 설명 반환
@app.get("/")
async def root():
    if last_image_data:
        return {
            "message": "FastAPI server is running",
            "description": last_description,
            "confidence": last_confidence,
            "summary": last_summary,
            "image_data": last_image_data
        }
    else:
        return {"message": "No image uploaded yet"}

@app.post("/upload")
async def upload_image(request: Request):
    data = await request.json()
    img_base64 = data['image']
    image_data = base64.b64decode(img_base64)
    image = Image.open(BytesIO(image_data))

    # CLIP 모델로 상황 설명과 신뢰도 추출
    inputs = processor(text=text_prompts, images=image, return_tensors="pt", padding=True)
    outputs = model(**inputs)
    logits_per_image = outputs.logits_per_image
    probs = logits_per_image.softmax(dim=1)

    max_prob_idx = probs.argmax()
    description = text_prompts[max_prob_idx]
    confidence = probs[0, max_prob_idx].item()

    # LangChain을 사용하여 요약 생성
    try:
        summary = llm_chain.run({"description": description, "confidence": confidence})
    except Exception as e:
        summary = f"요약 생성 실패. 오류: {e}"

    # 업로드된 이미지 정보 저장
    last_image_data = img_base64
    last_description = description
    last_confidence = confidence
    last_summary = summary

    return {
        "description": description,
        "confidence": confidence,
        "summary": summary
    }

# FastAPI 서버 실행
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8502)
