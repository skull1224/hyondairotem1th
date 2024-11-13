# fastapi_server.py

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import base64
from io import BytesIO
import torch

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

# 루트 엔드포인트: 최근 업로드된 이미지와 설명 반환
@app.get("/")
async def root():
    if last_image_data:
        return {
            "message": "FastAPI server is running",
            "description": last_description,
            "confidence": last_confidence,
            "image_data": last_image_data
        }
    else:
        return {"message": "No image uploaded yet"}

@app.post("/upload")
async def upload_image(request: Request):
    global last_image_data, last_description, last_confidence
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

    # 최근 이미지와 설명을 전역 변수에 저장
    last_image_data = img_base64
    last_description = description
    last_confidence = confidence

    return {
        "description": description,
        "confidence": confidence,
        "image_data": img_base64  # Base64로 인코딩된 이미지 데이터 반환
    }

# FastAPI 서버 실행
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8502)
