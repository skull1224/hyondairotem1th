# streamlit_app.py

import streamlit as st
import requests
from PIL import Image
import base64
from io import BytesIO

st.title("사진 요약 보고서 작성")

# FastAPI 서버에서 최근 업로드된 이미지와 설명 요청
def get_last_uploaded_image():
    # FastAPI 서버의 루트 URL
    fastapi_url = "http://127.0.0.1:8502/"

    # GET 요청을 통해 최근 이미지와 설명 가져오기
    response = requests.get(fastapi_url)
    if response.status_code == 200:
        result = response.json()
        if "image_data" in result:
            description = result["description"]
            confidence = result["confidence"]

            # Base64로 인코딩된 이미지 디코딩
            img_base64 = result["image_data"]
            image_data = base64.b64decode(img_base64)
            image = Image.open(BytesIO(image_data))

            # Streamlit에 이미지와 설명 표시
            st.image(image, caption=f"상황 설명: {description} (신뢰도: {confidence:.2f})")
        else:
            st.write(result["message"])
    else:
        st.write("서버와의 통신에 실패했습니다.")

# 최근 업로드된 이미지와 설명을 가져와서 표시
get_last_uploaded_image()
