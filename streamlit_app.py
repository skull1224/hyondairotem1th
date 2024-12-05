import streamlit as st
import requests
from PIL import Image
import base64
from io import BytesIO
from streamlit_option_menu import option_menu
import cv2

# FastAPI 서버 URL
fastapi_url = "http://127.0.0.1:8502/upload"
# RTSP 스트림 URL
RTSP_URL = "rtsp://127.0.0.1:8554"
# 상단 옵션 메뉴
selected = option_menu(
    None,
    ["Home", "Upload", "Stream"],
    icons=['house', 'cloud-upload', 'camera-video'],
    menu_icon="cast",
    default_index=0,
    orientation="horizontal",
    styles={
        "container": {"padding": "0!important", "background-color": "#fafafa"},
        "icon": {"color": "orange", "font-size": "25px"},
        "nav-link": {"font-size": "20px", "text-align": "center", "margin": "0px", "--hover-color": "#eee"},
        "nav-link-selected": {"background-color": "green"},
    },
)


# Home 메뉴
if selected == "Home":
    st.title("Welcome to the Image Summary Report!")
    st.write("이 애플리케이션은 웹캠을 사용하여 이미지를 캡처하고 분석하여 상황 설명을 제공합니다.")
    st.write("상단 메뉴를 통해 기능을 탐색하세요.")

# Upload 메뉴
elif selected == "Upload":
    st.title("사진 업로드 및 분석")
    image_file = st.file_uploader("이미지를 업로드하세요", type=["jpg", "jpeg", "png"])

    if image_file:
        # 업로드된 이미지를 Base64로 인코딩
        img = Image.open(image_file)
        buffered = BytesIO()
        img.save(buffered, format="JPEG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode()

        # FastAPI 서버로 POST 요청
        payload = {"image": img_base64}
        response = requests.post(fastapi_url, json=payload)

        if response.status_code == 200:
            result = response.json()
            description = result["description"]
            confidence = result["confidence"]
            summary = result["summary"]

            # 결과 표시
            st.image(img, caption=f"상황 설명: {description} (신뢰도: {confidence:.2f})")
            st.write("상황 요약:")
            st.success(summary)
        else:
            st.error("FastAPI 서버와 통신 실패")

# Stream 메뉴
elif selected == "Stream":
    st.title("실시간 스트림 분석")

    # RTSP URL (Jetson Nano의 실시간 스트림 URL)
    RTSP_URL = "rtsp://127.0.0.1:8554"  # Jetson Nano에서 송출하는 RTSP URL

    # FastAPI 서버 URL
    fastapi_url = "http://127.0.0.1:8502/upload"

    def read_rtsp_stream():
        # OpenCV를 사용하여 RTSP 스트림 읽기
        cap = cv2.VideoCapture(RTSP_URL)
        if not cap.isOpened():
            st.error("RTSP 스트림에 연결할 수 없습니다.")
            return

        stframe = st.empty()  # Streamlit의 빈 프레임
        while True:
            ret, frame = cap.read()
            if not ret:
                st.warning("스트림 중단됨")
                break

            # 스트림을 Streamlit에 표시
            stframe.image(frame, channels="BGR", caption="Jetson Nano 실시간 스트림")

            # 특정 프레임 분석 버튼
            if st.button("현재 프레임 분석"):
                _, buffer = cv2.imencode('.jpg', frame)
                img_base64 = base64.b64encode(buffer).decode()
                payload = {"image": img_base64}

                # FastAPI 서버로 프레임 데이터 전송
                response = requests.post(fastapi_url, json=payload)
                if response.status_code == 200:
                    result = response.json()
                    description = result["description"]
                    confidence = result["confidence"]
                    st.success(f"분석 결과: {description} (신뢰도: {confidence:.2f})")
                else:
                    st.error("FastAPI 서버와 통신 실패")

        cap.release()

    # Jetson Nano 스트림 처리 버튼
    if st.button("Jetson 스트림 시작"):
        read_rtsp_stream()
