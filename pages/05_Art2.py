import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.cluster import KMeans

# 페이지 기본 설정
st.set_page_config(
    page_title="AI 보석십자수 디자이너",
    page_icon="💎",
    layout="wide"
)

st.title("💎 AI 보석십자수 디자이너 프로젝트")
st.markdown("### **정보(k-평균 군집화와 압축) × 미술(보석십자수 도안 디자인) 융합 수업**")
st.write("인공지능을 이용해 명화의 색상과 해상도를 압축하고, 나만의 보석십자수 도안을 만들어봅시다.")
st.markdown("---")

# 사이드바 - 조작 포인트 설정
st.sidebar.header("⚙️ 도안 제작 설정 (조작 포인트)")

uploaded_file = st.sidebar.file_uploader("1. 명화 이미지 업로드 (JPG, PNG)", type=["jpg", "jpeg", "png"])

pixel_size = st.sidebar.slider(
    "2. 보석 1칸의 크기 (픽셀 크기)",
    min_value=4,
    max_value=32,
    value=8,
    step=2,
    help="값이 커질수록 도안의 격자가 커지고 해상도가 낮아집니다. (데이터 축소)"
)

k_value = st.sidebar.slider(
    "3. 사용할 보석 색상 수 (k값)",
    min_value=2,
    max_value=20,
    value=8,
    help="AI가 이미지에서 추출할 대표 보석 색상의 개수입니다. (색상 데이터 압축)"
)

st.sidebar.markdown("---")
st.sidebar.info(
    "💡 **수업 탐구 포인트**\n"
    "- **픽셀 크기 조절:** 공간 데이터의 압축\n"
    "- **k값 조절:** 색상 데이터의 압축(양자화)"
)

if uploaded_file is not None:
    # 이미지 로드 및 크기 정규화
    img = Image.open(uploaded_file).convert("RGB")
    img_np = np.array(img)
    orig_h, orig_w, _ = img_np.shape
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🖼️ 원본 명화 이미지")
        st.image(img, use_container_width=True)
        st.caption(f"원본 해상도: {orig_w} × {orig_h} 픽셀")
        
    with col2:
        st.subheader("🤖 AI 보석십자수 도안 결과")
        
        if st.button("🚀 보석십자수 도안 생성하기", type="primary"):
            with st.spinner("AI가 도안을 디자인하고 있습니다..."):
                
                # Step 1: 해상도 축소 (보석십자수 픽셀화 - 축소 시켰다가 다시 늘려서 격자감 표현)
                # 미술의 '점묘화' 혹은 격자 도안 효과를 내기 위한 전처리
                low_res_w = max(1, orig_w // pixel_size)
                low_res_h = max(1, orig_h // pixel_size)
                
                img_small = img.resize((low_res_w, low_res_h), resample=Image.Resampling.BILINEAR)
                small_np = np.array(img_small)
                
                # Step 2: k-Means를 이용한 색상 압축 (양자화)
                pixels = small_np.reshape(-1, 3) / 255.0  # 0~1 정규화 및 2차원 변환
                
                kmeans = KMeans(n_clusters=k_value, random_state=42, n_init=10)
                kmeans.fit(pixels)
                
                labels = kmeans.labels_
                centroids = kmeans.cluster_centers_
                
                # 대표 색상으로 픽셀 채우기
                quantized_pixels = centroids[labels]
                quantized_small_np = (quantized_pixels.reshape(low_res_h, low_res_w, 3) * 255).astype(np.uint8)
                
                # 학생들에게 도안 형태로 크게 보여주기 위해 다시 원본 크기로 확대 (네이버후드 방식)
                output_img = Image.fromarray(quantized_small_np).resize((orig_w, orig_h), resample=Image.Resampling.NEAREST)
                
                # 결과 출력
                st.image(output_img, use_container_width=True)
                st.success(f"도안 생성 완료! (가로 {low_res_w}칸 × 세로 {low_res_h}칸 = 총 {low_res_w*low_res_h:,}개의 보석 필요)")
                
                # Step 3: AI가 추출한 보석 팔레트 구성
                st.markdown("### 🎨 AI 보석 팔레트 (구매해야 할 보석 색상 리스트)")
                
                hex_colors = []
                for color in centroids:
                    r, g, b = int(color[0]*255), int(color[1]*255), int(color[2]*255)
                    hex_colors.append(f"#{r:02x}{g:02x}{b:02x}")
                
                # 컬러칩 배치
                palette_cols = st.columns(min(k_value, 10))  # 한 줄에 최대 10개씩 출력
                for idx, color_hex in enumerate(hex_colors):
                    col_target = palette_cols[idx % 10]
                    with col_target:
                        st.markdown(
                            f"<div style='background-color:{color_hex}; height:50px; border-radius:5px; border:2px solid #FFF; box-shadow: 1px 1px 5px rgba(0,0,0,0.3);'></div>", 
                            unsafe_allow_html=True
                        )
                        # 각 기호 지정 (보석십자수 도안의 기호 역할 대체)
                        st.caption(f"**보석 {idx+1}**\n{color_hex}")
                
                # 정보 교과 연계: 데이터 압축률 계산 정보 확장 영역
                with st.expander("📊 정보 과학 데이터 압축의 비밀 확인하기"):
                    st.write("원본 이미지와 AI 도안 이미지의 데이터 용량을 비교해보세요.")
                    orig_data_size = orig_w * orig_h * 3
                    compressed_data_size = (low_res_w * low_res_h * 1) + (k_value * 3) # (각 칸의 라벨인덱스 바이트) + (팔레트 색상 데이터)
                    compression_ratio = (1 - (compressed_data_size / orig_data_size)) * 100
                    
                    st.metric(label="데이터 압축률(공간 및 색상 통합)", value=f"{compression_ratio:.2f}% 감소")
                    st.caption("※ 인공지능 k-평균 군집화를 통해 필요한 색상 정보의 종류를 대폭 줄였기 때문에 용량이 크게 압축됩니다.")

else:
    st.info("👈 왼쪽 사이드바에서 원하는 명화 이미지(JPG/PNG)를 업로드하고 도안을 제작해보세요!")
