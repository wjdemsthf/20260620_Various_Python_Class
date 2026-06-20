import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.cluster import KMeans

# 1. 페이지 기본 설정
st.set_page_config(
    page_title="AI 보석십자수 & 데이터 압축",
    page_icon="💎",
    layout="wide"
)

st.title("💎 AI 보석십자수 디자이너 및 데이터 압축 실습")
st.markdown("### **정보(k-평균 군집화와 손실 압축) × 미술(보석십자수 도안) 융합 수업**")
st.write("인공지능이 무한한 아날로그 색상을 유한한 디지털 보석 데이터로 어떻게 '군집화'하고 '압축'하는지 단계별로 확인해봅시다.")
st.markdown("---")

# 2. 사이드바 - 조작 포인트 설정 (Control Panel)
st.sidebar.header("⚙️ 도안 제작 및 알고리즘 제어")

uploaded_file = st.sidebar.file_uploader("1. 명화 이미지 업로드 (JPG, PNG)", type=["jpg", "jpeg", "png"])

pixel_size = st.sidebar.slider(
    "2. 보석 1칸의 크기 (공간 해상도 조절)",
    min_value=4,
    max_value=32,
    value=8,
    step=2,
    help="값이 커질수록 격자가 커지고 데이터 해상도가 낮아집니다. (공간 데이터 압축)"
)

k_value = st.sidebar.slider(
    "3. 사용할 보석 색상 수 (k값 선택)",
    min_value=2,
    max_value=16,
    value=6,
    help="AI가 이미지에서 추출할 대표 보석 색상의 가짓수입니다. (색상 데이터 압축)"
)

# 학습 과정 시각화를 위한 핵심 조작 포인트
max_iter_value = st.sidebar.slider(
    "4. AI 알고리즘 반복 횟수 (Learning Steps)",
    min_value=1,
    max_value=10,
    value=1,
    step=1,
    help="1부터 시작해서 숫자를 늘려보세요. AI가 중심점을 중심값으로 이동시키며 도안을 정교화하는 과정을 볼 수 있습니다."
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    "### 📖 핵심 개념 가이드\n"
    "- **k-Means 군집화:** 수십만 개의 픽셀 색상 중 서로 유사한 것끼리 묶어 $k$개의 대표 보석 색상을 찾아내는 비지도학습 알고리즘입니다.\n"
    "- **손실 압축(Lossy Compression):** 인간이 눈으로 구별하기 어려운 미세한 정보(해상도, 미묘한 색상 차이)를 버리고 용량을 획기적으로 줄이는 방법입니다."
)

# 3. 메인 로직 작동
if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    img_np = np.array(img)
    orig_h, orig_w, _ = img_np.shape
    
    # 탭 기능을 활용해 화면을 깔끔하게 분할
    tab1, tab2 = st.tabs(["🖼️ 도안 제작 및 AI 군집화 과정", "📊 데이터 압축 및 수학적 원리"])
    
    # --- 핵심 연산 수행 ---
    # 공간 축소 (보석십자수 격자화)
    low_res_w = max(1, orig_w // pixel_size)
    low_res_h = max(1, orig_h // pixel_size)
    img_small = img.resize((low_res_w, low_res_h), resample=Image.Resampling.BILINEAR)
    small_np = np.array(img_small)
    
    # 2차원 배열 변환 및 정규화
    pixels = small_np.reshape(-1, 3) / 255.0
    
    # 학생들이 선택한 반복 횟수(max_iter)와 무작위성 고정(init='random')으로 과정 추적 유도
    kmeans = KMeans(
        n_clusters=k_value, 
        init='random', 
        max_iter=max_iter_value, 
        n_init=1, 
        random_state=42
    )
    kmeans.fit(pixels)
    
    labels = kmeans.labels_
    centroids = kmeans.cluster_centers_
    
    # 이미지 복원 및 확대
    quantized_pixels = centroids[labels]
    quantized_small_np = (quantized_pixels.reshape(low_res_h, low_res_w, 3) * 255).astype(np.uint8)
    output_img = Image.fromarray(quantized_small_np).resize((orig_w, orig_h), resample=Image.Resampling.NEAREST)
    
    # Hex 컬러 코드 변환
    hex_colors = []
    for color in centroids:
        r, g, b = int(color[0]*255), int(color[1]*255), int(color[2]*255)
        hex_colors.append(f"#{r:02x}{g:02x}{b:02x}")
        
    # --- TAB 1: 도안 및 군집화 과정 시각화 ---
    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("원본 명화 이미지")
            st.image(img, use_container_width=True)
            st.caption(f"원본 해상도: {orig_w} × {orig_h} 픽셀 (총 {orig_w*orig_h:,}개 픽셀 정보)")
            
        with col2:
            st.subheader(f"AI 보석십자수 도안 (현재 알고리즘 반복 단계: {max_iter_value}회)")
            st.image(output_img, use_container_width=True)
            st.caption(f"도안 해상도: 가로 {low_res_w}칸 × 세로 {low_res_h}칸 (총 {low_res_w*low_res_h:,}개의 보석 배치)")

        st.markdown("---")
        st.markdown(f"### 🎨 AI가 지정한 현재 단계의 보석 팔레트 (선택된 색상 수: k={k_value})")
        st.info("💡 **알고리즘 관찰 포인트:** 왼쪽 사이드바에서 '반복 횟수'를 1에서부터 하나씩 올려보세요! 무작위로 설정되었던 보석 색상들이 점차 명화의 지배적인 진짜 색상 묶음으로 수렴(변화)해가는 과정을 볼 수 있습니다.")
        
        palette_cols = st.columns(min(k_value, 8))
        for idx, color_hex in enumerate(hex_colors):
            col_target = palette_cols[idx % 8]
            with col_target:
                st.markdown(
                    f"<div style='background-color:{color_hex}; height:50px; border-radius:8px; border:2px solid #FFF;'></div>", 
                    unsafe_allow_html=True
                )
                st.caption(f"**보석 번호: {idx+1}**\n{color_hex}")

    # --- TAB 2: 데이터 압축 및 정보 과학 학습 ---
    with tab2:
        st.subheader("📉 인공지능 기법을 통한 데이터 압축 분석")
        st.write("컴퓨터 내부에서 일어나는 실제 비트(Bit)와 바이트(Byte) 용량 변화를 정량적으로 계산한 수치입니다.")
        
        # 이론적 비트 계산식 시각화
        # 원본: 픽셀당 24비트 (RGB 각각 8비트)
        orig_bytes = orig_w * orig_h * 3
        
        # 압축본: 각 격자 칸마다 어떤 색상 보석인지 나타내는 인덱스 정보 (log2(k) 비트 필요하지만 계산 편의상 바이트 단위 근사)
        # 각 격자 칸당 색상 인덱스 (1바이트로 충분) + 팔레트 색상 데이터 테이블 크기
        compressed_bytes = (low_res_w * low_res_h * 1) + (k_value * 3)
        compression_ratio = (1 - (compressed_bytes / orig_bytes)) * 100
        
        c_p1, c_p2, c_p3 = st.columns(3)
        with c_p1:
            st.metric(label="원래 디지털 이미지 데이터 용량", value=f"{orig_bytes:,} Bytes")
            st.caption("계산식: 가로 픽셀 × 세로 픽셀 × 3바이트(RGB)")
        with c_p2:
            st.metric(label="AI 압축 후 도안 데이터 용량", value=f"{compressed_bytes:,} Bytes")
            st.caption("계산식: (도안 가로칸 × 세로칸 × 1바이트) + (보석 색상 수 × 3바이트)")
        with c_p3:
            st.metric(label="최종 데이터 압축 효율", value=f"{compression_ratio:.2f}% 감소", delta=f"-{orig_bytes - compressed_bytes:,} Bytes")
            
        st.markdown("---")
        st.markdown("### 🧮 AI 내부 연산 데이터: 군집 중심점(Centroids)")
        st.write("아래 표는 AI가 수많은 픽셀들의 RGB 색상 거리를 유클리드 거리 공식($d = \\sqrt{\\Delta R^2 + \\Delta G^2 + \\Delta B^2}$)으로 계산하여 도출한 보석별 디지털 좌표(0.0 ~ 1.0)입니다.")
        
        df_centroids = pd.DataFrame(centroids, columns=['Red(빨강 성분)', 'Green(초록 성분)', 'Blue(파랑 성분)'])
        df_centroids.index = [f"보석 {i+1}" for i in range(k_value)]
        st.dataframe(df_centroids.style.background_gradient(cmap='Blues'))

else:
    st.info("👈 왼쪽 사이드바에서 분석할 명화 이미지를 먼저 업로드해 주세요! 학생들이 조작할 수 있는 대시보드가 활성화됩니다.")
