import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.cluster import KMeans
import plotly.express as px
import plotly.graph_objects as go
import math

# 1. 페이지 기본 설정
st.set_page_config(
    page_title="AI 보석십자수 & 압축 가속기",
    page_icon="💎",
    layout="wide"
)

st.title("💎 AI 보석십자수: 중심점 이동 시각화 및 실제 감소율 분석")
st.markdown("### **정보(고해상도 제어, 중심점 수렴, 실제 데이터 감소율) × 미술(보석십자수 도안) 융합 수업**")
st.write("가로 격자 최대 500칸, 색상 수 최대 64개까지 확장된 환경에서 AI 학습 알고리즘의 동적 변화와 손실 압축률을 정밀 분석합니다.")
st.markdown("---")

# 2. 사이드바 - 조작 포인트 설정 (범위 대폭 확장)
st.sidebar.header("⚙️ 고성능 도안 제작 및 알고리즘 제어")

uploaded_file = st.sidebar.file_uploader("1. 명화 이미지 업로드 (JPG, PNG)", type=["jpg", "jpeg", "png"])

# 변경포인트 3: 가로 칸수 최대 범주를 500으로 확장
target_width = st.sidebar.number_input(
    "2. 도안의 가로 보석 개수 (가로 칸수)",
    min_value=10,
    max_value=500,
    value=60,
    step=10,
    help="최대 500칸까지 설정 가능합니다. 칸수가 많아질수록 정교한 도안이 되지만 AI 연산 시간이 길어집니다."
)

# 변경포인트 3: k값 범주를 64까지 확장
k_value = st.sidebar.slider(
    "3. 사용할 보석 색상 수 (k값)",
    min_value=2,
    max_value=64,
    value=8,
    help="AI가 추출할 대표 보석 색상의 가짓수입니다. 최대 64개까지 지원합니다."
)

max_iter_value = st.sidebar.slider(
    "4. AI 알고리즘 반복 횟수 (Learning Steps)",
    min_value=1,
    max_value=15,
    value=2,
    step=1,
    help="반복 횟수를 1부터 늘려가며 우측 '중심점 이동 추적' 그래프에서 선들이 어떻게 제자리를 찾아가는지 관찰하세요."
)

# 3. 메인 로직 작동
if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    img_np = np.array(img)
    orig_h, orig_w, _ = img_np.shape
    
    # 비율 맞춤 세로 칸수 자동 연산
    aspect_ratio = orig_h / orig_w
    target_height = int(target_width * aspect_ratio)
    
    # 탭 구성
    tab1, tab2 = st.tabs(["🖼️ 보석십자수 도안화 결과", "📊 AI 중심점 이동 시각화 및 실제 감소율"])
    
    # --- 데이터 전처리 및 연산 ---
    img_small = img.resize((target_width, target_height), resample=Image.Resampling.BILINEAR)
    small_np = np.array(img_small)
    pixels = small_np.reshape(-1, 3)
    
    # K-Means 알고리즘 적용 (반복 횟수별 동적 관찰을 위해 고정 무작위성 부여)
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
    
    # 도안 이미지 생성 및 복원
    quantized_pixels = centroids[labels]
    quantized_small_np = quantized_pixels.reshape(target_height, target_width, 3).astype(np.uint8)
    output_img = Image.fromarray(quantized_small_np).resize((orig_w, orig_h), resample=Image.Resampling.NEAREST)
    
    # Hex 컬러 변환
    hex_colors = [f"#{int(c[0]):02x}{int(c[1]):02x}{int(c[2]):02x}" for c in centroids]
    
    # --- TAB 1: 도안 및 팔레트 출력 ---
    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("원본 명화 이미지")
            st.image(img, use_container_width=True)
            st.caption(f"원본 해상도: {orig_w} × {orig_h} 픽셀")
            
        with col2:
            st.subheader(f"AI 보석십자수 도안 결과")
            st.image(output_img, use_container_width=True)
            st.success(f"📐 비율 연산 결과: 가로 {target_width}칸 × 세로 {target_height}칸")
            st.caption(f"총 보석 개수: {target_width * target_height:,} 개")

        st.markdown("---")
        st.markdown(f"### 🎨 AI 보석 팔레트 (추출된 {k_value}개 색상 표)")
        
        # k값이 커짐에 따라 스크롤을 방지하기 위해 유연한 레이아웃 구성
        palette_html = "".join([
            f"<div style='display:inline-block; background-color:{c}; width:40px; height:40px; margin:4px; border-radius:4px; border:1px solid #FFF;' title='보석 {i+1}: {c}'></div>"
            for i, c in enumerate(hex_colors)
        ])
        st.markdown(palette_html, unsafe_allow_html=True)
        st.caption("마우스를 컬러칩에 올리면 보석 번호와 색상 코드를 볼 수 있습니다.")

    # --- TAB 2: 중심점 이동 및 감소율 정밀 분석 ---
    with tab2:
        st.subheader("📉 실제 데이터 크기 및 비트(Bit) 감소율 정밀 대조")
        
        # 고유 색상 수 및 최소 필요 비트 계산
        unique_colors_before = len(np.unique(pixels, axis=0))
        bits_needed_before = math.ceil(math.log2(max(2, unique_colors_before)))
        bits_needed_after = math.ceil(math.log2(k_value))
        
        # 변경포인트 1: 실제 물리적 바이트 감소율 계산 연산 보강
        orig_bytes = orig_w * orig_h * 3
        # 압축 후: 인덱스 정보를 담을 비트 데이터 공간 + 컬러맵(팔레트) 공간
        compressed_bytes = math.ceil((target_width * target_height * bits_needed_after) / 8) + (k_value * 3)
        actual_compression_ratio = (1 - (compressed_bytes / orig_bytes)) * 100
        
        c_m1, c_m2, c_m3 = st.columns(3)
        with c_m1:
            st.metric(label="원본 이미지 실제 크기", value=f"{orig_bytes:,} Bytes")
            st.caption(f"표현 색상 수: {unique_colors_before:,}가지 (필요 최소 비트: {bits_needed_before} Bits)")
        with c_p2 if 'c_p2' in locals() else c_m2:
            st.metric(label="도안 변환 후 실제 크기", value=f"{compressed_bytes:,} Bytes")
            st.caption(f"제한 색상 수: {k_value}가지 (필요 최소 비트: {bits_needed_after} Bits)")
        with c_p3 if 'c_p3' in locals() else c_m3:
            # 실시간 가시적 감소율 퍼센트 강조
            st.metric(label="💥 실제 데이터 용량 감소율", value=f"{actual_compression_ratio:.2f}% 감소", delta=f"-{orig_bytes - compressed_bytes:,} Bytes")
            st.caption("정보의 불필요한 해상도와 색상 낭비를 정리하여 압축한 정량적 수치입니다.")

        st.markdown("---")
        
        # 변경포인트 2: K군집화 반복 과정에서 중심점이 이동 및 수렴하는 시각화 대시보드
        st.subheader("🔄 K-Means 알고리즘 중심점(Centroid) 이동 및 수렴 추적")
        st.write("사이드바의 '알고리즘 반복 횟수'를 바꾸면, 무작위로 흩어져 있던 중심점들의 RGB 색상 좌표가 각 군집의 평균값으로 정렬되며 제자리를 찾는 과정을 시각적으로 추적합니다.")
        
        # 라인 그래프를 활용해 중심점 좌표의 변화 유도 (각 보석의 RGB 성분을 병렬 축 형태로 표현)
        fig_lines = go.Figure()
        
        for idx in range(k_value):
            fig_lines.add_trace(go.Scatter(
                x=['Red (빨강)', 'Green (초록)', 'Blue (파랑)'],
                y=[centroids[idx][0], centroids[idx][1], centroids[idx][2]],
                mode='lines+markers',
                name=f"보석 {idx+1} 중심점",
                line=dict(color=hex_colors[idx], width=3),
                marker=dict(size=10)
            ))
            
        fig_lines.update_layout(
            title=f"현재 반복 단계({max_iter_value}회차)에서의 보석별 RGB 위치 분포 평면도",
            xaxis_title="색상 성분 축",
            yaxis_title="디지털 수치 값 (0 ~ 255)",
            yaxis=dict(range=[0, 255]),
            height=450
        )
        st.plotly_chart(fig_lines, use_container_width=True)
        st.caption("🔍 **관찰 가이드:** 반복 횟수를 1에서 2, 3으로 늘릴 때마다 꺾은선들이 고정된 위치로 움직이지 않고 수렴하게 됩니다. 이것은 AI가 픽셀들의 최적의 평균 색상을 완전히 찾아내어 '학습이 완료'되었음을 의미합니다.")

else:
    st.info("👈 왼쪽 사이드바에서 고해상도 명화 도안 제작을 시작할 이미지를 업로드해 주세요.")
