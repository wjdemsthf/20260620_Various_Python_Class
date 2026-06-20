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
    page_title="AI 보석십자수 & 압축 분석기",
    page_icon="💎",
    layout="wide"
)

st.title("💎 AI 보석십자수 디자이너: 데이터 압축 및 중심점 궤적 분석")
st.markdown("### **정보(k-평균 군집화 궤적, 색상 비트, 실제 감소율) × 미술(보석십자수 도안) 융합 수업**")
st.markdown("---")

# 2. 사이드바 제어 패널
st.sidebar.header("⚙️ 도안 제작 및 알고리즘 제어")
uploaded_file = st.sidebar.file_uploader("1. 명화 이미지 업로드 (JPG, PNG)", type=["jpg", "jpeg", "png"])

target_width = st.sidebar.number_input(
    "2. 도안의 가로 보석 개수 (가로 칸수)",
    min_value=10, max_value=500, value=60, step=10,
    help="최대 500칸까지 설정 가능합니다."
)

k_value = st.sidebar.slider(
    "3. 사용할 보석 색상 수 (k값)",
    min_value=2, max_value=64, value=6,
    help="AI가 추출할 대표 보석 색상의 가짓수입니다. (최대 64개)"
)

max_iter_value = st.sidebar.slider(
    "4. AI 알고리즘 반복 횟수 (Learning Steps)",
    min_value=1, max_value=15, value=1, step=1,
    help="1부터 시작해서 숫자를 천천히 늘려보세요! 중심점(◆)이 이동하며 도안과 그래프가 정교해집니다."
)

if uploaded_file is not None:
    # 이미지 로드 및 격자화 전처리
    img = Image.open(uploaded_file).convert("RGB")
    img_np = np.array(img)
    orig_h, orig_w, _ = img_np.shape
    
    # 비율 맞춤 세로 칸수 자동 연산
    aspect_ratio = orig_h / orig_w
    target_height = int(target_width * aspect_ratio)
    
    img_small = img.resize((target_width, target_height), resample=Image.Resampling.BILINEAR)
    small_np = np.array(img_small)
    pixels = small_np.reshape(-1, 3)
    
    # --- 핵심 알고리즘 연산 (K-Means) ---
    # 현재 단계(max_iter)의 중심점과 라벨 계산
    kmeans_current = KMeans(n_clusters=k_value, init='random', max_iter=max_iter_value, n_init=1, random_state=42)
    kmeans_current.fit(pixels)
    labels_current = kmeans_current.labels_
    centroids_current = kmeans_current.cluster_centers_
    
    # [학습 이동 궤적 시각화용 연산] 이전 단계(max_iter - 1)의 중심점 계산
    if max_iter_value > 1:
        kmeans_prev = KMeans(n_clusters=k_value, init='random', max_iter=max_iter_value - 1, n_init=1, random_state=42)
        kmeans_prev.fit(pixels)
        centroids_prev = kmeans_prev.cluster_centers_
    else:
        centroids_prev = None

    # 도안 이미지 생성
    quantized_pixels = centroids_current[labels_current]
    quantized_small_np = quantized_pixels.reshape(target_height, target_width, 3).astype(np.uint8)
    output_img = Image.fromarray(quantized_small_np).resize((orig_w, orig_h), resample=Image.Resampling.NEAREST)
    hex_colors = [f"#{int(c[0]):02x}{int(c[1]):02x}{int(c[2]):02x}" for c in centroids_current]
    
    # 탭 구현
    tab1, tab2 = st.tabs(["🖼️ 보석십자수 도안화 결과", "📊 AI 중심점 이동 시각화 및 실제 감소율"])
    
    # --- TAB 1: 도안 및 팔레트 결과 ---
    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("원본 명화 이미지")
            st.image(img, use_container_width=True)
        with col2:
            st.subheader("AI 보석십자수 도안 결과")
            st.image(output_img, use_container_width=True)
            st.success(f"📐 비율 자동 연산: 가로 {target_width}칸 × 세로 {target_height}칸 (총 {target_width * target_height:,}개 보석)")

        st.markdown("---")
        st.markdown(f"### 🎨 AI 보석 팔레트 (추출된 {k_value}개 색상)")
        palette_html = "".join([
            f"<div style='display:inline-block; background-color:{c}; width:40px; height:40px; margin:4px; border-radius:4px; border:1px solid #FFF;' title='보석 {i+1}: {c}'></div>"
            for i, c in enumerate(hex_colors)
        ])
        st.markdown(palette_html, unsafe_allow_html=True)

    # --- TAB 2: 요청하신 4가지 핵심 데이터 요약 대시보드 ---
    with tab2:
        st.markdown("### 📊 정보 교과 탐구 및 알고리즘 정밀 분석")
        
        # 데이터 연산 (고유 색상 및 비트 수)
        unique_colors_before = len(np.unique(pixels, axis=0))
        bits_needed_before = math.ceil(math.log2(max(2, unique_colors_before)))
        bits_needed_after = math.ceil(math.log2(k_value))
        
        # 실제 물리적 바이트 용량 및 감소율 계산
        orig_bytes = orig_w * orig_h * 3
        compressed_bytes = math.ceil((target_width * target_height * bits_needed_after) / 8) + (k_value * 3)
        actual_compression_ratio = (1 - (compressed_bytes / orig_bytes)) * 100
        
        # 레이아웃 1: 1번(색상/비트 비교) & 2번(실제 감소율) 데이터 배치
        c1, c2 = st.columns([6, 4])
        
        with c1:
            st.markdown("#### **[데이터 1] 변환 전·후 색상 가짓수 및 표현 비트(Bit) 비교**")
            data_comparison = {
                "구분": ["변환 전 (격자화 직후)", "변환 후 (AI 도안 압축)"],
                "사용한 고유 색상 수": [f"{unique_colors_before:,} 가지", f"{k_value} 가지 (k값)"],
                "픽셀(칸)당 최소 표현 비트 수": [f"{bits_needed_before} Bits", f"{bits_needed_after} Bits"],
                "데이터 표현 방식": ["각 픽셀에 24비트 직접 저장", f"색상표 인덱스 방식 ({bits_needed_after}비트만 사용)"]
            }
            st.table(pd.DataFrame(data_comparison))
            
        with c2:
            st.markdown("#### **[데이터 2] 변환시 실제 데이터 크기 및 감소율**")
            st.metric(label="💥 실제 데이터 용량 감소율", value=f"{actual_compression_ratio:.2f}% 감소")
            st.caption(f"- **원본 이미지 실제 크기:** {orig_bytes:,} Bytes")
            st.caption(f"- **압축 도안 실제 크기:** {compressed_bytes:,} Bytes")
            st.caption(f"- **절약된 용량:** {orig_bytes - compressed_bytes:,} Bytes")

        st.markdown("---")
        
        # 레이아웃 2: 3번(3D 시각화) & 4번(중심점 이동 과정 궤적) 결합 그래프
        st.markdown("#### **[데이터 3 & 4] 3D RGB 색상 공간 군집화 및 중심점 이동 궤적(Trajectory) 추적**")
        st.write("사이드바의 '반복 횟수'를 1회에서 2회, 3회로 늘릴 때마다, 무작위로 생성된 보석 중심점(◆)이 픽셀 무리 한가운데로 **이동한 선(궤적)**을 눈으로 확인할 수 있습니다.")
        
        # Plotly Graph Objects를 이용한 정밀 커스텀 3D 플로팅
        fig = go.Figure()
        
        # 3. 3D RGB 색상 공간에서의 픽셀 데이터 군집화 시각화 (산점도 점)
        df_pixels = pd.DataFrame(pixels, columns=['Red', 'Green', 'Blue'])
        df_pixels['Cluster'] = labels_current
        
        # 최적화를 위해 데이터 포인트가 너무 많으면 최대 3000개만 샘플링하여 3D 차트 지연 방지
        if len(df_pixels) > 3000:
            df_pixels = df_pixels.sample(n=3000, random_state=42)
            
        for cluster_idx in range(k_value):
            cluster_data = df_pixels[df_pixels['Cluster'] == cluster_idx]
            fig.add_trace(go.Scatter3d(
                x=cluster_data['Red'], y=cluster_data['Green'], z=cluster_data['Blue'],
                mode='markers',
                marker=dict(size=2, opacity=0.4),
                name=f"군집 {cluster_idx+1} 픽셀 무리"
            ))
            
        # 4. K군집화 알고리즘 중심점의 이동 과정 (궤적 및 현재 위치 선 그리기)
        for i in range(k_value):
            current_ctr = centroids_current[i]
            
            # 만약 이전 단계 중심점이 존재한다면 이동 경로(선)를 그려줌
            if list(centroids_prev) is not None:
                prev_ctr = centroids_prev[i]
                fig.add_trace(go.Scatter3d(
                    x=[prev_ctr[0], current_ctr[0]],
                    y=[prev_ctr[1], current_ctr[1]],
                    z=[prev_ctr[2], current_ctr[2]],
                    mode='lines+markers',
                    line=dict(color='white', width=4),
                    marker=dict(size=4, color='yellow'),
                    name=f"보석 {i+1} 이동 경로",
                    showlegend=False
                ))
            
            # 현재 최종 중심점 위치 (큰 다이아몬드로 강조)
            fig.add_trace(go.Scatter3d(
                x=[current_ctr[0]], y=[current_ctr[1]], z=[current_ctr[2]],
                mode='markers',
                marker=dict(size=10, symbol='diamond', color=hex_colors[i], line=dict(color='white', width=2)),
                name=f"★ 보석 {i+1} 현재 중심점"
            ))
            
        fig.update_layout(
            scene=dict(
                xaxis=dict(title='Red (0-255)', range=[0, 255]),
                yaxis=dict(title='Green (0-255)', range=[0, 255]),
                zaxis=dict(title='Blue (0-255)', range=[0, 255])
            ),
            margin=dict(l=0, r=0, b=0, t=0),
            height=600
        )
        
        st.plotly_chart(fig, use_container_width=True)
        st.caption("💡 **탐구 발문 예시:** 반복 횟수(max_iter)가 4~5회 이상을 넘어가면 왜 이동 경로를 나타내는 노란색 선이 더 이상 나타나지 않고 고정될까요? AI의 학습 완료 조건과 연결 지어 생각해보세요.")
        
else:
    st.info("👈 왼쪽 제어 패널에서 분석할 명화 이미지를 업로드해 주세요.")
