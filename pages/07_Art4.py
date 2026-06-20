import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.cluster import KMeans
import plotly.graph_objects as go
import math

# 1. 페이지 기본 설정
st.set_page_config(
    page_title="AI 보석십자수 & 압축 마스터",
    page_icon="💎",
    layout="wide"
)

st.title("💎 AI 보석십자수 디자이너: 전수 데이터 시각화 및 압축률 정밀 분석")
st.markdown("### **정보(군집별 색상 통일, 중심점 가시성 극대화, 단계별 압축률) × 미술 융합 수업**")
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
    min_value=1, max_value=15, value=3, step=1,
    help="숫자를 늘릴수록 3D 그래프에 1회차부터 현재 회차까지 이동한 '누적 잔상(꼬리선)'이 길게 그려집니다."
)

if uploaded_file is not None:
    # [원본 데이터 수집]
    img = Image.open(uploaded_file).convert("RGB")
    img_np = np.array(img)
    orig_h, orig_w, _ = img_np.shape
    orig_total_pixels = orig_w * orig_h
    orig_bytes = orig_total_pixels * 3
    
    # 비율 맞춤 세로 칸수 자동 연산
    aspect_ratio = orig_h / orig_w
    target_height = int(target_width * aspect_ratio)
    target_total_pixels = target_width * target_height
    
    # [1단계 압축 연산: 공간 해상도 감소 (격자화)]
    img_small = img.resize((target_width, target_height), resample=Image.Resampling.BILINEAR)
    small_np = np.array(img_small)
    pixels = small_np.reshape(-1, 3)
    
    # [2단계 압축 연산: K-Means 색상 간소화]
    all_centroids = []
    for i in range(1, max_iter_value + 1):
        km = KMeans(n_clusters=k_value, init='random', max_iter=i, n_init=1, random_state=42)
        km.fit(pixels)
        all_centroids.append(km.cluster_centers_)
        if i == max_iter_value:
            labels_current = km.labels_
            centroids_current = km.cluster_centers_

    # 도안 이미지 생성 및 확대 복원
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
            st.success(f"📐 비율 자동 연산: 가로 {target_width}칸 × 세로 {target_height}칸 (총 {target_total_pixels:,}개 보석)")

        st.markdown("---")
        st.markdown(f"### 🎨 AI 보석 팔레트 (추출된 {k_value}개 색상)")
        palette_html = "".join([
            f"<div style='display:inline-block; background-color:{c}; width:40px; height:40px; margin:4px; border-radius:4px; border:1px solid #FFF;' title='보석 {i+1}: {c}'></div>"
            for i, c in enumerate(hex_colors)
        ])
        st.markdown(palette_html, unsafe_allow_html=True)

    # --- TAB 2: 데이터 압축 및 수렴 분석 (수직 레이아웃) ---
    with tab2:
        st.markdown("### 📊 정보 교과 정밀 데이터 분석 테이블")
        
        unique_colors_before = len(np.unique(pixels, axis=0))
        bits_needed_before = math.ceil(math.log2(max(2, unique_colors_before)))
        bits_needed_after = math.ceil(math.log2(k_value))
        
        st.markdown("#### **1️⃣ 변환 전·후 색상 가짓수 및 표현 비트(Bit) 비교**")
        data_comparison = {
            "구분": ["변환 전 (격자화 직후)", "변환 후 (AI 도안 압축)"],
            "사용한 고유 색상 수": [f"{unique_colors_before:,} 가지", f"{k_value} 가지 (k값)"],
            "픽셀(칸)당 최소 표현 비트 수": [f"{bits_needed_before} Bits", f"{bits_needed_after} Bits"],
            "데이터 표현 방식": ["각 픽셀에 24비트(RGB) 직접 저장", f"색상표 인덱스 방식 ({bits_needed_after}비트만 사용)"]
        }
        st.table(pd.DataFrame(data_comparison))
        
        st.markdown("---")
        
        st.markdown("#### **2️⃣ 데이터 감소율 및 압축률(Compression Ratio) 심층 비교**")
        spatial_compression_ratio = (1 - (target_total_pixels / orig_total_pixels)) * 100
        color_compression_ratio = (1 - (bits_needed_after / 24)) * 100
        
        compressed_bytes = math.ceil((target_total_pixels * bits_needed_after) / 8) + (k_value * 3)
        actual_compression_ratio = (1 - (compressed_bytes / orig_bytes)) * 100
        
        c_m1, c_m2, c_m3 = st.columns(3)
        with c_m1:
            st.metric(label="📉 [예상] 픽셀(공간) 감소 압축률", value=f"{spatial_compression_ratio:.2f}% 감소")
            st.caption(f"픽셀 수: {orig_total_pixels:,}개 ➔ {target_total_pixels:,}개")
        with c_m2:
            st.metric(label="📉 [예상] 색상 간소화 압축률", value=f"{color_compression_ratio:.2f}% 감소")
            st.caption(f"표현 비트: 24 Bits ➔ {bits_needed_after} Bits")
        with c_m3:
            st.metric(label="💥 [실제] 최종 물리 압축률 (합산)", value=f"{actual_compression_ratio:.2f}% 감소", delta=f"-{orig_bytes - compressed_bytes:,} Bytes")
            st.caption(f"용량: {orig_bytes:,} Bytes ➔ {compressed_bytes:,} Bytes")
            
        st.markdown("---")
        
        st.markdown("#### **3️⃣ 3D RGB 색상 공간 전수 데이터 군집화 및 중심점 누적 이동 궤적(Trace) 추적**")
        st.write("모든 픽셀 정보가 군집의 대표 색상으로 통일되어 시각화되며, AI 중심점 다이아몬드가 최상위 레이어에 선명하게 드러납니다.")
        st.info("💡 마우스 드래그로 회전 가능합니다. 흰색 실선 궤적 끝에 선명하게 위치한 다이아몬드(◆)를 관찰하세요.")
        
        fig = go.Figure()
        
        # 데이터프레임 구성
        df_pixels = pd.DataFrame(pixels, columns=['Red', 'Green', 'Blue'])
        # 변경포인트 1: 표시되는 색상을 명화 색상이 아닌 '해당 군집의 대표 보석 색상'으로 전수 통일
        df_pixels['Cluster_Hex'] = [hex_colors[l] for l in labels_current]
        df_pixels['Cluster_Name'] = [f"군집 {l+1}" for l in labels_current]
        
        # 1단계: 배경에 깔리는 픽셀 점들을 먼저 렌더링 (가시성 확보를 위해 매우 작게 설정)
        fig.add_trace(go.Scatter3d(
            x=df_pixels['Red'], y=df_pixels['Green'], z=df_pixels['Blue'],
            mode='markers',
            marker=dict(
                size=1.2,                     # 픽셀 점 크기를 줄여 중심점을 가리지 않게 조절
                color=df_pixels['Cluster_Hex'], # 군집 대표색으로 통일
                opacity=0.4                   # 투명도를 주어 내부 중심점 궤적이 투과되도록 유도
            ),
            name="군집화된 픽셀 데이터",
            hoverinfo='text',
            text=df_pixels['Cluster_Name']
        ))
            
        # 2단계: 그 위에 중심점 누적 이동 궤적(선)과 현재 중심점(다이아몬드)을 얹어서 렌더링 (레이아웃 순서상 위로 올라옴)
        for color_idx in range(k_value):
            trace_r = [step[color_idx][0] for step in all_centroids]
            trace_g = [step[color_idx][1] for step in all_centroids]
            trace_b = [step[color_idx][2] for step in all_centroids]
            
            # 이동 경로 누적 잔상선 (흰색 튜브 실선)
            if len(trace_r) > 1:
                fig.add_trace(go.Scatter3d(
                    x=trace_r, y=trace_g, z=trace_b,
                    mode='lines+markers',
                    line=dict(color='#FFFFFF', width=6),
                    marker=dict(size=3.5, color='#FFD700'),
                    name=f"보석 {color_idx+1} 이동 경로",
                    showlegend=False
                ))
            
            # 변경포인트 2: 픽셀에 묻히지 않도록 선명한 대비(검은 테두리 + 밝은 흰색 서브라인 효과) 및 세련된 크기 조절
            fig.add_trace(go.Scatter3d(
                x=[trace_r[-1]], y=[trace_g[-1]], z=[trace_b[-1]],
                mode='markers',
                marker=dict(
                    size=10,                      # 다이아몬드 크기를 너무 크지 않고 예리하게 살짝 줄임
                    symbol='diamond', 
                    color=hex_colors[color_idx], 
                    line=dict(color='#000000', width=3.5) # 두꺼운 검은 테두리를 주어 최전방에 확실히 격리되어 보이게 함
                ),
                name=f"◆ 보석 {color_idx+1} 현재 중심점"
            ))
            
        fig.update_layout(
            scene=dict(
                xaxis=dict(title='Red (0-255)', range=[0, 255], backgroundcolor="rgb(35, 35, 35)", gridcolor="gray"),
                yaxis=dict(title='Green (0-255)', range=[0, 255], backgroundcolor="rgb(35, 35, 35)", gridcolor="gray"),
                zaxis=dict(title='Blue (0-255)', range=[0, 255], backgroundcolor="rgb(35, 35, 35)", gridcolor="gray"),
                aspectmode='cube'
            ),
            margin=dict(l=0, r=0, b=0, t=0),
            height=750,
            template="plotly_dark"
        )
        
        st.plotly_chart(fig, use_container_width=True)

else:
    st.info("👈 왼쪽 제어 패널에서 명화 이미지를 업로드하시면 최적화된 3D 대시보드가 활성화됩니다.")
