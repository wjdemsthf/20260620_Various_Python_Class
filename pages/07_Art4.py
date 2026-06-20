import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.cluster import KMeans
import plotly.graph_objects as go
import math

# 1. 페이지 기본 설정
st.set_page_config(
    page_title="AI 보석십자수 & 궤적 추적기",
    page_icon="💎",
    layout="wide"
)

st.title("💎 AI 보석십자수 디자이너: 알고리즘 누적 궤적 분석")
st.markdown("### **정보(k-평균 군집화 누적 잔상, 색상 비트, 실제 감소율) × 미술(보석십자수 도안) 융합 수업**")
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
    
    # --- [핵심 알고리즘 연산] 단계별 중심점 누적 추적 ---
    # 1회차부터 현재 선택된 max_iter까지의 모든 중심점 위치를 기록할 리스트 배열
    # K-Means의 random_state와 init을 일치시켜 회차별 연속성을 보장합니다.
    all_centroids = []
    
    for i in range(1, max_iter_value + 1):
        km = KMeans(n_clusters=k_value, init='random', max_iter=i, n_init=1, random_state=42)
        km.fit(pixels)
        all_centroids.append(km.cluster_centers_)
        
        # 마지막 회차의 결과(현재 상태) 저장
        if i == max_iter_value:
            labels_current = km.labels_
            centroids_current = km.cluster_centers_

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

    # --- TAB 2: 4가지 요구사항 집약 대시보드 ---
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

        st.markdown("---")
        
        # 레이아웃 2: 3번(3D 시각화) & 4번(중심점 누적 잔상 이동 궤적) 결합 그래프
        st.markdown("#### **[데이터 3 & 4] 3D RGB 색상 공간 군집화 및 중심점 누적 이동 궤적(Trace) 추적**")
        st.write("사이드바의 '반복 횟수'를 늘릴수록, 처음에 지정된 무작위 보석 위치부터 현재 위치까지 **이동해 온 전체 경로가 실시간 잔상(흰 선)으로 누적**되어 나타납니다.")
        st.info("💡 슬라이더를 1부터 시작해 5, 10으로 천천히 올리며 다이아몬드가 꼬리를 길게 늘어뜨리며 픽셀 중심으로 기어 들어가는지 관찰해보세요.")
        
        fig = go.Figure()
        
        # [데이터 3] 3D RGB 색상 공간에서의 픽셀 데이터 군집화 시각화 (산점도 점)
        df_pixels = pd.DataFrame(pixels, columns=['Red', 'Green', 'Blue'])
        df_pixels['Cluster'] = labels_current
        
        # 3D 차트 랙 방지를 위해 데이터가 너무 많으면 최대 2000개만 랜덤 샘플링
        if len(df_pixels) > 2000:
            df_pixels = df_pixels.sample(n=2000, random_state=42)
            
        for cluster_idx in range(k_value):
            cluster_data = df_pixels[df_pixels['Cluster'] == cluster_idx]
            fig.add_trace(go.Scatter3d(
                x=cluster_data['Red'], y=cluster_data['Green'], z=cluster_data['Blue'],
                mode='markers',
                marker=dict(size=2, opacity=0.3),
                name=f"군집 {cluster_idx+1} 픽셀 무리"
            ))
            
        # [데이터 4] K군집화 알고리즘 중심점의 '누적 잔상' 이동 궤적 그리기
        # 각 보석(k)별로 회차별 이동 경로를 추적하여 선으로 연결합니다.
        for color_idx in range(k_value):
            # 특정 보석의 1회차부터 현재 회차까지의 R, G, B 좌표 모으기
            trace_r = [step[color_idx][0] for step in all_centroids]
            trace_g = [step[color_idx][1] for step in all_centroids]
            trace_b = [step[color_idx][2] for step in all_centroids]
            
            # 1회차부터 누적된 이동 경로 선(Line) 그리기
            if len(trace_r) > 1:
                fig.add_trace(go.Scatter3d(
                    x=trace_r, y=trace_g, z=trace_b,
                    mode='lines+markers',
                    line=dict(color='white', width=5),
                    marker=dict(size=4, color='yellow'),
                    name=f"보석 {color_idx+1} 이동 흔적",
                    showlegend=False
                ))
            
            # 최종 도달한 현재 회차의 중심점 위치 (큰 다이아몬드)
            fig.add_trace(go.Scatter3d(
                x=[trace_r[-1]], y=[trace_g[-1]], z=[trace_b[-1]],
                mode='markers',
                marker=dict(size=12, symbol='diamond', color=hex_colors[color_idx], line=dict(color='white', width=2)),
                name=f"◆ 보석 {color_idx+1} 현재 중심점"
            ))
            
        fig.update_layout(
            scene=dict(
                xaxis=dict(title='Red (0-255)', range=[0, 255]),
                yaxis=dict(title='Green (0-255)', range=[0, 255]),
                zaxis=dict(title='Blue (0-255)', range=[0, 255])
            ),
            margin=dict(l=0, r=0, b=0, t=0),
            height=650
        )
        
        st.plotly_chart(fig, use_container_width=True)

else:
    st.info("👈 왼쪽 제어 패널에서 분석할 명화 이미지를 업로드해 주세요.")
