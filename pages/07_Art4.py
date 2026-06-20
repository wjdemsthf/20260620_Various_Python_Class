import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.cluster import KMeans
import plotly.graph_objects as go
import math
import requests
from io import BytesIO

# ----------------------------------------------------------------
# 🌟 [선생님 설정 구역] 🌟
# 선생님의 깃허브 계정명과 저장소(레포지토리) 이름을 정확하게 적어주세요.
# ----------------------------------------------------------------
GITHUB_USERNAME = "wjdemsthf" 
GITHUB_REPO = "20260620_Various_Python_Class"
# ----------------------------------------------------------------

# 1. 페이지 기본 설정
st.set_page_config(
    page_title="AI 보석십자수 & 압축 마스터",
    page_icon="💎",
    layout="wide"
)

st.title("💎 AI 보석십자수 디자이너: 전수 데이터 시각화 및 압축률 정밀 분석")
st.markdown("### **정보(고대비 디폴트 군집화, 군집별 세트 On/Off, 단계별 압축률) × 미술 융합 수업**")
st.markdown("---")

# 2. 사이드바 제어 패널
st.sidebar.header("⚙️ 도안 제작 및 알고리즘 제어")

sample_option = st.sidebar.selectbox(
    "1. 실험할 명화 선택",
    ["직접 파일 업로드하기", "고흐 - 별이 빛나는 밤", "뭉크 - 절규"]
)

# 파일 업로더 제어
uploaded_file = None
if sample_option == "직접 파일 업로드하기":
    uploaded_file = st.sidebar.file_uploader("명화 이미지 업로드 (JPG, PNG)", type=["jpg", "jpeg", "png"])

target_width = st.sidebar.number_input(
    "2. 도안의 가로 보석 개수 (가로 칸수)",
    min_value=10, max_value=500, value=60, step=10,
    help="최대 500칸까지 설정 가능합니다."
)

k_value = st.sidebar.slider(
    "3. 사용할 보석 색상 수 (k값)",
    min_value=2, max_value=64, value=6,
    help="AI가 추출할 대표 보석 색상의 가짓수입니다."
)

max_iter_value = st.sidebar.slider(
    "4. AI 알고리즘 반복 횟수 (Learning Steps)",
    min_value=1, max_value=15, value=3, step=1,
    help="숫자를 늘릴수록 3D 그래프에 누적 잔상(꼬리선)이 길게 그려집니다."
)

# 🌟 방법 1: pages/ 하위 폴더 경로에 구애받지 않는 절대 주소형 깃허브 로더 함수
def load_image_from_github(file_name):
    # main/ 바로 뒤에 sample_images/를 고정하여 상대 경로 문제를 원천 차단합니다.
    url = f"https://raw.githubusercontent.com/{GITHUB_USERNAME}/{GITHUB_REPO}/main/sample_images/{file_name}"
    try:
        response = requests.get(url)
        if response.status_code == 200:
            return Image.open(BytesIO(response.content)).convert("RGB")
        else:
            st.sidebar.error(f"❌ 파일을 찾을 수 없습니다.\n요청 주소: {url}")
            return None
    except Exception as e:
        st.sidebar.error(f"이미지 로드 중 오류 발생: {e}")
        return None

# 이미지 객체 초기화
img = None

# 선택지에 따른 파일명 매핑 및 로딩
if sample_option == "직접 파일 업로드하기":
    if uploaded_file is not None:
        img = Image.open(uploaded_file).convert("RGB")
elif sample_option == "고흐 - 별이 빛나는 밤":
    img = load_image_from_github("Starry_Night.jpg")
elif sample_option == "뭉크 - 절규":
    img = load_image_from_github("The_Scream.jpg")
elif sample_option == "베르메르 - 진주 귀걸이를 한 소녀":
    img = load_image_from_github("진주 귀걸이를 한 소녀.jpg")

# 3. 메인 로직 작동
if img is not None:
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

    # --- TAB 2: 데이터 압축 및 수렴 분석 ---
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
        st.write("군집 간 경계가 뚜렷이 구분되도록 데이터의 색상을 고대비 디폴트 테마로 자동 지정했습니다.")
        st.info("💡 우측 범례에서 중심점 제어 버튼을 클릭하면 해당 군집의 픽셀 무리와 경로가 한꺼번에 켜고 꺼집니다.")
        
        fig = go.Figure()
        df_pixels = pd.DataFrame(pixels, columns=['Red', 'Green', 'Blue'])
        df_pixels['Cluster_Idx'] = labels_current
        
        for color_idx in range(k_value):
            cluster_pixel_data = df_pixels[df_pixels['Cluster_Idx'] == color_idx]
            
            # 1. 군집별 픽셀 전수 데이터 (디폴트 고대비 컬러 자동 지정)
            fig.add_trace(go.Scatter3d(
                x=cluster_pixel_data['Red'], y=cluster_pixel_data['Green'], z=cluster_pixel_data['Blue'],
                mode='markers',
                marker=dict(size=1.2, opacity=0.35),
                name=f"보석 {color_idx+1} 영역 픽셀 무리",
                legendgroup=f"group_{color_idx}",
                showlegend=False,
                hoverinfo='text',
                text=f"보석 {color_idx+1}번 군집"
            ))
            
            # 2. 군집별 이동 경로 누적 잔상선
            trace_r = [step[color_idx][0] for step in all_centroids]
            trace_g = [step[color_idx][1] for step in all_centroids]
            trace_b = [step[color_idx][2] for step in all_centroids]
            
            if len(trace_r) > 1:
                fig.add_trace(go.Scatter3d(
                    x=trace_r, y=trace_g, z=trace_b,
                    mode='lines+markers',
                    line=dict(color='#FFFFFF', width=6),
                    marker=dict(size=3.5, color='#FFD700'),
                    legendgroup=f"group_{color_idx}",
                    showlegend=False,
                    name=f"보석 {color_idx+1} 이동 경로"
                ))
            
            # 3. 현재 최종 중심점 (실제 보석 색상 칩 매핑 + 흰 테두리)
            fig.add_trace(go.Scatter3d(
                x=[trace_r[-1]], y=[trace_g[-1]], z=[trace_b[-1]],
                mode='markers',
                marker=dict(
                    size=11, symbol='diamond', color=hex_colors[color_idx], 
                    line=dict(color='#FFFFFF', width=2)
                ),
                name=f"◆ 보석 {color_idx+1} 중심점 제어",
                legendgroup=f"group_{color_idx}"
            ))
            
        fig.update_layout(
            scene=dict(
                xaxis=dict(title='Red (0-255)', range=[0, 255], backgroundcolor="rgb(35, 35, 35)", gridcolor="gray"),
                yaxis=dict(title='Green (0-255)', range=[0, 255], backgroundcolor="rgb(35, 35, 35)", gridcolor="gray"),
                zaxis=dict(title='Blue (0-255)', range=[0, 255], backgroundcolor="rgb(35, 35, 35)", gridcolor="gray"),
                aspectmode='cube'
            ),
            margin=dict(l=0, r=0, b=0, t=0),
            height=750, template="plotly_dark",
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01, font=dict(size=12))
        )
        st.plotly_chart(fig, use_container_width=True)

else:
    st.info("👈 왼쪽 제어 패널에서 명화 샘플을 선택하거나 직접 파일을 업로드하시면 인터랙티브 대시보드가 즉시 실행됩니다!")
