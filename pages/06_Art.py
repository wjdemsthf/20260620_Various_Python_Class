import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.cluster import KMeans
import plotly.express as px
import math

# 1. 페이지 기본 설정
st.set_page_config(
    page_title="AI 보석십자수 & 비트 압축",
    page_icon="💎",
    layout="wide"
)

st.title("💎 AI 보석십자수: 해상도 비율 연산 및 비트 압축 실습")
st.markdown("### **정보(이미지 비율, 고유 색상 수, 표현 비트) × 미술(보석십자수 도안) 융합 수업**")
st.write("명화의 가로 칸수를 정하면 비율에 맞춰 세로 칸수가 자동 계산되며, AI 군집화를 통한 비트(Bit) 단위 압축 효율을 보여줍니다.")
st.markdown("---")

# 2. 사이드바 - 조작 포인트 설정
st.sidebar.header("⚙️ 도안 제작 및 알고리즘 제어")

uploaded_file = st.sidebar.file_uploader("1. 명화 이미지 업로드 (JPG, PNG)", type=["jpg", "jpeg", "png"])

# 변경포인트 1: 픽셀 크기가 아닌 '가로 보석 개수(칸수)' 입력받기
target_width = st.sidebar.number_input(
    "2. 도안의 가로 보석 개수 (가로 칸수)",
    min_value=10,
    max_value=150,
    value=50,
    step=5,
    help="도안의 가로 보석 개수를 입력하세요. 세로 개수는 그림 비율에 맞춰 자동 계산됩니다."
)

k_value = st.sidebar.slider(
    "3. 사용할 보석 색상 수 (k값)",
    min_value=2,
    max_value=16,
    value=8,
    help="AI가 추출할 대표 보석 색상의 개수(군집 수)입니다."
)

max_iter_value = st.sidebar.slider(
    "4. AI 알고리즘 반복 횟수 (Learning Steps)",
    min_value=1,
    max_value=10,
    value=2,
    step=1,
    help="반복 횟수를 늘릴수록 3D 공간에서 보석 중심점들이 최적의 무게중심으로 수렴합니다."
)

# 3. 메인 로직 작동
if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    img_np = np.array(img)
    orig_h, orig_w, _ = img_np.shape
    
    # 변경포인트 1-2: 원본 비율에 맞춘 세로 픽셀(칸) 수 자동 연산
    aspect_ratio = orig_h / orig_w
    target_height = int(target_width * aspect_ratio)
    
    # 탭 구성
    tab1, tab2 = st.tabs(["🖼️ 보석십자수 도안화 결과", "📊 색상수 및 최소 표현 비트(Bit) 분석"])
    
    # --- 데이터 전처리 및 연산 ---
    # 학생들이 입력한 가로/세로 칸수로 이미지 축소
    img_small = img.resize((target_width, target_height), resample=Image.Resampling.BILINEAR)
    small_np = np.array(img_small)
    pixels = small_np.reshape(-1, 3)
    
    # K-Means 알고리즘 적용
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
            st.subheader(f"AI 보석십자수 도안 (비율 맞춤 연산 결과)")
            st.image(output_img, use_container_width=True)
            # 가로에 따른 세로 연산 결과 안내
            st.success(f"📐 연산 결과: 가로 {target_width}칸 × 세로 {target_height}칸 (그림 비율 {orig_w}:{orig_h})")
            st.caption(f"총 보석 개수: {target_width * target_height:,} 개")

        st.markdown("---")
        st.markdown(f"### 🎨 AI가 선정한 보석 팔레트 (선택된 색상 수: k={k_value})")
        palette_cols = st.columns(min(k_value, 10))
        for idx, color_hex in enumerate(hex_colors):
            with palette_cols[idx % 10]:
                st.markdown(f"<div style='background-color:{color_hex}; height:50px; border-radius:8px; border:2px solid #FFF;'></div>", unsafe_allow_html=True)
                st.caption(f"**보석 {idx+1}**\n{color_hex}")

    # --- TAB 2: 데이터 분석 및 최소 비트수 비교 ---
    with tab2:
        st.subheader("📉 색상 데이터 압축 및 이진수 표현 비트(Bit) 분석")
        st.write("컴퓨터가 아날로그 색상을 디지털로 표현할 때 필요한 최소 비트 수의 변화를 정량적으로 비교합니다.")
        
        # 변경포인트 2: 변환 전 실제 고유 색상 수 카운트
        # 원본 소형 이미지(도안의 해상도 기준)에서 고유한 RGB 조합의 개수를 계산합니다.
        unique_colors_before = len(np.unique(pixels, axis=0))
        
        # 최소 필요 비트 수 계산 공식: ceil(log2(색상수))
        # 만약 고유 색상이 1개 이하인 극단적 예외 상황을 방지하기 위해 max(1, ...) 처리
        bits_needed_before = math.ceil(math.log2(max(2, unique_colors_before)))
        bits_needed_after = math.ceil(math.log2(k_value))
        
        # 표 데이터 구성
        data_comparison = {
            "항목": ["변환 전 (도안 격자화 직후)", "변환 후 (AI 보석십자수 도안)"],
            "사용한 색상 수": [f"{unique_colors_before:,} 가짓수", f"{k_value:,} 가짓수 (k값)"],
            "1개 칸(픽셀)당 최소 표현 비트 수": [f"{bits_needed_before} Bits", f"{bits_needed_after} Bits"],
            "컴퓨터 표준 저장 비트 (참고)": ["24 Bits (True Color)", f"{bits_needed_after} Bits (인덱스 컬러 색상표 적용)"]
        }
        df_comp = pd.DataFrame(data_comparison)
        st.table(df_comp)
        
        # 정보 교과 이론 설명 보충
        st.markdown(
            f"💡 **정보 과학 개념 가이드:**\n"
            f"- 변환 전 그림에는 미묘하게 다른 색상이 총 **{unique_colors_before:,}가지**나 존재하여, 이를 구별해 저장하려면 이진수 최소 **{bits_needed_before}자리(Bits)**가 필요합니다.\n"
            f"- 하지만 AI가 **{k_value}개**의 군집으로 묶어버린 후에는, 각 칸에 어떤 보석을 박을지 결정하는 데 단 **{bits_needed_after}자리(Bits)**의 이진수 정보만 있으면 충분합니다.\n"
            f"- 결과적으로 색상 표현을 위한 칸당 데이터 용량이 약 **{((bits_needed_before - bits_needed_after)/bits_needed_before)*100:.1f}% 감소**하는 디지털 손실 압축이 일어났습니다."
        )
        
        st.markdown("---")
        st.subheader("📊 3D RGB 색상 공간 군집화 그래프")
        
        # Plotly 3D 산점도 빌드
        df_pixels = pd.DataFrame(pixels, columns=['Red', 'Green', 'Blue'])
        df_pixels['Cluster'] = [f"군집 {l+1}" for l in labels]
        df_centroids = pd.DataFrame(centroids, columns=['Red', 'Green', 'Blue'])
        df_centroids['Cluster'] = [f"★보석 {i+1} (중심점)" for i in range(k_value)]
        df_total = pd.concat([df_pixels, df_centroids], ignore_index=True)
        
        fig = px.scatter_3d(
            df_total, x='Red', y='Green', z='Blue', color='Cluster', opacity=0.6,
            size=[3 if "★" not in str(c) else 15 for c in df_total['Cluster']],
            symbol=['circle' if "★" not in str(c) else 'diamond' for c in df_total['Cluster']],
            range_x=[0, 255], range_y=[0, 255], range_z=[0, 255],
            title=f"3D 공간 색상 데이터 뭉치기와 중심점 (반복: {max_iter_value}회)"
        )
        fig.update_layout(margin=dict(l=0, r=0, b=0, t=40))
        st.plotly_chart(fig, use_container_width=True)

else:
    st.info("👈 왼쪽 사이드바에서 명화 이미지를 업로드하면 비율 맞춤 도안 및 이진수 비트 압축 분석기가 작동합니다.")
