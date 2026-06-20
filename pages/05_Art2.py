import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.cluster import KMeans
import plotly.express as px

# 1. 페이지 기본 설정
st.set_page_config(
    page_title="AI 보석십자수 & 데이터 시각화",
    page_icon="💎",
    layout="wide"
)

st.title("💎 AI 보석십자수: 군집화 및 데이터 시각화 실습")
st.markdown("### **정보(3D 색상 공간과 k-평균 군집화) × 미술(보석십자수 도안) 융합 수업**")
st.write("인공지능이 무한한 색상 데이터를 3차원 공간에서 어떻게 분류하고 대표 색상(중심점)을 선정하는지 시각적으로 확인해봅시다.")
st.markdown("---")

# 2. 사이드바 - 조작 포인트 설정
st.sidebar.header("⚙️ 도안 제작 및 알고리즘 제어")

uploaded_file = st.sidebar.file_uploader("1. 명화 이미지 업로드 (JPG, PNG)", type=["jpg", "jpeg", "png"])

pixel_size = st.sidebar.slider(
    "2. 보석 1칸의 크기 (공간 해상도)",
    min_value=6,
    max_value=30,
    value=12,
    step=2,
    help="값이 커질수록 격자가 커지고 3D 그래프에 표시될 데이터 포인트(픽셀) 수가 줄어들어 연산이 빨라집니다."
)

k_value = st.sidebar.slider(
    "3. 사용할 보석 색상 수 (k값)",
    min_value=2,
    max_value=10,
    value=4,
    help="AI가 추출할 대표 보석 색상의 개수이자, 3D 공간에 생성될 군집의 수입니다."
)

max_iter_value = st.sidebar.slider(
    "4. AI 알고리즘 반복 횟수 (Learning Steps)",
    min_value=1,
    max_value=10,
    value=1,
    step=1,
    help="1회부터 시작해 늘려보세요. 무작위였던 중심점들이 픽셀들의 무게중심으로 이동하며 색상이 정교해집니다."
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    "### 💡 시각화 관찰 가이드\n"
    "1. **반복 횟수를 1**로 두면 AI가 색상을 무작위로 선택해 도안과 그래프의 중심점이 어색합니다.\n"
    "2. **반복 횟수를 늘릴수록** 3D 공간의 데이터 무리 한가운데로 보석 중심점들이 정렬되는 과정을 볼 수 있습니다."
)

# 3. 메인 로직 작동
if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    img_np = np.array(img)
    orig_h, orig_w, _ = img_np.shape
    
    # 탭 구성
    tab1, tab2 = st.tabs(["🖼️ 보석십자수 도안 및 AI 학습 과정", "📊 3D 데이터 시각화 및 압축률 분석"])
    
    # --- 데이터 전처리 및 연산 ---
    low_res_w = max(1, orig_w // pixel_size)
    low_res_h = max(1, orig_h // pixel_size)
    img_small = img.resize((low_res_w, low_res_h), resample=Image.Resampling.BILINEAR)
    small_np = np.array(img_small)
    
    # 3차원 RGB 데이터를 2차원 행렬로 변환 (0~255 값 유지 - 그래프 시각화 직관성을 위해)
    pixels = small_np.reshape(-1, 3)
    
    # K-Means 알고리즘 적용 (과정 추적을 위해 init='random', n_init=1 설정)
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
    quantized_small_np = quantized_pixels.reshape(low_res_h, low_res_w, 3).astype(np.uint8)
    output_img = Image.fromarray(quantized_small_np).resize((orig_w, orig_h), resample=Image.Resampling.NEAREST)
    
    # Hex 컬러 변환
    hex_colors = []
    for color in centroids:
        r, g, b = int(color[0]), int(color[1]), int(color[2])
        hex_colors.append(f"#{r:02x}{g:02x}{b:02x}")
        
    # --- TAB 1: 도안 및 팔레트 출력 ---
    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("원본 명화 이미지")
            st.image(img, use_container_width=True)
            st.caption(f"원본: {orig_w} × {orig_h} 픽셀")
            
        with col2:
            st.subheader(f"AI 보석십자수 도안 (학습 반복: {max_iter_value}회)")
            st.image(output_img, use_container_width=True)
            st.caption(f"도안: 가로 {low_res_w}칸 × 세로 {low_res_h}칸 (총 {low_res_w*low_res_h:,}개 보석)")

        st.markdown("---")
        st.markdown(f"### 🎨 현재 단계에서 AI가 선정한 보석 팔레트")
        
        palette_cols = st.columns(min(k_value, 10))
        for idx, color_hex in enumerate(hex_colors):
            col_target = palette_cols[idx % 10]
            with col_target:
                st.markdown(
                    f"<div style='background-color:{color_hex}; height:50px; border-radius:8px; border:2px solid #FFF;'></div>", 
                    unsafe_allow_html=True
                )
                st.caption(f"**보석 {idx+1}**\n{color_hex}")

    # --- TAB 2: 3D 시각화 및 데이터 분석 ---
    with tab2:
        st.subheader("📊 3D RGB 색상 공간에서의 군집화 시각화")
        st.write("그림 속 모든 픽셀들을 빨강(R), 초록(G), 파랑(B) 축을 가진 3차원 공간에 점으로 찍은 그래프입니다.")
        st.info("💡 **그래프 조작법:** 마우스로 그래프를 드래그하면 3차원으로 회전 시켜볼 수 있습니다. 큰 다이아몬드(◆) 표시가 AI가 찾아낸 현재의 보석 색상(중심점)입니다.")
        
        # Plotly용 데이터프레임 빌드
        df_pixels = pd.DataFrame(pixels, columns=['Red', 'Green', 'Blue'])
        df_pixels['Cluster'] = [f"군집 {l+1}" for l in labels]
        
        # 실제 픽셀의 색상을 그래프 점 색상으로 표현하기 위해 Hex 값 부여
        df_pixels['Hex'] = [f"#{p[0]:02x}{p[1]:02x}{p[2]:02x}" for p in pixels]
        
        # 중심점(Centroids) 데이터도 그래프에 표시하기 위해 추가
        df_centroids = pd.DataFrame(centroids, columns=['Red', 'Green', 'Blue'])
        df_centroids['Cluster'] = [f"★보석 {i+1} (중심점)" for i in range(k_value)]
        df_centroids['Hex'] = hex_colors
        
        # 전체 데이터 통합
        df_total = pd.concat([df_pixels, df_centroids], ignore_index=True)
        
        # 3D 산점도 그리기
        fig = px.scatter_3d(
            df_total, 
            x='Red', y='Green', z='Blue',
            color='Cluster',
            opacity=0.6,
            size=[3 if "★" not in str(c) else 15 for c in df_total['Cluster']], # 중심점은 크게 표시
            symbol=[ 'circle' if "★" not in str(c) else 'diamond' for c in df_total['Cluster']],
            range_x=[0, 255], range_y=[0, 255], range_z=[0, 255],
            title=f"k-Means 색상 데이터 분포 (반복 횟수: {max_iter_value})"
        )
        
        fig.update_layout(margin=dict(l=0, r=0, b=0, t=40), scene=dict(
            xaxis_title='Red (0~255)',
            yaxis_title='Green (0~255)',
            zaxis_title='Blue (0~255)'
        ))
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        st.subheader("📉 데이터 손실 압축률 결과")
        
        orig_bytes = orig_w * orig_h * 3
        compressed_bytes = (low_res_w * low_res_h * 1) + (k_value * 3)
        compression_ratio = (1 - (compressed_bytes / orig_bytes)) * 100
        
        c_p1, c_p2, c_p3 = st.columns(3)
        with c_p1:
            st.metric(label="원래 이미지 데이터 용량", value=f"{orig_bytes:,} Bytes")
        with c_p2:
            st.metric(label="AI 도안 압축 후 용량", value=f"{compressed_bytes:,} Bytes")
        with c_p3:
            st.metric(label="최종 데이터 압축 효율", value=f"{compression_ratio:.2f}% 감소")

else:
    st.info("👈 왼쪽 사이드바에서 이미지를 업로드하시면 3D 색상 시각화 대시보드가 실시간으로 생성됩니다.")
