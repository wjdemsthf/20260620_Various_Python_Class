import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# 1. 웹앱 페이지 기본 설정
st.set_page_config(
    page_title="AI 미술 보존과학자",
    page_icon="🎨",
    layout="wide"
)

# 헤더 부분
st.title("🎨 AI 미술 보존과학자 프로젝트")
st.markdown("### **k-평균 군집화(k-Means)를 이용한 디지털 명화 색상 복원**")
st.write("컴퓨터가 이미지를 어떻게 숫자로 인식하고, 비지도학습을 통해 어떻게 대표 색상을 찾아내는지 실험해봅시다.")
st.markdown("---")

# 2. 사이드바 - 조작 포인트 (Control Panel)
st.sidebar.header("⚙️ 실험 설정 (조작 포인트)")

# 샘플 이미지 제공 (학생들이 준비된 이미지가 없을 때를 대비)
sample_option = st.sidebar.selectbox(
    "1. 실험할 명화 선택",
    ["직접 업로드하기", "고흐 - 별이 빛나는 밤", "모네 - 양산 든 여인", "정선 - 인왕제색도"]
)

# 군집 수 k 설정 슬라이더
k_value = st.sidebar.slider(
    "2. 추출할 색상 수 (k값 선택)",
    min_value=2,
    max_value=16,
    value=5,
    help="AI가 이미지를 몇 개의 대표 색상으로 묶을지 결정합니다."
)

st.sidebar.markdown("---")
st.sidebar.info(
    "💡 **탐구 팁:**\n"
    "k값을 아주 작게(2~3) 설정했을 때와 크게(12 이상) 설정했을 때, "
    "그림의 분위기와 복원도가 어떻게 달라지는지 관찰해보세요."
)

# 3. 이미지 로드 로직
img = None

if sample_option == "직접 업로드하기":
    uploaded_file = st.sidebar.file_uploader("명화 이미지 파일 업로드 (JPG, PNG)", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        img = Image.open(uploaded_file)
else:
    # 예시용 샘플 이미지 경로 (실제 배포 시에는 인터넷 URL이나 같은 폴더 내 이미지 경로 지정 가능)
    # 여기서는 안내 메시지로 대체하거나 예시 처리를 합니다.
    st.sidebar.warning(f"'{sample_option}' 샘플을 선택했습니다. 아래 [AI 미술 복원 실행] 버튼을 눌러주세요.")
    # 임의의 테스트용 데이터 생성 (실제 사용시 이미지 파일 준비 권장)
    if sample_option == "고흐 - 별이 빛나는 밤":
        img = Image.new('RGB', (300, 200), color='#1C39BB') # 임시 단색 이미지
    elif sample_option == "모네 - 양산 든 여인":
        img = Image.new('RGB', (300, 200), color='#7FAFD4')
    else:
        img = Image.new('RGB', (300, 200), color='#D3D3D3')

# 4. 메인 화면 구성 (2단 레이아웃)
if img is not None:
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🖼️ 원본 이미지")
        st.image(img, use_container_width=True)
        
        # 데이터 구조 시각화 (정보 교과 학습 요소)
        img_np = np.array(img)
        h, w, c = img_np.shape
        st.markdown(f"**📊 이미지 데이터 정보:**")
        st.caption(f"- 가로: {w} 픽셀, 세로: {h} 픽셀")
        st.caption(f"- 총 픽셀 수: {w} × {h} = {w*h:,} 개")
        st.caption(f"- 하나의 픽셀은 3차원 데이터로 구성: **[R, G, B]**")

    with col2:
        st.subheader("🤖 AI 알고리즘 복원 결과")
        
        # 버튼을 눌러야 알고리즘이 작동하도록 설정 (과부하 방지)
        if st.button("🚀 AI 미술 복원 실행", type="primary"):
            with st.spinner("AI가 수십만 개의 픽셀 색상을 분석 중입니다..."):
                
                # [데이터 전처리] 0~1 사이로 정규화 및 2차원 배열로 변환
                img_float = img_np / 255.0
                pixels = img_float.reshape(-1, 3)
                
                # [인공지능 모델 학습] K-Means 알고리즘 적용
                kmeans = KMeans(n_clusters=k_value, random_state=42, n_init=10)
                kmeans.fit(pixels)
                
                # [예측 및 복원] 각 픽셀을 가장 가까운 중심점(대표 색상)으로 치환
                labels = kmeans.labels_
                centroids = kmeans.cluster_centers_
                
                # 대표 색상으로만 채워진 새로운 이미지 생성
                quantized_pixels = centroids[labels]
                quantized_img = quantized_pixels.reshape(h, w, c)
                
                # 결과 이미지 출력
                st.image(quantized_img, use_container_width=True)
                st.success(f"성공적으로 {k_value}개의 색상으로 복원했습니다!")
                
                # 5. 화가의 팔레트 (추출된 대표 색상) 시각화
                st.markdown("### 🎨 AI가 추출한 화가의 대표 팔레트")
                
                # 팔레트를 그리기 위한 데이터프레임 구성
                hex_colors = []
                for color in centroids:
                    # RGB 값을 0~255 범위의 정수로 변환 후 Hex 코드로 변환
                    r, g, b = int(color[0]*255), int(color[1]*255), int(color[2]*255)
                    hex_colors.append(f"#{r:02x}{g:02x}{b:02x}")
                
                # 화면에 가로로 유색 박스 출력하기
                palette_cols = st.columns(k_value)
                for idx, col in enumerate(palette_cols):
                    with col:
                        # Streamlit 마크다운을 활용해 커스텀 색상 박스 그리기
                        st.markdown(
                            f"<div style='background-color:{hex_colors[idx]}; height:60px; border-radius:5px; border:1px solid #777;'></div>", 
                            unsafe_allow_html=True
                        )
                        st.caption(f"색상 {idx+1}\n{hex_colors[idx]}")
                        
                # 6. 탐구 활동용 데이터 테이블 출력
                with st.expander("📊 AI 내부 연산 데이터 확인하기"):
                    st.write("알고리즘에 의해 계산된 최종 중심점(Centroids)의 RGB 좌표값입니다.")
                    df_centroids = pd.DataFrame(centroids, columns=['Red', 'Green', 'Blue'])
                    st.dataframe(df_centroids)

else:
    # 이미지가 업로드되지 않았을 때의 초기 화면
    st.info("👈 왼쪽 사이드바에서 실험할 명화를 선택하거나 이미지를 업로드해 주세요!")
