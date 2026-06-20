import streamlit as st

# 1. 페이지 기본 설정 (가장 상단에 위치해야 합니다)
st.set_page_config(
    page_title="파이썬 웹 앱 연수",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. 사이드바 구성
with st.sidebar:
    st.title("⚙️ 연수 메뉴")
    st.info("왼쪽 메뉴를 통해 원하는 실습 페이지로 이동하세요.")
    
    st.markdown("---")
    st.markdown("### 👨‍💻 강사 및 문의")
    st.caption("이름: 정보 교사")
    st.caption("이메일: teacher@school.kr")

# 3. 메인 화면 대문 구성
# 헤더 부분 (타이틀 및 소개)
st.title("🚀 파이썬 & 스트림릿 웹 앱 프로그래밍 연수")
st.subheader("Python & Streamlit Web App Workshop")
st.markdown("""
    스트림릿(Streamlit)을 활용하여 복잡한 프론트엔드 지식 없이도 
    **단 몇 줄의 파이썬 코드만으로 강력한 웹 애플리케이션을 제작**하는 연수 페이지입니다.
""")

st.markdown("---")

# 4. 주요 특징 / 연수 내용 (3열 레이아웃 구성)
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("### ⚡ 생산성 극대화")
    st.write("HTML, CSS, JavaScript 없이 오직 파이썬 코드로만 UI를 빠르게 빌드합니다.")

with col2:
    st.markdown("### 📊 데이터 시각화")
    st.write("Pandas 데이터프레임, Plotly, Matplotlib 등 다양한 시각화 차트를 손쉽게 연동합니다.")

with col3:
    st.markdown("### 🤖 컴포넌트 활용")
    st.write("버튼, 슬라이더, 텍스트 입력창 등 풍부한 위젯을 활용해 인터랙티브한 앱을 만듭니다.")

st.markdown("---")

# 5. 하단 안내 및 인터랙션 피드백 구역
st.markdown("### 📌 연수 시작하기")
st.write("아래 버튼을 눌러 오늘 연수에 임하는 다짐을 체크해 보세요!")

# 간단한 상호작용 예시
ready = st.checkbox("오늘 멋진 웹 앱을 만들어 볼 준비가 되셨나요?")

if ready:
    st.success("🎉 좋습니다! 사이드바의 메뉴를 선택해 첫 번째 실습을 시작해 보세요!")
    st.balloons()
else:
    st.warning("위 체크박스를 눌러 준비 상태를 완료해 주세요!")

# 6. 푸터(Footer)
st.markdown("<br><br>", unsafe_allow_html=True)
st.caption("© 2026 파이썬 교사 연수 프로그램. All rights reserved.")
