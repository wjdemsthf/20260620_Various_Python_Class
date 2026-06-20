import streamlit as st
import requests
import pandas as pd
import numpy as np
import datetime
import plotly.express as px
from sklearn.linear_model import LinearRegression

# -----------------------------------------------------------------------------
# 1. 페이지 기본 설정 및 레이아웃 (반드시 코드 최상단에 위치해야 합니다)
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="실시간 지진 트래커 & AI 예측기",
    page_icon="🌋",
    layout="wide"
)

st.title("🌋 실시간 지진 모니터링 및 교육용 예측 대시보드")
st.markdown("""
이 애플리케이션은 **미국 지질조사국(USGS) API**로부터 실시간 전 세계 지진 데이터를 가져옵니다.  
학생들이 관심 있는 지역의 지진 발생 패턴을 분석하고, 머신러닝 기법을 통해 미래 추세를 데이터 과학 관점으로 예측해 봅니다.
""")

# -----------------------------------------------------------------------------
# 2. 사이드바 컨트롤 (데이터 필터 설정)
# -----------------------------------------------------------------------------
st.sidebar.header("🔍 검색 조건 설정")

# 날짜 범위 선택
today = datetime.date.today()
start_date = st.sidebar.date_input("조회 시작일", today - datetime.timedelta(days=30))
end_date = st.sidebar.date_input("조회 종료일", today)

# 지진 규모(Magnitude) 필터
min_magnitude = st.sidebar.slider("최소 지진 규모 (M)", 0.0, 9.0, 2.5, 0.5)

# 미리 정의된 주요 관심 지역 좌표 [위도, 경도, 검색 반경(km)]
regions = {
    "전 세계 (Global)": None,
    "대한민국 및 주변부": [35.9078, 127.7669, 500],
    "일본 및 주변부": [36.2048, 138.2529, 1000],
    "미국 서부 (캘리포니아)": [36.7783, -119.4179, 800],
    "사용자 직접 지정": "custom"
}

selected_region = st.sidebar.selectbox("분석 대상 지역 선택", list(regions.keys()))

lat, lon, radius = None, None, None
if selected_region == "사용자 직접 지정":
    lat = st.sidebar.number_input("위도 (Latitude)", -90.0, 90.0, 37.5665)  # 기본값: 서울
    lon = st.sidebar.number_input("경도 (Longitude)", -180.0, 180.0, 126.9780)
    radius = st.sidebar.slider("검색 반경 (km)", 10, 2000, 500)
elif regions[selected_region] is not None:
    lat, lon, radius = regions[selected_region]

# 데이터 수동 새로고침 버튼
if st.sidebar.button("🔄 실시간 데이터 업데이트"):
    st.cache_data.clear()
    st.rerun()

# -----------------------------------------------------------------------------
# 3. USGS API 데이터 수집 함수
# -----------------------------------------------------------------------------
@st.cache_data(ttl=300)  # 5분 동안 데이터 캐싱 (지속적인 API 호출 방지)
def fetch_earthquake_data(start, end, min_mag, lat=None, lon=None, max_rad_km=None):
    base_url = "https://earthquake.usgs.gov/fdsnws/event/1/query"
    params = {
        "format": "geojson",
        "starttime": start.strftime("%Y-%m-%d"),
        "endtime": end.strftime("%Y-%m-%d"),
        "minmagnitude": min_mag,
        "orderby": "time"
    }
    
    # 지역 필터가 설정된 경우 파라미터 추가
    if lat is not None and lon is not None and max_rad_km is not None:
        params["latitude"] = lat
        params["longitude"] = lon
        params["maxradiuskm"] = max_rad_km

    try:
        response = requests.get(base_url, params=params, timeout=10)
        if response.status_code == 200:
            data = response.json()
            features = data.get("features", [])
            
            eq_list = []
            for f in features:
                props = f["properties"]
                geom = f["geometry"]
                eq_list.append({
                    "일시": pd.to_datetime(props["time"], unit="ms"),
                    "규모": props["mag"],
                    "위치": props["place"],
                    "위도": geom["coordinates"][1],
                    "경도": geom["coordinates"][0],
                    "진원깊이_km": geom["coordinates"][2]
                })
            return pd.DataFrame(eq_list)
        else:
            st.error(f"API 오류: 상태 코드 {response.status_code}")
            return pd.DataFrame()
    except Exception as e:
        st.error(f"USGS API 연결 실패: {e}")
        return pd.DataFrame()

# 데이터 로드
df = fetch_earthquake_data(start_date, end_date, min_magnitude, lat, lon, radius)

# -----------------------------------------------------------------------------
# 4. 대시보드 UI 및 시각화 구동
# -----------------------------------------------------------------------------
if df.empty:
    st.warning("⚠️ 선택하신 조건에 해당하는 지진 데이터가 없습니다. 검색 조건을 넓혀보세요.")
else:
    # (1) 주요 지표 요약 (Key Metrics)
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("총 지진 발생 횟수", f"{len(df)} 건")
    with col2:
        st.metric("최대 규모", f"M {df['규모'].max():.1f}")
    with col3:
        st.metric("평균 규모", f"M {df['규모'].mean():.1f}")
    with col4:
        st.metric("최고 진원 깊이", f"{df['진원깊이_km'].max():.1f} km")

    st.markdown("---")

    # (2) 실시간 지진 발생 위치 지도
    st.subheader("🗺️ 실시간 지진 발생 지도")
    st.markdown("지도 위의 점 크기는 지진의 강도(규모)를 나타냅니다.")
    
    map_df = df.copy()
    # 지도가 시각적으로 명확하게 보이도록 규모에 따른 크기 보정
    map_df['원크기'] = (map_df['규모'] ** 2) * 2 
    st.map(map_df, latitude='위도', longitude='경도', size='원크기')

    st.markdown("---")

    # (3) 통계 분석 차트
    st.subheader("📊 지진 통계 데이터 시각화")
    chart_col1, chart_col2 = st.columns(2)
    
    with chart_col1:
        st.markdown("**[차트 1] 지진 규모별 빈도수 (히스토그램)**")
        fig_hist = px.histogram(df, x="규모", nbins=15, labels={"규모": "지진 규모 (Magnitude)"}, 
                                color_discrete_sequence=['#ef553b'])
        st.plotly_chart(fig_hist, use_container_width=True)
        
    with chart_col2:
        st.markdown("**[차트 2] 시간 흐름에 따른 지진 규모와 깊이**")
        fig_scatter = px.scatter(df, x="일시", y="규모", size="규모", color="진원깊이_km",
                                 labels={"일시": "발생 시간", "규모": "규모", "진원깊이_km": "깊이 (km)"},
                                 color_continuous_scale=px.colors.sequential.Plasma)
        st.plotly_chart(fig_scatter, use_container_width=True)

    # (4) 교육용 지진 확률 및 리스크 분석 모듈 (수학적 모델)
    st.markdown("---")
    st.subheader("🔮 교육용 지진 확률 예측 (수학적 리스크 분석)")
    
    with st.expander("💡 학생들을 위한 필수 과학 지식 읽어보기", expanded=True):
        st.info("""
        **"지진은 정확히 언제, 어디서 일어날지 초 단위로 맞출 수 있을까요?"** 현재 현대 과학 기술로는 특정 지진의 정확한 날짜와 시각을 예측하는 것이 불가능합니다.  
        대신 과학자들은 **과거에 지진이 얼마나 자주 일어났는지(통계)**를 바탕으로, 향후 특정 기간 내에 지진이 일어날 **'확률'**을 계산합니다.
        
        아래 모듈은 학생들이 설정한 기간과 지역의 데이터를 기반으로 **포아송 분포(Poisson Distribution)** 모델을 적용해 수립한 통계적 예측입니다.
        """)
    
    # 조회 기간 계산
    days_range = (end_date - start_date).days
    if days_range <= 0:
        days_range = 1
        
    # 일평균 지진 발생률 (λ, 람다) 계산
    lambda_rate = len(df) / days_range
    
    pred_col1, pred_col2 = st.columns(2)
    
    with pred_col1:
        st.markdown("### 📈 선택 지역의 지진 활성 프로파일")
        st.write(f"• 해당 조건 내 하루 평균 지진 발생 횟수: `{lambda_rate:.3f}` 회")
        
        # 일일 발생률에 따른 가상 위험도 등급 및 훈련 가이드라인
        if lambda_rate == 0:
            risk_level = "🟢 매우 낮음 (Very Low)"
            guide = "지진 안전지대로 보이나, 언제나 기본적인 대피 요령은 숙지하고 있어야 합니다."
        elif lambda_rate < 0.3:
            risk_level = "🟡 보통 (Moderate)"
            guide = "간헐적인 지진 활동이 있습니다. 비상용품 위치를 확인해 봅시다."
        elif lambda_rate < 1.5:
            risk_level = "🟠 높음 (High)"
            guide = "지진 활동이 활발한 편입니다! 교실에서 '머리 보호(Drop-Cover-Hold)' 자세를 연습하기에 최적의 데이터입니다."
        else:
            risk_level = "🔴 매우 높음 (Very Active)"
            guide = "지진이 매우 자주 발생하는 판의 경계 근처일 가능성이 높습니다! 정기적인 대피 훈련이 필수적입니다."
            
        st.markdown(f"• 통계 기반 지역 활성도 등급: **{risk_level}**")
        st.markdown(f"• **대피 훈련 추천 가이드:** *{guide}*")

    with pred_col2:
        st.markdown("### 🎲 미래 지진 발생 확률 예측")
        st.write(f"과거 {days_range}일간의 데이터를 바탕으로, 향후 이 지역에 **규모 {min_magnitude} 이상의 지진이 최소 1회 이상 발생할 확률**입니다.")
        
        # 포아송 분포 공식 적용: P(X >= 1) = 1 - e^(-λ * t)
        prob_3day = (1 - np.exp(-lambda_rate * 3)) * 100
        prob_7day = (1 - np.exp(-lambda_rate * 7)) * 100
        
        st.metric(label="향후 3일 이내 발생 확률", value=f"{prob_3day:.1f}%")
        st.metric(label="향후 7일 이내 발생 확률", value=f"{prob_7day:.1f}%")

    # (5) 머신러닝 기반 향후 추세 예측 모듈 (AI 분석)
    st.markdown("---")
    st.subheader("🤖 데이터 과학 기반 향후 지진 추세 예측 (Machine Learning)")

    # 시계열 학습을 위해 일자별 지진 발생 횟수 데이터셋 생성
    df_daily = df.copy()
    df_daily['날짜'] = df_daily['일시'].dt.date
    daily_counts = df_daily.groupby('날짜').size().reset_index(name='발생횟수')
    daily_counts = daily_counts.sort_values('날짜')

    if len(daily_counts) < 3:
        st.info("💡 데이터 기간이 너무 짧아 머신러닝 추세 예측을 진행할 수 없습니다. 사이드바에서 조회 기간을 더 길게 설정해 주세요.")
    else:
        # 인덱스를 이용한 날짜의 연속적 수치화 (X: 독립변수, y: 종속변수)
        daily_counts['날짜순서'] = np.arange(len(daily_counts))
        X = daily_counts[['날짜순서']]
        y = daily_counts['발생횟수']
        
        # 선형 회귀(Linear Regression) 모델 학습
        model = LinearRegression()
        model.fit(X, y)
        
        # 향후 5일간의 미래 날짜 예측 데이터 생성
        future_indices = np.array([[len(daily_counts) + i] for i in range(5)])
        future_predictions = model.predict(future_indices)
        
        # 음수 예측값 방지 (지진 발생 횟수는 최소 0)
        future_predictions = np.clip(future_predictions, 0, None)
        
        # 예측 결과 테이블 구성
        last_date = daily_counts['날짜'].max()
        future_dates = [last_date + datetime.timedelta(days=i+1) for i in range(5)]
        
        pred_df = pd.DataFrame({
            '예측 날짜': future_dates,
            '예측 발생 횟수 (건)': np.round(future_predictions, 1)
        })
        
        ml_col1, ml_col2 = st.columns([1, 1])
        
        with ml_col1:
            st.markdown("### 📈 선형 회귀(Linear Regression) 분석 결과")
            st.write("최근 지진 발생 빈도의 흐름을 분석하여, 지진 활동이 **증가 추세**인지 **감소 추세**인지 파악합니다.")
            
            slope = model.coef_[0]
            if slope > 0.05:
                trend_text = "🔺 **최근 해당 지역의 지진 발생 빈도가 증가하는 추세입니다.**"
                st.warning(trend_text)
            elif slope < -0.05:
                trend_text = "📉 **최근 해당 지역의 지진 발생 빈도가 감소하는 추세입니다.**"
                st.success(trend_text)
            else:
                trend_text = "➡️ **최근 지진 발생 빈도가 일정한 평형 상태를 유지하고 있습니다.**"
                st.info(trend_text)
                
            st.markdown(f"*(학습 모델의 추세선 기울기(Slope): `{slope:.4f}`)*")
            st.dataframe(pred_df, use_container_width=True)
            
        with ml_col2:
            st.markdown("### 📊 향후 5일간 일별 예상 발생 건수")
            
            # 시각화를 위해 기존 데이터와 예측 데이터 병합
            historical_trend = pd.DataFrame({
                '날짜': daily_counts['날짜'],
                '발생횟수': daily_counts['발생횟수'],
                '구분': '과거 실제 데이터'
            })
            future_trend = pd.DataFrame({
                '날짜': future_dates,
                '발생횟수': future_predictions,
                '구분': 'AI 모델 예측치'
            })
            total_trend = pd.concat([historical_trend, future_trend])
            
            # 선형 추세 시각화 차트
            fig_trend = px.line(total_trend, x='날짜', y='발생횟수', color='구분', markers=True,
                                labels={'발생횟수': '일일 지진 발생 횟수'},
                                color_discrete_map={'과거 실제 데이터': '#636EFA', 'AI 모델 예측치': '#EF553B'})
            st.plotly_chart(fig_trend, use_container_width=True)

    # (6) 상세 데이터 로그 테이블
    st.markdown("---")
    st.subheader("📋 상세 지진 기록 로그 (최근 100개 데이터)")
    st.dataframe(df[["일시", "규모", "위치", "진원깊이_km", "위도", "경도"]].head(100), use_container_width=True)
