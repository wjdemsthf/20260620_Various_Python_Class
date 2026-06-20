import streamlit as st
import requests
import pandas as pd
import numpy as np
import datetime
import plotly.express as px
from sklearn.linear_model import LinearRegression  # 예측 기능 사용 시 필요

# --- 여기서부터 기존 코드가 시작됩니다 ---
st.set_page_config(
    page_title="실시간 지진 트래커 & 교육용 예측기",
    page_icon="🌋",
    layout="wide"
)

st.title("🌋 실시간 지진 모니터링 및 교육용 예측 대시보드")
st.markdown("---")  # 에러가 났던 지점
-----------------------------------------------------------------------------
# [추가 기능] 머신러닝 기반 향후 지진 추세 예측 모듈
# -----------------------------------------------------------------------------

st.subheader("🤖 데이터 과학 기반 향후 지진 추세 예측 (Machine Learning)")

# 시계열 학습을 위해 일자별 지진 발생 횟수 데이터셋 생성
df_daily = df.copy()
df_daily['날짜'] = df_daily['일시'].dt.date
daily_counts = df_daily.groupby('날짜').size().reset_index(name='발생횟수')
daily_counts = daily_counts.sort_values('날짜')

if len(daily_counts) < 3:
    st.info("💡 데이터 기간이 너무 짧아 머신러닝 추세 예측을 진행할 수 없습니다. 사이드바에서 조회 기간을 더 길게 설정해 주세요.")
else:
    # 1. 인덱스를 이용한 날짜의 연속적 수치화 (X: 독립변수, y: 종속변수)
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
    
    # 화면 레이아웃 구성
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
            
        st.markdown(f"*(학습 모델의 기울기(Slope): `{slope:.4f}`)*")
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
