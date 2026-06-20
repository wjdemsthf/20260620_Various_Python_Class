import streamlit as st
import requests
import os
import pandas as pd
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("RIOT_API_KEY")

st.set_page_config(page_title="챔피언 스탯 분석", page_icon="📊")
st.title("📊 챔피언 숙련도 분석")

# 라이엇 Data Dragon 챔피언 매칭 함수
@st.cache_data
def get_champion_clean_data():
    url = "https://ddragon.leagueoflegends.com/cdn/14.23.1/data/ko_KR/champion.json"
    res = requests.get(url).json()
    champions = res['data']
    id_to_name = {meta['key']: meta['name'] for name, meta in champions.items()}
    return id_to_name

champ_dict = get_champion_clean_data()

col1, col2 = st.columns([3, 1])
with col1:
    game_name = st.text_input("소환사 이름", placeholder="Hide on bush", key="champ_name")
with col2:
    tag_line = st.text_input("태그", placeholder="KR1", key="champ_tag")

if st.button("숙련도 조회"):
    if not game_name or not tag_line:
        st.warning("이름과 태그를 모두 입력해주세요.")
    else:
        with st.spinner("숙련도 데이터를 분석 중..."):
            try:
                # 1. PUUID 조회
                account_url = f"https://asia.api.riotgames.com/riot/account/v1/accounts/by-riot-id/{game_name}/{tag_line}?api_key={API_KEY}"
                account_res = requests.get(account_url)
                
                if account_res.status_code != 200:
                    st.error("소환사를 찾을 수 없습니다.")
                else:
                    puuid = account_res.json()['puuid']
                    
                    # 2. Champion Mastery API 호출 (한국 서버 기준 kr 라우팅 이용)
                    mastery_url = f"https://kr.api.riotgames.com/lol/champion-mastery/v4/champion-masteries/by-puuid/{puuid}?api_key={API_KEY}"
                    mastery_data = requests.get(mastery_url).json()
                    
                    # TOP 5 챔피언 추출 및 가공
                    top_5 = mastery_data[:5]
                    
                    chart_data = []
                    for i, champ in enumerate(top_5, 1):
                        champ_id = str(champ['championId'])
                        champ_name = champ_dict.get(champ_id, f"Unknown({champ_id})")
                        points = champ['championPoints']
                        level = champ['championLevel']
                        
                        st.write(f"🏅 **TOP {i}** | {champ_name} (레벨 {level}) - {points:,} 점")
                        chart_data.append({"챔피언": champ_name, "숙련도 점수": points})
                    
                    # 3. 데이터프레임 변환 후 차트 그리기
                    df = pd.DataFrame(chart_data)
                    df = df.set_index("챔피언")
                    
                    st.write("---")
                    st.subheader("모스트 챔피언 점수 비교")
                    st.bar_chart(df)
                    
            except Exception as e:
                st.error(f"오류가 발생했습니다: {e}")
