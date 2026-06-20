import streamlit as st
import requests

# 로컬(.env)이든 클라우드(Secrets)든 Streamlit이 알아서 키를 찾아옵니다!
API_KEY = st.secrets["RIOT_API_KEY"]

st.set_page_config(page_title="최근 전적 검색", page_icon="⚔️")
st.title("⚔️ 최근 전적 조회")

# 라이엇 Data Dragon에서 최신 챔피언 정보 가져오기 (ID 변환용)
@st.cache_data
def get_champion_clean_data():
    # 2026년 기준 최신 버전 적용 (필요시 버전 문자열 업데이트)
    url = "https://ddragon.leagueoflegends.com/cdn/14.23.1/data/ko_KR/champion.json"
    res = requests.get(url).json()
    champions = res['data']
    # { '266': '아트록스', ... } 구조로 변경
    id_to_name = {meta['key']: meta['name'] for name, meta in champions.items()}
    return id_to_name

champ_dict = get_champion_clean_data()

# 검색 UI
col1, col2 = st.columns([3, 1])
with col1:
    game_name = st.text_input("소환사 이름", placeholder="Hide on bush")
with col2:
    tag_line = st.text_input("태그", placeholder="KR1")

if st.button("전적 검색"):
    if not game_name or not tag_line:
        st.warning("이름과 태그를 모두 입력해주세요.")
    else:
        with st.spinner("라이엇 서버에서 전적을 가져오는 중..."):
            try:
                # 1. Account-V1 API로 PUUID 조회
                account_url = f"https://asia.api.riotgames.com/riot/account/v1/accounts/by-riot-id/{game_name}/{tag_line}?api_key={API_KEY}"
                account_res = requests.get(account_url)
                
                if account_res.status_code != 200:
                    st.error("소환사를 찾을 수 없거나 API 키가 만료되었습니다.")
                else:
                    puuid = account_res.json()['puuid']
                    
                    # 2. Match-V5 API로 최근 5게임 Match ID 조회
                    match_list_url = f"https://asia.api.riotgames.com/lol/match/v5/matches/by-puuid/{puuid}/ids?start=0&count=5&api_key={API_KEY}"
                    match_ids = requests.get(match_list_url).json()
                    
                    st.subheader(f"🔥 {game_name}#{tag_line} 님의 최근 5게임 결과")
                    
                    # 3. 각 매치 상세 정보 가져오기
                    for match_id in match_ids:
                        detail_url = f"https://asia.api.riotgames.com/lol/match/v5/matches/{match_id}?api_key={API_KEY}"
                        match_detail = requests.get(detail_url).json()
                        
                        # 내 플레이어 정보 찾기
                        participants = match_detail['info']['participants']
                        my_data = next(p for p in participants if p['puuid'] == puuid)
                        
                        # 데이터 가공
                        win = "승리" if my_data['win'] else "패배"
                        champ_id = str(my_data['championId'])
                        champ_name = champ_dict.get(champ_id, f"챔피언({champ_id})")
                        kda = f"{my_data['kills']} / {my_data['deaths']} / {my_data['assists']}"
                        
                        # Streamlit UI로 시각화
                        bg_color = "#D6E4FF" if my_data['win'] else "#FFD8D8"
                        st.markdown(
                            f"""
                            <div style="background-color: {bg_color}; padding: 15px; border-radius: 8px; margin-bottom: 10px; color: black;">
                                <strong>[{win}]</strong> {champ_name} | <strong>KDA:</strong> {kda} | 딜량: {my_data['totalDamageDealtToChampions']:,}
                            </div>
                            """, 
                            unsafe_allow_html=True
                        )
            except Exception as e:
                st.error(f"오류가 발생했습니다: {e}")
