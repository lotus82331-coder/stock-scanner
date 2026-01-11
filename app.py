import streamlit as st
import FinanceDataReader as fdr
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import requests
import time
from datetime import datetime, timedelta
import pytz
from streamlit_autorefresh import st_autorefresh

# --- [1. 기본 설정 및 자동 갱신] ---
st.set_page_config(page_title="통합 주식 스캐너", layout="wide")
# 5분(300,000ms)마다 앱 자동 새로고침
st_autorefresh(interval=300000, key="auto_refresh")

# 한국 시간 설정
KST = pytz.timezone('Asia/Seoul')
now_kst = datetime.now(KST)

st.title("🚀 3-전략 통합 실시간 스캐너")
st.caption(f"최근 갱신 시각: {now_kst.strftime('%Y-%m-%d %H:%M:%S')} (5분마다 자동 업데이트)")

# --- [2. 사이드바 및 보안 설정] ---
with st.sidebar:
    st.header("⚙️ 설정")
    # Streamlit Cloud의 Secrets에 저장한 값을 불러옵니다.
    # 만약 로컬 테스트 중이라면 직접 문자열을 넣어도 되지만, GitHub 업로드 시엔 아래 형태를 유지하세요.
    try:
        TELEGRAM_TOKEN = st.secrets["TELEGRAM_TOKEN"]
    except:
        TELEGRAM_TOKEN = st.text_input("Telegram Token", type="password")
        
    chat_ids_input = st.text_area("Chat IDs (쉼표 구분)", value='7656279558, -1003530274162')
    CHAT_IDS = [cid.strip() for cid in chat_ids_input.split(',')]
    
    st.divider()
    vol_threshold = st.slider("전략2 거래량 배수 (전일 대비)", 1.0, 3.0, 1.2, 0.1)
    st.info("전략3은 5일선>20일선 정배열 상태에서 20일선을 터치하는 종목을 찾습니다.")
    
    scan_btn = st.button("🔍 즉시 수동 스캔", use_container_width=True)

# --- [3. 분석 함수 정의] ---
def get_indicators(df):
    """보조지표 계산: MA, Bollinger, CCI"""
    df['MA5'] = df['Close'].rolling(window=5).mean()
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['StdDev'] = df['Close'].rolling(window=20).std()
    df['Lower'] = df['MA20'] - (df['StdDev'] * 2)
    
    tp = (df['High'] + df['Low'] + df['Close']) / 3
    ma_tp = tp.rolling(window=20).mean()
    mad = tp.rolling(window=20).apply(lambda x: np.abs(x - x.mean()).mean())
    df['CCI'] = (tp - ma_tp) / (0.015 * mad + 1e-9)
    return df

def send_telegram(msg):
    """텔레그램 메시지 전송"""
    for cid in CHAT_IDS:
        try:
            url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
            requests.get(url, params={"chat_id": cid, "text": msg}, timeout=5)
        except: pass

# --- [4. 메인 분석 로직] ---
def run_integrated_analysis():
    # 대상 로드
    stocks_all = fdr.StockListing('KRX')
    kospi200 = fdr.StockListing('KOSPI').head(200)
    kosdaq150 = fdr.StockListing('KOSDAQ').head(150)
    large_caps = pd.concat([kospi200, kosdaq150])[['Code', 'Name']].drop_duplicates()
    
    THEMES = {
        '반도체/HBM': ['삼성전자', 'SK하이닉스', '한미반도체', '가온칩스', '리노공업', 'HPSP', 'DB하이텍'],
        '2차전지/ESS': ['LG에너지솔루션', '삼성SDI', '포스코홀딩스', '에코프로비엠', '엘앤에프', '엔켐'],
        '조선/방산': ['HD현대중공업', '한화오션', '한화에어로스페이스', '현대로템', 'LIG넥스원'],
        '로봇/원전': ['레인보우로보틱스', '두산로보틱스', '두산에너빌리티', '한전기술']
    }

    res1, res2, res3 = [], [], []
    progress_bar = st.progress(0)
    status_text = st.empty()

    # 스캔 시작
    combined_targets = []
    # 대형주 리스트 추가
    for _, r in large_caps.iterrows():
        combined_targets.append({'Code': r['Code'], 'Name': r['Name'], 'Type': 'Large'})
    # 테마주 리스트 추가 (중복 제거)
    theme_names = [n for ns in THEMES.values() for n in ns]
    theme_stocks = stocks_all[stocks_all['Name'].isin(theme_names)]
    for _, r in theme_stocks.iterrows():
        # 테마명 매칭
        theme_name = [t for t, ns in THEMES.items() if r['Name'] in ns][0]
        combined_targets.append({'Code': r['Code'], 'Name': r['Name'], 'Type': 'Theme', 'Theme': theme_name})
    
    target_df = pd.DataFrame(combined_targets).drop_duplicates('Code').reset_index(drop=True)

    for i, row in target_df.iterrows():
        try:
            progress_bar.progress((i + 1) / len(target_df))
            status_text.text(f"분석 중... {row['Name']}")
            
            df = fdr.DataReader(row['Code'], (datetime.now() - timedelta(days=60)).strftime('%Y-%m-%d'))
            if len(df) < 25: continue
            df = get_indicators(df)
            curr, prev = df.iloc[-1], df.iloc[-2]

            # [전략 1] 과매도 (대형주 위주)
            if row['Type'] == 'Large':
                if (prev['Close'] >= prev['Lower']) and (curr['Close'] < curr['Lower']) and (curr['CCI'] <= -100):
                    res1.append({'종목': row['Name'], '가격': int(curr['Close']), 'CCI': round(curr['CCI'],1), 'Code': row['Code']})

            # [전략 2] 지지선 이탈 (테마주 위주)
            vol_ratio = curr['Volume'] / prev['Volume'] if prev['Volume'] > 0 else 0
            if curr['Close'] < prev['Low'] and vol_ratio >= vol_threshold:
                res2.append({'테마': row.get('Theme','기타'), '종목': row['Name'], '가격': int(curr['Close']), '거래량': round(vol_ratio,1), 'Code': row['Code']})

            # [전략 3] 20일선 눌림목 (대형주 위주)
            if row['Type'] == 'Large':
                if curr['MA5'] > curr['MA20'] and curr['Low'] <= curr['MA20'] and curr['Close'] >= curr['MA20']*0.98:
                    res3.append({'종목': row['Name'], '가격': int(curr['Close']), '20일선': int(curr['MA20']), 'Code': row['Code']})
            
            time.sleep(0.01)
        except: continue

    progress_bar.empty()
    status_text.empty()
    return res1, res2, res3

# --- [5. 결과 출력 및 시각화] ---
if scan_btn or 'initialized' not in st.session_state:
    st.session_state['initialized'] = True
    r1, r2, r3 = run_integrated_analysis()
    
    st.divider()
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("🎯 전략1: 과매도 반등")
        if r1:
            df1 = pd.DataFrame(r1)
            st.dataframe(df1[['종목', '가격', 'CCI']], use_container_width=True)
            send_telegram(f"🔔 [전략1: 과매도]\n" + "\n".join([f"{x['종목']}: {x['가격']:,}원" for x in r1]))
        else: st.info("조건 부합 없음")

    with col2:
        st.subheader("🚨 전략2: 지지 이탈")
        if r2:
            df2 = pd.DataFrame(r2)
            st.dataframe(df2[['테마', '종목', '가격', '거래량']], use_container_width=True)
            send_telegram(f"🚨 [전략2: 지지이탈]\n" + "\n".join([f"[{x['테마']}] {x['종목']}: {x['가격']:,}원" for x in r2]))
        else: st.info("조건 부합 없음")

    with col3:
        st.subheader("📏 전략3: 20선 눌림목")
        if r3:
            df3 = pd.DataFrame(r3)
            st.dataframe(df3[['종목', '가격', '20일선']], use_container_width=True)
            send_telegram(f"💡 [전략3: 눌림목]\n" + "\n".join([f"{x['종목']}: {x['가격']:,}원" for x in r3]))
        else: st.info("조건 부합 없음")

    # 통합 차트 뷰
    all_found = r1 + r2 + r3
    if all_found:
        st.divider()
        st.subheader("📊 발견 종목 상세 차트")
        selected_name = st.selectbox("차트를 볼 종목 선택", list(dict.fromkeys([x['종목'] for x in all_found])))
        
        # 선택된 종목의 코드 찾기
        selected_code = [x['Code'] for x in all_found if x['종목'] == selected_name][0]
        chart_df = fdr.DataReader(selected_code, (datetime.now() - timedelta(days=90)).strftime('%Y-%m-%d'))
        chart_df = get_indicators(chart_df)
        
        fig = go.Figure(data=[go.Candlestick(x=chart_df.index, open=chart_df['Open'], high=chart_df['High'], low=chart_df['Low'], close=chart_df['Close'], name='봉차트')])
        fig.add_trace(go.Scatter(x=chart_df.index, y=chart_df['MA20'], line=dict(color='orange', width=2), name='20일선'))
        fig.add_trace(go.Scatter(x=chart_df.index, y=chart_df['Lower'], line=dict(color='gray', width=1, dash='dash'), name='BB하단'))
        fig.update_layout(xaxis_rangeslider_visible=False, height=500, title=f"{selected_name} 기술적 분석 차트")
        st.plotly_chart(fig, use_container_width=True)
