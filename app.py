import streamlit as st
import FinanceDataReader as fdr
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import requests
import time
from datetime import datetime, timedelta
from streamlit_autorefresh import st_autorefresh

# --- [1. 기본 설정 및 자동 갱신] ---
st.set_page_config(page_title="종합 주식 스캐너", layout="wide")
# 5분마다 자동 새로고침 (300,000ms)
st_autorefresh(interval=300000, key="auto_refresh")

st.title("📈 3-Strategy 실시간 통합 스캐너")
st.caption(f"최근 스캔 시각: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# --- [2. 사이드바 설정] ---
with st.sidebar:
    st.header("⚙️ 알림 및 필터 설정")
    token = st.text_input("Telegram Token", type="password", value='8434131542:AAE-DrGRHveco9-hwoWNcO2zJ_64cHjIcKA')
    chat_ids = st.text_area("Chat IDs (쉼표 구분)", value='7656279558, -1003530274162').split(',')
    
    st.divider()
    vol_filter = st.slider("전략 2 거래량 필터 (전일 대비 배수)", 1.0, 3.0, 1.5, 0.1)
    scan_btn = st.button("🔍 즉시 수동 스캔", use_container_width=True)

# --- [3. 보조 지표 계산 함수] ---
def get_indicators(df):
    # 이동평균
    df['MA5'] = df['Close'].rolling(window=5).mean()
    df['MA20'] = df['Close'].rolling(window=20).mean()
    # 볼린저밴드 (전략 1용)
    df['StdDev'] = df['Close'].rolling(window=20).std()
    df['Lower'] = df['MA20'] - (df['StdDev'] * 2)
    # CCI (전략 1용)
    tp = (df['High'] + df['Low'] + df['Close']) / 3
    ma_tp = tp.rolling(window=20).mean()
    mad = tp.rolling(window=20).apply(lambda x: np.abs(x - x.mean()).mean())
    df['CCI'] = (tp - ma_tp) / (0.015 * mad + 1e-9)
    return df

# --- [4. 텔레그램 알림 함수] ---
def send_msg(msg):
    for cid in chat_ids:
        try:
            url = f"https://api.telegram.org/bot{token}/sendMessage"
            requests.get(url, params={"chat_id": cid.strip(), "text": msg})
        except: pass

# --- [5. 전략 로직 정의] ---
def run_all_strategies():
    # 데이터 로드
    stocks = fdr.StockListing('KRX')
    kospi200 = fdr.StockListing('KOSPI').head(200)
    kosdaq150 = fdr.StockListing('KOSDAQ').head(150)
    large_caps = pd.concat([kospi200, kosdaq150])[['Code', 'Name']]
    
    # 테마 설정 (전략 2용)
    THEMES = {
        '반도체/HBM': ['삼성전자', 'SK하이닉스', '한미반도체', '가온칩스', '리노공업'],
        '2차전지/ESS': ['LG에너지솔루션', '삼성SDI', '포스코홀딩스', '에코프로비엠'],
        '조선': ['HD현대중공업', '삼성중공업', '한화오션'],
        '방산/항공': ['한화에어로스페이스', '현대로템', 'LIG넥스원']
    }
    
    res1, res2, res3 = [], [], []
    progress = st.progress(0)
    
    # 스캔 시작
    all_targets = pd.concat([large_caps.assign(Type='Large'), 
                            pd.DataFrame([{'Code':stocks[stocks['Name']==n]['Code'].values[0], 'Name':n, 'Type':'Theme'} 
                                          for t, ns in THEMES.items() for n in ns if n in stocks['Name'].values])])
    all_targets = all_targets.drop_duplicates('Code').reset_index(drop=True)

    for i, row in all_targets.iterrows():
        try:
            progress.progress((i+1)/len(all_targets))
            df = fdr.DataReader(row['Code'], (datetime.now() - timedelta(days=60)).strftime('%Y-%m-%d'))
            if len(df) < 25: continue
            df = get_indicators(df)
            curr, prev = df.iloc[-1], df.iloc[-2]

            # [전략 1] 대형주 과매도
            if row['Type'] == 'Large' and (prev['Close'] >= prev['Lower']) and (curr['Close'] < curr['Lower']) and (curr['CCI'] <= -100):
                res1.append(f"🎯 [반등] {row['Name']} ({int(curr['Close']):,}원)")

            # [전략 2] 저가 이탈 & 거래량 급증
            vol_ratio = curr['Volume'] / prev['Volume'] if prev['Volume'] > 0 else 0
            if curr['Close'] < prev['Low'] and vol_ratio >= vol_filter:
                res2.append(f"📉 [이탈] {row['Name']} ({int(curr['Close']):,}원 / 거래량 {vol_ratio:.1f}배)")

            # [전략 3] 20일선 눌림목 (정배열: 5 > 20)
            if row['Type'] == 'Large' and curr['MA5'] > curr['MA20'] and curr['Low'] <= curr['MA20'] and curr['Close'] >= curr['MA20']*0.98:
                res3.append(f"📏 [눌림] {row['Name']} ({int(curr['Close']):,}원)")
            
            time.sleep(0.01)
        except: continue
    
    progress.empty()
    return res1, res2, res3

# --- [6. 메인 실행 루프] ---
if scan_btn or 'first_run' not in st.session_state:
    st.session_state['first_run'] = True
    r1, r2, r3 = run_all_strategies()
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("💡 전략 1: 과매도")
        if r1: 
            st.success("\n".join(r1))
            send_msg("🔔 [전략1 과매도]\n" + "\n".join(r1))
        else: st.write("조건 부합 없음")

    with col2:
        st.subheader("🚨 전략 2: 지지 이탈")
        if r2: 
            st.error("\n".join(r2))
            send_msg("🚨 [전략2 지지이탈]\n" + "\n".join(r2))
        else: st.write("조건 부합 없음")

    with col3:
        st.subheader("📏 전략 3: 20선 눌림")
        if r3: 
            st.info("\n".join(r3))
            send_msg("💡 [전략3 눌림목]\n" + "\n".join(r3))
        else: st.write("조건 부합 없음")
