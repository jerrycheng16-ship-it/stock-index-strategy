import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# 頁面設定
st.set_page_config(page_title="0050 / 正2 雙均線緩衝策略回測", layout="wide")
st.title("📈 0050 / 0050正2 雙均線緩衝帶策略回測")

# 側邊欄參數設定
st.sidebar.header("⚙️ 策略參數設定")
start_date = st.sidebar.date_input("回測起始日", pd.to_datetime("2015-01-01"))
end_date = st.sidebar.date_input("回測結束日", pd.to_datetime("today"))

short_window = st.sidebar.number_input("短期均線 (日)", min_value=5, max_value=60, value=10)
long_window = st.sidebar.number_input("長期均線 (日)", min_value=60, max_value=300, value=250)
buffer_pct = st.sidebar.slider("均線緩衝帶 (%)", min_value=0.0, max_value=5.0, value=2.0, step=0.5) / 100.0
fee_rate = st.sidebar.number_input("單邊交易成本 (% 含手續費與證交稅)", min_value=0.0, max_value=1.0, value=0.2, step=0.05) / 100.0

@st.cache_data(ttl=3600)
def load_data(start, end):
    # 下載 0050 與 00631L (元大台灣50正2) 歷史資料
    # 往前多抓 400 天以確保計算 250 日均線時資料充足
    fetch_start = pd.to_datetime(start) - pd.Timedelta(days=450)
    df_0050 = yf.download("0050.TW", start=fetch_start, end=end, auto_adjust=True, progress=False)
    df_lev2 = yf.download("00631L.TW", start=fetch_start, end=end, auto_adjust=True, progress=False)
    
    # 處理 yfinance 多層欄位索引問題
    if isinstance(df_0050.columns, pd.MultiIndex):
        df_0050.columns = df_0050.columns.get_level_values(0)
    if isinstance(df_lev2.columns, pd.MultiIndex):
        df_lev2.columns = df_lev2.columns.get_level_values(0)

    df = pd.DataFrame({
        "0050_Close": df_0050["Close"],
        "00631L_Close": df_lev2["Close"]
    }).dropna()
    return df

def get_ma_state(prices, ma, buffer):
    """計算帶有緩衝帶的均線狀態 (True: 站上, False: 跌破)"""
    states = []
    # 初始狀態：收盤價是否大於均線
    curr_state = prices.iloc[0] > ma.iloc[0]
    
    for p, m in zip(prices, ma):
        if pd.isna(m):
            states.append(False)
            continue
        if p > m * (1 + buffer):
            curr_state = True
        elif p < m * (1 - buffer):
            curr_state = False
        states.append(curr_state)
    return pd.Series(states, index=prices.index)

def calculate_metrics(returns_series):
    """計算績效指標"""
    cum_ret = (1 + returns_series).cumprod()
    total_ret = cum_ret.iloc[-1] - 1
    days = (returns_series.index[-1] - returns_series.index[0]).days
    cagr = (1 + total_ret) ** (365.25 / days) - 1 if days > 0 else 0
    
    # 最大回撤 (MDD)
    rolling_max = cum_ret.cummax()
    drawdown = (cum_ret - rolling_max) / rolling_max
    mdd = drawdown.min()
    
    # 夏普值 (假設無風險利率 1.5%)
    rf_daily = 0.015 / 252
    excess_ret = returns_series - rf_daily
    sharpe = np.sqrt(252) * excess_ret.mean() / returns_series.std() if returns_series.std() != 0 else 0
    
    return total_ret, cagr, mdd, sharpe, cum_ret, drawdown

# 載入資料
with st.spinner("正在下載台灣 0050 與 00631L 歷史股價..."):
    raw_df = load_data(start_date, end_date)

if raw_df.empty:
    st.error("無法取得資料，請檢查網路連線或日期區間（00631L 上市日期為 2014/10/31）。")
    st.stop()

# 計算均線與狀態
df = raw_df.copy()
df["MA_Short"] = df["0050_Close"].rolling(short_window).mean()
df["MA_Long"] = df["0050_Close"].rolling(long_window).mean()
df = df.dropna().copy()

# 篩選使用者指定的回測區間
df["Above_Short"] = get_ma_state(df["0050_Close"], df["MA_Short"], buffer_pct)
df["Above_Long"] = get_ma_state(df["0050_Close"], df["MA_Long"], buffer_pct)
df = df[df.index >= pd.to_datetime(start_date)].copy()

# 計算每日報酬率
df["Ret_0050"] = df["0050_Close"].pct_change().fillna(0)
df["Ret_00631L"] = df["00631L_Close"].pct_change().fillna(0)

# ---------------------------------------------------------
# 策略訊號定義 (1: 正2, 0.5: 0050, 0: 現金)
# ---------------------------------------------------------
# 🟡 三態：兩條都站上抱正2(1)，只站上一條抱0050(0.5)，兩條都跌破拿現金(0)
cond_both_above = df["Above_Short"] & df["Above_Long"]
cond_both_below = (~df["Above_Short"]) & (~df["Above_Long"])

df["Pos_3State"] = np.select(
    [cond_both_above, cond_both_below],
    [1.0, 0.0],
    default=0.5
)

# 🟡 兩態：站上任一條就抱正2(1)，兩條都跌破退回0050(0.5)，不拿現金
df["Pos_2State"] = np.where(cond_both_below, 0.5, 1.0)

# 將訊號往後遞延一天 (今日收盤確認訊號，明日才持有該部位)
df["Exec_3State"] = df["Pos_3State"].shift(1).fillna(df["Pos_3State"].iloc[0])
df["Exec_2State"] = df["Pos_2State"].shift(1).fillna(df["Pos_2State"].iloc[0])

# 計算策略每日報酬與換倉手續費
def compute_strategy_return(exec_pos_series):
    ret = pd.Series(0.0, index=df.index)
    # 部位對應報酬
    ret = np.where(exec_pos_series == 1.0, df["Ret_00631L"],
          np.where(exec_pos_series == 0.5, df["Ret_0050"], 0.0))
    ret = pd.Series(ret, index=df.index)
    
    # 計算換倉次數與成本 (部位變動時扣除單邊交易成本)
    trades = exec_pos_series.diff().fillna(0) != 0
    ret[trades] -= fee_rate
    return ret, trades.sum()

df["Strat_3State_Ret"], trades_3 = compute_strategy_return(df["Exec_3State"])
df["Strat_2State_Ret"], trades_2 = compute_strategy_return(df["Exec_2State"])

# 計算各策略績效
m_3 = calculate_metrics(df["Strat_3State_Ret"])
m_2 = calculate_metrics(df["Strat_2State_Ret"])
m_50 = calculate_metrics(df["Ret_0050"])
m_lev = calculate_metrics(df["Ret_00631L"])

# ---------------------------------------------------------
# 介面視覺化呈現
# ---------------------------------------------------------
st.subheader("📊 策略績效總覽")

summary_data = {
    "策略名稱": [
        "🟡 三態策略 (正2 / 0050 / 現金)",
        "🟡 兩態策略 (正2 / 0050)",
        "基準：買進持有 0050",
        "基準：買進持有 0050正2"
    ],
    "總報酬率": [f"{m_3[0]:.1%}", f"{m_2[0]:.1%}", f"{m_50[0]:.1%}", f"{m_lev[0]:.1%}"],
    "年化報酬率 (CAGR)": [f"{m_3[1]:.2%}", f"{m_2[1]:.2%}", f"{m_50[1]:.2%}", f"{m_lev[1]:.2%}"],
    "最大回撤 (MDD)": [f"{m_3[2]:.2%}", f"{m_2[2]:.2%}", f"{m_50[2]:.2%}", f"{m_lev[2]:.2%}"],
    "夏普值 (Sharpe)": [f"{m_3[3]:.2f}", f"{m_2[3]:.2f}", f"{m_50[3]:.2f}", f"{m_lev[3]:.2f}"],
    "總換倉次數": [f"{trades_3} 次", f"{trades_2} 次", "0 次", "0 次"]
}
st.table(pd.DataFrame(summary_data))

# 繪製淨值曲線圖
st.subheader("📈 累積淨值走勢比較 (初始資金 = 1)")
fig_nav = go.Figure()
fig_nav.add_trace(go.Scatter(x=df.index, y=m_3[4], name="三態策略 (正2/0050/現金)", line=dict(width=2.5, color="#F59E0B")))
fig_nav.add_trace(go.Scatter(x=df.index, y=m_2[4], name="兩態策略 (正2/0050)", line=dict(width=2.5, color="#10B981")))
fig_nav.add_trace(go.Scatter(x=df.index, y=m_50[4], name="買進持有 0050", line=dict(width=1.5, dash="dot", color="#6B7280")))
fig_nav.add_trace(go.Scatter(x=df.index, y=m_lev[4], name="買進持有 0050正2", line=dict(width=1.5, dash="dash", color="#3B82F6")))

fig_nav.update_layout(height=450, hovermode="x unified", yaxis_title="累積淨值 (倍數)", legend=dict(orientation="h", y=1.08))
st.plotly_chart(fig_nav, use_container_width=True)

# 繪製回撤比較圖與部位變化
col1, col2 = st.columns(2)

with col1:
    st.subheader("📉 歷史回撤 (Drawdown) 比較")
    fig_dd = go.Figure()
    fig_dd.add_trace(go.Scatter(x=df.index, y=m_3[5]*100, name="三態策略", fill="tozeroy", line=dict(color="#F59E0B")))
    fig_dd.add_trace(go.Scatter(x=df.index, y=m_2[5]*100, name="兩態策略", line=dict(color="#10B981")))
    fig_dd.add_trace(go.Scatter(x=df.index, y=m_lev[5]*100, name="0050正2", line=dict(width=1, dash="dash", color="#3B82F6")))
    fig_dd.update_layout(height=350, hovermode="x unified", yaxis_title="回撤幅度 (%)")
    st.plotly_chart(fig_dd, use_container_width=True)

with col2:
    st.subheader("🧭 最新交易日訊號狀態")
    latest = df.iloc[-1]
    pos_map = {1.0: "🔥 100% 持有 0050正2 (00631L)", 0.5: "🛡️ 100% 持有 0050", 0.0: "💵 100% 持有現金"}
    
    st.info(f"""
    **資料最新日期**：{df.index[-1].strftime('%Y-%m-%d')}  
    * **0050 收盤價**：`{latest['0050_Close']:.2f}`
    * **10日均線 (±2%)**：`{latest['MA_Short']:.2f}` (`{latest['MA_Short']*0.98:.2f}` ~ `{latest['MA_Short']*1.02:.2f}`) → **{'站上' if latest['Above_Short'] else '跌破'}**
    * **250日均線 (±2%)**：`{latest['MA_Long']:.2f}` (`{latest['MA_Long']*0.98:.2f}` ~ `{latest['MA_Long']*1.02:.2f}`) → **{'站上' if latest['Above_Long'] else '跌破'}**
    
    ---
    * **🟡 三態策略建議部位**：{pos_map[latest['Pos_3State']]}
    * **🟡 兩態策略建議部位**：{pos_map[latest['Pos_2State']]}
    """)

# 顯示近期訊號明細表
with st.expander("📋 查看最近 30 個交易日訊號明細"):
    display_df = df[["0050_Close", "MA_Short", "MA_Long", "Above_Short", "Above_Long", "Pos_3State", "Pos_2State"]].tail(30).copy()
    display_df["Pos_3State"] = display_df["Pos_3State"].map({1.0: "正2", 0.5: "0050", 0.0: "現金"})
    display_df["Pos_2State"] = display_df["Pos_2State"].map({1.0: "正2", 0.5: "0050"})
    display_df.columns = ["0050收盤", "10MA", "250MA", "站上10MA", "站上250MA", "三態部位", "兩態部位"]
    st.dataframe(display_df.sort_index(ascending=False), use_container_width=True)
