import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from xgboost import XGBClassifier
import shap

# --- データ生成 ---
@st.cache_data
def load_data():
    n = 500
    timestamps = pd.date_range("2025-06-01", periods=n, freq="H")
    df = pd.DataFrame({
        "timestamp": timestamps,
        "device_id": np.random.choice(["sensor_A", "sensor_B", "sensor_C"], size=n),
        "temperature": 25 + np.random.randn(n) * 4,
        "humidity": 60 + np.random.randn(n) * 10
    })
    df["status"] = np.where((df["temperature"] > 30) | (df["humidity"] > 80), "error", "normal")
    df["label"] = np.where(df["status"] == "error", 1, 0)
    return df

df = load_data()

# --- センサー選択 ---
st.sidebar.title("センサー選択")
selected_devices = st.sidebar.multiselect("表示するセンサー", options=df["device_id"].unique(), default=df["device_id"].unique())
df = df[df["device_id"].isin(selected_devices)]

# --- 機械学習モデル（XGBoost） ---
X = df[["temperature", "humidity"]]
y = df["label"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
clf = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=0)
clf.fit(X_train, y_train)
df["predicted_label"] = clf.predict(X)

# --- 最新値 ---
latest = df.iloc[-1]
temp_now = latest["temperature"]
hum_now = latest["humidity"]
pred_now = latest["predicted_label"]

# --- UI表示 ---
st.title("IoTダッシュボード + 異常予測（XGBoost + 複数センサー対応）")

col1, col2, col3 = st.columns(3)
col1.metric("平均温度", f"{df['temperature'].mean():.2f} °C")
col2.metric("平均湿度", f"{df['humidity'].mean():.2f} %")
col3.metric("異常予測", "異常" if pred_now == 1 else "正常", delta=f"{int(df['predicted_label'].sum())} 件")

# --- タコメータ表示 ---
st.subheader("タコメータ")
gauge_temp = go.Figure(go.Indicator(
    mode="gauge+number+delta",
    value=temp_now,
    title={"text": "現在の温度"},
    delta={"reference": 25},
    gauge={"axis": {"range": [0, 50]},
           "steps": [{"range": [30, 50], "color": "red"}]}
))
st.plotly_chart(gauge_temp, use_container_width=True)

gauge_hum = go.Figure(go.Indicator(
    mode="gauge+number+delta",
    value=hum_now,
    title={"text": "現在の湿度"},
    delta={"reference": 60},
    gauge={"axis": {"range": [0, 100]},
           "steps": [{"range": [80, 100], "color": "red"}]}
))
st.plotly_chart(gauge_hum, use_container_width=True)

# --- 時系列 ---
st.subheader("時系列（異常点強調）")
fig_temp = px.scatter(df, x="timestamp", y="temperature", color="device_id", symbol=df["label"].map({0: "正常", 1: "異常"}))
st.plotly_chart(fig_temp, use_container_width=True)

fig_hum = px.scatter(df, x="timestamp", y="humidity", color="device_id", symbol=df["label"].map({0: "正常", 1: "異常"}))
st.plotly_chart(fig_hum, use_container_width=True)

# --- ステータス分布 ---
st.subheader("ステータス分布")
fig_pie = px.pie(df, names="status", title="ステータス構成比")
st.plotly_chart(fig_pie)

# --- アニメーション ---
st.subheader("異常傾向アニメーション")
fig_anim = px.scatter(df, x="temperature", y="humidity", 
                      animation_frame=df["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S"),
                      color="device_id", symbol="status", size_max=10)
st.plotly_chart(fig_anim)

# --- 評価指標 ---
with st.expander("モデル評価（分類レポート）"):
    y_pred_test = clf.predict(X_test)
    report = classification_report(y_test, y_pred_test, output_dict=True)
    st.dataframe(pd.DataFrame(report).transpose())

# --- 特徴量重要度（SHAP） ---
with st.expander("特徴量重要度（SHAP）"):
    explainer = shap.Explainer(clf)
    shap_values = explainer(X)
    st.set_option('deprecation.showPyplotGlobalUse', False)
    shap.summary_plot(shap_values, X, plot_type="bar")
    st.pyplot(bbox_inches='tight')

# --- 最新データ表示 ---
st.subheader("最新データ")
st.dataframe(df.tail(20))
