import streamlit as st
import pandas as pd
from sqlalchemy import create_engine, inspect
import plotly.express as px

def postgres_viewer():
    st.header("PostgreSQL")
    with st.expander("データベース接続情報"):
        host = st.text_input("ホスト", "localhost")
        port = st.text_input("ポート", "5432")
        dbname = st.text_input("データベース名", "your_db")
        user = st.text_input("ユーザー名", "postgres")
        password = st.text_input("パスワード", type="password")

    if st.button("接続してテーブル取得"):
        try:
            engine = create_engine(f"postgresql://{user}:{password}@{host}:{port}/{dbname}")
            inspector = inspect(engine)
            tables = inspector.get_table_names()
            st.session_state["engine"] = engine
            st.session_state["tables"] = tables
            st.success("接続に成功しました！")
        except Exception as e:
            st.error(f"接続に失敗しました: {e}")

    if "tables" in st.session_state:
        table = st.selectbox("テーブルを選択", st.session_state["tables"])
        query = f"SELECT * FROM {table} LIMIT 100"
        if st.button("クエリ実行"):
            try:
                df = pd.read_sql(query, con=st.session_state["engine"])
                st.dataframe(df)

                if not df.empty:
                    numeric_cols = df.select_dtypes(include="number").columns.tolist()
                    if numeric_cols:
                        x_axis = st.selectbox("X軸（数値列）", numeric_cols)
                        y_axis = st.selectbox("Y軸（数値列）", numeric_cols)
                        fig = px.scatter(df, x=x_axis, y=y_axis, title="PostgreSQL データ可視化")
                        st.plotly_chart(fig)
            except Exception as e:
                st.error(f"クエリ実行に失敗しました: {e}")
