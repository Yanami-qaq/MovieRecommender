"""
Streamlit demo app for the Movie Recommendation System.
Launch: streamlit run app/streamlit_app.py
"""
import os
import sys
import pickle
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, ROOT)

PROCESSED_DIR = os.path.join(ROOT, "data", "processed")
MODELS_DIR = os.path.join(PROCESSED_DIR, "models")

# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="电影推荐系统",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.metric-card {
    background: linear-gradient(135deg, #1e3a5f, #2c5f8a);
    padding: 16px 20px;
    border-radius: 12px;
    color: white;
    margin: 6px 0;
}
.metric-card h3 { margin: 0; font-size: 14px; opacity: 0.8; }
.metric-card p  { margin: 4px 0 0; font-size: 24px; font-weight: bold; }
.movie-card {
    border: 1px solid #e0e0e0;
    border-radius: 10px;
    padding: 14px;
    margin: 6px 0;
    background: #fafafa;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Data / model loading (cached)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading data and models...")
def load_everything():
    train = pd.read_pickle(os.path.join(PROCESSED_DIR, "train.pkl"))
    test  = pd.read_pickle(os.path.join(PROCESSED_DIR, "test.pkl"))
    movies = pd.read_pickle(os.path.join(PROCESSED_DIR, "movies.pkl"))
    content_features = pd.read_pickle(os.path.join(PROCESSED_DIR, "content_features.pkl"))

    model_names = ["UserCF", "ItemCF", "FunkSVD", "ContentBased", "Hybrid"]
    models = {}
    for name in model_names:
        pkl_path = os.path.join(MODELS_DIR, f"{name}.pkl")
        if os.path.exists(pkl_path):
            with open(pkl_path, "rb") as f:
                models[name] = pickle.load(f)

    # Load eval results if available
    import json
    results_path = os.path.join(PROCESSED_DIR, "eval_results.json")
    results = {}
    if os.path.exists(results_path):
        with open(results_path, "r") as f:
            results = json.load(f)

    return train, test, movies, content_features, models, results


def check_pipeline_ready():
    return os.path.exists(os.path.join(MODELS_DIR, "FunkSVD.pkl"))


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🎬 电影推荐系统")
    st.caption("数据分析与挖掘 · 期末项目")
    st.divider()

    if not check_pipeline_ready():
        st.error("未检测到训练好的模型，请先运行 `python main.py`")
        st.stop()

    train, test, movies, content_features, models, eval_results = load_everything()

    st.subheader("参数设置")
    algorithm = st.selectbox("推荐算法", list(models.keys()))
    top_k = st.slider("推荐数量 Top-K", min_value=5, max_value=20, value=10)

    all_users = sorted(train["user_id"].unique())
    user_id = st.selectbox("选择用户 ID", all_users, index=0)

    st.divider()
    st.subheader("页面导航")
    page = st.radio("页面导航", ["推荐结果", "算法对比", "数据探索"], label_visibility="collapsed")


# ─────────────────────────────────────────────────────────────────────────────
# Page: Recommendation
# ─────────────────────────────────────────────────────────────────────────────
if page == "推荐结果":
    st.header(f"为用户 {user_id} 推荐电影 · {algorithm}")

    user_train = train[train["user_id"] == user_id]
    rated_ids = set(user_train["movie_id"].tolist())

    col1, col2, col3 = st.columns(3)
    col1.metric("已评分电影数", len(rated_ids))
    col2.metric("平均评分", f"{user_train['rating'].mean():.2f}" if len(user_train) else "N/A")
    col3.metric("当前算法", algorithm)

    st.divider()

    # Get recommendations
    model = models[algorithm]
    with st.spinner("生成推荐中..."):
        rec_df = model.recommend(user_id, movies, rated_movie_ids=rated_ids, top_k=top_k)

    rec_df = rec_df.merge(movies[["movie_id", "title_clean", "genres", "year"]], on="movie_id", how="left")
    rec_df["predicted_rating"] = rec_df["predicted_rating"].round(2)

    st.subheader(f"Top-{top_k} 推荐结果")
    for i, row in rec_df.iterrows():
        cols = st.columns([0.5, 4, 2, 2])
        cols[0].markdown(f"**{i+1}**")
        cols[1].markdown(f"**{row['title_clean']}** ({row['year']})")
        cols[2].markdown(f"🎭 {row['genres']}")
        rating_val = row["predicted_rating"]
        stars = "⭐" * int(round(rating_val))
        cols[3].markdown(f"{stars} {rating_val:.2f}")

    st.divider()
    st.subheader("用户历史评分（最近20条）")
    hist = (user_train.merge(movies[["movie_id", "title_clean"]], on="movie_id", how="left")
            .sort_values("rating", ascending=False).head(20))
    st.dataframe(
        hist[["title_clean", "rating"]].rename(columns={"title_clean": "电影", "rating": "评分"}),
        width='stretch', hide_index=True
    )


# ─────────────────────────────────────────────────────────────────────────────
# Page: Algorithm Comparison
# ─────────────────────────────────────────────────────────────────────────────
elif page == "算法对比":
    st.header("算法性能对比")

    if not eval_results:
        st.warning("未找到评估结果，请先运行 `python main.py` 完成评估。")
        st.stop()

    from src.utils.helpers import results_to_dataframe
    df = results_to_dataframe(eval_results)

    st.subheader("综合指标表")
    st.dataframe(df.style.highlight_min(subset=["RMSE", "MAE"], color="#ffd6d6")
                        .highlight_max(subset=[c for c in df.columns if c not in ("RMSE","MAE")], color="#d6f5d6"),
                 width='stretch')

    st.divider()
    metrics_list = list(df.columns)
    col_sel = st.multiselect("选择展示的指标", metrics_list, default=metrics_list[:4])

    if col_sel:
        fig = go.Figure()
        for metric in col_sel:
            fig.add_trace(go.Bar(
                name=metric, x=df.index.tolist(),
                y=df[metric].tolist(),
                text=[f"{v:.4f}" for v in df[metric]],
                textposition="outside",
            ))
        fig.update_layout(
            barmode="group", title="算法对比（分组柱状图）",
            xaxis_title="算法", yaxis_title="分数",
            legend_title="指标", height=450,
        )
        st.plotly_chart(fig, width='stretch')

    st.divider()
    st.subheader("雷达图（排名类指标）")
    radar_metrics = [m for m in ["Precision@10", "Recall@10", "NDCG@10", "Coverage", "Diversity"]
                     if m in df.columns]
    if len(radar_metrics) >= 3:
        fig_radar = go.Figure()
        for model_name in df.index:
            vals = [df.loc[model_name, m] for m in radar_metrics]
            fig_radar.add_trace(go.Scatterpolar(
                r=vals + [vals[0]],
                theta=radar_metrics + [radar_metrics[0]],
                fill="toself", name=model_name,
            ))
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True)),
            showlegend=True, height=420, title="推荐质量雷达图"
        )
        st.plotly_chart(fig_radar, width='stretch')


# ─────────────────────────────────────────────────────────────────────────────
# Page: Data Exploration
# ─────────────────────────────────────────────────────────────────────────────
elif page == "数据探索":
    st.header("数据探索性分析 (EDA)")

    all_ratings = pd.concat([train, test], ignore_index=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("总评分数", f"{len(all_ratings):,}")
    c2.metric("用户数", f"{all_ratings['user_id'].nunique():,}")
    c3.metric("电影数", f"{all_ratings['movie_id'].nunique():,}")
    c4.metric("平均评分", f"{all_ratings['rating'].mean():.2f}")

    st.divider()
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("评分分布")
        rating_counts = all_ratings["rating"].value_counts().sort_index()
        fig1 = px.bar(x=rating_counts.index, y=rating_counts.values,
                      labels={"x": "评分", "y": "数量"},
                      color=rating_counts.values, color_continuous_scale="Blues")
        fig1.update_layout(height=320, showlegend=False)
        st.plotly_chart(fig1, width='stretch')

    with col2:
        st.subheader("用户评分数分布（前50%）")
        user_counts = all_ratings.groupby("user_id").size()
        cutoff = user_counts.quantile(0.5)
        fig2 = px.histogram(user_counts[user_counts <= cutoff], nbins=40,
                            labels={"value": "评分数", "count": "用户数"},
                            color_discrete_sequence=["#2c5f8a"])
        fig2.update_layout(height=320, showlegend=False)
        st.plotly_chart(fig2, width='stretch')

    st.subheader("Top-20 最多评分电影")
    movie_counts = (all_ratings.groupby("movie_id").size()
                    .reset_index(name="n_ratings")
                    .merge(movies[["movie_id", "title_clean"]], on="movie_id")
                    .sort_values("n_ratings", ascending=False).head(20))
    fig3 = px.bar(movie_counts, x="n_ratings", y="title_clean", orientation="h",
                  color="n_ratings", color_continuous_scale="Blues",
                  labels={"n_ratings": "评分数", "title_clean": "电影"})
    fig3.update_layout(height=520, yaxis=dict(autorange="reversed"), showlegend=False)
    st.plotly_chart(fig3, width='stretch')

    st.subheader("各类型电影数量")
    genre_series = movies["genre_list"].explode().value_counts()
    fig4 = px.pie(values=genre_series.values, names=genre_series.index,
                  title="电影类型分布", hole=0.35)
    fig4.update_layout(height=420)
    st.plotly_chart(fig4, width='stretch')


