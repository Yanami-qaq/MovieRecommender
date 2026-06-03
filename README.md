# 电影推荐系统

数据分析与挖掘课程期末项目 · 选项C：端到端应用系统开发

## 快速开始

```bash
# 安装依赖
pip install -r requirements.txt

# 训练所有模型（首次运行自动下载 MovieLens 1M 数据集，约 5 分钟）
python main.py

# 启动 Streamlit 演示界面
streamlit run app/streamlit_app.py
```

## 项目结构

```
MovieRecommender/
├── data/
│   ├── raw/ml-1m/          # MovieLens 1M 原始数据（自动下载）
│   └── processed/          # 预处理数据、模型文件、评估结果
├── src/
│   ├── data/
│   │   ├── loader.py       # 数据下载与加载
│   │   └── preprocessor.py # 数据清洗、特征工程、数据集划分
│   ├── models/
│   │   ├── base.py         # 推荐器基类
│   │   ├── user_cf.py      # 基于用户的协同过滤
│   │   ├── item_cf.py      # 基于物品的协同过滤
│   │   ├── svd_model.py    # FunkSVD 矩阵分解
│   │   ├── content_based.py # 基于内容的推荐
│   │   └── hybrid.py       # 混合推荐
│   ├── evaluation/
│   │   ├── metrics.py      # RMSE/MAE/P@K/R@K/NDCG/Coverage/Diversity
│   │   └── evaluator.py    # 评估框架
│   └── utils/
│       └── helpers.py      # 结果保存、图表生成
├── app/
│   └── streamlit_app.py    # Streamlit 交互界面
├── main.py                 # 一键运行主流程
└── requirements.txt
```

## 算法说明

| 算法 | 文件 | 核心参数 |
|------|------|---------|
| UserCF | `src/models/user_cf.py` | `n_neighbors=30` |
| ItemCF | `src/models/item_cf.py` | `n_neighbors=30` |
| FunkSVD | `src/models/svd_model.py` | `n_factors=50, n_epochs=20` |
| ContentBased | `src/models/content_based.py` | TF-IDF 500维 |
| Hybrid | `src/models/hybrid.py` | `cf_weight=0.6, cb_weight=0.4` |
