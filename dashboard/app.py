import sys
import streamlit as st
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

import pandas as pd
import lightgbm as lgb
from src.data_utils import read_parquet


st.set_page_config(page_title="Trader Insights — Advanced", layout="wide")
st.title("Trader Behavior Insights — Demo")

try:
    df = read_parquet()
except Exception as e:
    st.error(f"Processed data not found: {e}")
    st.stop()

st.sidebar.header("Quick experiment")
sample = st.sidebar.slider("Sample size (for quick training)", 1000, 50000, 5000)
if st.sidebar.button("Train quick LGB baseline"):
    features = ['score','score_3d','score_7d','leverage','log_notional']
    df2 = df.dropna(subset=features + ['closedpnl'])
    df2['target'] = (df2['closedpnl'] > 0).astype(int)
    d = df2.sample(n=min(sample, len(df2)), random_state=42)
    X = d[features]
    y = d['target']
    lgb_train = lgb.Dataset(X, y)
    params = {'objective':'binary','metric':'auc','verbosity':-1}
    model = lgb.train(params, lgb_train, num_boost_round=200)
    st.success("Trained quick LGB model (demo).")
    fi = pd.DataFrame({'feature':model.feature_name(), 'imp': model.feature_importance()}).sort_values('imp', ascending=False)
    st.table(fi)
