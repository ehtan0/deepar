#!/usr/bin/env python
# coding: utf-8

# In[2]:


import sys

sys.path

sys.path.append(r'C:\Users\AIVE\Anaconda3\envs\deepar\Lib\site-packages')

# %load_ext autoreload
# %autoreload 2

import itertools
import os
import warnings
import category_encoders
import gluonts
import mxnet
import numpy as np
import pandas as pd
import altair as alt
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler, OrdinalEncoder
from category_encoders.hashing import HashingEncoder
from gluonts.dataset.multivariate_grouper import MultivariateGrouper
from gluonts.evaluation import Evaluator, MultivariateEvaluator
from gluonts.evaluation.backtest import make_evaluation_predictions
from gluonts.dataset.common import ListDataset
from gluonts.dataset.field_names import FieldName
from gluonts.model.deepar import DeepAREstimator
from gluonts.model.deepvar import DeepVAREstimator
from gluonts.model.predictor import Predictor
from gluonts.mx.distribution import (
    LowrankMultivariateGaussianOutput,
    NegativeBinomialOutput, 
)
from gluonts.mx.trainer import Trainer
from mxnet.context import num_gpus
from utils.evaluation import calc_eval_metric, WRMSSEEvaluator

def main():
    mxnet.random.seed(42)
    np.random.seed(42)
    warnings.filterwarnings("ignore")

    DATA_PATH = "./data"
    MODEL_PATH = "models"

    calendar = pd.read_csv(os.path.join(DATA_PATH, "calendar.csv"))
    selling_prices = pd.read_csv(os.path.join(DATA_PATH, "sell_prices.csv"))
    df_train_valid = pd.read_csv(os.path.join(DATA_PATH, "sales_train_validation.csv"))
    df_train_eval = pd.read_csv(os.path.join(DATA_PATH, "sales_train_evaluation.csv"))
    sample_submission = pd.read_csv(os.path.join(DATA_PATH, "sample_submission.csv"))

    # key columns
    key_names = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]

    # 상품코드 + 지역코드
    all_ids = df_train_eval["id"].unique()

    date_names = ["d_" + str(i) for i in range(1, 1942)]
    dates = calendar["date"].unique()
    test_steps = 28


    # 개별 상품코드마다 모든 날짜 매핑 
    key_pairs = list(itertools.product(all_ids, dates))

    key_pairs = pd.DataFrame(key_pairs, columns=["id", "date"])


    #sample_ratio 설정 후 sample data 추출
    sample_ratio = 0.01

    if sample_ratio == 1.0:
        sampled_ids = all_ids
    else:
        sampled_ids = np.random.choice( all_ids, round(sample_ratio * len(all_ids)), replace=False).tolist()
        
    print( f"{len(sampled_ids)} out of {len(all_ids)} IDs were selected for testing.")

    target = df_train_eval[["id"] + date_names]
    target = target.set_index("id").T.reset_index()
    date_dict = calendar[["date", "d"]].set_index("d").to_dict()["date"]
    target["index"] = target["index"].replace(date_dict)
    target.columns = ["date"] + target.columns[1:].tolist()
    target = target.set_index("date")

    feature_names = ["event_name_1", "event_type_1", "event_name_2", "event_type_2"]
    events = calendar[["date"] + feature_names].fillna("NA")
    train = events[events["date"] < dates[-2 * test_steps]][feature_names]

    encoder = HashingEncoder(drop_invariant=True)
    _ = encoder.fit(train)
    encoded = encoder.transform(events[feature_names])
    events = pd.concat([events[["date"]], encoded], axis=1)

    time_related = calendar[["date", "wday", "month"]]
    time_related["day"] = time_related["date"].map(lambda x: int(x.split("-")[2]))

    feat_dynamic_cat = events.merge(time_related).set_index("date")

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(feat_dynamic_cat)
    feat_dynamic_cat = pd.DataFrame(scaled, columns=feat_dynamic_cat.columns, index=feat_dynamic_cat.index)
    n_feat_dynamic_cat = feat_dynamic_cat.shape[1]

    prices = (df_train_eval[["id", "store_id", "item_id"]].merge(selling_prices, how="left").drop(["store_id", "item_id"], axis=1))
    week_to_date = calendar[["date", "wm_yr_wk"]].drop_duplicates()
    prices = week_to_date.merge(prices, how="left").drop(["wm_yr_wk"], axis=1)

    scaler = MinMaxScaler()
    train = prices[prices["date"] < dates[-2 * test_steps]][["sell_price"]]

    _ = scaler.fit(train)
    prices["sell_price"] = scaler.transform(prices[["sell_price"]])
    prices = prices.pivot(index="date", columns="id", values="sell_price")
    prices = prices.fillna(method="bfill")

    snap = calendar[["date", "snap_CA", "snap_TX", "snap_WI"]]
    snap.columns = ["date", "CA", "TX", "WI"]
    snap = pd.melt(snap,id_vars="date",value_vars=["CA", "TX", "WI"],var_name="state_id",value_name="snap",)
    snap = key_pairs.merge(df_train_eval[["id", "state_id"]], how="left").merge(snap, on=["date", "state_id"], how="left")
    snap = snap.pivot(index="date", columns="id", values="snap")

    feat_dynamic_real = pd.concat([prices, snap], axis=1)
    n_feat_dynamic_real = int(feat_dynamic_real.shape[1] / target.shape[1])

    feature_names = ["item_id", "dept_id", "cat_id", "store_id", "state_id"]
    feat_static_cat = df_train_eval[["id"] + feature_names]

    encoder = OrdinalEncoder()
    feat_static_cat[feature_names] = encoder.fit_transform(feat_static_cat[feature_names])
    feat_static_cat[feature_names] = feat_static_cat[feature_names].astype(int)
    feat_static_cat = feat_static_cat.set_index("id").T

    cardinality = [len(category) for category in encoder.categories_]

    def split_into_n_array(x, n):
        return np.hsplit(x.values.T.ravel(), n)

    def train():
        train_list = []
        for test_sampled_id in sampled_ids:
            dict_by_id = {
                FieldName.TARGET: target[test_sampled_id].iloc[:-test_steps].values,
                FieldName.START: target.index[0],
                FieldName.FEAT_DYNAMIC_REAL: split_into_n_array(feat_dynamic_cat.iloc[: -2 * test_steps],  n_feat_dynamic_cat,)+ split_into_n_array(feat_dynamic_real[test_sampled_id].iloc[: -2 * test_steps], n_feat_dynamic_real,),
                FieldName.FEAT_STATIC_CAT: feat_static_cat[test_sampled_id].values,}
            train_list.append(dict_by_id)

        test_list = []
        for test_sampled_id in sampled_ids:
            dict_by_id = {
                FieldName.TARGET: target[test_sampled_id].values,
                FieldName.START: target.index[0],
                FieldName.FEAT_DYNAMIC_REAL: split_into_n_array(feat_dynamic_cat.iloc[: -test_steps], n_feat_dynamic_cat,)+ split_into_n_array(feat_dynamic_real[test_sampled_id].iloc[: -test_steps],n_feat_dynamic_real,),
                FieldName.FEAT_STATIC_CAT: feat_static_cat[test_sampled_id].values,
            }
            test_list.append(dict_by_id)
        return train_list, test_list 
    train_list, test_list = train()
        
    train_dataset = ListDataset(train_list, freq="D")
    test_dataset = ListDataset(test_list, freq="D")

    device = "gpu" if num_gpus() > 0 else "cpu"
    trainer = Trainer(ctx=device,
        epochs=10, # before : 200
        num_batches_per_epoch=50,
        learning_rate=0.001,
        hybridize=True,
    )

    deepar_estimator = DeepAREstimator(
        freq="D", 
        prediction_length=test_steps,  #예측기간
        trainer=trainer,
        context_length=2 * test_steps, #
        num_layers=2, # RNN hidden layer
        num_cells=40, # 각 hidden layer 셀 수
        cell_type="lstm",
        dropout_rate=0.1,
        use_feat_dynamic_real=True,
        use_feat_static_cat=True,
        use_feat_static_real=False,
        cardinality=cardinality,
        distr_output=NegativeBinomialOutput(), #음이항?
        batch_size=30,
    )


    deepar_predictor = deepar_estimator.train(train_dataset)

    # os.makedirs(os.path.join(MODEL_PATH, "deepar"), exist_ok=True)
    # deepar_predictor.serialize(Path(os.path.join(MODEL_PATH, "deepar")))

    # deepar_predictor = Predictor.deserialize(Path(os.path.join(MODEL_PATH, "deepar")))


    forecast_iter, ts_iter = make_evaluation_predictions( 
        dataset=test_dataset,
        predictor=deepar_predictor, 
        num_samples=100,
    ) 
    forecasts = list(forecast_iter)
    tss = list(ts_iter)

    num_series = len(sampled_ids)

    evaluator = Evaluator(quantiles=[0.1, 0.5, 0.9])
    agg_metrics, item_metrics = evaluator(
        iter(tss), iter(forecasts), num_series=num_series
    )

    string = ""
    for key, value in agg_metrics.items():
        if not np.isnan(value):
            string += key + ": " + f"{value:.4f}\n"
            
    print(string[:-2])

    df_sampled = (df_train_eval.set_index("id").loc[sampled_ids].reset_index())
    df_train_sampled = df_sampled.loc[:, key_names + date_names[:-test_steps]]
    df_test_sampled = df_sampled.loc[:, date_names[-test_steps:]]

    wrmsse_evaluator = WRMSSEEvaluator(df_train_sampled, df_test_sampled, calendar, selling_prices, test_steps)

    predictions = [forecast.mean for forecast in forecasts]
    df_pred_sampled = pd.DataFrame(predictions, columns=df_test_sampled.columns)
    eval_metrics = calc_eval_metric(df_test_sampled, df_pred_sampled)

    wrmsse = wrmsse_evaluator.score(df_pred_sampled)

    print(f"DeepAR WRMSSE: {wrmsse:.6f}")


if __name__=="__main__":
    main()