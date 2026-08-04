def build_xgboost_model():
    from xgboost import XGBClassifier

    return XGBClassifier(
        n_estimators=350, max_depth=4, learning_rate=0.035, subsample=0.8,
        colsample_bytree=0.8, eval_metric="logloss", random_state=42,
    )

