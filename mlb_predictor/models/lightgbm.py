def build_lightgbm_model():
    from lightgbm import LGBMClassifier

    return LGBMClassifier(
        n_estimators=350, num_leaves=24, learning_rate=0.035, subsample=0.8,
        colsample_bytree=0.8, random_state=42, verbosity=-1,
    )
