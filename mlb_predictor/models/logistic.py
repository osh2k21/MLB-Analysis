def build_logistic_model():
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    return Pipeline([
        ("scale", StandardScaler()),
        ("model", LogisticRegression(C=0.25, max_iter=2000)),
    ])

