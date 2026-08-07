def build_logistic_model():
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    # add_indicator=True is deliberately off: a feature that's missing in only
    # a handful of training rows (e.g. wind_speed_mph, which Open-Meteo failed
    # on for ~1-2 of ~12,000 historical rows) gets a near-zero-variance
    # "missingness" indicator column from StandardScaler. Live inference hits
    # that same missingness far more often (forecast API timeouts, games too
    # far out, missing venue coordinates) than training ever did, so the rare
    # indicator's tiny fitted std blows up an ordinary 0/1 flag into a scaled
    # value of ~90+ standard deviations -- enough on its own to collapse a
    # reasonable prediction to near 0% or 100%. Plain median imputation with
    # no indicator avoids this failure mode entirely.
    return Pipeline([
        ("impute", SimpleImputer(strategy="median", add_indicator=False)),
        ("scale", StandardScaler()),
        ("model", LogisticRegression(C=0.25, max_iter=2000)),
    ])
