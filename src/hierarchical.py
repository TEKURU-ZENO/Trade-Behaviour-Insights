import pandas as pd
import numpy as np
import statsmodels.formula.api as smf

def fit_mixed_effects(df: pd.DataFrame, outcome='return_pct', group='account'):
    df2 = df.dropna(subset=[outcome,'score_3d','leverage','notional']).copy()
    df2['log_notional'] = np.log1p(df2['notional'])
    formula = f"{outcome} ~ score_3d + leverage + log_notional + C(side)"
    try:
        md = smf.mixedlm(formula, df2, groups=df2[group])
        mdf = md.fit(method='lbfgs')
        return mdf
    except Exception as e:
        print("Mixed model failed:", e)
        return None
