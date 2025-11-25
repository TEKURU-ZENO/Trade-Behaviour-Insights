import shap

def shap_summary(model, X, max_display=20):
    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(X)
    shap.summary_plot(sv, X, max_display=max_display)
    return sv
