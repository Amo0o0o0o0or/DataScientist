"""Interactive, educational dashboard for a small medical ML case study."""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import confusion_matrix, roc_curve

from model import calculate_metrics, load_dataset, train_and_evaluate


st.set_page_config(
    page_title="Wisconsin Diagnostic Explorer",
    page_icon="W",
    layout="wide",
)


@st.cache_resource(show_spinner="Training and evaluating the models...")
def get_results():
    return train_and_evaluate()


features, target = load_dataset()
results = get_results()

st.title("Wisconsin Diagnostic Explorer")
st.caption(
    "A small, reproducible classification case study using the packaged "
    "Breast Cancer Wisconsin Diagnostic dataset."
)
st.warning(
    "Educational portfolio project only. This model is not validated for "
    "clinical use and must not be used to diagnose or guide care."
)

summary, performance, explain, examples = st.tabs(
    ["Dataset", "Model performance", "Feature analysis", "Explore a case"]
)

with summary:
    malignant_count = int(target.sum())
    benign_count = int((target == 0).sum())
    total = len(target)
    col1, col2, col3 = st.columns(3)
    col1.metric("Samples", f"{total:,}")
    col2.metric("Measured features", f"{features.shape[1]}")
    col3.metric("Malignant cases", f"{malignant_count:,} ({malignant_count / total:.1%})")

    st.subheader("What is in the data?")
    st.write(
        "Each row describes measurements computed from a digitized image of a "
        "fine needle aspirate. The target is whether the sample was labeled "
        "malignant or benign in the source dataset."
    )
    left, right = st.columns(2)
    class_counts = pd.DataFrame(
        {"Diagnosis": ["Benign", "Malignant"], "Samples": [benign_count, malignant_count]}
    )
    left.plotly_chart(
        px.bar(
            class_counts,
            x="Diagnosis",
            y="Samples",
            color="Diagnosis",
            color_discrete_map={"Benign": "#168c83", "Malignant": "#d45c47"},
            title="Class balance",
        ),
        use_container_width=True,
    )
    chart_data = features[["mean radius", "mean texture"]].copy()
    chart_data["Diagnosis"] = target.map({0: "Benign", 1: "Malignant"})
    right.plotly_chart(
        px.scatter(
            chart_data,
            x="mean radius",
            y="mean texture",
            color="Diagnosis",
            color_discrete_map={"Benign": "#168c83", "Malignant": "#d45c47"},
            opacity=0.75,
            title="Two example measurements",
        ),
        use_container_width=True,
    )

with performance:
    st.subheader("Model selection and held-out evaluation")
    st.write(
        f"Selected **{results['model_name']}** by mean ROC-AUC over 5 stratified "
        "cross-validation folds on the training split. The test split was kept "
        "separate until final evaluation."
    )
    st.dataframe(
        results["leaderboard"].style.format(
            {"Mean CV ROC-AUC": "{:.3f}", "CV SD": "{:.3f}"}
        ),
        hide_index=True,
        use_container_width=True,
    )

    threshold = st.slider("Decision threshold", 0.05, 0.95, 0.5, 0.05)
    metrics = calculate_metrics(results["y_test"], results["probabilities"], threshold)
    metric_cols = st.columns(4)
    for col, label, key in zip(
        metric_cols,
        ["ROC-AUC", "Recall", "Specificity", "Precision"],
        ["roc_auc", "recall", "specificity", "precision"],
    ):
        col.metric(label, f"{metrics[key]:.3f}")

    left, right = st.columns(2)
    matrix = confusion_matrix(
        results["y_test"],
        (results["probabilities"] >= threshold).astype(int),
        labels=[0, 1],
    )
    left.plotly_chart(
        px.imshow(
            matrix,
            text_auto=True,
            x=["Predicted benign", "Predicted malignant"],
            y=["Actual benign", "Actual malignant"],
            color_continuous_scale="Teal",
            title=f"Confusion matrix at threshold {threshold:.2f}",
        ),
        use_container_width=True,
    )
    false_positive_rate, true_positive_rate, _ = roc_curve(
        results["y_test"], results["probabilities"]
    )
    roc_chart = go.Figure()
    roc_chart.add_trace(
        go.Scatter(x=false_positive_rate, y=true_positive_rate, mode="lines", name="Model")
    )
    roc_chart.add_trace(
        go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Chance", line={"dash": "dash"})
    )
    roc_chart.update_layout(
        title="ROC curve on the held-out test set",
        xaxis_title="False positive rate",
        yaxis_title="True positive rate",
        yaxis_range=[0, 1.02],
    )
    right.plotly_chart(roc_chart, use_container_width=True)
    st.caption(
        "Lowering the threshold may catch more malignant cases (higher recall), "
        "but usually increases false positives. The threshold is a teaching "
        "control, not a clinical recommendation."
    )

with explain:
    st.subheader("Which measurements matter to this model?")
    st.write(
        "Permutation importance measures how much test-set ROC-AUC changes when "
        "one feature is shuffled. It describes this fitted model; it does not "
        "establish a causal relationship."
    )
    top_features = results["importance"].head(12).sort_values("Importance")
    st.plotly_chart(
        px.bar(
            top_features,
            x="Importance",
            y="Feature",
            orientation="h",
            error_x="SD",
            color="Importance",
            color_continuous_scale="Teal",
            title="Top 12 permutation importances",
        ),
        use_container_width=True,
    )

with examples:
    st.subheader("Inspect a held-out sample")
    st.write("Choose a test-set row to inspect its measurements and model output.")
    test_features = results["x_test"]
    sample_index = st.selectbox(
        "Test sample",
        options=test_features.index.tolist(),
        format_func=lambda index: f"Sample {index + 1}",
    )
    sample = test_features.loc[[sample_index]]
    probability = float(results["model"].predict_proba(sample)[0, 1])
    actual_class = int(results["y_test"].loc[sample_index])
    predicted_class = int(probability >= 0.5)
    probability_col, predicted_col, actual_col = st.columns(3)
    probability_col.metric("Model-estimated malignant probability", f"{probability:.1%}")
    predicted_col.metric("Prediction at 0.50", "Malignant" if predicted_class else "Benign")
    actual_col.metric("Dataset label", "Malignant" if actual_class else "Benign")
    st.dataframe(
        sample.T.rename(columns={sample_index: "Measurement"}),
        use_container_width=True,
    )