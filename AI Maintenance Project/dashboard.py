from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import cross_val_score, train_test_split

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "ai4i2020.csv"

FEATURES = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]
TARGET = "Machine failure"
CRITICALITY_MAP = {"Low": 1, "Medium": 2, "High": 3}

st.set_page_config(
    page_title="AI Maintenance Planner",
    page_icon="⚙️",
    layout="wide",
)


@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.insert(0, "Machine ID", df["UDI"].astype(str).map(lambda x: f"M{x}"))
    return df


@st.cache_resource
def train_model(df: pd.DataFrame):
    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    smote = SMOTE(random_state=42)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

    model = RandomForestClassifier(
        n_estimators=300,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train_balanced, y_train_balanced)

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    metrics = {
        "Accuracy": accuracy_score(y_test, predictions),
        "Precision": precision_score(y_test, predictions, zero_division=0),
        "Recall": recall_score(y_test, predictions, zero_division=0),
        "F1 Score": f1_score(y_test, predictions, zero_division=0),
        "ROC-AUC": roc_auc_score(y_test, probabilities),
        "Confusion Matrix": confusion_matrix(y_test, predictions),
    }

    # Cross-validation with SMOTE inside each fold avoids data leakage.
    cv_pipeline = Pipeline(
        [
            ("smote", SMOTE(random_state=42)),
            (
                "rf",
                RandomForestClassifier(
                    n_estimators=200,
                    class_weight="balanced",
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    cv_recall = cross_val_score(cv_pipeline, X, y, cv=5, scoring="recall", n_jobs=1)

    return model, metrics, cv_recall


def add_criticality(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["Usage Score"] = (
        out["Rotational speed [rpm]"]
        + out["Torque [Nm]"]
        + out["Tool wear [min]"]
    )
    out["Criticality"] = pd.qcut(
        out["Usage Score"],
        q=3,
        labels=["Low", "Medium", "High"],
    )
    out["Criticality Score"] = out["Criticality"].astype(str).map(CRITICALITY_MAP)
    return out


def add_maintenance_plan(df: pd.DataFrame, model: RandomForestClassifier) -> pd.DataFrame:
    out = add_criticality(df)
    out["Failure Risk %"] = model.predict_proba(out[FEATURES])[:, 1] * 100
    out["Risk Prediction"] = model.predict(out[FEATURES]).astype(int)

    # Keep the priority definition from the original project:
    # failure prediction (0/1) × operational criticality (1/2/3).
    out["Priority Score"] = out["Risk Prediction"] * out["Criticality Score"]

    def schedule(score: int) -> str:
        return {
            3: "Immediate (within 3 days)",
            2: "Short-term (within 2 weeks)",
            1: "Routine (within 1 month)",
            0: "No action needed",
        }[int(score)]

    def resources(score: int) -> str:
        return {
            3: "2 technicians · Spare parts: in-stock priority",
            2: "1 technician · Spare parts: standard order",
            1: "1 technician · Spare parts: routine check",
            0: "No immediate resources needed",
        }[int(score)]

    out["Maintenance Schedule"] = out["Priority Score"].map(schedule)
    out["Resource Plan"] = out["Priority Score"].map(resources)
    out["Risk Band"] = np.where(out["Risk Prediction"] == 1, "Predicted Failure Risk", "Predicted Safe")

    out = out.sort_values(
        ["Priority Score", "Failure Risk %", "Criticality Score"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    out.insert(0, "Priority Rank", np.arange(1, len(out) + 1))
    return out


def simulate_delay(
    row: pd.Series,
    model: RandomForestClassifier,
    delay_days: int,
    wear_per_day: float,
    torque_per_day: float,
    temp_per_day: float,
):
    current = row[FEATURES].to_frame().T.astype(float)
    delayed = current.copy()

    delayed["Tool wear [min]"] = np.clip(
        delayed["Tool wear [min]"] + delay_days * wear_per_day,
        0,
        1000,
    )
    delayed["Torque [Nm]"] = np.maximum(
        delayed["Torque [Nm]"] + delay_days * torque_per_day,
        0,
    )
    delayed["Air temperature [K]"] += delay_days * temp_per_day
    delayed["Process temperature [K]"] += delay_days * temp_per_day

    current_probability = float(model.predict_proba(current)[0, 1] * 100)
    delayed_probability = float(model.predict_proba(delayed)[0, 1] * 100)
    current_prediction = int(model.predict(current)[0])
    delayed_prediction = int(model.predict(delayed)[0])

    # Re-use the original dataset's criticality scale via tertile thresholds.
    usage = (
        delayed["Rotational speed [rpm]"]
        + delayed["Torque [Nm]"]
        + delayed["Tool wear [min]"]
    ).iloc[0]
    thresholds = np.quantile(
        BASE_DF["Rotational speed [rpm]"]
        + BASE_DF["Torque [Nm]"]
        + BASE_DF["Tool wear [min]"],
        [1 / 3, 2 / 3],
    )
    if usage <= thresholds[0]:
        delayed_criticality = "Low"
    elif usage <= thresholds[1]:
        delayed_criticality = "Medium"
    else:
        delayed_criticality = "High"

    current_priority = int(row["Priority Score"])
    delayed_priority = delayed_prediction * CRITICALITY_MAP[delayed_criticality]

    schedule_map = {
        3: "Immediate (within 3 days)",
        2: "Short-term (within 2 weeks)",
        1: "Routine (within 1 month)",
        0: "No action needed",
    }

    return {
        "current_probability": current_probability,
        "delayed_probability": delayed_probability,
        "risk_delta": delayed_probability - current_probability,
        "current_prediction": current_prediction,
        "delayed_prediction": delayed_prediction,
        "current_priority": current_priority,
        "delayed_priority": delayed_priority,
        "current_criticality": str(row["Criticality"]),
        "delayed_criticality": delayed_criticality,
        "current_schedule": str(row["Maintenance Schedule"]),
        "delayed_schedule": schedule_map[delayed_priority],
        "current_wear": float(current["Tool wear [min]"].iloc[0]),
        "delayed_wear": float(delayed["Tool wear [min]"].iloc[0]),
    }


if not DATA_PATH.exists():
    st.error(f"Dataset not found: {DATA_PATH}")
    st.stop()

BASE_DF = load_data(str(DATA_PATH))
MODEL, METRICS, CV_RECALL = train_model(BASE_DF)
PLAN = add_maintenance_plan(BASE_DF, MODEL)

st.title("⚙️ AI Maintenance Planner")
st.caption(
    "Predictive maintenance dashboard built around the project's existing Random Forest + SMOTE pipeline"
)

# Sidebar filters
with st.sidebar:
    st.header("Filters")
    machine_search = st.text_input("Search Machine ID", placeholder="e.g. M25")
    criticality_filter = st.multiselect(
        "Criticality",
        ["High", "Medium", "Low"],
        default=["High", "Medium", "Low"],
    )
    priority_filter = st.multiselect(
        "Maintenance priority",
        [3, 2, 1, 0],
        default=[3, 2, 1, 0],
        format_func=lambda x: {
            3: "3 · Immediate",
            2: "2 · Short-term",
            1: "1 · Routine",
            0: "0 · No action",
        }[x],
    )

FILTERED = PLAN[
    PLAN["Criticality"].astype(str).isin(criticality_filter)
    & PLAN["Priority Score"].isin(priority_filter)
].copy()
if machine_search.strip():
    FILTERED = FILTERED[
        FILTERED["Machine ID"].str.contains(machine_search.strip(), case=False, na=False)
    ]

# KPIs
predicted_failures = int(PLAN["Risk Prediction"].sum())
immediate_count = int((PLAN["Priority Score"] == 3).sum())
high_criticality_failures = int(
    ((PLAN["Risk Prediction"] == 1) & (PLAN["Criticality"] == "High")).sum()
)
actual_failures = int(BASE_DF[TARGET].sum())

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Total Machines", f"{len(PLAN):,}")
k2.metric("Predicted Failures", f"{predicted_failures:,}")
k3.metric("Immediate Priority", f"{immediate_count:,}")
k4.metric("High-Criticality Risk", f"{high_criticality_failures:,}")
k5.metric("Recorded Failures", f"{actual_failures:,}")

st.divider()

overview_tab, queue_tab, whatif_tab, model_tab = st.tabs(
    ["Overview", "Maintenance Queue", "What-if Simulation", "Model Performance"]
)

with overview_tab:
    a, b = st.columns(2)
    with a:
        crit_counts = (
            PLAN["Criticality"]
            .value_counts()
            .reindex(["High", "Medium", "Low"])
            .fillna(0)
            .rename_axis("Criticality")
            .reset_index(name="Machines")
        )
        fig = px.bar(
            crit_counts,
            x="Criticality",
            y="Machines",
            text_auto=True,
            title="Machines by Criticality",
        )
        st.plotly_chart(fig, use_container_width=True)

    with b:
        priority_counts = (
            PLAN["Priority Score"]
            .value_counts()
            .reindex([3, 2, 1, 0])
            .fillna(0)
            .rename_axis("Priority Score")
            .reset_index(name="Machines")
        )
        fig = px.bar(
            priority_counts,
            x="Priority Score",
            y="Machines",
            text_auto=True,
            title="Maintenance Priority Distribution",
        )
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Top Machines to Review First")
    top = PLAN.head(10)[
        [
            "Priority Rank",
            "Machine ID",
            "Type",
            "Failure Risk %",
            "Risk Prediction",
            "Criticality",
            "Priority Score",
            "Maintenance Schedule",
            "Resource Plan",
        ]
    ].copy()
    top["Failure Risk %"] = top["Failure Risk %"].round(1)
    st.dataframe(top, use_container_width=True, hide_index=True)

    st.info(
        "Priority Score follows the original project logic: predicted failure (0/1) × criticality (1/2/3). "
        "Risk probability is shown separately so the team can see how confident the model is."
    )

with queue_tab:
    st.subheader("Maintenance Queue")
    queue_columns = [
        "Priority Rank",
        "Machine ID",
        "Type",
        "Failure Risk %",
        "Risk Prediction",
        "Criticality",
        "Priority Score",
        "Maintenance Schedule",
        "Resource Plan",
        "Air temperature [K]",
        "Process temperature [K]",
        "Rotational speed [rpm]",
        "Torque [Nm]",
        "Tool wear [min]",
    ]
    queue = FILTERED[queue_columns].copy()
    for col in [
        "Failure Risk %",
        "Air temperature [K]",
        "Process temperature [K]",
        "Torque [Nm]",
    ]:
        queue[col] = queue[col].round(2)
    st.dataframe(queue, use_container_width=True, hide_index=True, height=580)

    st.download_button(
        "⬇️ Download filtered maintenance queue",
        queue.to_csv(index=False).encode("utf-8"),
        file_name="maintenance_queue.csv",
        mime="text/csv",
    )

with whatif_tab:
    st.subheader("What-if: What happens if maintenance is delayed?")
    st.write(
        "Choose a machine and test a maintenance delay. Because the AI4I dataset does not contain a real future time-series, "
        "this feature is an explicit scenario simulation: it increases selected sensor values and reruns the trained classifier."
    )

    left, right = st.columns(2)
    with left:
        selected_machine = st.selectbox("Machine", PLAN["Machine ID"].tolist())
        delay_days = st.slider("Delay (days)", 1, 30, 7)
    with right:
        wear_per_day = st.slider("Tool wear increase / day (min)", 0.0, 30.0, 10.0, 1.0)
        torque_per_day = st.slider("Torque increase / day (Nm)", 0.0, 2.0, 0.3, 0.1)
        temp_per_day = st.slider("Temperature increase / day (K)", 0.0, 1.0, 0.2, 0.1)

    selected_row = PLAN.loc[PLAN["Machine ID"] == selected_machine].iloc[0]
    scenario = simulate_delay(
        selected_row,
        MODEL,
        delay_days,
        wear_per_day,
        torque_per_day,
        temp_per_day,
    )

    r1, r2, r3 = st.columns(3)
    r1.metric("Current risk", f"{scenario['current_probability']:.1f}%")
    r2.metric(
        "Risk after delay",
        f"{scenario['delayed_probability']:.1f}%",
        delta=f"{scenario['risk_delta']:+.1f} pp",
    )
    r3.metric(
        "Tool wear",
        f"{scenario['current_wear']:.0f} → {scenario['delayed_wear']:.0f} min",
    )

    st.subheader("Priority and scheduling impact")
    q1, q2, q3, q4 = st.columns(4)
    q1.metric("Current priority", str(scenario["current_priority"]))
    q2.metric("Delayed priority", str(scenario["delayed_priority"]))
    q3.metric("Criticality", f"{scenario['current_criticality']} → {scenario['delayed_criticality']}")
    q4.metric("Schedule", "Changed" if scenario["current_schedule"] != scenario["delayed_schedule"] else "Unchanged")

    comparison = pd.DataFrame(
        {
            "Item": ["Failure probability", "Failure prediction", "Criticality", "Priority score", "Maintenance schedule"],
            "Current": [
                f"{scenario['current_probability']:.1f}%",
                "Failure" if scenario["current_prediction"] else "Safe",
                scenario["current_criticality"],
                scenario["current_priority"],
                scenario["current_schedule"],
            ],
            "After delay": [
                f"{scenario['delayed_probability']:.1f}%",
                "Failure" if scenario["delayed_prediction"] else "Safe",
                scenario["delayed_criticality"],
                scenario["delayed_priority"],
                scenario["delayed_schedule"],
            ],
        }
    )
    st.dataframe(comparison, use_container_width=True, hide_index=True)

    st.warning(
        "Important: the delay assumptions are demonstration inputs, not a claim about real Suzlon sensor behavior. "
        "Calibrate them against historical maintenance/failure data before presenting this as a production forecast."
    )

with model_tab:
    st.subheader("Model Performance")
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Accuracy", f"{METRICS['Accuracy'] * 100:.1f}%")
    m2.metric("Precision", f"{METRICS['Precision'] * 100:.1f}%")
    m3.metric("Recall", f"{METRICS['Recall'] * 100:.1f}%")
    m4.metric("F1 Score", f"{METRICS['F1 Score'] * 100:.1f}%")
    m5.metric("ROC-AUC", f"{METRICS['ROC-AUC'] * 100:.1f}%")

    st.write(
        "For this imbalanced dataset, recall is important because missing an actual failure is more costly "
        "than a false alarm. The values above come from the held-out test set."
    )

    cm = METRICS["Confusion Matrix"]
    cm_df = pd.DataFrame(
        cm,
        index=["Actual Safe", "Actual Failure"],
        columns=["Predicted Safe", "Predicted Failure"],
    )
    st.subheader("Confusion Matrix")
    st.dataframe(cm_df, use_container_width=True)

    cv_df = pd.DataFrame(
        {
            "Fold": [f"Fold {i}" for i in range(1, 6)],
            "Recall": CV_RECALL,
        }
    )
    cv_df["Recall"] = cv_df["Recall"] * 100
    fig = px.bar(
        cv_df,
        x="Fold",
        y="Recall",
        text_auto=".1f",
        title=f"5-Fold Cross-Validated Recall (average {CV_RECALL.mean() * 100:.1f}%)",
    )
    st.plotly_chart(fig, use_container_width=True)

    importance = pd.DataFrame(
        {
            "Feature": FEATURES,
            "Importance": MODEL.feature_importances_,
        }
    ).sort_values("Importance", ascending=False)
    fig = px.bar(
        importance,
        x="Importance",
        y="Feature",
        orientation="h",
        text_auto=".3f",
        title="Random Forest Feature Importance",
    )
    st.plotly_chart(fig, use_container_width=True)

st.caption(
    "Prototype scope: AI4I 2020 is a public benchmark dataset. For a real Suzlon deployment, replace it with plant data and calibrate criticality, maintenance durations, resource rules and delay assumptions."
)
