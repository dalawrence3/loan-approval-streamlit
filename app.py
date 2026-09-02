import pickle

import numpy as np
import pandas as pd
import shap
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(page_title="Loan Decision Analysis", layout="wide")

with open("model.pkl", "rb") as file:
    model_data = pickle.load(file)

model = model_data["model"]
scaler = model_data["scaler"]
features = model_data["features"]

# The pickle stores no explicit label semantics for model.classes_, so the mapping
# "class 1 == approved" is an assumption inherited from the original app (supported by
# coefficient signs, e.g. FICO_score is positive and Ever_Bankrupt_or_Foreclose is
# negative). Guard it explicitly rather than silently trusting index [1] forever.
assert list(model.classes_) == [0, 1], (
    "model.classes_ no longer matches the expected [0, 1] ordering this app assumes "
    "for 'approval probability' — stop and re-verify the class mapping before trusting output."
)

# Background for shap.LinearExplainer: the original training dataset is not available in
# this repository, so we cannot estimate a real feature covariance matrix. We use the all-
# zero vector in already-SCALED feature space as the single background sample. Because
# StandardScaler maps each training feature's mean to 0, this scaled-zero vector represents
# the model evaluated at the training-feature means (mathematically identical to using
# scaler.mean_ as the background row), under an independent/interventional feature-
# perturbation assumption (we don't fabricate a correlation structure we have no evidence
# for). expected_value therefore equals the model's intercept, which is directly verifiable.
#
# This is a distinct concept from any single *applicant's* raw zero-valued feature (e.g. a
# zero-filled Reason_* or Employment_Sector_* dummy below): StandardScaler generally maps a
# raw 0 to (0 - mean) / std, which is NOT 0 in scaled space unless that feature's training
# mean happens to be 0. A zero-filled dummy can therefore still carry a non-trivial SHAP
# value relative to the training population — that is expected and mathematically valid,
# not a bug. It is exactly why those dummies are excluded from the primary display below by
# name/prefix (they were never supplied by the applicant), never by checking their SHAP
# magnitude.
_background = np.zeros((1, len(features)))
explainer = shap.LinearExplainer(model, masker=_background)

# Structural/non-independent features: Granted_Loan_Amount is hard-set equal to
# Requested_Loan_Amount below (this app has no separate underwriting-approved-amount
# workflow), which makes Loan_Gap always 0. Neither is an independently supplied
# applicant characteristic, so both are excluded from the human-facing top-driver
# list/chart even though they remain part of the full SHAP calculation.
STRUCTURAL_FEATURES = {"Granted_Loan_Amount", "Loan_Gap"}

# Loan Purpose (Reason) and Employment Sector are NOT collected in this Streamlit UI: the
# original exploratory analysis found both categorical fields provided relatively little
# practical distinction in approval rates compared to stronger predictors (FICO, lender,
# employment status, etc.), so the deployed interface intentionally omits them. Their
# dummy columns are therefore always zero-filled in the encoded input (see the feature-
# alignment loop below) and remain part of the complete 29-feature model vector and the
# full SHAP calculation — they are excluded from the primary applicant-facing driver
# list/chart purely because the applicant never supplied a value for them, NOT because
# their SHAP contribution is assumed to be small (see the scaling note above: a zero-filled
# dummy can still have a non-zero scaled value and a non-trivial SHAP contribution).
NON_APPLICANT_PROVIDED_PREFIXES = ("Reason_", "Employment_Sector_")

FEATURE_LABELS = {
    "Granted_Loan_Amount": "Granted Loan Amount",
    "Requested_Loan_Amount": "Requested Loan Amount",
    "FICO_score": "FICO Score",
    "Monthly_Gross_Income": "Monthly Income",
    "Monthly_Housing_Payment": "Housing Payment",
    "Ever_Bankrupt_or_Foreclose": "Prior Bankruptcy / Foreclosure",
    "Loan_to_Income": "Loan-to-Income Ratio",
    "Payment_to_Income": "Housing-Payment-to-Income Ratio",
    "Loan_Gap": "Requested vs. Granted Loan Gap",
    "Reason_credit_card_refinancing": "Loan Purpose: Credit Card Refinancing",
    "Reason_debt_conslidation": "Loan Purpose: Debt Consolidation",
    "Reason_home_improvement": "Loan Purpose: Home Improvement",
    "Reason_major_purchase": "Loan Purpose: Major Purchase",
    "Reason_other": "Loan Purpose: Other",
    "Employment_Status_part_time": "Employment Status: Part Time",
    "Employment_Status_unemployed": "Employment Status: Unemployed",
    "Employment_Sector_communication_services": "Employment Sector: Communication Services",
    "Employment_Sector_consumer_discretionary": "Employment Sector: Consumer Discretionary",
    "Employment_Sector_consumer_staples": "Employment Sector: Consumer Staples",
    "Employment_Sector_energy": "Employment Sector: Energy",
    "Employment_Sector_financials": "Employment Sector: Financials",
    "Employment_Sector_health_care": "Employment Sector: Health Care",
    "Employment_Sector_industrials": "Employment Sector: Industrials",
    "Employment_Sector_information_technology": "Employment Sector: Information Technology",
    "Employment_Sector_materials": "Employment Sector: Materials",
    "Employment_Sector_real_estate": "Employment Sector: Real Estate",
    "Employment_Sector_utilities": "Employment Sector: Utilities",
    "Lender_B": "Selected Lender: B",
    "Lender_C": "Selected Lender: C",
}

DECISION_THRESHOLD = 0.65


def label_for(feature_name: str) -> str:
    return FEATURE_LABELS.get(feature_name, feature_name)


# One-hot dummy columns are mutually exclusive within each of these groups. A dummy's
# SHAP value reflects a shift from the population-average membership in that single
# category, so labeling it with the category name alone is misleading whenever the
# applicant's real value is 0 (e.g. a full-time applicant could otherwise show up under
# "Employment Status: Unemployed"). Instead we sum the SHAP values across every dummy in
# a group — a mathematically valid use of SHAP additivity — into one entry labeled with
# the applicant's actual selected category, which is always accurate.
CATEGORY_GROUPS = [
    ("Employment_Status_", "Employment Status"),
    ("Lender_", "Selected Lender"),
]


def build_display_contributions(shap_row, feature_names, exclude, current_labels, top_n=8):
    """Rank contributions by |SHAP value| for the business-facing view.

    Structural features and non-applicant-provided features (see STRUCTURAL_FEATURES and
    NON_APPLICANT_PROVIDED_PREFIXES) are dropped regardless of their SHAP magnitude — they
    are excluded because the applicant never supplied them, not because their contribution
    happens to be small. One-hot categorical groups that ARE applicant-supplied are
    collapsed into a single entry per group, labeled with the applicant's actual category.
    """
    grouped_indices = set()
    entries = []

    for prefix, group_name in CATEGORY_GROUPS:
        idxs = [i for i, f in enumerate(feature_names) if f.startswith(prefix)]
        if not idxs:
            continue
        grouped_indices.update(idxs)
        total = sum(shap_row[i] for i in idxs)
        entries.append((f"{group_name}: {current_labels[prefix]}", total))

    for i, f in enumerate(feature_names):
        if i in grouped_indices or f in exclude or f.startswith(NON_APPLICANT_PROVIDED_PREFIXES):
            continue
        entries.append((label_for(f), shap_row[i]))

    entries.sort(key=lambda pair: abs(pair[1]), reverse=True)
    return entries[:top_n]


st.title("Loan Decision Analysis")
st.write(
    "Enter applicant information to generate a model-based approval probability "
    "and see which factors drove the result."
)

st.subheader("Applicant Information")

col1, col2 = st.columns(2)
with col1:
    fico = st.number_input("FICO Score", 300, 850, 650)
    income = st.number_input("Monthly Gross Income ($)", 1, 20000, 5000)
    requested_loan = st.number_input("Requested Loan Amount ($)", 1000, 2500000, 20000)
    housing = st.number_input("Housing Payment (Monthly) ($)", 0, 50000, 1500)
with col2:
    employment_status_label = st.selectbox(
        "Employment Status", ["Full time", "Part time", "Unemployed"]
    )
    bankrupt_label = st.selectbox("Ever Bankrupt or Foreclosed?", ["No", "Yes"])
    lender = st.selectbox("Select Lender", ["A", "B", "C"])

employment_status_map = {
    "Full time": "full_time",
    "Part time": "part_time",
    "Unemployed": "unemployed",
}
employment_status = employment_status_map[employment_status_label]
bankrupt = 1 if bankrupt_label == "Yes" else 0

if st.button("Analyze Application", type="primary"):

    input_df = pd.DataFrame({
        "Granted_Loan_Amount": [requested_loan],
        "Requested_Loan_Amount": [requested_loan],
        "FICO_score": [fico],
        "Employment_Status": [employment_status],
        "Monthly_Gross_Income": [income],
        "Monthly_Housing_Payment": [housing],
        "Ever_Bankrupt_or_Foreclose": [bankrupt],
        "Lender": [lender],
        # Reason and Employment_Sector are intentionally not collected in this UI (see
        # NON_APPLICANT_PROVIDED_PREFIXES above) — omitting them here means the
        # feature-alignment loop below fills every Reason_*/Employment_Sector_* dummy
        # with 0, and scaler.transform then maps those zeros into scaled space normally.
    })

    input_df["Loan_to_Income"] = input_df["Requested_Loan_Amount"] / input_df["Monthly_Gross_Income"]
    input_df["Payment_to_Income"] = input_df["Monthly_Housing_Payment"] / input_df["Monthly_Gross_Income"]
    input_df["Loan_Gap"] = input_df["Requested_Loan_Amount"] - input_df["Granted_Loan_Amount"]

    input_encoded = pd.get_dummies(input_df)

    for col in features:
        if col not in input_encoded.columns:
            input_encoded[col] = 0

    input_encoded = input_encoded[features]

    input_scaled = scaler.transform(input_encoded)

    # class 1 == approved (see assertion + comment above)
    prob = model.predict_proba(input_scaled)[0][1]
    prediction = int(prob >= DECISION_THRESHOLD)

    st.divider()
    st.subheader("Loan Decision Analysis")

    m1, m2, m3 = st.columns(3)
    m1.metric("Predicted Outcome", "APPROVED" if prediction == 1 else "NOT APPROVED")
    m2.metric("Approval Probability", f"{prob:.0%}")
    m3.metric("Decision Threshold", f"{DECISION_THRESHOLD:.0%}")

    if prediction == 1:
        st.success(f"Predicted: Approved — Lender {lender}")
    else:
        st.error(f"Predicted: Not Approved — Lender {lender}")

    # --- SHAP explanation ---
    shap_row = explainer.shap_values(input_scaled)[0]
    expected_value = explainer.expected_value
    if hasattr(expected_value, "__len__"):
        expected_value = expected_value[0]

    decision = model.decision_function(input_scaled)[0]
    reconstructed = expected_value + shap_row.sum()
    local_accuracy_diff = abs(decision - reconstructed)

    if local_accuracy_diff > 1e-6:
        st.warning(
            "The SHAP explanation failed its local-accuracy check for this input "
            "(expected_value + sum(SHAP values) did not reconstruct the model's "
            "decision function within tolerance), so the explanation is hidden to "
            "avoid showing a potentially incorrect result. The decision and "
            "probability above are unaffected."
        )
    else:
        current_labels = {
            "Employment_Status_": employment_status_label,
            "Lender_": f"Lender {lender}",
        }
        ranked = build_display_contributions(
            shap_row, features, STRUCTURAL_FEATURES, current_labels, top_n=8
        )
        supporting = [(label, v) for label, v in ranked if v > 0]
        against = [(label, v) for label, v in ranked if v < 0]

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Factors Supporting Approval**")
            if supporting:
                for label, _ in supporting:
                    st.write(f"- {label}")
            else:
                st.caption("No factors materially supported approval for this applicant.")
        with c2:
            st.markdown("**Factors Against Approval**")
            if against:
                for label, _ in against:
                    st.write(f"- {label}")
            else:
                st.caption("No factors materially worked against approval for this applicant.")

        st.caption(
    "Model explanations reflect both applicant inputs and derived financial ratios. "
    "Less informative categorical fields are excluded from the primary view but "
    "remain part of the underlying model."
        )

        st.subheader("Model Contribution Analysis")
        st.caption(
            "Each bar shows how much that factor pushed this applicant's prediction "
            "toward approval (right, green) or away from it (left, red), ranked by "
            "the size of its effect."
        )

        chart_data = sorted(ranked, key=lambda pair: pair[1])
        labels = [label for label, _ in chart_data]
        values = [v for _, v in chart_data]
        colors = ["#2e7d32" if v > 0 else "#c62828" for v in values]

        fig, ax = plt.subplots(figsize=(6, 0.45 * len(values) + 1))
        ax.barh(labels, values, color=colors)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_xlabel("Contribution to prediction (log-odds)")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        fig.tight_layout()
        st.pyplot(fig)

        with st.expander("Technical Model Explanation"):
            st.write(
                "Complete model feature explanation: this table lists all "
                f"{len(features)} features the trained model actually uses, including "
                "ones intentionally excluded from the applicant-facing view above."
            )
            st.write(
                "Explainer: `shap.LinearExplainer` on the exact scaled input passed "
                "to the logistic regression model (values are in log-odds units, "
                "the model's native decision-function scale — not probability)."
            )
            st.write(
                "Background / masking strategy: a single all-zero vector in SCALED "
                "feature space. Because `StandardScaler` maps each training feature's "
                "mean to 0, this scaled-zero background represents the model evaluated "
                "at the training-feature means — mathematically equivalent to using "
                "`scaler.mean_` as the background row — without assuming any feature "
                "correlation structure that can't be verified from the artifacts stored "
                "in this repository (no training dataset is available). Note this is "
                "distinct from any single applicant's raw zero-valued feature, which "
                "generally does NOT map to scaled zero (see below)."
            )
            st.write(f"Explainer expected value (model intercept): `{expected_value:.6f}`")
            st.write(f"Model decision_function output: `{decision:.6f}`")
            st.write(
                f"Local-accuracy check — expected_value + sum(SHAP values): "
                f"`{reconstructed:.6f}` (difference: `{local_accuracy_diff:.2e}`)"
            )
            full_table = pd.DataFrame(
                {
                    "Feature": [label_for(f) for f in features],
                    "Raw Feature": features,
                    "SHAP Value": shap_row,
                    "Applicant-Provided": [
                        f not in STRUCTURAL_FEATURES
                        and not f.startswith(NON_APPLICANT_PROVIDED_PREFIXES)
                        for f in features
                    ],
                }
            ).sort_values("SHAP Value", key=abs, ascending=False)
            st.caption(
                "`Applicant-Provided = False` marks features that are structurally "
                "fixed (Granted Loan Amount / Loan Gap) or intentionally not collected "
                "in this UI (Loan Purpose, Employment Sector) and therefore zero-filled "
                "in the input vector — see caption above the contribution chart. Their "
                "SHAP values are real and included in the local-accuracy check above; "
                "they are simply not attributable to anything this applicant told us."
            )
            st.dataframe(full_table, width="stretch", hide_index=True)

    st.caption(
        "This application is an educational portfolio demonstration of a "
        "machine-learning decision-support workflow and does not represent an "
        "actual lending decision or underwriting policy."
    )
