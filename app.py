import streamlit as st
import pandas as pd
import joblib


# -------------------------------------------------
# Page configuration
# -------------------------------------------------

st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📊",
    layout="wide"
)


# -------------------------------------------------
# Load model
# -------------------------------------------------

@st.cache_resource
def load_model():
    return joblib.load("customer_churn_pipeline.pkl")


model = load_model()


# -------------------------------------------------
# Styling
# -------------------------------------------------

st.markdown("""
<style>
    .block-container {
        max-width: 1100px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    .hero {
        padding: 28px;
        border-radius: 18px;
        color: white;
        background: linear-gradient(135deg, #172554, #2563eb);
        margin-bottom: 25px;
    }

    .hero h1 {
        margin: 0;
        font-size: 36px;
    }

    .hero p {
        margin: 8px 0 0;
        color: #dbeafe;
        font-size: 16px;
    }

    .result-card {
        padding: 24px;
        border-radius: 16px;
        margin-top: 16px;
    }

    .low-risk {
        background: #f0fdf4;
        border: 1px solid #86efac;
        color: #166534;
    }

    .medium-risk {
        background: #fffbeb;
        border: 1px solid #fcd34d;
        color: #92400e;
    }

    .high-risk {
        background: #fef2f2;
        border: 1px solid #fca5a5;
        color: #991b1b;
    }

    div.stFormSubmitButton > button {
        width: 100%;
        height: 48px;
        border-radius: 10px;
        background-color: #2563eb;
        color: white;
        border: none;
        font-weight: 600;
        font-size: 16px;
    }
</style>
""", unsafe_allow_html=True)


# -------------------------------------------------
# Sidebar
# -------------------------------------------------

with st.sidebar:

    st.title("📊 About")

    st.write("**Model:** Logistic Regression")
    st.write("**Task:** Binary Classification")
    st.write("**Output:** Churn probability")

    st.divider()

    st.write("### Risk levels")
    st.write("🟢 Low: Below 40%")
    st.write("🟡 Medium: 40%–69%")
    st.write("🔴 High: 70% or above")

    st.divider()

    st.caption(
        "This prediction should support customer-retention "
        "decisions and not be treated as a guaranteed outcome."
    )


# -------------------------------------------------
# Header
# -------------------------------------------------

st.markdown("""
<div class="hero">
    <h1>Customer Churn Predictor</h1>
    <p>
        Analyse customer behaviour and identify potential churn risk.
    </p>
</div>
""", unsafe_allow_html=True)


# -------------------------------------------------
# Input form
# -------------------------------------------------

with st.form("churn_form"):

    left_column, right_column = st.columns(2)

    with left_column:

        st.subheader("Customer and Services")

        # Field 1
        senior_citizen = st.selectbox(
            "1. Senior Citizen",
            ["No", "Yes"]
        )

        # Field 2
        tenure = st.slider(
            "2. Tenure (months)",
            min_value=0,
            max_value=72,
            value=12
        )

        # Field 3
        phone_plan = st.selectbox(
            "3. Phone Plan",
            [
                "No phone service",
                "Single line",
                "Multiple lines"
            ]
        )

        # Field 4
        internet_service = st.selectbox(
            "4. Internet Service",
            ["DSL", "Fiber optic", "No"]
        )

        # Field 5
        additional_services = st.multiselect(
            "5. Additional Services",
            [
                "Online Security",
                "Online Backup",
                "Device Protection",
                "Tech Support",
                "Streaming TV",
                "Streaming Movies"
            ],
            help="Select every additional service used by the customer."
        )

    with right_column:

        st.subheader("Contract and Billing")

        # Field 6
        contract = st.selectbox(
            "6. Contract Type",
            ["Month-to-month", "One year", "Two year"]
        )

        # Field 7
        paperless_billing = st.selectbox(
            "7. Paperless Billing",
            ["Yes", "No"]
        )

        # Field 8
        payment_method = st.selectbox(
            "8. Payment Method",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)"
            ]
        )

        # Field 9
        monthly_charges = st.number_input(
            "9. Monthly Charges",
            min_value=0.0,
            max_value=200.0,
            value=70.0,
            step=1.0
        )

        # Field 10
        total_charges = st.number_input(
            "10. Total Charges",
            min_value=0.0,
            value=840.0,
            step=10.0
        )

    submitted = st.form_submit_button("Predict Churn Risk")


# -------------------------------------------------
# Prediction
# -------------------------------------------------

if submitted:

    # Convert Phone Plan into original model columns
    if phone_plan == "No phone service":
        phone_service = "No"
        multiple_lines = "No phone service"

    elif phone_plan == "Single line":
        phone_service = "Yes"
        multiple_lines = "No"

    else:
        phone_service = "Yes"
        multiple_lines = "Yes"

    # Convert service multiselect into original columns
    if internet_service == "No":

        online_security = "No internet service"
        online_backup = "No internet service"
        device_protection = "No internet service"
        tech_support = "No internet service"
        streaming_tv = "No internet service"
        streaming_movies = "No internet service"

    else:

        online_security = (
            "Yes"
            if "Online Security" in additional_services
            else "No"
        )

        online_backup = (
            "Yes"
            if "Online Backup" in additional_services
            else "No"
        )

        device_protection = (
            "Yes"
            if "Device Protection" in additional_services
            else "No"
        )

        tech_support = (
            "Yes"
            if "Tech Support" in additional_services
            else "No"
        )

        streaming_tv = (
            "Yes"
            if "Streaming TV" in additional_services
            else "No"
        )

        streaming_movies = (
            "Yes"
            if "Streaming Movies" in additional_services
            else "No"
        )

    # Feature engineering: TenureGroup
    if tenure <= 12:
        tenure_group = "New"
    elif tenure <= 24:
        tenure_group = "Regular"
    elif tenure <= 48:
        tenure_group = "Loyal"
    else:
        tenure_group = "Highly Loyal"

    # Feature engineering: TotalServices
    total_services = sum([
        phone_service == "Yes",
        multiple_lines == "Yes",
        internet_service != "No",
        online_security == "Yes",
        online_backup == "Yes",
        device_protection == "Yes",
        tech_support == "Yes",
        streaming_tv == "Yes",
        streaming_movies == "Yes"
    ])

    # Feature engineering: HasLongTermContract
    has_long_term_contract = int(
        contract != "Month-to-month"
    )

    # Feature engineering: AverageMonthlySpend
    average_monthly_spend = (
        total_charges / tenure
        if tenure > 0
        else 0
    )

    customer_data = pd.DataFrame([{

        # Hidden defaults required by the current full model
        "gender": "Female",
        "Partner": "No",
        "Dependents": "No",

        # Visible inputs
        "SeniorCitizen": int(senior_citizen == "Yes"),
        "tenure": tenure,
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless_billing,
        "PaymentMethod": payment_method,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,

        # Engineered features
        "TenureGroup": tenure_group,
        "TotalServices": total_services,
        "HasLongTermContract": has_long_term_contract,
        "AverageMonthlySpend": average_monthly_spend
    }])

    prediction = model.predict(customer_data)[0]

    churn_probability = model.predict_proba(
        customer_data
    )[0][1]

    probability_percentage = churn_probability * 100

    st.divider()
    st.subheader("Prediction Result")

    metric1, metric2, metric3 = st.columns(3)

    metric1.metric(
        "Churn Probability",
        f"{probability_percentage:.1f}%"
    )

    metric2.metric(
        "Total Services",
        total_services
    )

    metric3.metric(
        "Customer Segment",
        tenure_group
    )

    st.progress(float(churn_probability))

    if probability_percentage >= 70:

        risk_title = "🔴 High Churn Risk"
        risk_class = "high-risk"
        risk_message = (
            "This customer has a high probability of leaving."
        )

    elif probability_percentage >= 40:

        risk_title = "🟡 Medium Churn Risk"
        risk_class = "medium-risk"
        risk_message = (
            "This customer should be monitored and engaged proactively."
        )

    else:

        risk_title = "🟢 Low Churn Risk"
        risk_class = "low-risk"
        risk_message = (
            "This customer currently has a relatively low churn risk."
        )

    st.markdown(
        f"""
        <div class="result-card {risk_class}">
            <h2>{risk_title}</h2>
            <p>{risk_message}</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.subheader("Recommended Actions")

    recommendations = []

    if contract == "Month-to-month":
        recommendations.append(
            "Offer a discount for switching to a long-term contract."
        )

    if tech_support != "Yes":
        recommendations.append(
            "Provide a technical-support trial or service bundle."
        )

    if online_security != "Yes":
        recommendations.append(
            "Recommend an online-security package."
        )

    if monthly_charges > 80:
        recommendations.append(
            "Review the plan and suggest a cost-effective alternative."
        )

    if probability_percentage < 40:
        recommendations.append(
            "Continue regular engagement and service monitoring."
        )

    for recommendation in recommendations:
        st.write(f"• {recommendation}")