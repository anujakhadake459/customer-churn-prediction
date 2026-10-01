import streamlit as st
import pandas as pd
import joblib


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)


# ============================================================
# LOAD SAVED MODEL ARTIFACTS
# ============================================================

@st.cache_resource
def load_artifacts():

    model = joblib.load(
        "models/final_xgboost_model.pkl"
    )

    preprocessor = joblib.load(
        "models/preprocessor.pkl"
    )

    threshold = joblib.load(
        "models/best_threshold.pkl"
    )

    return model, preprocessor, threshold


try:

    model, preprocessor, threshold = load_artifacts()

except Exception as e:

    st.error("Could not load the saved model files.")
    st.exception(e)
    st.stop()


# ============================================================
# PREPARE CUSTOMER DATA
# SAME FEATURE ENGINEERING USED DURING TRAINING
# ============================================================

def prepare_customer_data(customer_data):

    customer_data = customer_data.copy()

    # --------------------------------------------------------
    # Convert Signup_Date
    # --------------------------------------------------------

    if "Signup_Date" in customer_data.columns:

        customer_data["Signup_Date"] = pd.to_datetime(
            customer_data["Signup_Date"],
            errors="coerce"
        )

        # Create same date features used during training
        customer_data["Signup_Year"] = (
            customer_data["Signup_Date"].dt.year
        )

        customer_data["Signup_Month"] = (
            customer_data["Signup_Date"].dt.month
        )

        customer_data["Signup_Quarter"] = (
            customer_data["Signup_Date"].dt.quarter
        )

        customer_data["Signup_DayOfWeek"] = (
            customer_data["Signup_Date"].dt.dayofweek
        )

        # Remove original date
        customer_data = customer_data.drop(
            columns=["Signup_Date"]
        )

    # --------------------------------------------------------
    # Remove Customer_ID
    # SAME AS NOTEBOOK
    # --------------------------------------------------------

    if "Customer_ID" in customer_data.columns:

        customer_data = customer_data.drop(
            columns=["Customer_ID"]
        )

    # --------------------------------------------------------
    # SAME FEATURE ENGINEERING USED DURING TRAINING
    # --------------------------------------------------------

    # 1. Average Monthly Total Charges

    customer_data["Avg_Monthly_Total_Charges"] = (
        customer_data["Total_Charges"]
        / customer_data["Tenure_Months"].replace(0, 1)
    )

    # 2. Charge Ratio

    customer_data["Charge_Ratio"] = (
        customer_data["Monthly_Charges"]
        / customer_data[
            "Avg_Monthly_Total_Charges"
        ].replace(0, 1)
    )

    # 3. Activity Risk

    customer_data["Activity_Risk"] = (
        customer_data["Days_Since_Last_Activity"]
        / customer_data["Tenure_Months"].replace(0, 1)
    )

    # 4. Service Issue Score

    customer_data["Service_Issue_Score"] = (
        customer_data["Complaint_Count"]
        + customer_data["Support_Tickets_Raised"]
    )

    return customer_data


# ============================================================
# TITLE
# ============================================================

st.title("📊 Customer Churn Prediction")

st.write(
    "Upload customer data and predict the probability "
    "of customer churn."
)

st.info(
    f"Model: XGBoost | "
    f"Decision Threshold: {threshold:.2f}"
)


# ============================================================
# UPLOAD CUSTOMER DATA
# ============================================================

st.subheader("1. Upload Customer Data")

uploaded_file = st.file_uploader(
    "Upload a CSV file",
    type=["csv"]
)


if uploaded_file is not None:

    # ========================================================
    # READ CSV
    # ========================================================

    try:

        customer_data = pd.read_csv(
            uploaded_file
        )

        st.success(
            "Customer data uploaded successfully."
        )

        st.write("### Uploaded Data")

        st.dataframe(
            customer_data,
            use_container_width=True
        )

    except Exception as e:

        st.error(
            "Could not read the CSV file."
        )

        st.exception(e)

        st.stop()


    # ========================================================
    # PREDICTION
    # ========================================================

    st.subheader("2. Predict Churn")

    if st.button("🔮 Predict Churn"):

        try:

            # ------------------------------------------------
            # PREPARE CUSTOMER DATA
            # ------------------------------------------------

            prepared_data = prepare_customer_data(
                customer_data
            )

            # ------------------------------------------------
            # PREPROCESS
            # ------------------------------------------------

            processed_data = preprocessor.transform(
                prepared_data
            )

            # ------------------------------------------------
            # CHURN PROBABILITY
            # ------------------------------------------------

            churn_probability = (
                model.predict_proba(
                    processed_data
                )[:, 1]
            )

            # ------------------------------------------------
            # APPLY SAVED THRESHOLD
            # ------------------------------------------------

            prediction = [
                "Will Churn"
                if probability >= threshold
                else "Will Not Churn"
                for probability in churn_probability
            ]

            # ------------------------------------------------
            # RISK LEVEL
            # ------------------------------------------------

            risk_level = []

            for probability in churn_probability:

                if probability >= 0.70:

                    risk_level.append(
                        "High Risk"
                    )

                elif probability >= 0.40:

                    risk_level.append(
                        "Medium Risk"
                    )

                else:

                    risk_level.append(
                        "Low Risk"
                    )

            # ------------------------------------------------
            # CREATE RESULTS TABLE
            # ------------------------------------------------

            results = customer_data.copy()

            results["Churn Probability"] = (
                churn_probability * 100
            ).round(2)

            results["Prediction"] = prediction

            results["Risk Level"] = risk_level

            # ------------------------------------------------
            # DISPLAY RESULTS
            # ------------------------------------------------

            st.subheader(
                "3. Prediction Results"
            )

            st.dataframe(
                results,
                use_container_width=True
            )

            # ------------------------------------------------
            # SUMMARY
            # ------------------------------------------------

            total_customers = len(
                results
            )

            predicted_churn = sum(
                results["Prediction"]
                == "Will Churn"
            )

            predicted_no_churn = sum(
                results["Prediction"]
                == "Will Not Churn"
            )

            st.subheader(
                "Prediction Summary"
            )

            col1, col2, col3 = st.columns(3)

            with col1:

                st.metric(
                    "Total Customers",
                    total_customers
                )

            with col2:

                st.metric(
                    "Likely to Churn",
                    predicted_churn
                )

            with col3:

                st.metric(
                    "Likely to Stay",
                    predicted_no_churn
                )

            # ------------------------------------------------
            # DOWNLOAD RESULTS
            # ------------------------------------------------

            csv = results.to_csv(
                index=False
            ).encode("utf-8")

            st.download_button(
                label="⬇️ Download Predictions",
                data=csv,
                file_name="churn_predictions.csv",
                mime="text/csv"
            )

        except Exception as e:

            st.error(
                "Prediction failed."
            )

            st.exception(e)


else:

    st.warning(
        "Please upload a CSV file to begin."
    )