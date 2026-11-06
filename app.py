import json
from pathlib import Path
import pandas as pd
import streamlit as st
import plotly.express as px

from modules.data_processing import prepare_dataframe, validate_dataset, data_quality_summary, FEATURES
from modules.exploratory_analysis import render_eda
from modules.prediction import predict_one
from modules.dashboard import render_dashboard

ROOT = Path(__file__).resolve().parent
DEMO_PATH = ROOT / "sample_cleaned_data.csv"
METRICS_PATH = ROOT / "models" / "model_metrics.json"

st.set_page_config(page_title="Student Service Request Monitoring System", page_icon="🎓", layout="wide")

@st.cache_data
def load_demo_data():
    return pd.read_csv(DEMO_PATH)

def get_data():
    if "service_df" not in st.session_state:
        st.session_state.service_df = prepare_dataframe(load_demo_data())
    return st.session_state.service_df

df = get_data()
menu = st.sidebar.selectbox(
    "Navigate to",
    ["Home", "Dataset", "Data Analysis", "Prediction", "Dashboard", "System Design"]
)
st.sidebar.markdown("---")
st.sidebar.caption("Section B | Systems Analysis and Design")
st.sidebar.caption("Decision-support application")

if menu == "Home":
    st.title("🎓 Student Service Request Monitoring System")
    st.subheader("Predictive SLA Breach Monitoring")
    st.write("A decision-support application for monitoring student service workloads, exploring historical request patterns, and identifying pending requests that may need staff attention.")
    st.info("This application reuses the Decision Tree pipeline from the submitted midterm notebook. Predictions are advisory and do not authorize automatic denial of student services.")
    a,b,c = st.columns(3)
    a.metric("Dataset records", f"{len(df):,}")
    b.metric("Model", "Decision Tree")
    c.metric("Prediction target", "SLA_Breached")
    st.markdown("### Application modules")
    st.markdown("""
    - **Dataset:** upload a CSV, validate its schema, preview records, and inspect data quality.
    - **Data Analysis:** examine descriptive statistics, class distribution, service volume, and breach patterns.
    - **Prediction:** enter request details and obtain an advisory SLA-breach prediction.
    - **Dashboard:** monitor workload indicators and review flagged pending requests.
    - **System Design:** review system actors, requirements, use cases, and the human-review workflow.
    """)
    st.caption("Demo data is loaded initially. Upload your approved CSV in the Dataset page to replace it.")

elif menu == "Dataset":
    st.title("Dataset Upload and Validation")
    uploaded = st.file_uploader("Upload student service request CSV", type=["csv"])
    if uploaded is not None:
        try:
            new_df = pd.read_csv(uploaded)
            new_df = prepare_dataframe(new_df)
            ok, errors, missing = validate_dataset(new_df)
            if not ok:
                for error in errors: st.error(error)
                st.caption("Required fields: " + ", ".join(["Request_ID", *FEATURES, "SLA_Breached"]))
            else:
                st.session_state.service_df = new_df
                df = new_df
                st.success("CSV uploaded and required schema validated.")
        except Exception as exc:
            st.error(f"Unable to read this CSV. Check that it is a valid comma-separated file. Details: {exc}")
    df = get_data()
    ok, errors, missing = validate_dataset(df)
    if ok:
        st.success("Current dataset contains all required model and target columns.")
    else:
        for error in errors: st.error(error)
    st.markdown("### Data preview")
    st.dataframe(df.head(20), use_container_width=True)
    summary = data_quality_summary(df)
    a,b,c,d = st.columns(4)
    a.metric("Rows", f"{summary['rows']:,}")
    b.metric("Columns", summary["columns"])
    c.metric("Missing cells", summary["missing_cells"])
    d.metric("Duplicate Request IDs", summary["duplicate_request_ids"])
    st.markdown("### Missing values by column")
    missing_df = pd.Series(summary["missing_by_column"], name="Missing values").reset_index().rename(columns={"index":"Column"})
    st.dataframe(missing_df, use_container_width=True)
    st.download_button("Download current dataset as CSV", df.to_csv(index=False).encode("utf-8"), "student_service_requests_current.csv", "text/csv")

elif menu == "Data Analysis":
    st.title("Data Analysis")
    render_eda(get_data())

elif menu == "Prediction":
    st.title("Predictive Modeling")
    st.write("Enter values known at the request-submission checkpoint. Do not enter future processing outcomes.")
    df = get_data()
    def choices(col, fallback):
        vals = df[col].dropna().astype(str).unique().tolist() if col in df else []
        return sorted(vals) if vals else fallback
    with st.form("prediction_form"):
        c1,c2 = st.columns(2)
        with c1:
            service = st.selectbox("Service Type", choices("Service_Type", ["Certification","Enrollment","Records","IT Support"]))
            requester = st.selectbox("Requester Type", choices("Requester_Type", ["Student","Faculty","Staff"]))
            channel = st.selectbox("Requested Channel", choices("Requested_Channel", ["Portal","Email","Walk-in"]))
            priority = st.selectbox("Priority", choices("Priority", ["Low","Normal","High"]))
            office = st.selectbox("Assigned Office", choices("Assigned_Office", ["Registrar","Student Affairs","Accounting Office"]))
        with c2:
            queue = st.number_input("Queue Length", min_value=0, max_value=10000, value=8, step=1)
            docs = st.selectbox("Required Documents Complete", choices("Required_Documents_Complete", ["Yes","No"]))
            sla = st.number_input("SLA Days", min_value=0, max_value=365, value=5, step=1)
            weekday = st.selectbox("Submission Weekday", ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"])
            qcat = "Low" if queue <= 5 else ("Medium" if queue <= 15 else "High")
            st.caption(f"Queue category derived using the midterm bins: {qcat}")
        submitted = st.form_submit_button("Predict SLA Breach")
    if submitted:
        values = {
            "Service_Type":service, "Requester_Type":requester, "Requested_Channel":channel,
            "Priority":priority, "Assigned_Office":office, "Queue_Length":queue,
            "Required_Documents_Complete":docs, "SLA_Days":sla,
            "Submission_Weekday":weekday, "Queue_Category":qcat
        }
        try:
            pred, prob = predict_one(values)
            st.markdown("### Prediction result")
            if pred == 1:
                st.warning("Predicted outcome: Possible SLA breach — send for staff review.")
            else:
                st.success("Predicted outcome: No SLA breach predicted by the model.")
            if prob is not None:
                st.metric("Model-estimated breach probability", f"{prob*100:.1f}%")
            st.caption("Probability is the model's class score and has not been established as calibrated confidence.")
            st.write("**Request factors supplied:**")
            st.json(values)
            st.info("This prediction is advisory. An authorized staff member must verify the request, documents, queue conditions, and SLA before deciding on follow-up.")
            st.caption("Prediction timestamp: " + pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"))
        except Exception as exc:
            st.error(f"Prediction could not be completed: {exc}")
    st.markdown("### Midterm test-set metrics")
    if METRICS_PATH.exists():
        m = json.loads(METRICS_PATH.read_text())
        a,b,c,d = st.columns(4)
        a.metric("Accuracy", f"{m['accuracy']*100:.2f}%")
        b.metric("Breach precision", f"{m['precision_breach']*100:.2f}%")
        c.metric("Breach recall", f"{m['recall_breach']*100:.2f}%")
        d.metric("Breach F1", f"{m['f1_breach']*100:.2f}%")
        st.write("Confusion matrix (actual rows; predicted columns; order: 0 = no breach, 1 = breach)")
        st.dataframe(pd.DataFrame(m["confusion_matrix"], index=["Actual no breach","Actual breach"], columns=["Predicted no breach","Predicted breach"]))
        st.caption(f"Original midterm Decision Tree split: {m['train_rows']} training rows and {m['test_rows']} test rows. Metrics are reproduced using the submitted notebook settings.")
    st.warning("The model was trained on completed historical requests. Predictions for pending requests are an advisory use and should be monitored for performance drift.")

elif menu == "Dashboard":
    st.title("Request Delay and Workload Dashboard")
    render_dashboard(get_data())

elif menu == "System Design":
    st.title("Systems Analysis and Design")
    st.subheader("Actors")
    st.markdown("""
    - **Administrator / Supervisor:** views service volume, SLA performance, and workload summaries; monitors model performance and system use.
    - **Student Services Staff:** views assigned requests, reviews delay-risk flags, checks supporting details, and records the final follow-up decision in the authorized operational process.
    - **Student Requester:** submits service requests through the existing student-services process; this monitoring application does not make service eligibility decisions.
    """)
    st.subheader("Five functional requirements")
    requirements = [
        ("FR1 — Service volume monitoring","The system shall display the number of requests by service type to help administrators identify high-demand services."),
        ("FR2 — SLA performance monitoring","The system shall display the number and percentage of SLA-breached completed requests by assigned office."),
        ("FR3 — Delay risk review","The system shall flag pending requests with predicted breach risk and present them to authorized staff for human review."),
        ("FR4 — Dataset validation and analysis","The system shall accept CSV data, validate required columns, preview records, and summarize missing and duplicate values."),
        ("FR5 — Explainable decision support","The system shall display the predicted classification, relevant request attributes, model metrics, prediction timestamp, and advisory limitations.")
    ]
    for title,desc in requirements:
        st.markdown(f"**{title}**  \n{desc}")
    st.subheader("Three non-functional requirements")
    st.markdown("""
    1. **Security and privacy:** only authorized users should access identifiable student request records; public deployments must use synthetic or properly authorized data.
    2. **Usability:** navigation, labels, validation messages, and charts should be understandable to student-services personnel.
    3. **Reliability:** the application should validate input schemas, handle missing or invalid values safely, and load the saved model without retraining on each page refresh.
    """)
    st.subheader("Updated use case diagram")
    diagram = ROOT / "assets" / "use_case_diagram.png"
    if diagram.exists():
        st.image(str(diagram), caption="Updated use case diagram for the Student Service Request Monitoring System", use_container_width=True)
    else:
        st.image(str(ROOT / "assets" / "use_case_diagram_original.png"), caption="Uploaded midterm use case diagram", use_container_width=True)
    st.subheader("Analytical and human decision workflow")
    st.code("""
CSV / request records
        ↓
Validate schema, missing values, and duplicates
        ↓
Explore service volume, workload, and historical SLA outcomes
        ↓
Apply saved preprocessing + Decision Tree pipeline
        ↓
Display predicted risk and relevant request attributes
        ↓
Authorized staff reviews documents, queue, priority, and SLA
        ↓
Staff records follow-up / final operational decision
        ↓
Continue monitoring outcomes and model limitations
    """, language="text")
    st.info("A model flag is not an automatic rejection, denial, or authorization. Final service decisions remain with authorized personnel.")
