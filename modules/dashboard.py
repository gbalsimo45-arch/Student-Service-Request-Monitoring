import streamlit as st
import pandas as pd
import plotly.express as px
from .data_processing import FEATURES
from .prediction import load_model

def render_dashboard(df):
    st.subheader("Decision-Support Dashboard")
    labeled = df.dropna(subset=["SLA_Breached"]) if "SLA_Breached" in df else pd.DataFrame()
    total = len(df)
    completed = int((df["Status"] == "Completed").sum()) if "Status" in df else len(labeled)
    pending = int((df["Status"] == "Pending").sum()) if "Status" in df else 0
    breaches = int((labeled["SLA_Breached"] == 1).sum()) if len(labeled) else 0
    rate = (breaches/len(labeled)*100) if len(labeled) else 0
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Requests received", f"{total:,}")
    c2.metric("Completed requests", f"{completed:,}")
    c3.metric("Pending requests", f"{pending:,}")
    c4.metric("Observed breach rate", f"{rate:.1f}%" if len(labeled) else "N/A")
    st.caption("Observed breach rate is calculated only from rows with a known SLA_Breached label.")
    if "Queue_Category" in df:
        queue = df["Queue_Category"].value_counts(dropna=False).rename_axis("Queue category").reset_index(name="Requests")
        st.plotly_chart(px.bar(queue, x="Queue category", y="Requests", title="Current workload by queue category"), use_container_width=True)
    # Review queue: only pending records can be predicted here; target is never used as an input.
    if "Status" in df:
        pending_df = df[df["Status"].astype(str).str.lower().eq("pending")].copy()
    else:
        pending_df = pd.DataFrame()
    st.markdown("#### Staff review queue")
    if len(pending_df):
        usable = pending_df.dropna(subset=FEATURES, how="all").copy()
        try:
            model = load_model()
            X = usable[FEATURES]
            usable["Predicted_SLA_Breach"] = model.predict(X).astype(int)
            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(X)
                cls = list(model.classes_)
                if 1 in cls:
                    usable["Predicted_Breach_Probability"] = probs[:, cls.index(1)]
            flagged = usable[usable["Predicted_SLA_Breach"] == 1].copy()
            cols = [c for c in ["Request_ID","Service_Type","Assigned_Office","Priority","Queue_Length","SLA_Days","Predicted_SLA_Breach","Predicted_Breach_Probability"] if c in flagged]
            if len(flagged):
                st.warning(f"{len(flagged)} pending request(s) were flagged for authorized staff review. A flag is advisory, not a final decision.")
                st.dataframe(flagged[cols].sort_values("Predicted_Breach_Probability", ascending=False) if "Predicted_Breach_Probability" in cols else flagged[cols], use_container_width=True)
                st.info("Suggested follow-up: verify document completeness, check queue conditions and SLA target, then record a staff decision in the official service system.")
            else:
                st.success("No pending requests were flagged by the current model. Continue normal monitoring.")
        except Exception as exc:
            st.error(f"Could not create the review queue: {exc}")
    else:
        st.info("No pending records are available for the review queue. Upload a dataset containing Status = Pending.")
    st.caption("Model predictions support prioritization only. Authorized staff must review the underlying request and make the final decision.")
