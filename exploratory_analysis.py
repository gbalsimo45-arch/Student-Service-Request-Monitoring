import streamlit as st
import plotly.express as px
import pandas as pd

def render_eda(df):
    st.subheader("Exploratory Data Analysis")
    st.caption("Charts describe the uploaded records. SLA breach rates use labeled records only.")
    a,b,c = st.columns(3)
    a.metric("Records", f"{len(df):,}")
    c.metric("Missing cells", f"{int(df.isna().sum().sum()):,}")
    if "SLA_Breached" in df:
        labeled = df["SLA_Breached"].dropna()
        b.metric("Labeled records", f"{len(labeled):,}")
        if len(labeled):
            counts = labeled.map({0:"No breach",1:"SLA breached"}).value_counts().rename_axis("Outcome").reset_index(name="Count")
            st.plotly_chart(px.pie(counts, names="Outcome", values="Count", title="SLA breach class distribution"), use_container_width=True)
    st.markdown("#### Descriptive statistics")
    st.dataframe(df.describe(include="all").transpose(), use_container_width=True)
    if "Service_Type" in df:
        svc = df["Service_Type"].value_counts().rename_axis("Service type").reset_index(name="Requests")
        st.plotly_chart(px.bar(svc, x="Service type", y="Requests", title="Request volume by service type"), use_container_width=True)
    if {"Queue_Category","SLA_Breached"}.issubset(df.columns):
        labeled = df.dropna(subset=["SLA_Breached"])
        q = labeled.groupby("Queue_Category", observed=True)["SLA_Breached"].agg(["count","mean"]).reset_index()
        q["Breach rate (%)"] = q["mean"]*100
        st.plotly_chart(px.bar(q, x="Queue_Category", y="Breach rate (%)", hover_data=["count"], title="Observed SLA breach rate by queue category"), use_container_width=True)
    if {"Assigned_Office","SLA_Breached"}.issubset(df.columns):
        labeled = df.dropna(subset=["SLA_Breached"])
        office = labeled.groupby("Assigned_Office")["SLA_Breached"].agg(["count","mean"]).reset_index()
        office["Breach rate (%)"] = office["mean"]*100
        st.plotly_chart(px.bar(office, x="Assigned_Office", y="Breach rate (%)", hover_data=["count"], title="Observed SLA breach rate by assigned office"), use_container_width=True)
