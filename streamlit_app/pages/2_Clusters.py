import streamlit as st
import plotly.express as px
from utils import load_data

st.set_page_config(page_title="Clusters", layout="wide")

DATA_PATH = "data/processed/characters_final.csv"
df = load_data(DATA_PATH)

st.title("Clusters")

if "Cluster" not in df.columns:
    st.warning("No Cluster column found in your CSV.")
    st.stop()

st.write("### Cluster counts")
counts = df["Cluster"].value_counts().reset_index()
counts.columns = ["Cluster","Count"]
fig = px.bar(counts, x="Cluster", y="Count")
st.plotly_chart(fig, use_container_width=True)

if all(c in df.columns for c in ["pc1","pc2","pc3"]):
    st.write("### PCA Space (3D)")
    fig3d = px.scatter_3d(df, x="pc1", y="pc2", z="pc3", color="Cluster", hover_name="Character")
    st.plotly_chart(fig3d, use_container_width=True)
