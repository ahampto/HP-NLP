import streamlit as st
import plotly.express as px

from utils import load_data, load_descriptions, merge_descriptions, default_tagline, validate_names

st.set_page_config(page_title="HP Personality Explorer", layout="wide")

DATA_PATH = "data/processed/characters_final.csv"
DESC_PATH = "data/processed/descriptions.json"

st.title("Harry Potter Personality Explorer (Movie Dialogue)")

st.caption("Profiles reflect linguistic expression in dialogue — not clinical diagnoses.")

# Load files
try:
    df = load_data(DATA_PATH)
except FileNotFoundError:
    st.error(f"Missing file: {DATA_PATH}. Put your CSV there to run the app.")
    st.stop()

desc_map = {}
try:
    desc_map = load_descriptions(DESC_PATH)
except FileNotFoundError:
    st.warning(f"Missing descriptions file: {DESC_PATH}. Descriptions will show as TBD.")
    desc_map = {}

df = merge_descriptions(df, desc_map)

# Sidebar navigation
st.sidebar.header("Controls")
char = st.sidebar.selectbox("Choose a character", sorted(df["Character"].unique().tolist()))

row = df[df["Character"] == char].iloc[0]
sloan = row.get("Hybrid_SLOAN", "")
cluster = row.get("Cluster", "")
tagline = row.get("tagline_override") or default_tagline(str(sloan), str(cluster))
points = row.get("points_override", None)

# Header info
c1, c2, c3 = st.columns([2,1,1])
with c1:
    st.subheader(char)
    st.write(tagline)
with c2:
    st.metric("Cluster", cluster if cluster else "—")
with c3:
    st.metric("Hybrid SLOAN", sloan if sloan else "—")

# Main content
left, right = st.columns([1.2, 1])

with left:
    st.markdown("### Narrative Notes")
    if isinstance(points, list) and len(points) > 0:
        for p in points:
            st.markdown(f"- {p}")
    else:
        st.info("No curated description yet for this character. Add it to descriptions.json.")

with right:
    st.markdown("### Personality Profile")
    # Big Five bars if present
    big5_cols = [
        "Openness_scaled","Conscientiousness_scaled","Extraversion_scaled",
        "Agreeableness_scaled","Neuroticism_scaled"
    ]
    available = [c for c in big5_cols if c in df.columns]
    if len(available) == 5:
        plot_df = df[df["Character"] == char][available].melt(var_name="Trait", value_name="Score")
        fig = px.bar(plot_df, x="Trait", y="Score")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.caption("Big Five scaled columns not found in CSV (optional).")

    # PCA scatter if present
    if all(c in df.columns for c in ["pc1","pc2","pc3"]):
        st.markdown("### PCA Space (3D)")
        fig3d = px.scatter_3d(
            df, x="pc1", y="pc2", z="pc3",
            color="Cluster" if "Cluster" in df.columns else None,
            hover_name="Character"
        )
        # highlight selected character
        sel = df[df["Character"] == char]
        fig3d.add_scatter3d(x=sel["pc1"], y=sel["pc2"], z=sel["pc3"], mode="markers", marker=dict(size=6))
        st.plotly_chart(fig3d, use_container_width=True)
    else:
        st.caption("PCA columns pc1/pc2/pc3 not found in CSV (optional).")

with st.expander("Name validation (CSV vs descriptions.json)"):
    if desc_map:
        report = validate_names(df, desc_map)
        st.write("**In descriptions.json but not in CSV:**", report["in_json_not_in_df"])
        st.write("**In CSV but missing descriptions:**", report["in_df_not_in_json"][:50])
        if len(report["in_df_not_in_json"]) > 50:
            st.write(f"... and {len(report['in_df_not_in_json'])-50} more")
    else:
        st.write("No descriptions.json loaded.")
