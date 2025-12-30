import streamlit as st
import pandas as pd
import json
import os
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
import base64
import re
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import pdist
import random
import numpy as np 

# --- 1. CONFIG & SETTINGS ---
st.set_page_config(page_title="HP Personality Analytics", layout="wide", page_icon="🪄")

V1, V2, V3 = "88.01%", "7.15%", "2.41%"
AXIS_LABEL_1, AXIS_LABEL_2, AXIS_LABEL_3 = f"PC1 ({V1})", f"PC2 ({V2})", f"PC3 ({V3})"
COLOR_PALETTE = px.colors.qualitative.Prism

def display_pdf(file_path):
    with open(file_path, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')
    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="800" type="application/pdf"></iframe>'
    st.markdown(pdf_display, unsafe_allow_html=True)

# --- 2. THE ULTIMATE CLEANER ---
def clean_text(text):
    if isinstance(text, list):
        return [clean_text(i) for i in text if i]
    if pd.isna(text) or text is None:
        return ""
    t = str(text)
    while any(char in t for char in "[]'\""):
        t = re.sub(r"[\[\]'\"“”‘’]", "", t)
    return t.strip()

# --- 3. AUDIO HANDLER ---
def add_bg_audio(audio_file):
    if os.path.exists(audio_file):
        with open(audio_file, "rb") as f:
            data = f.read()
            b64 = base64.b64encode(data).decode()
            md = f"""<audio id="bg-audio" loop autoplay><source src="data:audio/mp3;base64,{b64}" type="audio/mp3"></audio>"""
            st.markdown(md, unsafe_allow_html=True)

# --- 4. GAME: WHO AM I? (THE SORTING HAT) ---
def run_who_am_i_game(df, target_cluster_col):
    st.header("🕵️‍♂️ The Sorting Hat: Who am I?")
    st.write("A random character has been chosen. Identify them by their statistical 'Shadow Profile'.")

    if 'who_target' not in st.session_state:
        sample = df.sample(n=4)
        st.session_state.who_target = sample.iloc[0]
        options = sample['Character'].tolist()
        random.shuffle(options)
        st.session_state.who_options = options
        st.session_state.who_answered = False

    target = st.session_state.who_target
    traits = ['Openness_scaled', 'Conscientiousness_scaled', 'Extraversion_scaled', 'Agreeableness_scaled', 'Neuroticism_scaled']
    labels = [t.replace('_scaled', '') for t in traits]
    radar_values = [(target[t] - df[t].min()) / (df[t].max() - df[t].min()) for t in traits]

    col1, col2 = st.columns([2, 1])
    with col1:
            # Dynamically get cluster color and fill
            target_cluster = int(target[target_cluster_col])
            cluster_color = COLOR_PALETTE[target_cluster % len(COLOR_PALETTE)]
            fill_color = cluster_color.replace('rgb', 'rgba').replace(')', ', 0.3)')
            
            fig_radar = go.Figure(data=go.Scatterpolar(
                r=radar_values, 
                theta=labels, 
                fill='toself', 
                line_color=cluster_color,
                fillcolor=fill_color
            ))
            
            fig_radar.update_layout(
                title=f"Statistical Shadow Profile",
                polar=dict(
                    radialaxis=dict(visible=True, gridcolor="gray", showticklabels=False),
                    angularaxis=dict(gridcolor="gray")
                ),
                paper_bgcolor="rgba(0,0,0,0)"
            )
            st.plotly_chart(fig_radar, use_container_width=True)

    with col2:
        st.write("### Cast Your Guess")
        user_choice = st.radio("Who matches this data?", st.session_state.who_options)
        
        if st.button("Revelio!"):
            st.session_state.who_answered = True
            if user_choice == target['Character']:
                st.success(f"✨ **Correct!** This is {target['Character']}.")
                st.balloons()
            else:
                st.error(f"❌ **Wrong.** This profile belongs to {target['Character']}.")

        if st.session_state.who_answered:
            if st.button("Play Next Round"):
                for key in ['who_target', 'who_options', 'who_answered']:
                    if key in st.session_state: del st.session_state[key]
                st.rerun()

# --- 5. GAME: CHARACTER MIRROR ---
def run_character_mirror(df, descriptions, accent_color):
    st.header("🪞 The Character Mirror")
    
    with st.expander("🔬 The Math: Why Euclidean Distance?"):
        st.markdown("""
        Use **Euclidean Distance** to find the match because it respects **Intensity**.
        """)

    traits = ['Openness_scaled', 'Conscientiousness_scaled', 'Extraversion_scaled', 'Agreeableness_scaled', 'Neuroticism_scaled']
    labels = [t.replace('_scaled', '') for t in traits]
    user_scores = []

    st.subheader("Rate Yourself (-3 to +3)")
    cols = st.columns(5)
    for i, trait in enumerate(traits):
        with cols[i]:
            val = st.slider(labels[i], -3.0, 3.0, 0.0, step=0.1, key=f"mirror_{trait}")
            user_scores.append(val)

    if st.button("Find My Statistical Twins"):
        user_vec = np.array(user_scores)
        char_vecs = df[traits].values
        distances = np.linalg.norm(char_vecs - user_vec, axis=1) 
        
        df_mirror = df.copy()
        df_mirror['dist'] = distances
        top_3 = df_mirror.sort_values('dist').head(3)

        st.divider()
        st.subheader("🔮 Your Top Match Comparison")
        
        match1 = top_3.iloc[0]
        user_norm = [(v - (-3)) / 6 for v in user_scores]
        twin_norm = [(match1[t] - (-3)) / 6 for t in traits]

        fig_comp = go.Figure()
        fig_comp.add_trace(go.Scatterpolar(r=user_norm, theta=labels, fill='toself', name='You'))
        fig_comp.add_trace(go.Scatterpolar(r=twin_norm, theta=labels, fill='toself', name=match1['Character'], line_color=accent_color))
        # Removed hardcoded font/bgcolor for Chrome compatibility
        fig_comp.update_layout(
            polar=dict(radialaxis=dict(visible=True, gridcolor="gray")), 
            paper_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(fig_comp, use_container_width=True)

        st.write("### Your Top 3 Archetypes")
        m_cols = st.columns(3)
        for i in range(3):
            match = top_3.iloc[i]
            with m_cols[i]:
                st.markdown(f"#### #{i+1}: {match['Character']}")
                match_pct = max(0, 100 - (match['dist'] * 15))
                st.metric("Similarity", f"{match_pct:.1f}%")
                for ext in ['.png', '.jpg', '.jpeg']:
                    path = os.path.join("images", f"{match['Character']}{ext}")
                    if os.path.exists(path): st.image(path, use_container_width=True); break
                st.write(f"*{descriptions.get(match['Character'], {}).get('tagline', '')}*")

# --- 6. ORIGINAL MATCHMAKER & QUIZ ---
def run_matching_game(df):
    st.header("🧩 SLOAN Matchmaker")
    with st.expander("📚 The SLOAN Decoder Key", expanded=False):
        legend_data = {
            "Dimension": ["Extraversion", "Neuroticism", "Conscientiousness", "Agreeableness", "Openness"],
            "High Pole": ["**S**ocial", "**L**imbic", "**O**rganized", "**A**greeable", "**I**nquisitive"],
            "Low Pole": ["**R**eserved", "**C**alm", "**U**nstructured", "**E**gocentric", "**N**on-curious"],
            "Soft Codes": ["s / r", "l / c", "o / u", "a / e", "i / n"]
        }
        st.table(legend_data)
    
    if 'match_chars' not in st.session_state:
        sample_df = df.sample(n=5)
        st.session_state.match_chars = sample_df['Character'].tolist()
        st.session_state.match_codes = sample_df['Hybrid_SLOAN'].tolist()
        shuffled_names = st.session_state.match_chars.copy()
        random.shuffle(shuffled_names)
        st.session_state.display_names = shuffled_names

    user_answers = {}
    for i in range(5):
        code = st.session_state.match_codes[i]
        col1, col2 = st.columns([1, 2])
        with col1: st.markdown(f"#### `{code}`") 
        with col2: user_answers[code] = st.selectbox(f"Select for {code}", ["-- Select Character --"] + st.session_state.display_names, label_visibility="collapsed", key=f"match_select_{i}")

    if st.button("Check My Matches"):
        correct_count = sum(1 for i in range(5) if user_answers[st.session_state.match_codes[i]] == st.session_state.match_chars[i])
        st.subheader(f"Final Score: {correct_count}/5")
        if correct_count == 5: st.balloons()
        for i in range(5): st.write(f"🧬 `{st.session_state.match_codes[i]}` is **{st.session_state.match_chars[i]}**")

    if st.button("New Round / Shuffle"):
        for key in ['match_chars', 'match_codes', 'display_names']:
            if key in st.session_state: del st.session_state[key]
        st.rerun()

def run_personality_quiz(df):
    st.header("🧙‍♂️ Wizarding Personality Mastery Quiz")
    quiz_data = [
        {"question": "Which character possesses the highest statistical 'Openness'?", "options": ["Albus Dumbledore", "Luna Lovegood", "Hermione Granger", "Voldemort"], "answer": "Voldemort", "explanation": "Voldemort's dialogue is statistically focused on complex magical theory."},
        {"question": "Who has all lowercase SLOAN codes?", "options": ["Sirius Black", "Arthur Weasley", "Neville Longbottom", "Minerva McGonagall"], "answer": "Sirius Black", "explanation": "Sirius and Fudge are the dataset 'Middle-men'."}
    ]
    if 'quiz_idx' not in st.session_state: st.session_state.quiz_idx, st.session_state.quiz_score, st.session_state.quiz_done, st.session_state.show_explanation = 0, 0, False, False
    
    if not st.session_state.quiz_done:
        q = quiz_data[st.session_state.quiz_idx % len(quiz_data)]
        st.write(f"### {q['question']}")
        user_choice = st.radio("Choose:", q['options'], key=f"q_{st.session_state.quiz_idx}")
        if st.button("Submit Answer"): st.session_state.show_explanation = True; st.rerun()
        if st.session_state.show_explanation:
            if user_choice == q['answer']: st.success("Correct!")
            else: st.error(f"Incorrect. Answer: {q['answer']}")
            if st.button("Next Question"): st.session_state.quiz_idx += 1; st.session_state.show_explanation = False; st.rerun()
    else:
        st.balloons(); st.metric("Final Score", f"{st.session_state.quiz_score}"); st.button("Restart Quiz", on_click=lambda: st.session_state.clear())

# --- 7. DATA LOADERS ---
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('df5.csv')
        df.columns = df.columns.str.strip()
        if os.path.exists('character_descriptions2.json'):
            with open('character_descriptions2.json', 'r', encoding='utf-8') as f: descriptions = json.load(f)
        else: descriptions = {}
        cleaned_desc = {char: {"tagline": clean_text(d.get("tagline", "")), "points": clean_text(d.get("points", []))} for char, d in descriptions.items()}
        return df, cleaned_desc
    except Exception as e:
        st.error(f"⚠️ Load Error: {e}"); return None, None

df, descriptions = load_data()

if df is not None:
    # --- 8. SIDEBAR CONTROLS ---
    st.sidebar.title("🧪 Statistical Controls")
    k_val = st.sidebar.slider("Number of Clusters (k):", 2, 10, 3)
    target_cluster_col = 'Cluster' if k_val == 3 else f'Cluster_k{k_val}'
    selected_char = st.sidebar.selectbox("Choose a Persona:", sorted(df['Character'].unique()))
    char_row = df[df['Character'] == selected_char].iloc[0]
    ACCENT_COLOR = COLOR_PALETTE[int(char_row[target_cluster_col]) % len(COLOR_PALETTE)]
    
    if st.sidebar.checkbox("🔊 Play Background Music", value=False):
        add_bg_audio("hp_theme.mp3")

    st.markdown(f"<style>h1, h2, .stSubheader {{ color: {ACCENT_COLOR} !important; }}</style>", unsafe_allow_html=True)

    # --- 9. TABS LAYOUT ---
    tabs = st.tabs(["👤 Profile", "🌌 Galaxy", "🪄 Quiz", "🧩 Matchmaker", "🕵️‍♂️ Who am I?", "🪞 Mirror","Technical Deep Dive" ])

    with tabs[0]:
        st.title("Harry Potter: Personality Archetypes")
        col1, col2 = st.columns([1, 1.5])
        with col1:
            st.header(selected_char)
            for ext in ['.png', '.jpg', '.jpeg']:
                path = os.path.join("images", f"{selected_char}{ext}")
                if os.path.exists(path): st.image(path, use_container_width=True); break
            st.markdown(f"### SLOAN: <span style='color:{ACCENT_COLOR}'>`{char_row['Hybrid_SLOAN']}`</span>", unsafe_allow_html=True)
            st.markdown(f"**{descriptions.get(selected_char, {}).get('tagline', '')}**")
            for point in descriptions.get(selected_char, {}).get('points', []):
                if point: st.markdown(f"• {point}")
        with col2:
            traits = ['Openness_scaled', 'Conscientiousness_scaled', 'Extraversion_scaled', 'Agreeableness_scaled', 'Neuroticism_scaled']
            labels = [t.replace('_scaled', '') for t in traits]
            radar_values = [(char_row[t] - df[t].min()) / (df[t].max() - df[t].min()) for t in traits]
            fill_color = ACCENT_COLOR.replace('rgb', 'rgba').replace(')', ', 0.3)')
            
            fig_radar = go.Figure(data=go.Scatterpolar(r=radar_values, theta=labels, fill='toself', line_color=ACCENT_COLOR, fillcolor=fill_color))
            fig_radar.update_layout(
                polar=dict(radialaxis=dict(visible=True, gridcolor="gray", showticklabels=False), angularaxis=dict(gridcolor="gray")),
                paper_bgcolor="rgba(0,0,0,0)"
            ) 
            st.plotly_chart(fig_radar, use_container_width=True)

    with tabs[1]:
        st.header(f"The Personality Galaxy (k={k_val})")
        df_plot = df.copy()
        df_plot[target_cluster_col] = df_plot[target_cluster_col].astype(str)
        fig_pca = px.scatter_3d(df_plot, x='pc1', y='pc2', z='pc3', color=target_cluster_col, text='Character', hover_name='Character', height=800, color_discrete_sequence=COLOR_PALETTE)
        # Removed hardcoded dark background for 3D plot
        fig_pca.update_layout(paper_bgcolor="rgba(0,0,0,0)", scene=dict(xaxis_title=AXIS_LABEL_1, yaxis_title=AXIS_LABEL_2, zaxis_title=AXIS_LABEL_3))
        st.plotly_chart(fig_pca, use_container_width=True)
        st.divider()
        st.header(f"Hierarchical Similarity Tree")
        features = df[['pc1', 'pc2', 'pc3']]
        Z = linkage(features, method='complete', metric='cosine')
        fig_dendro = ff.create_dendrogram(features, orientation='bottom', labels=df['Character'].values)
        # Removed hardcoded font colors for theme compatibility
        fig_dendro.update_layout(width=1200, height=450, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_dendro, use_container_width=True)

    with tabs[2]: run_personality_quiz(df)
    with tabs[3]: run_matching_game(df)
    with tabs[4]: run_who_am_i_game(df, target_cluster_col)
    with tabs[5]: run_character_mirror(df, descriptions, ACCENT_COLOR)
    with tabs[6]:
        st.header("Technical Deep Dive & Methodology")
        st.write("Explore the methodology behind the NLP extraction, PCA decorrelation, and clustering.")

        github_pdf_url = "https://github.com/ahampto/HP-NLP/blob/main/hp_personality_presentation.pdf"
        col1, col2 = st.columns([1, 1])
        with col1:
            st.link_button(label="🚀 Open Full Presentation (GitHub Viewer)", url=github_pdf_url, use_container_width=True, type="primary")
        with col2:
            try:
                with open("hp_personality_presentation.pdf", "rb") as f:
                    st.download_button(label="📥 Download PDF to Device", data=f, file_name="HP_Personality_Deep_Dive.pdf", mime="application/pdf", use_container_width=True)
            except FileNotFoundError:
                st.warning("Download file only available after deployment.")
        st.divider()
        st.info("💡 Tip: The GitHub viewer allows you to scroll through all 90 pages with full search and zoom functionality.")