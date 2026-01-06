# -*- coding: utf-8 -*-
"""
Created on Fri Dec 19 03:17:33 2025

@author: Marcin Deszczka
"""

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import plotly.express as px
import plotly.graph_objects as go

# --- KONFIGURACJA DANYCH ---
URL_VOTES = "https://github.com/HowTheyVote/data/releases/download/2026-01-03/member_votes.csv.gz"
URL_MEMBERS = "https://github.com/HowTheyVote/data/releases/download/2026-01-03/members.csv.gz"
URL_ROLLCALLS = "https://github.com/HowTheyVote/data/releases/download/2026-01-03/votes.csv.gz"

st.set_page_config(page_title="EuroMatrix AI", layout="wide", page_icon="🇪🇺")

# --- SŁOWNIK JĘZYKOWY ---
LANGS = {
    'PL': {
        'tab_comp': "🤝 Porównywarka", 'tab_fra': "🧭 Frakcje", 'tab_ai': "🤖 Klastry AI", 'tab_top': "🔥 Tematy",
        'search': "🔍 Znajdź posła:", 'info_pca': "Mapa bliskość poglądów (PCA).",
        'friends': "Sojusznicy (zgodność):", 'enemies': "Oponenci (różnica):",
        'cl_summary': "Zestawienie składu klastrów (kogo AI połączyło ze sobą):",
        'mep_belongs': "Wybrany poseł należy do:", 'no_mep': "Wybierz posła, aby go podświetlić."
    },
    'EN': {
        'tab_comp': "🤝 Comparator", 'tab_fra': "🧭 Groups", 'tab_ai': "🤖 AI Clusters", 'tab_top': "🔥 Topics",
        'search': "🔍 Find MEP:", 'info_pca': "Ideological proximity map (PCA).",
        'friends': "Allies (agreement):", 'enemies': "Opponents (divergence):",
        'cl_summary': "Cluster composition overview (who was grouped together):",
        'mep_belongs': "Selected MEP belongs to:", 'no_mep': "Select an MEP to highlight them."
    }
}

with st.sidebar:
    lang_code = st.radio("Language / Język", ["PL", "EN"], horizontal=True)
    L = LANGS[lang_code]

@st.cache_data(show_spinner=False)
def load_data():
    rollcalls = pd.read_csv(URL_ROLLCALLS, compression='gzip')
    rollcalls['timestamp'] = pd.to_datetime(rollcalls['timestamp'], errors='coerce')
    current_term_ids = rollcalls[rollcalls['timestamp'] > '2024-07-16']['id']
    votes = pd.read_csv(URL_VOTES, compression='gzip')
    votes = votes[votes['vote_id'].isin(current_term_ids)]
    members = pd.read_csv(URL_MEMBERS, compression='gzip')
    members['full_name'] = members['last_name'] + ' ' + members['first_name']
    data = pd.merge(votes, members[['id', 'full_name', 'country_code']], left_on='member_id', right_on='id')
    data['numeric'] = data['position'].map({'FOR': 1, 'AGAINST': -1, 'ABSTENTION': 0, 'ABSTAIN': 0}).fillna(0)
    mep_groups = data.groupby('member_id')['group_code'].agg(lambda x: x.mode()[0] if not x.mode().empty else 'NI').to_dict()
    mep_names = members.set_index('id')['full_name'].to_dict()
    return data, mep_groups, mep_names

data_raw, groups_dict, names_dict = load_data()
pivot_all = data_raw.pivot_table(index='vote_id', columns='member_id', values='numeric').fillna(0)

pca_coords = PCA(n_components=2).fit_transform(pivot_all.T)
df_base = pd.DataFrame(pca_coords, columns=['X', 'Y'], index=pivot_all.columns)
df_base['name'], df_base['group'] = df_base.index.map(names_dict), df_base.index.map(groups_dict)

tabs = st.tabs([L['tab_comp'], L['tab_fra'], L['tab_ai'], L['tab_top']])

# --- TAB 1: Porównywarka ---
with tabs[0]:
    sel_mep_comp = st.selectbox(L['search'], sorted(list(names_dict.values())), key="c1")
    if sel_mep_comp:
        mid = {v: k for k, v in names_dict.items()}.get(sel_mep_comp)
        corr = pivot_all.corrwith(pivot_all[mid]).sort_values(ascending=False)
        res = pd.DataFrame({'Zgodność': corr, 'Poseł': corr.index.map(names_dict), 'Frakcja': corr.index.map(groups_dict)}).dropna()
        c1, c2 = st.columns(2)
        c1.subheader(L['friends'])
        c1.dataframe(res.head(11).iloc[1:], use_container_width=True, hide_index=True)
        c2.subheader(L['enemies'])
        c2.dataframe(res.tail(10).sort_values('Zgodność'), use_container_width=True, hide_index=True)

# --- TAB 3: Klastry AI z raportem składu ---
with tabs[2]:
    nk = st.slider("Liczba klastrów AI", 2, 20, 8)
    df_base['cluster'] = [f"Grupa {c+1}" for c in KMeans(n_clusters=nk, random_state=42, n_init=10).fit_predict(pivot_all.T)]
    
    sel_mep_ai = st.selectbox(L['search'], [""] + sorted(list(names_dict.values())), key="a1")
    
    fig = px.scatter(df_base, x='X', y='Y', color='cluster', hover_name='name', height=500, template="plotly_white")
    if sel_mep_ai:
        mid_ai = {v: k for k, v in names_dict.items()}.get(sel_mep_ai)
        fig.add_trace(go.Scatter(x=[df_base.loc[mid_ai, 'X']], y=[df_base.loc[mid_ai, 'Y']], 
                                 mode='markers+text', marker=dict(color='black', size=15, symbol='star'), 
                                 text=[f"★ {sel_mep_ai}"], textposition="top center", name="Cel"))
    st.plotly_chart(fig, use_container_width=True)

    # RAPORT SKŁADU KLASTRÓW
    st.subheader(L['cl_summary'])
    
    # Przygotowanie danych do tabeli zbiorczej
    all_clusters = sorted(df_base['cluster'].unique(), key=lambda x: int(x.split()[1]))
    
    for cl_name in all_clusters:
        with st.expander(f"📊 {cl_name}"):
            members_in_cl = df_base[df_base['cluster'] == cl_name]
            stats = members_in_cl['group'].value_counts().reset_index()
            stats.columns = ['Frakcja', 'Liczba posłów']
            
            c1, c2 = st.columns([1, 2])
            c1.dataframe(stats, hide_index=True)
            # Pokazujemy top 15 nazwisk w klastrze
            c2.write("**Przykładowi posłowie w tej grupie:**")
            c2.caption(", ".join(members_in_cl['name'].head(15).tolist()) + "...")

