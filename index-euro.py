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
        'search': "🔍 Znajdź posła:", 'info_pca': "Mapa bliskości poglądów (PCA).",
        'friends': "Najwięksi sojusznicy (zgodność):", 'enemies': "Najwięksi oponenci (różnica):",
        'cl_stat_title': "Skład frakcyjny klastra, do którego należy", 'no_mep': "Wybierz posła, aby zobaczyć analizę klastra."
    },
    'EN': {
        'tab_comp': "🤝 Comparator", 'tab_fra': "🧭 Groups", 'tab_ai': "🤖 AI Clusters", 'tab_top': "🔥 Topics",
        'search': "🔍 Find MEP:", 'info_pca': "Ideological proximity map (PCA).",
        'friends': "Closest allies (agreement):", 'enemies': "Main opponents (divergence):",
        'cl_stat_title': "Party composition of the cluster belonging to", 'no_mep': "Select an MEP to see cluster analysis."
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

# Obliczenia bazowe
pca_coords = PCA(n_components=2).fit_transform(pivot_all.T)
df_base = pd.DataFrame(pca_coords, columns=['X', 'Y'], index=pivot_all.columns)
df_base['name'], df_base['group'] = df_base.index.map(names_dict), df_base.index.map(groups_dict)

tabs = st.tabs([L['tab_comp'], L['tab_fra'], L['tab_ai'], L['tab_top']])

# --- TAB 1: Porównywarka (Sojusznicy i Wrogowie) ---
with tabs[0]:
    sel_mep_comp = st.selectbox(L['search'], sorted(list(names_dict.values())), key="comp")
    if sel_mep_comp:
        mid = {v: k for k, v in names_dict.items()}.get(sel_mep_comp)
        corr = pivot_all.corrwith(pivot_all[mid]).sort_values(ascending=False)
        res = pd.DataFrame({'Zgodność': corr, 'Poseł': corr.index.map(names_dict), 'Frakcja': corr.index.map(groups_dict)}).dropna()
        
        col1, col2 = st.columns(2)
        col1.subheader(L['friends'])
        col1.dataframe(res.head(11).iloc[1:], use_container_width=True, hide_index=True)
        col2.subheader(L['enemies'])
        col2.dataframe(res.tail(10).sort_values('Zgodność'), use_container_width=True, hide_index=True)

# --- TAB 3: Klastry AI (Intuicyjne) ---
with tabs[2]:
    nk = st.slider("Liczba klastrów AI", 2, 20, 8)
    df_base['cluster'] = [f"Grupa AI {c+1}" for c in KMeans(n_clusters=nk, random_state=42, n_init=10).fit_predict(pivot_all.T)]
    
    sel_mep_ai = st.selectbox(L['search'], [""] + sorted(list(names_dict.values())), key="ai_search")
    
    if sel_mep_ai:
        mid_ai = {v: k for k, v in names_dict.items()}.get(sel_mep_ai)
        my_cluster = df_base.loc[mid_ai, 'cluster']
        
        # Wykres z podświetleniem
        fig = px.scatter(df_base, x='X', y='Y', color='cluster', hover_name='name', 
                         title=f"Podział na {nk} klastrów AI", height=500, template="plotly_white")
        fig.add_trace(go.Scatter(x=[df_base.loc[mid_ai, 'X']], y=[df_base.loc[mid_ai, 'Y']], 
                                 mode='markers+text', marker=dict(color='black', size=15, symbol='star'), 
                                 text=[f"★ {sel_mep_ai}"], textposition="top center", name="Wybrany poseł"))
        st.plotly_chart(fig, use_container_width=True)
        
        # STATYSTYKI KLASTRA - To o co prosiłeś
        st.subheader(f"{L['cl_stat_title']} {sel_mep_ai} ({my_cluster}):")
        cluster_members = df_base[df_base['cluster'] == my_cluster]
        stats = cluster_members['group'].value_counts().reset_index()
        stats.columns = ['Frakcja', 'Liczba posłów']
        
        c1, c2 = st.columns([1, 2])
        c1.write("Kogo AI połączyło w jedną grupę?")
        c1.dataframe(stats, hide_index=True)
        c2.write(f"Przykładowi posłowie z tej samej grupy AI:")
        c2.caption(", ".join(cluster_members['name'].sample(min(20, len(cluster_members))).tolist()))
    else:
        st.info(L['no_mep'])
        fig_empty = px.scatter(df_base, x='X', y='Y', color='cluster', hover_name='name', template="plotly_white", height=500)
        st.plotly_chart(fig_empty, use_container_width=True)

