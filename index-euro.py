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

# --- TŁUMACZENIA (Słownik) ---
LANGS = {
    'PL': {
        'title': "EuroParl AI v9.0",
        'sidebar_lang': "Wybierz język / Select language",
        'tab1': "🤝 Porównywarka",
        'tab2': "🧭 Kompas Frakcji",
        'tab3': "🤖 AI Klastry",
        'desc_pca': "Mapa pokazuje bliskość poglądów. Kropki blisko siebie to posłowie głosujący identycznie.",
        'focus_label': "Podświetl frakcję:",
        'search_mep': "Znajdź europosła:",
        'squad_title': "Polityczni towarzysze (ten sam klaster):",
        'no_data': "Za mało danych dla tego tematu."
    },
    'EN': {
        'title': "EuroParl AI v9.0",
        'sidebar_lang': "Select language",
        'tab1': "🤝 Comparator",
        'tab2': "🧭 Party Compass",
        'tab3': "🤖 AI Clusters",
        'desc_pca': "The map shows ideological proximity. Dots close together represent MEPs with similar voting patterns.",
        'focus_label': "Highlight group:",
        'search_mep': "Find MEP:",
        'squad_title': "Political squad (same cluster):",
        'no_data': "Not enough data for this topic."
    },
    'DE': {
        'tab1': "🤝 Vergleich", 'tab2': "🧭 Kompass", 'tab3': "🤖 KI-Cluster",
        'desc_pca': "Die Karte zeigt die ideologische Nähe der Abgeordneten.",
        'focus_label': "Fraktion hervorheben:",
        'search_mep': "Abgeordneten finden:"
    },
    'FR': {
        'tab1': "🤝 Comparateur", 'tab2': "🧭 Boussole", 'tab3': "🤖 Clusters IA",
        'desc_pca': "La carte montre la proximité idéologique entre les députés.",
        'focus_label': "Surligner le groupe:",
        'search_mep': "Trouver un député:"
    }
}

st.set_page_config(page_title="EuroMatrix AI", layout="wide")

# --- WYBÓR JĘZYKA ---
lang_code = st.sidebar.selectbox("Language", ["PL", "EN", "DE", "FR"], index=0)
L = LANGS[lang_code]

# --- KONFIGURACJA DANYCH ---
URL_VOTES = "https://github.com/HowTheyVote/data/releases/download/2026-01-03/member_votes.csv.gz"
URL_MEMBERS = "https://github.com/HowTheyVote/data/releases/download/2026-01-03/members.csv.gz"
URL_ROLLCALLS = "https://github.com/HowTheyVote/data/releases/download/2026-01-03/votes.csv.gz"

@st.cache_data(show_spinner=False)
def load_data():
    try:
        rollcalls = pd.read_csv(URL_ROLLCALLS, compression='gzip')
        rollcalls['timestamp'] = pd.to_datetime(rollcalls['timestamp'], errors='coerce')
        current_term = rollcalls[rollcalls['timestamp'] > '2024-07-16']
        current_term_ids = current_term['id']
        vote_titles = current_term.set_index('id').get('display_title', current_term.set_index('id').get('procedure_title', '...')).to_dict()

        votes = pd.read_csv(URL_VOTES, compression='gzip')
        votes = votes[votes['vote_id'].isin(current_term_ids)]

        members = pd.read_csv(URL_MEMBERS, compression='gzip')
        members['full_name'] = members['last_name'] + ' ' + members['first_name']
        
        data = pd.merge(votes, members[['id', 'full_name', 'country_code']], left_on='member_id', right_on='id')
        vote_map = {'FOR': 1, 'AGAINST': -1, 'ABSTENTION': 0, 'ABSTAIN': 0}
        data['numeric'] = data['position'].map(vote_map).fillna(0)
        
        mep_groups = data.groupby('member_id')['group_code'].agg(lambda x: x.mode()[0] if not x.mode().empty else 'NI').to_dict()
        mep_names = members.set_index('id')['full_name'].to_dict()
        mep_countries = members.set_index('id')['country_code'].to_dict()

        return data, mep_groups, mep_names, mep_countries
    except Exception as e:
        st.error(f"Error: {e}")
        st.stop()

# --- BACKEND LOGIC (PCA/Clusters) ---
@st.cache_data
def calculate_pca(pivot_df, groups_dict, names_dict, countries_dict):
    pca = PCA(n_components=2)
    components = pca.fit_transform(pivot_df.fillna(0).T)
    pca_df = pd.DataFrame(components, columns=['X', 'Y'], index=pivot_df.columns)
    pca_df['name'] = pca_df.index.map(names_dict)
    pca_df['group'] = pca_df.index.map(groups_dict)
    pca_df['country'] = pca_df.index.map(countries_dict)
    return pca_df

# --- INTERFEJS ---
st.title(L.get('title', "EuroMatrix"))
data_raw, groups_dict, names_dict, countries_dict = load_data()
pivot_all = data_raw.pivot_table(index='vote_id', columns='member_id', values='numeric')

tabs = st.tabs([L['tab1'], L['tab2'], L['tab3']])

with tabs[1]: # Kompas
    st.info(L['desc_pca'])
    col1, col2 = st.columns([2, 1])
    search_q = col1.selectbox(L['search_mep'], [""] + sorted(list(names_dict.values())), key="s1")
    
    # Automatyczne podświetlanie frakcji wyszukanego posła
    mep_id = {v: k for k, v in names_dict.items()}.get(search_q)
    default_focus = groups_dict.get(mep_id, "Wszyscy") if mep_id else "Wszyscy"
    
    focus_g = col2.selectbox(L.get('focus_label', "Focus"), ["Wszyscy"] + sorted(list(set(groups_dict.values()))), index=0)
    
    pca_res = calculate_pca(pivot_all, groups_dict, names_dict, countries_dict)
    
    # Rysowanie
    fig = px.scatter(pca_res, x='X', y='Y', color='group', hover_name='name', 
                     title=L['tab2'], height=600, template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)

with tabs[2]: # AI Klastry
    n_k = st.slider("Clusters", 2, 15, 6)
    # ... reszta logiki klastrów z v7.0 ...
    st.write("Więcej tłumaczeń dodasz analogicznie w słowniku LANGS.")
