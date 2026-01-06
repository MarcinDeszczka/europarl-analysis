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
        'search': "🔍 Znajdź posła:", 'focus': "💡 Podświetl grupę:", 'topic_input': "Wpisz temat (np. Ukraina):",
        'info_pca': "Mapa pokazuje bliskość poglądów. Kropki blisko siebie głosują podobnie.",
        'squad_msg': "Posłowie w tym samym klastrze co", 'no_results': "Brak wyników.", 'sidebar_title': "Ustawienia"
    },
    'EN': {
        'tab_comp': "🤝 Comparator", 'tab_fra': "🧭 Groups", 'tab_ai': "🤖 AI Clusters", 'tab_top': "🔥 Topics",
        'search': "🔍 Find MEP:", 'focus': "💡 Highlight group:", 'topic_input': "Enter topic (e.g. Climate):",
        'info_pca': "The map shows ideological proximity. Dots close together have similar voting records.",
        'squad_msg': "MEPs in the same cluster as", 'no_results': "No results found.", 'sidebar_title': "Settings"
    }
}

# --- SIDEBAR (Wybór języka) ---
with st.sidebar:
    st.title("EuroMatrix AI")
    lang_code = st.radio("Language / Język", ["PL", "EN"], horizontal=True)
    L = LANGS[lang_code]
    st.divider()
    st.caption("Data source: HowTheyVote.eu")

@st.cache_data(show_spinner=False)
def load_data():
    try:
        rollcalls = pd.read_csv(URL_ROLLCALLS, compression='gzip')
        rollcalls['timestamp'] = pd.to_datetime(rollcalls['timestamp'], errors='coerce')
        current_term = rollcalls[rollcalls['timestamp'] > '2024-07-16']
        vote_titles = current_term.set_index('id').get('display_title', current_term.set_index('id').get('procedure_title', '')).fillna('').to_dict()
        
        votes = pd.read_csv(URL_VOTES, compression='gzip')
        votes = votes[votes['vote_id'].isin(current_term['id'])]
        
        members = pd.read_csv(URL_MEMBERS, compression='gzip')
        members['full_name'] = members['last_name'] + ' ' + members['first_name']
        
        data = pd.merge(votes, members[['id', 'full_name', 'country_code']], left_on='member_id', right_on='id')
        data['numeric'] = data['position'].map({'FOR': 1, 'AGAINST': -1, 'ABSTENTION': 0, 'ABSTAIN': 0}).fillna(0)
        data['vote_title'] = data['vote_id'].map(vote_titles)
        
        mep_groups = data.groupby('member_id')['group_code'].agg(lambda x: x.mode()[0] if not x.mode().empty else 'NI').to_dict()
        mep_names = members.set_index('id')['full_name'].to_dict()
        mep_countries = members.set_index('id')['country_code'].to_dict()
        return data, mep_groups, mep_names, mep_countries
    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.stop()

def draw_map(df, color_col, title, highlight_id=None, focus_group=None):
    df_plot = df.copy()
    color_map = {'EPP': '#0055aa', 'S&D': '#f0001c', 'Renew': '#ffcc00', 'Greens/EFA': '#44aa00', 'ECR': '#000080', 'PfE': '#202040', 'The Left': '#800000', 'NI': '#999999'}
    
    if focus_group and focus_group != "Wszyscy":
        df_plot['ColorGroup'] = df_plot[color_col].apply(lambda x: x if x == focus_group else 'Other')
        color_discrete = {'Other': '#e0e0e0', focus_group: color_map.get(focus_group, '#ff4b4b')}
        c_col = 'ColorGroup'
    else:
        c_col = color_col
        color_discrete = color_map

    fig = px.scatter(df_plot.sort_values(c_col, ascending=False), x='X', y='Y', color=c_col, 
                     hover_name='name', color_discrete_map=color_discrete, height=600, template="plotly_white")
    
    if highlight_id and highlight_id in df.index:
        t = df.loc[highlight_id]
        fig.add_trace(go.Scatter(x=[t['X']], y=[t['Y']], mode='markers+text', marker=dict(color='black', size=15, symbol='star'), text=[t['name']], textposition="top center", name="Target"))
    return fig

# --- MAIN ---
data_raw, groups_dict, names_dict, countries_dict = load_data()
pivot_all = data_raw.pivot_table(index='vote_id', columns='member_id', values='numeric')

tabs = st.tabs([L['tab_comp'], L['tab_fra'], L['tab_ai'], L['tab_top']])

# TAB 1: Comparator
with tabs[0]:
    sel_c = st.selectbox("Select Country:", sorted(list(set(countries_dict.values()))), index=0)
    meps_in_c = {v: k for k, v in names_dict.items() if countries_dict.get(k) == sel_c and k in pivot_all.columns}
    sel_mep = st.selectbox(L['search'], sorted(meps_in_c.keys()))
    if sel_mep:
        mid = meps_in_c[sel_mep]
        corr = pivot_all.corrwith(pivot_all[mid]).dropna()
        res = pd.DataFrame({'Correlation': corr, 'Name': corr.index.map(names_dict), 'Group': corr.index.map(groups_dict)})
        st.dataframe(res.sort_values('Correlation', ascending=False).head(15), use_container_width=True)

# TAB 2: Groups (Compass)
with tabs[1]:
    st.info(L['info_pca'])
    pca_all = PCA(n_components=2).fit_transform(pivot_all.fillna(0).T)
    df_pca = pd.DataFrame(pca_all, columns=['X', 'Y'], index=pivot_all.columns)
    df_pca['name'], df_pca['group'] = df_pca.index.map(names_dict), df_pca.index.map(groups_dict)
    
    c1, c2 = st.columns(2)
    s_mep = c1.selectbox(L['search'], [""] + sorted(list(names_dict.values())), key="p1")
    f_grp = c2.selectbox(L['focus'], ["Wszyscy"] + sorted(list(set(groups_dict.values()))))
    
    mid_h = {v: k for k, v in names_dict.items()}.get(s_mep)
    st.plotly_chart(draw_map(df_pca, 'group', "Map", highlight_id=mid_h, focus_group=f_grp), use_container_width=True)

# TAB 3: AI Clusters
with tabs[2]:
    nk = st.slider("Number of Clusters", 2, 12, 6)
    clusters = KMeans(n_clusters=nk, random_state=42).fit_predict(pivot_all.fillna(0).T)
    df_pca['cluster'] = [f"Cluster {c+1}" for c in clusters]
    
    s_mep_ai = st.selectbox(L['search'], [""] + sorted(list(names_dict.values())), key="p2")
    mid_ai = {v: k for k, v in names_dict.items()}.get(s_mep_ai)
    f_cl = df_pca.loc[mid_ai, 'cluster'] if mid_ai else "Wszyscy"
    
    st.plotly_chart(draw_map(df_pca, 'cluster', "AI", highlight_id=mid_ai, focus_group=f_cl), use_container_width=True)
    if mid_ai:
        st.subheader(f"{L['squad_msg']} {s_mep_ai}:")
        st.table(df_pca[df_pca['cluster'] == f_cl][['name', 'group']].head(20))

# TAB 4: Topics
with tabs[3]:
    query = st.text_input(L['topic_input'], "Ukraine")
    if query:
        mask = data_raw['vote_title'].str.contains(query, case=False, na=False)
        t_data = data_raw[mask]
        if t_data['vote_id'].nunique() > 2:
            p_t = t_data.pivot_table(index='vote_id', columns='member_id', values='numeric')
            pca_t = PCA(n_components=2).fit_transform(p_t.fillna(0).T)
            df_t = pd.DataFrame(pca_t, columns=['X', 'Y'], index=p_t.columns)
            df_t['name'], df_t['group'] = df_t.index.map(names_dict), df_t.index.map(groups_dict)
            st.plotly_chart(draw_map(df_t, 'group', query), use_container_width=True)
        else:
            st.warning(L['no_results'])
