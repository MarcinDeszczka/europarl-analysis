# -*- coding: utf-8 -*-
"""
Created on Fri Dec 19 03:17:33 2025

@author: Użytkownik
"""

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import plotly.express as px
import plotly.graph_objects as go
import os

st.set_page_config(page_title="EuroParl Matrix", layout="wide")

# ==========================================
# KONFIGURACJA LINKÓW (Pobieranie bezpośrednio z GitHub)
# ==========================================
URL_VOTES = "https://github.com/HowTheyVote/data/releases/download/2026-01-03/member_votes.csv.gz"
URL_MEMBERS = "https://github.com/HowTheyVote/data/releases/download/2026-01-03/members.csv.gz"
URL_ROLLCALLS = "https://github.com/HowTheyVote/data/releases/download/2026-01-03/votes.csv.gz"

@st.cache_data(show_spinner=False)
def load_data():
    # METADANE - wczytywanie bezpośrednio z URL
    try:
        rollcalls = pd.read_csv(URL_ROLLCALLS, compression='gzip')
        rollcalls['timestamp'] = rollcalls['timestamp'].astype(str).str.strip()
        rollcalls['timestamp'] = pd.to_datetime(rollcalls['timestamp'], errors='coerce')
        rollcalls = rollcalls.dropna(subset=['timestamp'])
        
        # Filtr kadencji (od 16.07.2024)
        current_term = rollcalls[rollcalls['timestamp'] > '2024-07-16']
        current_term_ids = current_term['id']
        d_min, d_max = current_term['timestamp'].min(), current_term['timestamp'].max()
        vote_titles = current_term.set_index('id').get('display_title', current_term.set_index('id')['procedure_title']).fillna('').to_dict()
    except Exception as e: raise RuntimeError(f"Błąd pobierania votes: {e}")

    # GŁOSY
    try:
        votes = pd.read_csv(URL_VOTES, compression='gzip')
        votes = votes[votes['vote_id'].isin(current_term_ids)]
    except Exception as e: raise RuntimeError(f"Błąd pobierania member_votes: {e}")

    # POSŁOWIE
    try:
        members = pd.read_csv(URL_MEMBERS, compression='gzip')
        members['full_name'] = members['last_name'] + ' ' + members['first_name']
    except Exception as e: raise RuntimeError(f"Błąd pobierania members: {e}")
    
    data = pd.merge(votes, members[['id', 'full_name', 'country_code']], left_on='member_id', right_on='id')
    vote_map = {'FOR': 1, 'AGAINST': -1, 'ABSTAIN': 0}
    data['numeric'] = data['position'].map(vote_map)
    data['vote_title'] = data['vote_id'].map(vote_titles)
    
    mep_groups = data.groupby('member_id')['group_code'].agg(lambda x: x.mode()[0] if not x.mode().empty else 'NI').to_dict()
    mep_names = members.set_index('id')['full_name'].to_dict()
    mep_countries = members.set_index('id')['country_code'].to_dict()

    return data, mep_groups, mep_names, mep_countries, d_min, d_max

@st.cache_data
def calculate_pca(pivot_df, groups_dict, names_dict, countries_dict, flip_x=False, flip_y=False):
    pivot_filled = pivot_df.fillna(0)
    if pivot_filled.shape[1] < 3: return None, [0, 0]

    pca = PCA(n_components=3)
    components = pca.fit_transform(pivot_filled.T)
    if flip_x: components[:, 0] = -components[:, 0]
    if flip_y: components[:, 1] = -components[:, 1]
    
    variance = pca.explained_variance_ratio_
    pca_df = pd.DataFrame(data=components, columns=['X', 'Y', 'Z'], index=pivot_filled.columns)
    pca_df['nazwisko'] = pca_df.index.map(names_dict)
    pca_df['frakcja'] = pca_df.index.map(groups_dict)
    pca_df['kraj'] = pca_df.index.map(countries_dict)
    return pca_df, variance

@st.cache_data
def calculate_clusters(pivot_df, n_clusters):
    pivot_filled = pivot_df.fillna(0)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(pivot_filled.T)
    return labels

# ==========================================
# ZAAWANSOWANA FUNKCJA RYSUJĄCA (FOCUS MODE)
# ==========================================
def draw_smart_map(df, color_col, title, labels, highlight_id=None, focus_group=None):
    """
    Rysuje mapę z opcją 'wygaszania' tła (Focus Mode).
    """
    df_plot = df.copy()
    
    # 1. Logika kolorów (Focus Mode)
    # Jeśli wybrano grupę do podświetlenia, resztę zamieniamy na "Inne" (szare)
    if focus_group and focus_group != "Wszyscy":
        # Tworzymy tymczasową kolumnę koloru
        df_plot['ColorGroup'] = df_plot[color_col].apply(lambda x: x if x == focus_group else 'Reszta')
        
        # Definiujemy mapę kolorów: Wybrana grupa ma swój kolor, reszta jest szara
        color_discrete_map = {
            'Reszta': '#e0e0e0', # Szary
            focus_group: '#ff0000' if 'Grupa' in str(focus_group) else None # Czerwony dla klastrów, domyślny dla frakcji
        }
        
        # Jeśli to frakcje, używamy oficjalnych kolorów
        OFFICIAL_COLORS = {
            'EPP': '#0055aa', 'S&D': '#f0001c', 'Renew': '#ffcc00', 
            'Greens/EFA': '#44aa00', 'ECR': '#000080', 'PfE': '#202040', 
            'ESN': '#112233', 'The Left': '#800000', 'NI': '#999999'
        }
        if focus_group in OFFICIAL_COLORS:
            color_discrete_map[focus_group] = OFFICIAL_COLORS[focus_group]
            
        color_column_to_use = 'ColorGroup'
        # Sortujemy tak, żeby 'Reszta' była rysowana pierwsza (pod spodem), a Focus na wierzchu
        df_plot = df_plot.sort_values('ColorGroup', ascending=False)
        
    else:
        # Tryb normalny
        color_column_to_use = color_col
        color_discrete_map = {
            'EPP': '#0055aa', 'S&D': '#f0001c', 'Renew': '#ffcc00', 
            'Greens/EFA': '#44aa00', 'ECR': '#000080', 'PfE': '#202040', 
            'ESN': '#112233', 'The Left': '#800000', 'NI': '#999999'
        }

    # 2. Rysowanie
    fig = px.scatter(
        df_plot, x='X', y='Y', 
        color=color_column_to_use,
        hover_name='nazwisko', hover_data=['kraj', 'frakcja'],
        labels=labels,
        title=title,
        height=700,
        color_discrete_map=color_discrete_map
    )
    
    # Styl kropek
    opacity = 0.9 if focus_group else 0.7
    fig.update_traces(marker=dict(size=9, opacity=opacity, line=dict(width=0.5, color='DarkSlateGrey')))

    # 3. Podświetlenie konkretnego posła (Gwiazda)
    if highlight_id and highlight_id in df.index:
        target = df.loc[highlight_id]
        fig.add_trace(go.Scatter(
            x=[target['X']], y=[target['Y']],
            mode='markers+text',
            marker=dict(color='red', size=22, symbol='star', line=dict(width=2, color='white')),
            text=[target['nazwisko']], textposition="top center",
            textfont=dict(size=14, color='black', family="Arial Black"),
            name="Szukany Poseł"
        ))

    return fig

# ==========================================
# 2. FRONTEND
# ==========================================
st.title("🇪🇺 EuroParl Matrix")

try:
    with st.spinner("Ładowanie..."):
        data_raw, groups_dict, names_dict, countries_dict, d_min, d_max = load_data()
        pivot_all = data_raw.pivot_table(index='vote_id', columns='member_id', values='numeric')
    st.toast(f"📅 Dane: {d_min.date()} — {d_max.date()}")
except Exception as e: st.error("Błąd"); st.code(str(e)); st.stop()

# Lista wszystkich nazwisk do wyszukiwarki
all_mep_names_inv = {name: mid for mid, name in names_dict.items() if mid in pivot_all.columns}
all_mep_names = sorted(list(all_mep_names_inv.keys()))

tabs = st.tabs(["👥 Porównywarka", "🧭 Kompas (Frakcje)", "🤖 AI Klastry", "🔥 Tematy"])

# --- TAB 1: PORÓWNYWARKA ---
with tabs[0]:
    c1, c2 = st.columns(2)
    with c1:
        v_countries = sorted([c for c in list(set(countries_dict.values())) if isinstance(c, str)])
        sel_country = st.selectbox("Kraj:", v_countries, index=v_countries.index('PL') if 'PL' in v_countries else 0)
    with c2:
        meps_ids = [m for m, c in countries_dict.items() if c == sel_country and m in pivot_all.columns]
        mep_map = {names_dict.get(m, f"ID {m}"): m for m in meps_ids}
        sel_mep = st.selectbox("Europoseł:", sorted(mep_map.keys())) if mep_map else None

    if sel_mep:
        tid = mep_map[sel_mep]
        res = pd.DataFrame({'korelacja': pivot_all.corrwith(pivot_all[tid])})
        res['nazwisko'] = res.index.map(names_dict)
        res['frakcja'] = res.index.map(groups_dict)
        res['kraj'] = res.index.map(countries_dict)
        res = res.drop(tid, errors='ignore').dropna()
        
        cl, cr = st.columns(2)
        cl.subheader("🤝 Sojusznicy"); cl.dataframe(res.sort_values('korelacja', ascending=False).head(10).style.format({"korelacja": "{:.2f}"}), use_container_width=True)
        cr.subheader("⚔️ Przeciwnicy"); cr.dataframe(res.sort_values('korelacja').head(10).style.format({"korelacja": "{:.2f}"}), use_container_width=True)

# --- TAB 2: KOMPAS FRAKCJI ---
with tabs[1]:
    col_c1, col_c2, col_c3 = st.columns([2, 1, 1])
    
    # 1. Wyszukiwarka
    search_q = col_c1.selectbox("🔍 Znajdź posła (automatyczny focus):", [""] + all_mep_names, key="s_fra")
    
    # 2. Manualny Focus
    all_groups = sorted(list(set(groups_dict.values())))
    focus_manual = col_c2.selectbox("💡 Podświetl tylko frakcję:", ["Wszyscy"] + all_groups)
    
    do_refresh = col_c3.button("Odśwież")

    if do_refresh or 'pca_done' in st.session_state:
        st.session_state['pca_done'] = True
        pca_res, variance = calculate_pca(pivot_all, groups_dict, names_dict, countries_dict)
        
        # Logika Focusu
        high_id = all_mep_names_inv.get(search_q)
        active_focus = "Wszyscy"
        
        # Jeśli wybrano posła, automatycznie ustawiamy focus na jego frakcję
        if high_id:
            user_group = pca_res.loc[high_id, 'frakcja']
            # Jeśli manualny focus nie jest ustawiony, użyj grupy posła
            if focus_manual == "Wszyscy":
                active_focus = user_group
                st.info(f"📍 Znaleziono posła: **{search_q}**. Należy do frakcji: **{user_group}**. Podświetlam tę grupę.")
            else:
                active_focus = focus_manual
        else:
            active_focus = focus_manual

        fig = draw_smart_map(
            pca_res, color_col='frakcja', title="Mapa Frakcji",
            labels={'X': 'Wymiar 1', 'Y': 'Wymiar 2'},
            highlight_id=high_id,
            focus_group=active_focus
        )
        st.plotly_chart(fig, use_container_width=True)

# --- TAB 3: AI KLASTRY (SQUAD INSPECTOR) ---
with tabs[2]:
    st.write("Algorytm K-Means dzieli parlament na bloki głosowania.")
    
    ck1, ck2 = st.columns([2, 2])
    n_k = ck1.slider("Liczba Klastrów", 2, 20, 5)
    search_ai = ck2.selectbox("🔍 Znajdź posła i jego towarzyszy:", [""] + all_mep_names, key="s_ai")
    
    if st.button("Uruchom Klastrowanie"):
        labels = calculate_clusters(pivot_all, n_k)
        pca_res, _ = calculate_pca(pivot_all, groups_dict, names_dict, countries_dict)
        pca_res['AI_Cluster'] = [f"Grupa {l+1}" for l in labels]
        
        # Logika Focusu dla Klastrów
        high_id_ai = all_mep_names_inv.get(search_ai)
        focus_cluster = "Wszyscy"
        
        if high_id_ai:
            user_cluster = pca_res.loc[high_id_ai, 'AI_Cluster']
            focus_cluster = user_cluster
            st.success(f"🤖 **{search_ai}** został przydzielony przez AI do: **{user_cluster}**.")
        
        fig_cl = draw_smart_map(
            pca_res, color_col='AI_Cluster', title=f"Podział na {n_k} klastrów",
            labels={'X': 'Wymiar 1', 'Y': 'Wymiar 2'},
            highlight_id=high_id_ai,
            focus_group=focus_cluster
        )
        st.plotly_chart(fig_cl, use_container_width=True)
        
        # === SQUAD INSPECTOR (TABELA TOWARZYSZY) ===
        if high_id_ai:
            st.divider()
            st.subheader(f"👥 Kto jest w '{focus_cluster}' razem z wybranym posłem?")
            st.write("Oto lista Twoich politycznych 'bliźniaków' w tym klastrze:")
            
            # Filtrujemy tylko ten klaster
            squad_df = pca_res[pca_res['AI_Cluster'] == focus_cluster].copy()
            # Sortujemy po kraju i nazwisku
            squad_df = squad_df.sort_values(['kraj', 'nazwisko'])
            
            st.dataframe(
                squad_df[['nazwisko', 'frakcja', 'kraj', 'AI_Cluster']], 
                use_container_width=True,
                hide_index=True
            )
        else:
            # Jeśli nikt nie jest wybrany, pokaż podsumowanie klastrów
            st.caption("Wybierz posła w wyszukiwarce powyżej, aby zobaczyć dokładną listę członków jego klastra.")
            crosstab = pd.crosstab(pca_res['frakcja'], pca_res['AI_Cluster'])
            st.plotly_chart(px.imshow(crosstab, text_auto=True, aspect="auto", color_continuous_scale="Blues"), use_container_width=True)

# --- TAB 4 ---
with tabs[3]:
    st.write("Wpisz temat (np. Ukraine, Nature):")
    key = st.text_input("Słowo kluczowe:", "Ukraine")
    if key:
        mask = data_raw['vote_title'].str.contains(key, case=False, na=False)
        t_data = data_raw[mask]
        if t_data['vote_id'].nunique() > 3:
            p_t = t_data.pivot_table(index='vote_id', columns='member_id', values='numeric')
            pca_t, var_t = calculate_pca(p_t, groups_dict, names_dict, countries_dict)
            if pca_t is not None:
                st.plotly_chart(draw_smart_map(
                    pca_t, color_col='frakcja', title=f"Temat: {key}",
                    labels={'X': 'W', 'Y': 'W'}
                ), use_container_width=True)
        else: st.warning("Za mało danych.")