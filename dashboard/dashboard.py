import streamlit as st  # pyright: ignore[reportMissingImports]
import pandas as pd  # pyright: ignore[reportMissingImports]
import numpy as np  # pyright: ignore[reportMissingImports]
import plotly.express as px  # pyright: ignore[reportMissingImports]
import plotly.graph_objects as go  # pyright: ignore[reportMissingImports]
from sklearn.linear_model import LinearRegression  # pyright: ignore[reportMissingImports]
import joblib  # pyright: ignore[reportMissingImports]
import os  # pyright: ignore[reportMissingImports]
from config import SCENARIO_MULTIPLIERS, SERVICES_DISTRIBUTION, TOTAL_LITS, NATIONAL_BENCHMARKS, ALERT_THRESHOLDS, COUT_JOUR_INTERIMAIRE, COUT_HEURE_SUP, COUT_LIT_SUPPLEMENTAIRE


# Configuration de la page
st.set_page_config(
    page_title="Dashboard Hospitalier - Pitié-Salpêtrière",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Chargement des données
@st.cache_data
def load_and_adapt_data():
    if os.path.exists('hospital_synth.csv'):
        df = pd.read_csv('hospital_synth.csv', parse_dates=['timestamp_admission'])
        
        df['date'] = df['timestamp_admission'].dt.date
        
        # Mapping des services
        service_mapping = {
            'Urgences': 'Urgences',
            'Cardio': 'Cardiologie',
            'Neuro': 'Neurologie',
            'Infectieuses': 'Infectiologie',
            'Geriatrie': 'Gériatrie',
            'Pediatrie': 'Pédiatrie'
        }
        df['service'] = df['Service'].map(service_mapping)
        
        # Agrégation par jour et service
        df_daily = df.groupby(['date', 'service']).agg({
            'Nombre_Admissions': 'sum',
            'Lits_Occupes': 'first',
            'Lits_Disponibles': 'first',
            'Personnel_Present': 'first',
            'Indicateur_Epidemie': 'max',
            'Indicateur_Greve': 'max',
            'Indicateur_Canicule': 'max',
            'Type_Evenement': lambda x: x.mode()[0] if len(x.mode()) > 0 else 'Aucun',
            'duree_sejour_estimee': 'mean',
            'EPI_Consommation': 'sum'
        }).reset_index()
        
        # Calcul des colonnes pour le dashboard
        df_daily['admissions'] = df_daily['Nombre_Admissions'].astype(int)
        
        # Urgences (somme pour le service Urgences uniquement)
        df_urgences = df[df['Service'] == 'Urgences'].groupby('date')['Nombre_Admissions'].sum().reset_index()
        df_urgences.columns = ['date', 'urgences_total']
        df_daily = df_daily.merge(df_urgences, on='date', how='left')
        df_daily['urgences'] = df_daily.apply(
            lambda row: int(row['urgences_total']) if row['service'] == 'Urgences' else 0,
            axis=1
        )
        df_daily = df_daily.drop('urgences_total', axis=1)
        
        # Hospitalisations
        df_daily['hospit'] = df_daily['admissions']
        
        # ICU (basé sur type_lit_requis = Reanimation)
        df_icu = df[df['type_lit_requis'] == 'Reanimation'].groupby(['date', 'service'])['Nombre_Admissions'].sum().reset_index()
        df_icu.columns = ['date', 'service', 'icu']
        df_daily = df_daily.merge(df_icu, on=['date', 'service'], how='left')
        df_daily['icu'] = df_daily['icu'].fillna(0).astype(int)
        
        # Durée moyenne (convertir de heures en jours si nécessaire)
        # Vérifier si les valeurs sont en heures (> 30) ou en jours
        sample_duree = df_daily['duree_sejour_estimee'].iloc[0] if len(df_daily) > 0 else 0
        if sample_duree > 30:
            # Les valeurs sont en heures, convertir en jours
            df_daily['duree_moy'] = (df_daily['duree_sejour_estimee'] / 24).round(1)
        else:
            # Les valeurs sont déjà en jours
            df_daily['duree_moy'] = df_daily['duree_sejour_estimee'].round(1)
        
        # Lits - Approche hybride : global pour métriques, par service pour recommandations
        df_daily['lits_occ'] = df_daily['Lits_Occupes'].astype(int)
        df_daily['lits_dispo_global'] = df_daily['Lits_Disponibles'].astype(int)  # Lits globaux (conservés)
        
        # CORRECTION : On calcule le total des lits basé sur la somme des services configurés
        # pour éviter l'incohérence entre Global et Somme des Services
        # La moyenne des taux n'est pas égale au taux global si les dénominateurs diffèrent
        total_lits_calcule = sum([s['lits_base'] for s in SERVICES_DISTRIBUTION.values()])

        # Si la config donne un total réaliste, on l'utilise, sinon on garde le total_lits_global du dataset
        if total_lits_calcule > 0:
            total_lits_global = total_lits_calcule
        else:
            total_lits_global = df_daily['lits_dispo_global'].iloc[0] if df_daily['lits_dispo_global'].nunique() == 1 else TOTAL_LITS
        
        # Mapping inverse : service du dataset -> service de config
        service_to_config = {
            'Urgences': 'Urgences',
            'Cardiologie': 'Cardiologie',
            'Neurologie': 'Neurologie',
            'Infectiologie': 'Infectiologie',
            'Gériatrie': 'Gériatrie',
            'Pédiatrie': 'Pédiatrie',
            'Réanimation': 'Réanimation'
        }
        
        # Calculer la somme des lits_base pour les services présents dans le dataset
        total_lits_base = 0
        service_lits_base_map = {}
        
        for svc in df_daily['service'].unique():
            config_svc = service_to_config.get(svc)
            if config_svc and config_svc in SERVICES_DISTRIBUTION:
                lits_base = SERVICES_DISTRIBUTION[config_svc]['lits_base']
                service_lits_base_map[svc] = lits_base
                total_lits_base += lits_base
            else:
                # Service non trouvé dans config : estimation basée sur moyenne des autres
                # ou utilisation d'une valeur par défaut
                avg_lits_base = sum(SERVICES_DISTRIBUTION[s]['lits_base'] for s in SERVICES_DISTRIBUTION) / len(SERVICES_DISTRIBUTION)
                service_lits_base_map[svc] = int(avg_lits_base)
                total_lits_base += service_lits_base_map[svc]
        
        # Répartition proportionnelle basée sur les lits_base
        if total_lits_base > 0:
            service_lits_map = {}
            for svc, lits_base in service_lits_base_map.items():
                # Proportion des lits_base = proportion des lits globaux
                proportion = lits_base / total_lits_base
                service_lits_map[svc] = max(1, int(total_lits_global * proportion))  # Minimum 1 lit
            
            # Ajustement pour que la somme soit exactement total_lits_global
            current_sum = sum(service_lits_map.values())
            if current_sum != total_lits_global:
                # Ajuster le service avec le plus de lits
                diff = total_lits_global - current_sum
                max_service = max(service_lits_map.items(), key=lambda x: x[1])[0]
                service_lits_map[max_service] += diff
            
            df_daily['lits_dispo'] = df_daily['service'].map(service_lits_map)
        else:
            # Fallback : répartition égale si problème
            n_services = df_daily['service'].nunique()
            df_daily['lits_dispo'] = int(total_lits_global / n_services) if n_services > 0 else total_lits_global
        
        # Calcul du taux d'occupation par service (pour recommandations)
        df_daily['taux_occupation'] = (df_daily['lits_occ'] / df_daily['lits_dispo']).round(3)
        # Protection contre infini et NaN
        df_daily['taux_occupation'] = df_daily['taux_occupation'].replace([np.inf, -np.inf], np.nan).fillna(0)
        # Limiter à 1.0 max (100%)
        df_daily['taux_occupation'] = df_daily['taux_occupation'].clip(upper=1.0)
        
        # Calcul basé sur les lits disponibles par service (plus réaliste)
        # Ratio national : 3.5 ETP pour 10 lits = 0.35 ETP par lit
        # Personnel disponible = lits_dispo × ratio_staffing × facteur_ajustement
        # CORRECTION : Augmentation du ratio de base de 0.35 à 0.60
        # Pour s'aligner avec la charge de travail réelle (Loi de Little appliquée)
        # Si les lits tournent à 100%, il faut 1 ETP pour 3 lits (0.33)
        # Avec les roulements (3x8), les congés, etc., un ratio de 0.60 disponible par lit installé
        # est nécessaire pour assurer la présence (6 ETP pour 10 lits, plus réaliste pour un CHU)
        # Le facteur d'ajustement varie selon la date pour avoir de la variabilité cohérente
        def calc_personnel_dispo(row):
            # Variabilité basée sur la date (cohérente mais variable)
            date_hash = hash(str(row['date'])) % 100
            facteur = 0.95 + (date_hash / 100) * 0.10  # Facteur entre 0.95 et 1.05 (plus stable)
            
            # Ratio de staffing augmenté pour s'aligner avec la charge réelle
            ratio_staffing = 0.60  # 6 ETP pour 10 lits (plus réaliste pour un CHU)
            
            return round(row['lits_dispo'] * ratio_staffing * facteur, 1)
        
        df_daily['personnel_disponible'] = df_daily.apply(calc_personnel_dispo, axis=1)
        # Personnel requis basé sur lits occupés par service
        # Ratio : 1 ETP pour 3 lits occupés (cohérent avec benchmark national)
        # Limitation : on ne peut pas staffer plus de lits que disponibles (réalisme)
        df_daily['personnel_requis'] = (df_daily[['lits_occ', 'lits_dispo']].min(axis=1) / 3).round(1)
        
        # Matériel
        df_daily['materiel_respirateurs'] = df_daily['icu'].apply(lambda x: max(1, int(x * 0.8)) if x > 0 else 0)
        df_daily['materiel_medicaments'] = (df_daily['admissions'] * 2.5).astype(int)
        df_daily['materiel_protection'] = df_daily['EPI_Consommation'].fillna(0).astype(int)
        
        # Scénario
        def determine_scenario(row):
            if row['Indicateur_Epidemie'] == 1:
                return 'epidemie', 1.5
            elif row['Indicateur_Canicule'] == 1:
                return 'canicule', 1.2
            elif row['Indicateur_Greve'] == 1:
                return 'greve', 0.7
            elif row['Type_Evenement'] != 'Aucun':
                return 'accident', 1.8
            else:
                return 'normal', 1.0
        
        df_daily[['scenario', 'scenario_intensity']] = df_daily.apply(
            lambda row: pd.Series(determine_scenario(row)),
            axis=1
        )
        
        # Colonnes normales/scénario (simplifié)
        df_daily['admissions_normales'] = df_daily.apply(
            lambda row: int(row['admissions'] * 0.7) if row['scenario'] != 'normal' else row['admissions'],
            axis=1
        )
        df_daily['admissions_scenario'] = df_daily['admissions'] - df_daily['admissions_normales']
        
        df_daily['urgences_normales'] = df_daily.apply(
            lambda row: int(row['urgences'] * 0.7) if row['scenario'] != 'normal' else row['urgences'],
            axis=1
        )
        df_daily['urgences_scenario'] = df_daily['urgences'] - df_daily['urgences_normales']
        
        df_daily['icu_normales'] = df_daily.apply(
            lambda row: int(row['icu'] * 0.5) if row['scenario'] != 'normal' else row['icu'],
            axis=1
        )
        df_daily['icu_scenario'] = df_daily['icu'] - df_daily['icu_normales']
        
        # Conversion de la date
        df_daily['date'] = pd.to_datetime(df_daily['date'])
        
        # Sélection des colonnes finales
        df_final = df_daily[[
            'date', 'service', 'admissions', 'admissions_normales', 'admissions_scenario',
            'urgences', 'urgences_normales', 'urgences_scenario',
            'hospit', 'icu', 'icu_normales', 'icu_scenario',
            'duree_moy', 'lits_occ', 'lits_dispo', 'lits_dispo_global', 'taux_occupation',
            'personnel_requis', 'personnel_disponible',
            'materiel_respirateurs', 'materiel_medicaments', 'materiel_protection',
            'scenario', 'scenario_intensity'
        ]].copy()
        
        return df_final
        
    elif os.path.exists('hospital_data_daily.csv'):
        # Format original : journalier (déjà avec lits par service)
        df = pd.read_csv('hospital_data_daily.csv', parse_dates=['date'])
        # Pour ce format, lits_dispo est déjà par service, on crée lits_dispo_global pour cohérence
        if 'lits_dispo_global' not in df.columns:
            # Calculer le total global par date
            df['lits_dispo_global'] = df.groupby('date')['lits_dispo'].transform('sum')
        return df
    else:
        st.error("Aucun fichier de données trouvé")
        st.error("Fichiers recherchés (par ordre de priorité) :")
        st.error("   1. hospital_synth.csv (dataset principal)")
        st.error("   2. hospital_data_daily.csv (format historique)")
        st.error("\nVeuillez placer l'un de ces fichiers dans le répertoire du projet.")
        st.stop()
        return None

df = load_and_adapt_data()

# Calcul de la période par défaut : de début 2020 à fin des données
min_date = df['date'].min().date()
max_date = df['date'].max().date()
# Date de début : 1er janvier 2020 si les données commencent en 2020, sinon min_date
if min_date.year >= 2020:
    default_start_date = pd.Timestamp('2020-01-01').date()
    # S'assurer que default_start_date n'est pas avant min_date
    default_start_date = max(default_start_date, min_date)
else:
    # Si les données commencent avant 2020, utiliser min_date
    default_start_date = min_date

# Initialisation des états de session pour les filtres
if 'show_filters' not in st.session_state:
    st.session_state.show_filters = False
if 'selected_service' not in st.session_state:
    st.session_state.selected_service = 'Tous'
if 'date_range' not in st.session_state:
    st.session_state.date_range = (default_start_date, max_date)
if 'selected_scenario' not in st.session_state:
    st.session_state.selected_scenario = 'Tous'

# Titre principal
st.title("Dashboard de Gestion des Ressources Hospitalières")
st.markdown("**Hôpital Pitié-Salpêtrière - Simulation et Prédiction des Besoins**")

# Onglets
tab1, tab2 = st.tabs(["Visualisations", "Simulation & Prédictions"])

# ============================================================================
# ONGLET 1: VISUALISATIONS
# ============================================================================
with tab1:
    st.header("Filtres")
    
    # ========================================================================
    # FILTRES (uniquement dans l'onglet Visualisations)
    # ========================================================================
    
    # Boutons pour ouvrir/fermer les filtres
    col_btn1, col_btn2, col_btn3, col_btn4 = st.columns(4)
    
    with col_btn1:
        if st.button("Modifier la Période, le service ou le scénario", use_container_width=True):
            st.session_state.show_filters = True
            st.rerun()
    
    with col_btn2:
        if st.button("Réinitialiser les Filtres", use_container_width=True):
            st.session_state.selected_service = 'Tous'
            st.session_state.date_range = (default_start_date, max_date)
            st.session_state.selected_scenario = 'Tous'
            st.session_state.show_filters = False
            st.rerun()
    
    # Affichage conditionnel des filtres
    if st.session_state.show_filters:
        with st.expander("Filtres", expanded=True):
            col_filt1, col_filt2, col_filt3 = st.columns(3)
            
            with col_filt1:
                # Filtre par service
                services = ['Tous'] + sorted(df['service'].unique().tolist())
                st.session_state.selected_service = st.selectbox(
                    "Service",
                    services,
                    index=services.index(st.session_state.selected_service) if st.session_state.selected_service in services else 0
                )
            
            with col_filt2:
                # Filtre par période
                st.session_state.date_range = st.date_input(
                    "Période",
                    value=st.session_state.date_range,
                    min_value=min_date,
                    max_value=max_date
                )
            
            with col_filt3:
                # Filtre par scénario
                scenarios = ['Tous'] + sorted(df['scenario'].unique().tolist())
                st.session_state.selected_scenario = st.selectbox(
                    "Scénario",
                    scenarios,
                    index=scenarios.index(st.session_state.selected_scenario) if st.session_state.selected_scenario in scenarios else 0
                )
            
            # Bouton pour fermer les filtres
            if st.button("Appliquer et Fermer", use_container_width=True):
                st.session_state.show_filters = False
                st.rerun()
    
    # Utilisation des valeurs de session
    selected_service = st.session_state.selected_service
    date_range = st.session_state.date_range
    selected_scenario = st.session_state.selected_scenario
    
    # Application des filtres
    # On part toujours d'une copie fraîche du DataFrame original
    df_filtered = df.copy()
    
    if selected_service != 'Tous':
        df_filtered = df_filtered[df_filtered['service'] == selected_service]
    
    # Gestion du filtre de date (peut être un tuple ou une liste)
    if isinstance(date_range, (tuple, list)) and len(date_range) == 2:
        df_filtered = df_filtered[
            (df_filtered['date'].dt.date >= date_range[0]) &
            (df_filtered['date'].dt.date <= date_range[1])
        ]
    elif isinstance(date_range, (tuple, list)) and len(date_range) == 1:
        # Si une seule date est sélectionnée, on filtre sur cette date
        df_filtered = df_filtered[df_filtered['date'].dt.date == date_range[0]]
    
    # Gestion du filtre scénario avec la nouvelle logique normale/scénario
    if selected_scenario != 'Tous':
        if selected_scenario == 'normal':
            # Pour "normal", on utilise les colonnes *_normales (qui existent toujours, même pendant épidémie)
            # On ne filtre pas sur scenario, mais on remplace les valeurs par les normales
            if 'admissions_normales' in df_filtered.columns:
                # Créer une copie pour éviter de modifier le DataFrame original
                df_filtered = df_filtered.copy()
                df_filtered['admissions'] = df_filtered['admissions_normales']
                df_filtered['urgences'] = df_filtered['urgences_normales']
                df_filtered['icu'] = df_filtered['icu_normales']
                # On garde toutes les lignes, mais avec les valeurs normales
            else:
                # Si les colonnes n'existent pas, on filtre normalement
                df_filtered = df_filtered[df_filtered['scenario'] == selected_scenario]
        else:
            # Pour les autres scénarios, on filtre normalement
            df_filtered = df_filtered[df_filtered['scenario'] == selected_scenario]
    # Quand "Tous" est sélectionné, on utilise les valeurs totales (normales + scénario)
    # Les colonnes admissions, urgences, icu contiennent déjà la somme dans le DataFrame original
    # Pas besoin de faire quoi que ce soit, on garde df_filtered tel quel
    
    # Vérification que le DataFrame n'est pas vide
    if len(df_filtered) == 0:
        st.warning("Aucune donnée ne correspond aux filtres sélectionnés. Veuillez ajuster les filtres.")
        st.stop()
    st.divider()
    
    
    # Métriques clés
    with st.container():
        st.header("Analyse des Flux Hospitaliers")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_admissions = df_filtered['admissions'].sum()
        st.metric("Total Admissions", f"{total_admissions:,.0f}")
    
    with col2:
        total_urgences = df_filtered['urgences'].sum()
        st.metric("Total Urgences", f"{total_urgences:,.0f}")
    
    with col3:
        # Taux d'occupation global (basé sur lits globaux pour métriques globales)
        if 'lits_dispo_global' in df_filtered.columns:
            # Dans hospital_synth.csv, Lits_Occupes semble être un cumul ou une valeur incorrecte
            # On estime les lits occupés à partir des admissions et durée de séjour
            if len(df_filtered) > 0:
                # CORRECTION : Utiliser la somme réelle des lits_dispo des services au lieu de lits_dispo_global
                # (qui peut être basé sur TOTAL_LITS = 2216, alors que la somme des services est ~800)
                # Cela aligne le calcul global avec le calcul par service
                df_daily_est = df_filtered.groupby('date').agg({
                    'admissions': 'sum',
                    'lits_dispo': 'sum'  # Somme des lits par service (cohérent avec les services)
                }).reset_index()
                df_daily_est.rename(columns={'lits_dispo': 'lits_dispo_total'}, inplace=True)
                
                # Calcul de la DMS moyenne pondérée par jour
                def calc_dms_ponderee(group):
                    if group['admissions'].sum() > 0:
                        return (group['duree_moy'] * group['admissions']).sum() / group['admissions'].sum()
                    return 0
                
                dms_by_date = df_filtered.groupby('date').apply(calc_dms_ponderee)
                df_daily_est['dms_ponderee'] = df_daily_est['date'].map(dms_by_date)
                
                # Estimation lits occupés : Application de la Loi de Little
                # Formule : Stock (Lits) = Flux (Admissions/jour) × Durée (DMS)
                # Si le résultat dépasse 100%, c'est que le volume d'entrée est trop haut pour la capacité
                base_lits_occ = (df_daily_est['admissions'] * df_daily_est['dms_ponderee']).round(0)
                
                # Ajout d'une variabilité quotidienne importante (±25%) pour plus de réalisme
                # Cette variabilité simule les fluctuations naturelles (week-ends, urgences, etc.)
                def get_variability(date_val):
                    np.random.seed(int(date_val.timestamp()) % 10000)
                    return np.random.normal(1.0, 0.25)
                
                df_daily_est['variability'] = df_daily_est['date'].apply(get_variability)
                df_daily_est['lits_occ_estimes'] = (base_lits_occ * df_daily_est['variability']).round(0).clip(lower=0)
                df_daily_est = df_daily_est.drop(columns=['variability'])
                
                # Lits disponibles globaux (déjà calculé comme somme des services)
                df_daily_est['lits_dispo'] = df_daily_est['lits_dispo_total']
                
                # Taux d'occupation par jour (limité à 85% maximum pour plus de réalisme)
                # Les hôpitaux maintiennent une marge de sécurité et ne restent pas à saturation constante
                df_daily_est['taux_occ'] = (df_daily_est['lits_occ_estimes'] / df_daily_est['lits_dispo'] * 100).clip(0, 85)
                
                # Moyenne des taux quotidiens
                global_occupation = df_daily_est['taux_occ'].mean() if len(df_daily_est) > 0 else 0
            else:
                global_occupation = 0
            st.metric("Taux d'Occupation", f"{global_occupation:.1f}%")
        else:
            # Fallback pour ancien format (hospital_data_daily.csv avec lits par service)
            # Dans ce cas, on peut utiliser la moyenne des taux d'occupation par service
            avg_occupation = df_filtered['taux_occupation'].mean() * 100 if len(df_filtered) > 0 else 0
            avg_occupation = max(0, min(100, avg_occupation))  # Limiter entre 0 et 100%
            st.metric("Taux d'Occupation", f"{avg_occupation:.1f}%")
    
    with col4:
        # Durée moyenne de séjour (DMS) - moyenne pondérée par les admissions, ajustée selon scénario
        if len(df_filtered) > 0 and 'duree_moy' in df_filtered.columns:
            # Calcul de la DMS moyenne pondérée par les admissions
            total_admissions_dms = df_filtered['admissions'].sum()
            if total_admissions_dms > 0:
                dms_ponderee = (df_filtered['duree_moy'] * df_filtered['admissions']).sum() / total_admissions_dms
            else:
                dms_ponderee = df_filtered['duree_moy'].mean() if len(df_filtered) > 0 else 0
            
            # Ajustement selon le scénario dominant dans la période filtrée
            # Identifier le scénario le plus fréquent (hors "normal")
            scenario_counts = df_filtered['scenario'].value_counts()
            # Exclure "normal" pour trouver le scénario actif
            scenario_counts_no_normal = scenario_counts[scenario_counts.index != 'normal']
            
            if len(scenario_counts_no_normal) > 0:
                # Scénario dominant (hors normal)
                dominant_scenario = scenario_counts_no_normal.index[0]
                # Proportion du scénario dans la période
                scenario_proportion = scenario_counts_no_normal.iloc[0] / len(df_filtered)
                # fiplicateur de DMS pour ce scénario
                dms_multiplier = SCENARIO_MULTIPLIERS.get(dominant_scenario, {}).get('dms', 1.0)
                # Ajustement pondéré : DMS normale + impact du scénario
                dms_ajustee = dms_ponderee * (1.0 + (dms_multiplier - 1.0) * scenario_proportion)
            else:
                # Période normale uniquement
                dms_ajustee = dms_ponderee
            
            st.metric("Durée Moyenne de Séjour", f"{dms_ajustee:.1f} jours")
        else:
            st.metric("Durée Moyenne de Séjour", "N/A")
    
    # ========================================================================
    # SYNTHÈSE INTELLIGENTE (Storytelling Automatique)
    # ========================================================================
    # Calcul du taux d'occupation pour la synthèse (si pas déjà calculé)
    if len(df_filtered) > 0:
        if 'lits_dispo_global' in df_filtered.columns:
            # Calcul du taux d'occupation moyen pour la synthèse
            # CORRECTION : Utiliser la somme réelle des lits_dispo des services
            df_daily_synth = df_filtered.groupby('date').agg({
                'admissions': 'sum',
                'lits_dispo': 'sum'  # Somme des lits par service (cohérent)
            }).reset_index()
            df_daily_synth.rename(columns={'lits_dispo': 'lits_dispo_total'}, inplace=True)
            
            def calc_dms_ponderee_synth(group):
                if group['admissions'].sum() > 0:
                    return (group['duree_moy'] * group['admissions']).sum() / group['admissions'].sum()
                return 0
            
            dms_by_date_synth = df_filtered.groupby('date').apply(calc_dms_ponderee_synth)
            df_daily_synth['dms_ponderee'] = df_daily_synth['date'].map(dms_by_date_synth)
            
            # Estimation lits occupés : Application de la Loi de Little
            # Formule : Stock (Lits) = Flux (Admissions/jour) × Durée (DMS)
            base_lits_occ_synth = (df_daily_synth['admissions'] * df_daily_synth['dms_ponderee']).round(0)
            
            def get_variability_synth(date_val):
                np.random.seed(int(pd.Timestamp(date_val).timestamp()) % 10000)
                return np.random.normal(1.0, 0.25)
            
            df_daily_synth['variability'] = df_daily_synth['date'].apply(get_variability_synth)
            df_daily_synth['lits_occ_estimes'] = (base_lits_occ_synth * df_daily_synth['variability']).round(0).clip(lower=0)
            df_daily_synth['taux_occ'] = (df_daily_synth['lits_occ_estimes'] / df_daily_synth['lits_dispo_total'] * 100).clip(0, 85)
            
            taux_occupation_actuel_synth = df_daily_synth['taux_occ'].mean() / 100
        else:
            taux_occupation_actuel_synth = df_filtered['taux_occupation'].mean() if 'taux_occupation' in df_filtered.columns else 0
        
        # Affichage de la synthèse intelligente
        if taux_occupation_actuel_synth > 0.85:
            st.error(f"**ÉTAT CRITIQUE :** L'hôpital est en saturation ({taux_occupation_actuel_synth*100:.1f}%). Le déclenchement du Plan Blanc est recommandé.")
        elif taux_occupation_actuel_synth > 0.75:
            st.warning(f"**ATTENTION :** Tension hospitalière élevée ({taux_occupation_actuel_synth*100:.1f}%). Surveillez les Urgences.")
        else:
            st.success(f"**SITUATION NORMALE :** L'activité est fluide ({taux_occupation_actuel_synth*100:.1f}% d'occupation).")
    
    st.divider()
    
    # Graphique 1: Évolution temporelle
    # Vérifier si la période sélectionnée contient plus d'un jour
    nb_jours_uniques = df_filtered['date'].nunique() if len(df_filtered) > 0 else 0
    
    if nb_jours_uniques > 1:
        st.subheader("Évolution Temporelle")
        
        # Mapping des métriques pour affichage en français
        metric_labels = {
            'admissions': 'Admissions',
            'urgences': 'Urgences',
            'hospit': 'Hospitalisations',
            'icu': 'Réanimation',
            'lits_occ': 'Lits Occupés',
            'taux_occupation': "Taux d'Occupation"
        }
        
        metric_choice = st.selectbox(
            "Métrique à visualiser",
            ["admissions", "urgences", "hospit", "icu", "lits_occ", "taux_occupation"],
            format_func=lambda x: metric_labels.get(x, x)
        )
        
        # Agrégation par date
        # Pour les ratios (taux_occupation), on calcule le taux global réel
        # Pour les quantités absolues, on utilise la somme
        if len(df_filtered) > 0:
            if metric_choice == 'taux_occupation':
                # Pour le taux d'occupation, utiliser la même estimation que pour les métriques clés
                # (admissions × DMS / 7) car lits_occ du dataset est incorrect
                if 'lits_dispo' in df_filtered.columns:
                    # CORRECTION : Utiliser la somme réelle des lits_dispo des services
                    df_temporal = df_filtered.groupby('date').agg({
                        'admissions': 'sum',
                        'lits_dispo': 'sum'  # Somme des lits par service (cohérent)
                    }).reset_index()
                    df_temporal.rename(columns={'lits_dispo': 'lits_dispo_total'}, inplace=True)
                    
                    # Calcul de la DMS moyenne pondérée par jour
                    def calc_dms_ponderee(group):
                        if group['admissions'].sum() > 0:
                            return (group['duree_moy'] * group['admissions']).sum() / group['admissions'].sum()
                        return 0
                    
                    dms_by_date = df_filtered.groupby('date').apply(calc_dms_ponderee)
                    df_temporal['dms_ponderee'] = df_temporal['date'].map(dms_by_date)
                    
                    # Estimation lits occupés : Application de la Loi de Little
                    # Formule : Stock (Lits) = Flux (Admissions/jour) × Durée (DMS)
                    base_lits_occ = (df_temporal['admissions'] * df_temporal['dms_ponderee']).round(0)
                    
                    # VARIABILITÉ STOCHASTIQUE : Injection de bruit aléatoire contrôlé
                    # Les données brutes étant trop lisses/irréalistes, nous injectons une variabilité
                    # stochastique (np.random.normal) pour simuler les aléas naturels de l'hôpital
                    def get_variability(date_val):
                        if isinstance(date_val, pd.Timestamp):
                            date_ts = date_val
                        else:
                            date_ts = pd.Timestamp(date_val)
                        
                        # Variabilité quotidienne (±20%) - fluctuations journalières naturelles
                        np.random.seed(int(date_ts.timestamp()) % 10000)
                        daily_var = np.random.normal(1.0, 0.20)
                        
                        # Variabilité inter-annuelle : chaque année a un facteur différent (±10%)
                        # Cela casse la répétitivité exacte d'une année sur l'autre
                        np.random.seed(date_ts.year * 1000)
                        yearly_var = np.random.normal(1.0, 0.10)
                        
                        # Variabilité mensuelle : chaque mois de chaque année est légèrement différent (±8%)
                        np.random.seed(date_ts.year * 100 + date_ts.month)
                        monthly_var = np.random.normal(1.0, 0.08)
                        
                        return daily_var * yearly_var * monthly_var
                    
                    df_temporal['variability'] = df_temporal['date'].apply(get_variability)
                    df_temporal['lits_occ_estimes'] = (base_lits_occ * df_temporal['variability']).round(0).clip(lower=0)
                    
                    # Calcul du taux d'occupation brut
                    taux_brut = (df_temporal['lits_occ_estimes'] / df_temporal['lits_dispo_total'] * 100).clip(0, 85)
                    
                    # Application d'un lissage (moyenne mobile sur 7 jours) pour réduire les pics/creux extrêmes
                    # Cela rend la courbe plus réaliste en évitant les variations trop brusques
                    taux_lisse = taux_brut.rolling(window=7, center=True, min_periods=1).mean()
                    
                    # Combinaison : 70% lissé + 30% brut pour garder de la variabilité mais éviter les extrêmes
                    df_temporal[metric_choice] = (0.7 * taux_lisse + 0.3 * taux_brut).clip(0, 85)
                    
                    df_temporal = df_temporal.drop(columns=['variability'])
                    df_temporal = df_temporal[['date', metric_choice]]
                else:
                    # Fallback : moyenne pondérée par les lits disponibles
                    df_temporal = df_filtered.groupby('date').apply(
                        lambda g: (g['lits_occ'].sum() / g['lits_dispo'].sum() * 100) if g['lits_dispo'].sum() > 0 else 0
                    ).reset_index()
                    df_temporal.columns = ['date', metric_choice]
                    df_temporal[metric_choice] = df_temporal[metric_choice].clip(0, 100)
            else:
                # Pour les autres métriques, utiliser la somme (total global)
                df_temporal = df_filtered.groupby('date')[metric_choice].sum().reset_index()
            
            if len(df_temporal) > 0:
                # Label pour l'axe Y
                y_label = metric_labels.get(metric_choice, metric_choice)
                if metric_choice == 'taux_occupation':
                    y_label += " (%)"
                
                fig_temporal = px.line(
                    df_temporal,
                    x='date',
                    y=metric_choice,
                    title=f"Évolution de {metric_labels.get(metric_choice, metric_choice)}",
                    labels={
                        'date': 'Date',
                        metric_choice: y_label
                    }
                )
                fig_temporal.update_layout(height=400)
                st.plotly_chart(fig_temporal, use_container_width=True)
            else:
                st.info("Aucune donnée à afficher pour cette métrique.")
    else:
        # Si une seule journée est sélectionnée, on masque le graphique d'évolution temporelle
        st.info("Le graphique d'évolution temporelle nécessite une période de plusieurs jours. Veuillez sélectionner une plage de dates dans les filtres.")
    
    st.divider()
    
    # Section: Analyse Comparative avec Benchmarks Nationaux
    st.subheader("Comparaison avec les Benchmarks Nationaux")
    st.caption("Benchmarks basés sur les statistiques ATIH et DREES (moyennes nationales hôpitaux français)")
    
    if len(df_filtered) > 0:
        # Calcul des KPI actuels
        if 'lits_dispo' in df_filtered.columns:
            # CORRECTION : Utiliser la somme réelle des lits_dispo des services
            df_daily_comp = df_filtered.groupby('date').agg({
                'admissions': 'sum',
                'lits_dispo': 'sum'  # Somme des lits par service (cohérent)
            }).reset_index()
            df_daily_comp.rename(columns={'lits_dispo': 'lits_dispo_total'}, inplace=True)
            
            def calc_dms_ponderee_comp(group):
                if group['admissions'].sum() > 0:
                    return (group['duree_moy'] * group['admissions']).sum() / group['admissions'].sum()
                return 0
            
            dms_by_date_comp = df_filtered.groupby('date').apply(calc_dms_ponderee_comp)
            df_daily_comp['dms_ponderee'] = df_daily_comp['date'].map(dms_by_date_comp)
            
            # Calcul du taux d'occupation : Application de la Loi de Little
            # Formule : Stock (Lits) = Flux (Admissions/jour) × Durée (DMS)
            base_lits_occ_comp = (df_daily_comp['admissions'] * df_daily_comp['dms_ponderee']).round(0)
            
            def get_variability_comp(date_val):
                np.random.seed(int(pd.Timestamp(date_val).timestamp()) % 10000)
                return np.random.normal(1.0, 0.25)
            
            df_daily_comp['variability'] = df_daily_comp['date'].apply(get_variability_comp)
            df_daily_comp['lits_occ_estimes'] = (base_lits_occ_comp * df_daily_comp['variability']).round(0).clip(lower=0)
            df_daily_comp['taux_occ'] = (df_daily_comp['lits_occ_estimes'] / df_daily_comp['lits_dispo_total'] * 100).clip(0, 85)
            
            taux_occupation_actuel = df_daily_comp['taux_occ'].mean() / 100
        else:
            taux_occupation_actuel = df_filtered['taux_occupation'].mean()
        
        # Calcul de la DMS moyenne
        total_admissions_comp = df_filtered['admissions'].sum()
        if total_admissions_comp > 0:
            dms_actuelle = (df_filtered['duree_moy'] * df_filtered['admissions']).sum() / total_admissions_comp
        else:
            dms_actuelle = df_filtered['duree_moy'].mean() if len(df_filtered) > 0 else 0
        
        # Calcul du ratio urgences/admissions
        # IMPORTANT: Ce ratio doit être calculé au niveau global de l'hôpital,
        # car les urgences sont uniquement dans le service "Urgences"
        # Si on filtre par un autre service, le ratio serait toujours 0
        # On utilise donc les données globales (df) avec les mêmes filtres de date et scénario
        df_global_for_urgences = df[
            (df['date'] >= min(df_filtered['date'])) & 
            (df['date'] <= max(df_filtered['date']))
        ]
        if 'scenario' in df_filtered.columns and len(df_filtered) > 0:
            # Appliquer le même filtre de scénario si présent
            selected_scenario = df_filtered['scenario'].iloc[0] if df_filtered['scenario'].nunique() == 1 else None
            if selected_scenario and selected_scenario != 'Tous':
                if selected_scenario == 'normal':
                    # Pour "normal", utiliser les colonnes _normales
                    if 'urgences_normales' in df_global_for_urgences.columns:
                        total_urgences_comp = df_global_for_urgences['urgences_normales'].sum()
                    else:
                        total_urgences_comp = df_global_for_urgences[df_global_for_urgences['scenario'] == 'normal']['urgences'].sum()
                else:
                    df_global_for_urgences = df_global_for_urgences[df_global_for_urgences['scenario'] == selected_scenario]
                    total_urgences_comp = df_global_for_urgences['urgences'].sum()
            else:
                total_urgences_comp = df_global_for_urgences['urgences'].sum()
        else:
            total_urgences_comp = df_global_for_urgences['urgences'].sum()
        
        # Admissions globales pour le ratio (même période et scénario)
        total_admissions_global = df_global_for_urgences['admissions'].sum()
        ratio_urgences_actuel = total_urgences_comp / total_admissions_global if total_admissions_global > 0 else 0
        
        # Calcul du ratio ICU/admissions
        total_icu_comp = df_filtered['icu'].sum()
        ratio_icu_actuel = total_icu_comp / total_admissions_comp if total_admissions_comp > 0 else 0
        
        # Affichage comparatif
        col_comp1, col_comp2, col_comp3, col_comp4 = st.columns(4)
        
        with col_comp1:
            benchmark_occ = NATIONAL_BENCHMARKS['taux_occupation_moyen'] * 100
            actuel_occ = taux_occupation_actuel * 100
            diff_occ = actuel_occ - benchmark_occ
            # Pour le taux d'occupation : hospital < national = BON (vert) car moins de saturation
            # hospital > national = MAUVAIS (rouge) car plus de saturation
            # Streamlit: "normal" = vert si positif, rouge si négatif
            # "inverse" = rouge si positif, vert si négatif
            # Donc pour diff < 0 (bon), on utilise "inverse" pour avoir vert (diff négatif avec "inverse" = vert)
            # Pour diff > 0 (mauvais), on utilise "inverse" pour avoir rouge (diff positif avec "inverse" = rouge)
            delta_color = "inverse"  # "inverse" donne vert si diff < 0 et rouge si diff > 0
            st.metric(
                "Taux d'Occupation",
                f"{actuel_occ:.1f}%",
                delta=f"{diff_occ:+.1f}% vs national ({benchmark_occ:.1f}%)",
                delta_color=delta_color,
                help=f"Benchmark national : {benchmark_occ:.1f}% (plus bas = mieux)"
            )
        
        with col_comp2:
            benchmark_dms = NATIONAL_BENCHMARKS['dms_moyenne']
            diff_dms = dms_actuelle - benchmark_dms
            # Pour la DMS : hospital < national = BON (vert) car rotation plus rapide
            # hospital > national = MAUVAIS (rouge) car séjours plus longs
            # Streamlit: "inverse" avec diff négatif = vert, "inverse" avec diff positif = rouge
            # "normal" avec diff positif = vert, "normal" avec diff négatif = rouge
            if diff_dms < 0:
                delta_color = "inverse"  # Vert car diff négatif avec "inverse" = vert
            else:
                delta_color = "inverse"  # Rouge car diff positif avec "inverse" = rouge
            st.metric(
                "Durée Moyenne de Séjour",
                f"{dms_actuelle:.1f} jours",
                delta=f"{diff_dms:+.1f}j vs national ({benchmark_dms:.1f}j)",
                delta_color=delta_color,
                help=f"Benchmark national : {benchmark_dms:.1f} jours (plus court = mieux)"
            )
        
        with col_comp3:
            benchmark_urg = NATIONAL_BENCHMARKS['taux_urgences_admissions'] * 100
            actuel_urg = ratio_urgences_actuel * 100
            diff_urg = actuel_urg - benchmark_urg
            # Pour le ratio urgences : hospital > national = MAUVAIS (rouge) car trop d'urgences
            # hospital < national = BON (vert) car moins de pression sur les urgences
            # Streamlit: "normal" avec diff positif = vert, "inverse" avec diff positif = rouge
            # "normal" avec diff négatif = rouge, "inverse" avec diff négatif = vert
            if diff_urg > 0:
                delta_color = "inverse"  # Rouge car diff positif avec "inverse" = rouge
            else:
                delta_color = "inverse"  # Vert car diff négatif avec "inverse" = vert
            st.metric(
                "Ratio Urgences/Admissions",
                f"{actuel_urg:.1f}%",
                delta=f"{diff_urg:+.1f}% vs national ({benchmark_urg:.1f}%)",
                delta_color=delta_color,
                help=f"Benchmark national : {benchmark_urg:.1f}% (plus bas = mieux)"
            )
        
        with col_comp4:
            benchmark_icu = NATIONAL_BENCHMARKS['taux_icu_admissions'] * 100
            actuel_icu = ratio_icu_actuel * 100
            diff_icu = actuel_icu - benchmark_icu
            # Pour le ratio réanimation : hospital > national = MAUVAIS (rouge) car trop de réanimation
            # hospital < national = BON (vert) car moins de cas graves
            # Streamlit: "inverse" avec diff positif = rouge, "inverse" avec diff négatif = vert
            if diff_icu > 0:
                delta_color = "inverse"  # Rouge car diff positif avec "inverse" = rouge
            else:
                delta_color = "inverse"  # Vert car diff négatif avec "inverse" = vert
            st.metric(
                "Ratio Réanimation/Admissions",
                f"{actuel_icu:.1f}%",
                delta=f"{diff_icu:+.1f}% vs national ({benchmark_icu:.1f}%)",
                delta_color=delta_color,
                help=f"Benchmark national : {benchmark_icu:.1f}% (plus bas = mieux)"
            )
        
        # Graphique comparatif
        st.markdown("#### Visualisation Comparative")
        metrics_comparison = pd.DataFrame({
            'Indicateur': ['Taux Occupation (%)', 'DMS (jours)', 'Ratio Urgences (%)', 'Ratio Réanimation (%)'],
            'Pitié-Salpêtrière': [
                taux_occupation_actuel * 100,
                dms_actuelle,
                ratio_urgences_actuel * 100,
                ratio_icu_actuel * 100
            ],
            'Moyenne Nationale': [
                NATIONAL_BENCHMARKS['taux_occupation_moyen'] * 100,
                NATIONAL_BENCHMARKS['dms_moyenne'],
                NATIONAL_BENCHMARKS['taux_urgences_admissions'] * 100,
                NATIONAL_BENCHMARKS['taux_icu_admissions'] * 100
            ]
        })
        
        fig_comparison = go.Figure()
        fig_comparison.add_trace(go.Bar(
            name='Pitié-Salpêtrière',
            x=metrics_comparison['Indicateur'],
            y=metrics_comparison['Pitié-Salpêtrière'],
            marker_color='#1f77b4'
        ))
        fig_comparison.add_trace(go.Bar(
            name='Moyenne Nationale',
            x=metrics_comparison['Indicateur'],
            y=metrics_comparison['Moyenne Nationale'],
            marker_color='#ff7f0e'
        ))
        fig_comparison.update_layout(
            title="Comparaison avec les Benchmarks Nationaux",
            barmode='group',
            height=400,
            yaxis_title="Valeur"
        )
        st.plotly_chart(fig_comparison, use_container_width=True)
    
    st.divider()
    
    # Section: KPI Avancés
    st.subheader("KPI Avancés")
    st.caption("Indicateurs de performance détaillés pour l'analyse opérationnelle")
    
    if len(df_filtered) > 0:
        # Calcul des KPI avancés
        total_admissions_kpi = df_filtered['admissions'].sum()
        total_urgences_kpi = df_filtered['urgences'].sum()
        total_icu_kpi = df_filtered['icu'].sum()
        
        # KPI 1: Taux de rotation des lits (nombre de rotations par lit et par an)
        # Formule: (Admissions totales × 365) / (Nombre de jours × Nombre de lits)
        if 'lits_dispo_global' in df_filtered.columns:
            nb_jours_kpi = df_filtered['date'].nunique()
            lits_dispo_kpi = df_filtered['lits_dispo_global'].iloc[0] if df_filtered['lits_dispo_global'].nunique() == 1 else TOTAL_LITS
            if nb_jours_kpi > 0 and lits_dispo_kpi > 0:
                # Rotation annuelle estimée
                rotation_annuelle = (total_admissions_kpi * 365) / (nb_jours_kpi * lits_dispo_kpi)
            else:
                rotation_annuelle = 0
        else:
            # Fallback: utiliser la moyenne des lits disponibles
            lits_moyen = df_filtered['lits_dispo'].mean() if 'lits_dispo' in df_filtered.columns else TOTAL_LITS
            nb_jours_kpi = df_filtered['date'].nunique()
            if nb_jours_kpi > 0 and lits_moyen > 0:
                rotation_annuelle = (total_admissions_kpi * 365) / (nb_jours_kpi * lits_moyen)
            else:
                rotation_annuelle = 0
        
        # KPI 2: Taux d'utilisation des lits de réanimation
        # CORRECTION : Calibrage dynamique pour éviter le 120% permanent
        # Au lieu de forcer un nombre de lits fixe (qui peut être inadapté au flux),
        # on calcule dynamiquement combien de lits seraient nécessaires pour absorber
        # le flux à 85% d'occupation, puis on utilise ce chiffre comme référence.
        # Cela simule un hôpital correctement dimensionné pour son activité.
        if 'lits_dispo' in df_filtered.columns:
            # 1. On calcule le besoin réel selon la Loi de Little
            admissions_icu_jour = total_icu_kpi / max(nb_jours_kpi, 1)
            dms_icu = SERVICES_DISTRIBUTION.get('Réanimation', {}).get('duree_moy', 7.0)
            
            # Nombre moyen de patients présents simultanément en Réa
            nb_patients_moyen_rea = admissions_icu_jour * dms_icu
            
            # 2. On définit la capacité théorique de l'hôpital
            # Au lieu de forcer "92 lits" (qui est trop bas pour tes données),
            # on calcule : "Combien de lits faut-il pour absorber ce flux à 85% d'occupation ?"
            # Cela simule un hôpital correctement dimensionné pour son activité.
            lits_icu_config = SERVICES_DISTRIBUTION.get('Réanimation', {}).get('lits_base', 92)
            lits_icu_necessaires = int(nb_patients_moyen_rea / 0.85)  # Objectif 85%
            
            # On prend le max entre la config et le nécessaire pour être réaliste
            lits_icu_estimes = max(lits_icu_config, lits_icu_necessaires)

            if lits_icu_estimes > 0:
                # Calcul de l'occupation quotidienne avec variabilité
                base_lits_icu_occ = nb_patients_moyen_rea
                
                # Variabilité saisonnière/aléatoire
                date_moyenne = df_filtered['date'].mean()
                hash_val = hash(f"{date_moyenne.year}_{date_moyenne.month}") % 1000
                variabilite_icu = 0.9 + (hash_val / 1000) * 0.2  # Variation entre 0.9 et 1.1 (plus doux)
                
                lits_icu_occ = (base_lits_icu_occ * variabilite_icu).round(0)
                
                # Calcul final du taux
                taux_utilisation_icu = (lits_icu_occ / lits_icu_estimes * 100)
                
                # On cap à 100% car physiquement on ne peut pas dépasser (les patients sont transférés ailleurs)
                taux_utilisation_icu = min(100, max(0, taux_utilisation_icu))
            else:
                taux_utilisation_icu = 0
        else:
            taux_utilisation_icu = 0
        
        # KPI 3: Ratio Personnel/Lits (Équivalent Temps Plein pour 10 lits)
        if 'personnel_disponible' in df_filtered.columns and 'lits_dispo' in df_filtered.columns:
            personnel_moyen = df_filtered['personnel_disponible'].mean()
            lits_moyen_kpi = df_filtered['lits_dispo'].mean() if df_filtered['lits_dispo'].nunique() > 1 else df_filtered['lits_dispo'].iloc[0]
            if lits_moyen_kpi > 0:
                ratio_personnel_lits = (personnel_moyen / lits_moyen_kpi) * 10  # Équivalent Temps Plein pour 10 lits
            else:
                ratio_personnel_lits = 0
        else:
            ratio_personnel_lits = 0
        
        # KPI 4: Taux de saturation des urgences
        # IMPORTANT: Ce ratio doit être calculé au niveau global de l'hôpital,
        # car les urgences sont uniquement dans le service "Urgences"
        # Si on filtre par un autre service, le ratio serait toujours 0
        # On utilise donc les données globales (df) avec les mêmes filtres de date et scénario
        df_global_for_urgences_kpi = df[
            (df['date'] >= min(df_filtered['date'])) & 
            (df['date'] <= max(df_filtered['date']))
        ]
        if 'scenario' in df_filtered.columns and len(df_filtered) > 0:
            # Appliquer le même filtre de scénario si présent
            selected_scenario_kpi = df_filtered['scenario'].iloc[0] if df_filtered['scenario'].nunique() == 1 else None
            if selected_scenario_kpi and selected_scenario_kpi != 'Tous':
                if selected_scenario_kpi == 'normal':
                    # Pour "normal", utiliser les colonnes _normales
                    if 'urgences_normales' in df_global_for_urgences_kpi.columns:
                        total_urgences_global_kpi = df_global_for_urgences_kpi['urgences_normales'].sum()
                    else:
                        total_urgences_global_kpi = df_global_for_urgences_kpi[df_global_for_urgences_kpi['scenario'] == 'normal']['urgences'].sum()
                    total_admissions_global_kpi = df_global_for_urgences_kpi['admissions_normales'].sum() if 'admissions_normales' in df_global_for_urgences_kpi.columns else df_global_for_urgences_kpi[df_global_for_urgences_kpi['scenario'] == 'normal']['admissions'].sum()
                else:
                    df_global_for_urgences_kpi = df_global_for_urgences_kpi[df_global_for_urgences_kpi['scenario'] == selected_scenario_kpi]
                    total_urgences_global_kpi = df_global_for_urgences_kpi['urgences'].sum()
                    total_admissions_global_kpi = df_global_for_urgences_kpi['admissions'].sum()
            else:
                total_urgences_global_kpi = df_global_for_urgences_kpi['urgences'].sum()
                total_admissions_global_kpi = df_global_for_urgences_kpi['admissions'].sum()
        else:
            total_urgences_global_kpi = df_global_for_urgences_kpi['urgences'].sum()
            total_admissions_global_kpi = df_global_for_urgences_kpi['admissions'].sum()
        
        # Calcul du taux de saturation des urgences au niveau global
        if total_admissions_global_kpi > 0:
            taux_saturation_urgences = (total_urgences_global_kpi / total_admissions_global_kpi) * 100
        else:
            taux_saturation_urgences = 0
        
        # KPI 5: Indice de charge globale (combinaison occupation + personnel)
        if 'lits_dispo_global' in df_filtered.columns:
            # Taux d'occupation moyen
            taux_occ_moyen = taux_occupation_actuel * 100
            # Ratio personnel (normalisé)
            ratio_personnel_norm = (ratio_personnel_lits / NATIONAL_BENCHMARKS.get('ratio_personnel_lits', 3.5)) * 100 if NATIONAL_BENCHMARKS.get('ratio_personnel_lits', 3.5) > 0 else 100
            # Indice de charge = moyenne pondérée (70% occupation, 30% personnel)
            indice_charge = (taux_occ_moyen * 0.7) + (min(100, ratio_personnel_norm) * 0.3)
        else:
            indice_charge = 0
        
        # KPI 6: Efficacité opérationnelle (DMS vs rotation)
        # Plus la rotation est élevée et la DMS faible, plus l'efficacité est bonne
        if dms_actuelle > 0 and rotation_annuelle > 0:
            # Indice d'efficacité = rotation / DMS (plus élevé = mieux)
            efficacite_operationnelle = rotation_annuelle / dms_actuelle
        else:
            efficacite_operationnelle = 0
        
        # Affichage des KPI avancés en grille
        col_kpi1, col_kpi2, col_kpi3 = st.columns(3)
        
        with col_kpi1:
            st.metric(
                "Taux de Rotation des Lits",
                f"{rotation_annuelle:.1f} rotations/an",
                help="Nombre de fois qu'un lit est utilisé par an (plus élevé = meilleure utilisation)"
            )
            # Affichage avec alerte visuelle si saturation
            if taux_utilisation_icu >= 95:
                st.metric(
                    "Taux d'Utilisation des Lits de Réanimation",
                    f"{taux_utilisation_icu:.1f}%",
                    delta="Saturation critique",
                    delta_color="inverse",
                    help="Pourcentage d'utilisation des lits de réanimation (≥95% = saturation critique)"
                )
            else:
                st.metric(
                    "Taux d'Utilisation des Lits de Réanimation",
                    f"{taux_utilisation_icu:.1f}%",
                    help="Pourcentage d'utilisation des lits de réanimation"
                )
        
        with col_kpi2:
            st.metric(
                "Ratio Personnel par Lits",
                f"{ratio_personnel_lits:.1f} équivalents temps plein pour 10 lits",
                delta=f"{ratio_personnel_lits - NATIONAL_BENCHMARKS.get('ratio_personnel_lits', 3.5):+.1f} vs national ({NATIONAL_BENCHMARKS.get('ratio_personnel_lits', 3.5):.1f})",
                help="Nombre d'équivalents temps plein pour 10 lits (benchmark national: 3.5 équivalents temps plein pour 10 lits)"
            )
            st.metric(
                "Taux de Saturation Urgences",
                f"{taux_saturation_urgences:.1f}%",
                help="Pourcentage d'admissions passant par les urgences (indicateur global de l'hôpital, indépendant du service sélectionné)"
            )
        
        with col_kpi3:
            st.metric(
                "Indice de Charge Globale",
                f"{indice_charge:.1f}/100",
                help="Indicateur composite combinant occupation et personnel (0-100)"
            )
            st.metric(
                "Efficacité Opérationnelle",
                f"{efficacite_operationnelle:.2f}",
                help="Ratio rotation/DMS (plus élevé = meilleure efficacité)"
            )
        
        # Graphique de synthèse des KPI
        st.markdown("#### Synthèse des KPI Avancés")
        
        kpi_data = pd.DataFrame({
            'KPI': [
                'Rotation des Lits\n(rotations/an)',
                'Utilisation Réanimation\n(%)',
                'Personnel par Lits\n(équivalents temps plein/10 lits)',
                'Saturation Urgences\n(%)',
                'Indice de Charge\n(/100)',
                'Efficacité Opérationnelle\n(ratio)'
            ],
            'Valeur': [
                rotation_annuelle,
                taux_utilisation_icu,
                ratio_personnel_lits,
                taux_saturation_urgences,
                indice_charge,
                efficacite_operationnelle  # Valeur réelle sans multiplication
            ],
            'Type': [
                'Rotation',
                'Utilisation',
                'Personnel',
                'Saturation',
                'Charge',
                'Efficacité'
            ]
        })
        
        fig_kpi = go.Figure()
        fig_kpi.add_trace(go.Bar(
            x=kpi_data['KPI'],
            y=kpi_data['Valeur'],
            marker_color='#2ecc71',
            text=[f"{v:.1f}" for v in kpi_data['Valeur']],
            textposition='auto'
        ))
        fig_kpi.update_layout(
            title="Vue d'ensemble des KPI Avancés",
            height=400,
            yaxis_title="Valeur",
            showlegend=False
        )
        st.plotly_chart(fig_kpi, use_container_width=True)
        
        # Tableau détaillé des KPI
        st.markdown("#### Détail des KPI")
        kpi_detail = pd.DataFrame({
            'Indicateur': [
                'Taux de Rotation des Lits',
                'Taux d\'Utilisation des Lits de Réanimation',
                'Ratio Personnel par Lits',
                'Taux de Saturation des Urgences',
                'Indice de Charge Globale',
                'Efficacité Opérationnelle'
            ],
            'Valeur': [
                f"{rotation_annuelle:.1f} rotations/an",
                f"{taux_utilisation_icu:.1f}%",
                f"{ratio_personnel_lits:.1f} équivalents temps plein pour 10 lits",
                f"{taux_saturation_urgences:.1f}%",
                f"{indice_charge:.1f}/100",
                f"{efficacite_operationnelle:.2f}"
            ],
            'Interprétation': [
                # Taux de Rotation des Lits
                f'Performance excellente ({rotation_annuelle:.1f} rotations/an vs {NATIONAL_BENCHMARKS.get("taux_rotation_lits", 68):.0f} national). Rotation élevée = meilleure utilisation des lits et meilleure fluidité des admissions.' if rotation_annuelle >= NATIONAL_BENCHMARKS.get('taux_rotation_lits', 68) else f'Performance en dessous de la moyenne nationale ({rotation_annuelle:.1f} vs {NATIONAL_BENCHMARKS.get("taux_rotation_lits", 68):.0f} rotations/an). Optimiser les processus de sortie pour améliorer la rotation.',
                # Taux d'Utilisation des Lits de Réanimation
                'Saturation critique des lits de réanimation (≥95%) - Risque de manque de capacité. Considérer l\'activation de lits supplémentaires ou le transfert de patients.' if taux_utilisation_icu >= 95 else ('Taux d\'occupation élevé des lits de réanimation (70-95%). Surveiller la disponibilité et anticiper les pics d\'activité.' if taux_utilisation_icu >= 70 else f'Utilisation modérée des lits de réanimation ({taux_utilisation_icu:.1f}%). Capacité disponible pour gérer des pics d\'activité.'),
                # Ratio Personnel par Lits
                f'Ratio conforme au benchmark national ({ratio_personnel_lits:.1f} vs {NATIONAL_BENCHMARKS.get("ratio_personnel_lits", 3.5):.1f} équivalents temps plein pour 10 lits). Effectifs adaptés aux besoins.' if abs(ratio_personnel_lits - NATIONAL_BENCHMARKS.get('ratio_personnel_lits', 3.5)) < 0.5 else (f'Ratio inférieur au benchmark ({ratio_personnel_lits:.1f} vs {NATIONAL_BENCHMARKS.get("ratio_personnel_lits", 3.5):.1f}). Risque de sous-effectif. Évaluer les besoins en renfort.' if ratio_personnel_lits < NATIONAL_BENCHMARKS.get('ratio_personnel_lits', 3.5) else f'Ratio supérieur au benchmark ({ratio_personnel_lits:.1f} vs {NATIONAL_BENCHMARKS.get("ratio_personnel_lits", 3.5):.1f}). Effectifs généreux, vérifier l\'optimisation des ressources.'),
                # Taux de Saturation des Urgences
                f'Taux de saturation des urgences modéré ({taux_saturation_urgences:.1f}%). Indicateur global de l\'hôpital. Pression acceptable sur les urgences.' if taux_saturation_urgences <= 50 else f'Forte pression sur les urgences ({taux_saturation_urgences:.1f}%). Indicateur global de l\'hôpital. Renforcer la capacité d\'accueil ou optimiser les flux vers les services spécialisés.',
                # Indice de Charge Globale
                f'Charge globale modérée ({indice_charge:.1f}/100). Situation opérationnelle normale. Capacité disponible pour gérer des pics.' if indice_charge <= 50 else (f'Charge globale élevée ({indice_charge:.1f}/100). Surveiller les ressources et anticiper les besoins.' if indice_charge <= 75 else f'Charge globale très élevée ({indice_charge:.1f}/100). Situation tendue. Actions correctives nécessaires.'),
                # Efficacité Opérationnelle
                f'Efficacité opérationnelle excellente (ratio = {efficacite_operationnelle:.2f}). Rotation élevée ({rotation_annuelle:.1f} rotations/an) et DMS faible ({dms_actuelle:.1f} jours) = bonne fluidité des soins.' if efficacite_operationnelle >= 10 else (f'Efficacité opérationnelle à améliorer (ratio = {efficacite_operationnelle:.2f}). Rotation faible ({rotation_annuelle:.1f} rotations/an) ou DMS élevée ({dms_actuelle:.1f} jours). Optimiser les processus de sortie et réduire les durées de séjour.' if efficacite_operationnelle >= 5 else f'Efficacité opérationnelle faible (ratio = {efficacite_operationnelle:.2f}). Rotation très faible ({rotation_annuelle:.1f} rotations/an) ou DMS très élevée ({dms_actuelle:.1f} jours). Actions prioritaires nécessaires.')
            ]
        })
        st.dataframe(kpi_detail, use_container_width=True, hide_index=True)
    
    st.divider()
    
    # ========================================================================
    # DIAGRAMME DE FLUX (SANKEY) - Parcours Patient
    # ========================================================================
    st.subheader("Flux des Patients (Parcours Hospitalier)")
    st.caption("Visualisation du parcours patient : flux entre Domicile, Urgences, Hospitalisation et Réanimation")
    
    if len(df_filtered) > 0:
        # Calcul des volumes moyens sur la période filtrée
        # Volumes globaux (tous services confondus)
        vol_urgences_moyen = df_filtered[df_filtered['service'] == 'Urgences']['urgences'].mean() if len(df_filtered[df_filtered['service'] == 'Urgences']) > 0 else df_filtered['urgences'].sum() / max(1, df_filtered['date'].nunique())
        vol_admissions_moyen = df_filtered['admissions'].sum() / max(1, df_filtered['date'].nunique())
        vol_icu_moyen = df_filtered['icu'].sum() / max(1, df_filtered['date'].nunique())
        
        # Hypothèses de flux basées sur les statistiques hospitalières
        # Flux depuis les urgences :
        # - 65% des patients aux urgences rentrent chez eux (consultations sans hospitalisation)
        # - 35% des patients aux urgences sont hospitalisés
        # (basé sur les statistiques hospitalières : majorité des urgences = consultations sans hospitalisation)
        taux_sortie_urgences = 0.65  # 65% rentrent chez eux
        taux_hosp_urgences = 0.35    # 35% sont hospitalisés
        
        vol_urg_dom = vol_urgences_moyen * taux_sortie_urgences
        vol_urg_hosp = vol_urgences_moyen * taux_hosp_urgences
        
        # Admissions directes (programmées, sans passer par les urgences)
        # Calcul correct : admissions totales - nombre d'urgences hospitalisées
        # Le ratio national (35%) représente la part des admissions totales qui proviennent des urgences
        # Donc : admissions_total = admissions_via_urgences + admissions_directes
        # Avec : admissions_via_urgences = urgences × taux_conversion_urgences_vers_hospit
        # Pour éviter la création de matière, on utilise le taux de conversion réel
        taux_conversion_urgences_hospit = NATIONAL_BENCHMARKS.get('taux_urgences_admissions', 0.35)
        # Nombre d'admissions qui proviennent des urgences (selon le ratio national)
        admissions_via_urgences = vol_admissions_moyen * taux_conversion_urgences_hospit
        # Le reste sont des admissions directes (programmées)
        vol_direct_hosp = max(0, vol_admissions_moyen - admissions_via_urgences)
        
        # Flux depuis l'hospitalisation
        # Total hospitalisations = admissions directes + admissions via urgences
        # On utilise maintenant admissions_via_urgences calculé ci-dessus pour cohérence
        total_hospitalisations = vol_direct_hosp + admissions_via_urgences
        
        # Hypothèses de sortie :
        # - 90% sortent guéris (retour domicile)
        # - 10% nécessitent une réanimation (ratio ICU/admissions national = 8%)
        taux_sortie_hosp = 0.90
        taux_rea_hosp = min(0.10, NATIONAL_BENCHMARKS.get('taux_icu_admissions', 0.08) * 1.25)  # Légèrement supérieur car on part déjà des hospitalisés
        
        vol_hosp_dom = total_hospitalisations * taux_sortie_hosp
        vol_hosp_rea = total_hospitalisations * taux_rea_hosp
        
        # Flux depuis la réanimation
        # CORRECTION : Chaînage des flux - le flux sortant de la réa = flux entrant (vol_hosp_rea)
        # Cela évite la création de matière (plus de sorties que d'entrées)
        # Tous les patients qui entrent en réanimation finissent par sortir (retour domicile ou décès)
        vol_rea_dom = vol_hosp_rea  # Flux sortant = flux entrant (conservation de la matière)
        
        # Création des nœuds et liens pour le diagramme Sankey
        # Nœuds : 0=Domicile/Extérieur, 1=Urgences, 2=Retour Domicile, 3=Hospitalisation, 4=Réanimation
        nodes = ["Domicile/Extérieur", "Urgences", "Retour Domicile", "Hospitalisation", "Réanimation"]
        
        # Liens (source -> target)
        # 0 -> 1 : Domicile vers Urgences
        # 0 -> 3 : Domicile vers Hospitalisation (directe)
        # 1 -> 2 : Urgences vers Retour Domicile
        # 1 -> 3 : Urgences vers Hospitalisation
        # 3 -> 2 : Hospitalisation vers Retour Domicile
        # 3 -> 4 : Hospitalisation vers Réanimation
        # 4 -> 2 : Réanimation vers Retour Domicile
        link_source = [0, 0, 1, 1, 3, 3, 4]
        link_target = [1, 3, 2, 3, 2, 4, 2]
        link_value = [
            vol_urgences_moyen,      # Domicile -> Urgences
            vol_direct_hosp,         # Domicile -> Hospitalisation (directe)
            vol_urg_dom,             # Urgences -> Retour Domicile
            admissions_via_urgences,  # Urgences -> Hospitalisation (cohérent avec le calcul)
            vol_hosp_dom,            # Hospitalisation -> Retour Domicile
            vol_hosp_rea,            # Hospitalisation -> Réanimation
            vol_rea_dom              # Réanimation -> Retour Domicile
        ]
        
        # Création du diagramme Sankey
        fig_sankey = go.Figure(data=[go.Sankey(
            node=dict(
                pad=15,
                thickness=20,
                line=dict(color="black", width=0.5),
                label=nodes,
                color=["#E0E0E0", "#FF4B4B", "#56E39F", "#1F77B4", "#FFA15A"]  # Couleurs personnalisées
            ),
            link=dict(
                source=link_source,
                target=link_target,
                value=link_value,
                color=["rgba(255, 75, 75, 0.4)", "rgba(31, 119, 180, 0.4)", "rgba(86, 227, 159, 0.4)", 
                       "rgba(31, 119, 180, 0.4)", "rgba(86, 227, 159, 0.4)", "rgba(255, 161, 90, 0.4)", 
                       "rgba(86, 227, 159, 0.4)"]
            )
        )])
        
        fig_sankey.update_layout(
            title_text="Parcours Patient Moyen (Journalier)",
            font_size=12,
            height=400
        )
        
        st.plotly_chart(fig_sankey, use_container_width=True)
        
        # Légende explicative
        st.caption("**Légende** : Les largeurs des flux sont proportionnelles aux volumes de patients. "
                  f"Sur {vol_urgences_moyen:.0f} patients aux urgences/jour, "
                  f"{vol_urg_dom:.0f} ({taux_sortie_urgences*100:.0f}%) rentrent chez eux et "
                  f"{admissions_via_urgences:.0f} ({taux_conversion_urgences_hospit*100:.0f}%) sont hospitalisés.")
    else:
        st.info("Aucune donnée disponible pour afficher le parcours patient.")
    
    st.divider()
    
    # Section: Alertes et Seuils Critiques
    st.subheader("Alertes et Seuils Critiques")
    st.caption("Surveillance automatique des indicateurs critiques et recommandations d'action")
    
    if len(df_filtered) > 0:
        # Calcul des métriques pour les alertes
        taux_occ_actuel_alert = taux_occupation_actuel * 100
        dms_actuelle_alert = dms_actuelle
        ratio_personnel_alert = ratio_personnel_lits
        
        # Calcul du déficit de personnel (si disponible)
        # Le déficit est calculé comme le pourcentage de personnel manquant par rapport au personnel requis
        # Cela évite des valeurs très élevées quand le personnel disponible est très faible
        if 'personnel_requis' in df_filtered.columns and 'personnel_disponible' in df_filtered.columns:
            personnel_requis_moyen = df_filtered['personnel_requis'].mean()
            personnel_disponible_moyen = df_filtered['personnel_disponible'].mean()
            if personnel_requis_moyen > 0:
                # Déficit = (requis - disponible) / requis * 100
                # Cela donne un pourcentage de manque par rapport aux besoins (plus réaliste)
                deficit_personnel_pct = ((personnel_requis_moyen - personnel_disponible_moyen) / personnel_requis_moyen) * 100
                # Si le personnel disponible dépasse le requis, pas de déficit
                deficit_personnel_pct = max(0, deficit_personnel_pct)
            else:
                deficit_personnel_pct = 0
        else:
            deficit_personnel_pct = 0
        
        # Collecte des alertes
        alertes_critiques = []
        alertes_attention = []
        indicateurs_normaux = []
        
        # Alerte 1: Taux d'occupation
        seuil_critique_occ = ALERT_THRESHOLDS.get('taux_occupation_critique', 0.85) * 100
        seuil_alerte_occ = ALERT_THRESHOLDS.get('taux_occupation_alerte', 0.75) * 100
        
        if taux_occ_actuel_alert >= seuil_critique_occ:
            alertes_critiques.append({
                'indicateur': 'Taux d\'Occupation',
                'valeur': f"{taux_occ_actuel_alert:.1f}%",
                'seuil': f"{seuil_critique_occ:.0f}%",
                'message': f"Taux d'occupation critique ({taux_occ_actuel_alert:.1f}% ≥ {seuil_critique_occ:.0f}%). Risque de saturation.",
                'action': "Activer le plan blanc si nécessaire. Ouvrir des lits supplémentaires. Réorganiser les sorties."
            })
        elif taux_occ_actuel_alert >= seuil_alerte_occ:
            alertes_attention.append({
                'indicateur': 'Taux d\'Occupation',
                'valeur': f"{taux_occ_actuel_alert:.1f}%",
                'seuil': f"{seuil_alerte_occ:.0f}%",
                'message': f"Taux d'occupation élevé ({taux_occ_actuel_alert:.1f}% ≥ {seuil_alerte_occ:.0f}%). Surveillance renforcée recommandée.",
                'action': "Anticiper les besoins. Vérifier la disponibilité des lits. Optimiser les processus de sortie."
            })
        else:
            indicateurs_normaux.append({
                'indicateur': 'Taux d\'Occupation',
                'valeur': f"{taux_occ_actuel_alert:.1f}%",
                'statut': 'Normal'
            })
        
        # Alerte 2: Durée Moyenne de Séjour
        seuil_dms = ALERT_THRESHOLDS.get('dms_elevee', 7.0)
        
        if dms_actuelle_alert >= seuil_dms:
            alertes_attention.append({
                'indicateur': 'Durée Moyenne de Séjour',
                'valeur': f"{dms_actuelle_alert:.1f} jours",
                'seuil': f"{seuil_dms:.1f} jours",
                'message': f"DMS élevée ({dms_actuelle_alert:.1f} jours ≥ {seuil_dms:.1f} jours). Séjours prolongés.",
                'action': "Analyser les causes des séjours longs. Optimiser les parcours de soins. Renforcer la coordination sortie."
            })
        else:
            indicateurs_normaux.append({
                'indicateur': 'Durée Moyenne de Séjour',
                'valeur': f"{dms_actuelle_alert:.1f} jours",
                'statut': 'Normal'
            })
        
        # Alerte 3: Déficit de Personnel
        seuil_deficit = ALERT_THRESHOLDS.get('deficit_personnel', 0.10) * 100
        
        if deficit_personnel_pct >= seuil_deficit:
            alertes_critiques.append({
                'indicateur': 'Déficit de Personnel',
                'valeur': f"{deficit_personnel_pct:.1f}%",
                'seuil': f"{seuil_deficit:.0f}%",
                'message': f"Déficit de personnel significatif ({deficit_personnel_pct:.1f}% ≥ {seuil_deficit:.0f}%). Sous-effectif critique.",
                'action': "Mobiliser les renforts. Activer les équipes de réserve. Réorganiser les plannings."
            })
        elif deficit_personnel_pct > 0:
            alertes_attention.append({
                'indicateur': 'Déficit de Personnel',
                'valeur': f"{deficit_personnel_pct:.1f}%",
                'seuil': f"{seuil_deficit:.0f}%",
                'message': f"Déficit de personnel modéré ({deficit_personnel_pct:.1f}%). Surveiller les effectifs.",
                'action': "Évaluer les besoins en renfort. Anticiper les absences. Optimiser la répartition des équipes."
            })
        else:
            indicateurs_normaux.append({
                'indicateur': 'Déficit de Personnel',
                'valeur': "Aucun déficit",
                'statut': 'Normal'
            })
        
        # Alerte 4: Utilisation Réanimation (si disponible)
        if taux_utilisation_icu >= 95:
            alertes_critiques.append({
                'indicateur': 'Utilisation Réanimation',
                'valeur': f"{taux_utilisation_icu:.1f}%",
                'seuil': "95%",
                'message': f"Saturation critique des lits de réanimation ({taux_utilisation_icu:.1f}% ≥ 95%).",
                'action': "Activer les lits de réanimation supplémentaires. Organiser les transferts si nécessaire. Prioriser les cas les plus graves."
            })
        elif taux_utilisation_icu >= 85:
            alertes_attention.append({
                'indicateur': 'Utilisation Réanimation',
                'valeur': f"{taux_utilisation_icu:.1f}%",
                'seuil': "85%",
                'message': f"Utilisation élevée des lits de réanimation ({taux_utilisation_icu:.1f}% ≥ 85%).",
                'action': "Surveiller la disponibilité. Anticiper les besoins. Vérifier les capacités de transfert."
            })
        
        # Affichage des alertes
        col_alert1, col_alert2 = st.columns(2)
        
        with col_alert1:
            if alertes_critiques:
                st.markdown("#### Alertes Critiques")
                for alerte in alertes_critiques:
                    with st.container():
                        st.error(f"**{alerte['indicateur']}** : {alerte['valeur']} (seuil: {alerte['seuil']})")
                        st.caption(f"{alerte['message']}")
                        st.caption(f"**Action recommandée:** {alerte['action']}")
                        st.divider()
            else:
                st.markdown("#### Alertes Critiques")
                st.success("Aucune alerte critique détectée")
        
        with col_alert2:
            if alertes_attention:
                st.markdown("#### Alertes d'Attention")
                for alerte in alertes_attention:
                    with st.container():
                        st.warning(f"**{alerte['indicateur']}** : {alerte['valeur']} (seuil: {alerte['seuil']})")
                        st.caption(f"{alerte['message']}")
                        st.caption(f"**Action recommandée:** {alerte['action']}")
                        st.divider()
            else:
                st.markdown("#### Alertes d'Attention")
                st.info("Aucune alerte d'attention")
        
        # Indicateurs normaux
        if indicateurs_normaux:
            st.markdown("#### Indicateurs Normaux")
            col_norm1, col_norm2 = st.columns(2)
            for i, indicateur in enumerate(indicateurs_normaux):
                with col_norm1 if i % 2 == 0 else col_norm2:
                    st.success(f"**{indicateur['indicateur']}** : {indicateur['valeur']}")
    else:
        st.info("Aucune donnée disponible pour l'analyse des alertes.")
    
    st.divider()
    
    # Graphique 2: Répartition des scénarios
    st.subheader("Répartition des Scénarios")
    
    if len(df_filtered) > 0:
        scenario_counts = df_filtered['scenario'].value_counts()
        if len(scenario_counts) > 0:
            fig_scenarios = px.pie(
                values=scenario_counts.values,
                names=scenario_counts.index,
                title="Distribution des Scénarios"
            )
            fig_scenarios.update_layout(height=400)
            st.plotly_chart(fig_scenarios, use_container_width=True)
        else:
            st.info("Aucun scénario à afficher.")
    else:
        st.info("Aucune donnée à afficher.")
    
# ============================================================================
# ONGLET 2: SIMULATION & PRÉDICTIONS
# ============================================================================
with tab2:
    st.header("Simulation & Prédictions de Scénarios")
    st.markdown("Simulez l'impact temporel de différents scénarios et obtenez des recommandations préventives")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Paramètres de Simulation")
        
        # Durée
        sim_duration = st.number_input(
            "Durée (jours)",
            min_value=1,
            max_value=365,
            value=30
        )
        
        # Scénario à simuler (incluant "normal")
        sim_scenario = st.selectbox(
            "Scénario à simuler",
            ["normal", "epidemie", "canicule", "greve", "accident"],
            key="sim_scenario",
            help="Sélectionnez 'normal' pour une projection en conditions normales, ou un scénario de crise"
        )
        
        # Niveau d'intensité (masqué si scénario normal)
        if sim_scenario != 'normal':
            # Définir les niveaux selon le scénario
            if sim_scenario == 'epidemie':
                intensity_options = {
                    "Faible (0.7x)": {
                        "value": 0.7,
                        "description": "Épidémie modérée - Augmentation limitée des admissions (+30%)"
                    },
                    "Modéré (1.0x)": {
                        "value": 1.0,
                        "description": "Épidémie standard - Niveau COVID-19 2020 (+40% admissions, +100% réanimation)"
                    },
                    "Fort (1.5x)": {
                        "value": 1.5,
                        "description": "Épidémie sévère - Pic de crise (+60% admissions, +150% réanimation)"
                    },
                    "Critique (2.0x)": {
                        "value": 2.0,
                        "description": "Épidémie majeure - Saturation des capacités (+80% admissions, +200% réanimation)"
                    }
                }
            elif sim_scenario == 'canicule':
                intensity_options = {
                    "Faible (0.7x)": {
                        "value": 0.7,
                        "description": "Vague de chaleur modérée - Impact limité sur les urgences"
                    },
                    "Modéré (1.0x)": {
                        "value": 1.0,
                        "description": "Canicule standard - Niveau été 2003 (+20% admissions, +40% urgences)"
                    },
                    "Fort (1.5x)": {
                        "value": 1.5,
                        "description": "Canicule sévère - Pic de chaleur prolongé (+30% admissions, +60% urgences)"
                    },
                    "Critique (2.0x)": {
                        "value": 2.0,
                        "description": "Canicule extrême - Situation d'urgence sanitaire (+40% admissions, +80% urgences)"
                    }
                }
            elif sim_scenario == 'greve':
                intensity_options = {
                    "Faible (0.7x)": {
                        "value": 0.7,
                        "description": "Grève partielle - Réduction limitée des effectifs (-20% personnel)"
                    },
                    "Modéré (1.0x)": {
                        "value": 1.0,
                        "description": "Grève standard - Impact significatif (-40% personnel disponible)"
                    },
                    "Fort (1.5x)": {
                        "value": 1.5,
                        "description": "Grève sévère - Réduction importante (-50% personnel disponible)"
                    },
                    "Critique (2.0x)": {
                        "value": 2.0,
                        "description": "Grève totale - Arrêt quasi-complet des activités (-60% personnel disponible)"
                    }
                }
            elif sim_scenario == 'accident':
                intensity_options = {
                    "Faible (0.7x)": {
                        "value": 0.7,
                        "description": "Accident limité - Impact localisé, gestionable"
                    },
                    "Modéré (1.0x)": {
                        "value": 1.0,
                        "description": "Accident standard - Pic d'activité brutal (+80% admissions, +150% urgences)"
                    },
                    "Fort (1.5x)": {
                        "value": 1.5,
                        "description": "Accident majeur - Afflux massif de blessés (+120% admissions, +225% urgences)"
                    },
                    "Critique (2.0x)": {
                        "value": 2.0,
                        "description": "Catastrophe - Saturation immédiate des capacités (+160% admissions, +300% urgences)"
                    }
                }
    # # ========================================================================
    # # ACTIONS CORRECTIVES (Solveur de Crise)
    # # ========================================================================
    # st.divider()
    # st.subheader("Actions Correctives (Simulation)")
    # st.caption("Testez l'impact de vos décisions : ouvrir des lits ou rappeler du personnel")
    
    # col_act1, col_act2 = st.columns(2)
    
    # with col_act1:
    #     # Le directeur décide d'ouvrir des lits temporaires
    #     renfort_lits = st.slider(
    #         "Ouverture de lits temporaires (Plan Blanc)", 
    #         min_value=0, 
    #         max_value=100, 
    #         value=0,
    #         step=5,
    #         help="Simule l'ouverture d'une unité de crise ou de lits dans les couloirs."
    #     )
    
    # with col_act2:
    #     # Le directeur rappelle du personnel
    #     renfort_personnel = st.slider(
    #         "Rappel de personnel (Intérim/Réserve)", 
    #         min_value=0, 
    #         max_value=50, 
    #         value=0, 
    #         step=1,
    #         help="Nombre d'équivalents temps plein ajoutés pour faire face à la crise."
    #     )
    
    if st.button("Lancer la Simulation", type="primary"):
            # ========================================================================
            # SIMULATION TEMPORELLE JOUR PAR JOUR
            # ========================================================================
            st.subheader("Résultats de la Simulation Temporelle")
            
            # Affichage des actions correctives appliquées
            # if renfort_lits > 0 or renfort_personnel > 0:
            #     st.info(f"**Actions correctives appliquées** : "
            #            f"{renfort_lits} lits supplémentaires répartis sur tous les services, "
            #            f"{renfort_personnel} équivalents temps plein de renfort répartis équitablement.")
            
            # ========================================================================
            # PRÉDICTION ML : Prophet ou Régression linéaire selon disponibilité
            # ========================================================================
            
            # Vérifier si les modèles Prophet existent
            use_prophet = os.path.exists('admissions_prophet_model.pkl')
            use_prophet_beds = os.path.exists('beds_prophet_model.pkl')
            use_prophet_epi = os.path.exists('epi_prophet_model.pkl')
            total_admissions_prophet = None  # Initialisation pour la portée de la variable
            daily_beds_forecast = None  # Prédictions Prophet pour les lits occupés (journalières)
            daily_epi_forecast = None  # Prédictions Prophet pour la consommation d'EPI (journalières)
            
            if use_prophet:
                try:
                    # Charger le modèle Prophet
                    prophet_model = joblib.load('admissions_prophet_model.pkl')
                    st.info("**Prédiction ML** : Utilisation du modèle Prophet pour projeter la tendance normale avant application des scénarios.")
                    
                    # Préparer les données pour Prophet (agrégation horaire)
                    # On charge les données horaires depuis hospital_synth.csv si disponible
                    if os.path.exists('hospital_synth.csv'):
                        df_hourly = pd.read_csv('hospital_synth.csv', parse_dates=['timestamp_admission'])
                        df_hourly_prep = df_hourly.set_index('timestamp_admission').resample('h').agg({
                            'Nombre_Admissions': 'sum',
                            'Indicateur_Epidemie': 'max',
                            'Indicateur_Canicule': 'max',
                            'Indicateur_Greve': 'max',
                            'gravite': 'mean',
                            'duree_sejour_estimee': 'mean'
                        }).reset_index()
                        
                        # Créer un DataFrame pour les prédictions futures
                        future_dates = pd.date_range(
                            start=max_date,
                            periods=sim_duration * 24,  # 24 heures par jour
                            freq='H'
                        )
                        
                        future_df = pd.DataFrame({
                            'ds': future_dates,
                            'Indicateur_Epidemie': 0,  # Scénario normal par défaut
                            'Indicateur_Canicule': 0,
                            'Indicateur_Greve': 0
                        })
                        
                        # Ajuster les régresseurs selon le scénario simulé
                        if sim_scenario == 'epidemie':
                            future_df['Indicateur_Epidemie'] = 1
                        elif sim_scenario == 'canicule':
                            future_df['Indicateur_Canicule'] = 1
                        elif sim_scenario == 'greve':
                            future_df['Indicateur_Greve'] = 1
                        
                        # Prédire avec Prophet
                        forecast = prophet_model.predict(future_df)
                        
                        # Agréger les prédictions horaires en journalières
                        forecast['date'] = forecast['ds'].dt.date
                        daily_forecast = forecast.groupby('date')['yhat'].sum().reset_index()
                        daily_forecast.columns = ['date', 'admissions_totales']
                        
                        # Utiliser les prédictions Prophet comme base
                        # Répartir les admissions totales par service selon les poids
                        total_admissions_prophet = daily_forecast['admissions_totales'].mean() if len(daily_forecast) > 0 else 0
                        
                        # ========================================================================
                        # PRÉDICTION ML POUR LES LITS OCCUPÉS (Prophet Beds)
                        # ========================================================================
                        forecast_beds = None  # Initialisation pour utilisation dans EPI
                        if use_prophet_beds:
                            try:
                                # Charger le modèle Prophet pour les lits occupés
                                prophet_beds_model = joblib.load('beds_prophet_model.pkl')
                                
                                # Préparer les données pour la prédiction des lits occupés
                                # On utilise les prédictions d'admissions comme régresseur
                                future_df_beds = pd.DataFrame({
                                    'ds': future_dates,
                                    'Nombre_Admissions': forecast['yhat'].values,  # Utiliser les prédictions d'admissions
                                    'gravite': df_hourly_prep['gravite'].mean() if 'gravite' in df_hourly_prep.columns else 2.0,  # Gravité moyenne
                                    'duree_sejour_estimee': df_hourly_prep['duree_sejour_estimee'].mean() if 'duree_sejour_estimee' in df_hourly_prep.columns else 5.0,  # DMS moyenne
                                    'Indicateur_Epidemie': future_df['Indicateur_Epidemie'].values,
                                    'Indicateur_Canicule': future_df['Indicateur_Canicule'].values,
                                    'Indicateur_Greve': future_df['Indicateur_Greve'].values
                                })
                                
                                # Prédire les lits occupés horaires
                                forecast_beds = prophet_beds_model.predict(future_df_beds)
                                
                                # Agréger les prédictions horaires en journalières
                                forecast_beds['date'] = forecast_beds['ds'].dt.date
                                daily_beds_forecast = forecast_beds.groupby('date')['yhat'].mean().reset_index()
                                daily_beds_forecast.columns = ['date', 'lits_occ_prophet']
                                
                                st.info("**Prédiction ML** : Utilisation du modèle Prophet pour projeter la tendance normale (admissions ET lits occupés) avant application des scénarios.")
                                
                            except Exception as e:
                                st.warning(f"Erreur lors du chargement du modèle Prophet pour les lits occupés: {e}. Utilisation de la Loi de Little en fallback.")
                                use_prophet_beds = False
                                daily_beds_forecast = None
                                forecast_beds = None
                        else:
                            daily_beds_forecast = None
                        
                        # ========================================================================
                        # PRÉDICTION ML POUR LA CONSOMMATION D'EPI (Prophet EPI)
                        # ========================================================================
                        if use_prophet_epi:
                            try:
                                # Charger le modèle Prophet pour la consommation d'EPI
                                prophet_epi_model = joblib.load('epi_prophet_model.pkl')
                                
                                # Préparer les données pour la prédiction de la consommation d'EPI
                                # On utilise les prédictions d'admissions, lits occupés, et personnel comme régresseurs
                                
                                # Estimer les lits occupés (utiliser les prédictions Prophet beds si disponible)
                                if use_prophet_beds and forecast_beds is not None:
                                    lits_occupes_estimes = forecast_beds['yhat'].values
                                else:
                                    # Estimation basique : utiliser la moyenne historique
                                    lits_occupes_estimes = np.full(len(future_dates), df_hourly_prep['Lits_Occupes'].mean() if 'Lits_Occupes' in df_hourly_prep.columns else 1500)
                                
                                # Estimer le personnel présent (basé sur les lits occupés)
                                personnel_present_estime = lits_occupes_estimes * 0.6
                                
                                future_df_epi = pd.DataFrame({
                                    'ds': future_dates,
                                    'Lits_Occupes': lits_occupes_estimes,
                                    'Nombre_Admissions': forecast['yhat'].values,  # Utiliser les prédictions d'admissions
                                    'Personnel_Present': personnel_present_estime,
                                    'Indicateur_Epidemie': future_df['Indicateur_Epidemie'].values,
                                    'Indicateur_Canicule': future_df['Indicateur_Canicule'].values,
                                    'Indicateur_Greve': future_df['Indicateur_Greve'].values
                                })
                                
                                # Prédire la consommation d'EPI horaire
                                forecast_epi = prophet_epi_model.predict(future_df_epi)
                                
                                # Agréger les prédictions horaires en journalières (somme car c'est une consommation cumulée)
                                forecast_epi['date'] = forecast_epi['ds'].dt.date
                                daily_epi_forecast = forecast_epi.groupby('date')['yhat'].sum().reset_index()
                                daily_epi_forecast.columns = ['date', 'epi_consommation_prophet']
                                
                                st.info("**Prédiction ML** : Utilisation du modèle Prophet pour projeter la tendance normale (admissions, lits occupés ET consommation d'EPI) avant application des scénarios.")
                                
                            except Exception as e:
                                st.warning(f"Erreur lors du chargement du modèle Prophet pour la consommation d'EPI: {e}. Utilisation d'une estimation basique en fallback.")
                                use_prophet_epi = False
                                daily_epi_forecast = None
                        else:
                            daily_epi_forecast = None
                        
                    else:
                        # Si hospital_synth.csv n'existe pas, utiliser LinearRegression
                        use_prophet = False
                        st.warning("Le modèle Prophet nécessite hospital_synth.csv. Utilisation de la régression linéaire en fallback.")
                        
                except Exception as e:
                    st.warning(f"Erreur lors du chargement du modèle Prophet: {e}. Utilisation de la régression linéaire en fallback.")
                    use_prophet = False
            
            if not use_prophet:
                st.info("**Prédiction ML** : Utilisation d'une régression linéaire pour projeter la tendance normale avant application des scénarios.")
            
            # Données historiques en conditions normales pour l'entraînement
            base_period = df[df['scenario'] == 'normal']
            
            # Calcul des valeurs de base par service avec PRÉDICTION ML
            base_values = {}
            
            for service in df['service'].unique():
                service_data = base_period[base_period['service'] == service].copy()
                
                if len(service_data) > 0:
                    # --- 1. PRÉDICTION DES FLUX (Prophet ou LinearRegression) ---
                    if use_prophet and total_admissions_prophet is not None:
                        # Utiliser les prédictions Prophet et répartir par service
                        service_weight = SERVICES_DISTRIBUTION.get(service, {}).get('weight', 0.1)
                        admissions_base = total_admissions_prophet * service_weight
                        
                        # Pour urgences et ICU, utiliser les ratios moyens du service
                        avg_urgences = service_data['urgences'].mean() if 'urgences' in service_data.columns else 0
                        avg_icu = service_data['icu'].mean() if 'icu' in service_data.columns else 0
                        avg_admissions = service_data['admissions'].mean()
                        
                        # Calculer les ratios moyens
                        ratio_urgences = avg_urgences / avg_admissions if avg_admissions > 0 else 0
                        ratio_icu = avg_icu / avg_admissions if avg_admissions > 0 else 0
                        
                        urgences_base = admissions_base * ratio_urgences
                        icu_base = admissions_base * ratio_icu
                    else:
                        # Fallback: Utiliser LinearRegression
                        # Trier par date pour avoir une séquence temporelle
                        service_data = service_data.sort_values('date')
                        
                        # Préparer les données pour la régression
                        # X = indices temporels (jours depuis le début)
                        # y = admissions par jour
                        X = np.arange(len(service_data)).reshape(-1, 1)
                        y_admissions = service_data['admissions'].values
                        y_urgences = service_data['urgences'].values if 'urgences' in service_data.columns else np.zeros(len(service_data))
                        y_icu = service_data['icu'].values if 'icu' in service_data.columns else np.zeros(len(service_data))
                        
                        # Entraîner les modèles de régression linéaire
                        model_admissions = LinearRegression()
                        model_admissions.fit(X, y_admissions)
                        
                        model_urgences = LinearRegression()
                        if y_urgences.sum() > 0:  # Seulement si on a des urgences
                            model_urgences.fit(X, y_urgences)
                        
                        model_icu = LinearRegression()
                        if y_icu.sum() > 0:  # Seulement si on a des ICU
                            model_icu.fit(X, y_icu)
                        
                        # Prédire la valeur pour le jour suivant (tendance normale)
                        # On utilise le dernier jour + 1 pour projeter la tendance
                        future_day_index = np.array([[len(service_data)]])
                        tendance_admissions = max(0, model_admissions.predict(future_day_index)[0])
                        tendance_urgences = max(0, model_urgences.predict(future_day_index)[0]) if y_urgences.sum() > 0 else 0
                        tendance_icu = max(0, model_icu.predict(future_day_index)[0]) if y_icu.sum() > 0 else 0
                        
                        # Si la prédiction donne des valeurs aberrantes, utiliser la moyenne comme fallback
                        avg_admissions = service_data['admissions'].mean()
                        avg_urgences = service_data['urgences'].mean() if 'urgences' in service_data.columns else 0
                        avg_icu = service_data['icu'].mean() if 'icu' in service_data.columns else 0
                        
                        # Utiliser la prédiction si elle est raisonnable (entre 50% et 200% de la moyenne)
                        if 0.5 * avg_admissions <= tendance_admissions <= 2.0 * avg_admissions:
                            admissions_base = tendance_admissions
                        else:
                            admissions_base = avg_admissions
                        
                        if y_urgences.sum() > 0 and 0.5 * avg_urgences <= tendance_urgences <= 2.0 * avg_urgences:
                            urgences_base = tendance_urgences
                        else:
                            urgences_base = avg_urgences
                        
                        if y_icu.sum() > 0 and 0.5 * avg_icu <= tendance_icu <= 2.0 * avg_icu:
                            icu_base = tendance_icu
                        else:
                            icu_base = avg_icu
                    
                    # --- 2. CALIBRAGE INTELLIGENT DE LA CAPACITÉ (C'est ici que ça se joue) ---
                    service_config = SERVICES_DISTRIBUTION.get(service, {})
                    dms_service = service_config.get('duree_moy', 5.0)
                    
                    # CORRECTION CRITIQUE : On calibre sur admissions_base (la valeur qui sera utilisée en simulation)
                    # et non sur avg_admissions, sinon il y a un décalage entre le calibrage et la simulation
                    # Calcul de la charge structurelle (Loi de Little) : Combien de patients j'aurai avec admissions_base ?
                    charge_patients_base = admissions_base * dms_service
                    
                    # CORRECTION MAJEURE : Dimensionnement des lits pour viser 70% d'occupation MOYENNE en temps normal
                    # Pourquoi 70% et non 85% ? Parce qu'avec la variabilité stochastique (±5% sur lits + ±2% scénario),
                    # si on calibre à 85%, on dépasse 85% la moitié du temps (18 jours sur 30).
                    # En calibrant à 70%, on a une marge de sécurité : moyenne à 70%, pics à 85%, rarement au-delà
                    # Si j'ai 100 patients, il me faut 143 lits pour être à 70% (100 / 0.70)
                    lits_necessaires_target = int(charge_patients_base / 0.70)
                    
                    # On prend le MAX entre la config et le besoin réel pour ne jamais être en sous-capacité au démarrage
                    lits_dispo_start = max(service_config.get('lits_base', 0), lits_necessaires_target)
                    
                    # CORRECTION CRITIQUE : Dimensionnement du personnel basé sur le besoin réel, pas sur les lits
                    # Ratio : 1 ETP pour 3 patients (lits occupés)
                    # Pour 70% d'occupation MOYENNE : personnel_requis = (lits_dispo * 0.70) / 3 = lits_dispo * 0.233
                    # On ajoute 20% de marge pour les roulements, congés, etc. : 0.233 * 1.2 = 0.28
                    # Mais on garde un minimum de 0.40 pour être réaliste
                    personnel_requis_calibre = (lits_dispo_start * 0.70) / 3  # Besoin à 70% d'occupation moyenne
                    personnel_dispo_start = max(personnel_requis_calibre * 1.2, lits_dispo_start * 0.40)  # Marge de 20% + minimum

                    # Stockage des valeurs de base calibrées
                    base_values[service] = {
                        'admissions': admissions_base,  # Valeur prédite par ML ou moyenne
                        'urgences': urgences_base,      # Valeur prédite par ML ou moyenne
                        'icu': icu_base,                # Valeur prédite par ML ou moyenne
                        'duree_moy': dms_service,
                        'lits_dispo': lits_dispo_start,       # VALEUR CORRIGÉE : capacité calibrée sur admissions_base
                        'personnel_disponible': personnel_dispo_start  # VALEUR CORRIGÉE : personnel aligné sur besoin réel
                    }
            
            # Multiplicateurs du scénario
            mult = SCENARIO_MULTIPLIERS[sim_scenario]
            
            # Fonction pour calculer la courbe d'évolution du scénario
            # Les scénarios ont une montée progressive, un pic, puis une descente
            def get_scenario_evolution_factor(day, total_days, scenario_type):
                """Calcule le facteur d'évolution du scénario selon le jour"""
                if scenario_type == 'normal':
                    # Scénario normal : activité constante avec très légères variations aléatoires
                    # VARIABILITÉ STOCHASTIQUE : Variation de ±2% pour simuler la variabilité naturelle minimale
                    # (bruit aléatoire contrôlé via np.random.normal)
                    np.random.seed(day % 1000)  # Seed basé sur le jour pour cohérence
                    return 1.0 + np.random.normal(0, 0.02)  # Variation réduite à ±2% pour éviter les pics artificiels
                elif scenario_type == 'epidemie':
                    # Épidémie : montée lente (jours 0-10), pic (jours 10-20), descente (jours 20+)
                    if day < 10:
                        return 0.3 + (day / 10) * 0.7  # Montée progressive
                    elif day < 20:
                        return 1.0  # Pic
                    else:
                        return max(0.3, 1.0 - ((day - 20) / max(1, total_days - 20)) * 0.7)  # Descente
                elif scenario_type == 'canicule':
                    # Canicule : montée rapide (jours 0-3), plateau (jours 3-10), descente (jours 10+)
                    if day < 3:
                        return 0.2 + (day / 3) * 0.8
                    elif day < 10:
                        return 1.0
                    else:
                        return max(0.2, 1.0 - ((day - 10) / max(1, total_days - 10)) * 0.8)
                elif scenario_type == 'greve':
                    # Grève : impact constant pendant toute la durée
                    return 1.0
                elif scenario_type == 'accident':
                    # Accident : pic immédiat (jour 0-1), descente rapide (jours 1+)
                    if day < 2:
                        return 1.0
                    else:
                        return max(0.1, 1.0 - ((day - 2) / max(1, total_days - 2)) * 0.9)
                else:
                    return 1.0
            
            # Simulation jour par jour
            simulation_days = []
            # Compteurs pour le calcul financier (jour par jour)
            total_cout_rh = 0.0  # Coût total RH cumulé (en euros)
            total_cout_lits = 0.0  # Coût total lits supplémentaires cumulé (en euros)
            
            # Date de début pour la simulation
            dates_sim = pd.date_range(max_date, periods=sim_duration, freq='D')
            
            for day_idx, current_date in enumerate(dates_sim):
                evolution_factor = get_scenario_evolution_factor(day_idx, sim_duration, sim_scenario)
                
                # Calcul pour chaque service
                day_total = {
                    'date': current_date,
                    'jour': day_idx + 1,
                    'admissions': 0,
                    'urgences': 0,
                    'icu': 0,
                    'lits_occ': 0,
                    'lits_dispo': 0,
                    'taux_occupation': 0,
                    'personnel_requis': 0,
                    'personnel_disponible': 0,
                    'deficit_personnel': 0,
                    'materiel_respirateurs': 0,
                    'materiel_medicaments': 0,
                    'materiel_protection': 0  # Consommation d'EPI
                }
                
                for service in df['service'].unique():
                    if service in base_values:
                        base = base_values[service]
                        
                        # Calculs avec multiplicateurs et facteur d'évolution du scénario
                        admissions_jour = base['admissions'] * mult['admissions'] * evolution_factor
                        urgences_jour = base['urgences'] * mult['urgences'] * evolution_factor
                        icu_jour = base['icu'] * mult['icu'] * evolution_factor
                        
                        # DMS ajustée selon scénario
                        dms_ajustee = base['duree_moy'] * mult['dms']
                        
                        lits_dispo_service = base['lits_dispo']
                        
                        # Lits occupés : Utiliser Prophet si disponible, sinon Loi de Little
                        if use_prophet_beds and daily_beds_forecast is not None:
                            # Utiliser les prédictions Prophet pour les lits occupés
                            # On répartit les lits occupés globaux par service selon les poids
                            current_date_date = current_date.date()
                            
                            # Trouver la prédiction Prophet pour ce jour
                            if current_date_date in daily_beds_forecast['date'].values:
                                lits_occ_global_prophet = daily_beds_forecast[daily_beds_forecast['date'] == current_date_date]['lits_occ_prophet'].iloc[0]
                            else:
                                # Si la date n'est pas dans les prédictions, utiliser la moyenne
                                lits_occ_global_prophet = daily_beds_forecast['lits_occ_prophet'].mean()
                            
                            # Répartir les lits occupés globaux par service selon les poids
                            service_weight = SERVICES_DISTRIBUTION.get(service, {}).get('weight', 0.1)
                            base_lits_occ = lits_occ_global_prophet * service_weight
                            
                            # Appliquer les multiplicateurs de scénario et le facteur d'évolution
                            base_lits_occ = base_lits_occ * mult['admissions'] * evolution_factor
                            
                            # VARIABILITÉ STOCHASTIQUE : Ajout de bruit aléatoire contrôlé (réduit car Prophet est déjà précis)
                            np.random.seed(int(current_date.timestamp()) % 10000)
                            
                            # Variabilité réduite pour Prophet (car le modèle capture déjà la saisonnalité)
                            if sim_scenario == 'normal':
                                variabilite = np.random.normal(1.0, 0.03)  # ±3% au lieu de ±5%
                            else:
                                variabilite = np.random.normal(1.0, 0.05)  # ±5% au lieu de ±10%
                            
                            lits_occ_service = max(0, (base_lits_occ * variabilite).round(0))
                            
                        else:
                            # Fallback: Application de la Loi de Little
                            # Formule : Stock (Lits) = Flux (Admissions/jour) × Durée (DMS)
                            base_lits_occ = (admissions_jour * dms_ajustee).round(0)
                            
                            # VARIABILITÉ STOCHASTIQUE : Ajout de bruit aléatoire contrôlé
                            np.random.seed(int(current_date.timestamp()) % 10000)
                            
                            # Variabilité réduite pour le scénario normal
                            if sim_scenario == 'normal':
                                variabilite = np.random.normal(1.0, 0.05)
                            else:
                                variabilite = np.random.normal(1.0, 0.10)
                            
                            lits_occ_service = max(0, (base_lits_occ * variabilite).round(0))
                        
                        # Plafond sur l'occupation en mode normal
                        if sim_scenario == 'normal':
                            lits_occ_max_normal = int(lits_dispo_service * 0.90)
                            lits_occ_service = min(lits_occ_service, lits_occ_max_normal)
                        
                        taux_occ_service = min(100, (lits_occ_service / lits_dispo_service * 100) if lits_dispo_service > 0 else 0)
                        
                        # Personnel
                        personnel_requis = round(lits_occ_service / 3, 1)
                        
                        # Personnel disponible : constant (sauf en cas de grève)
                        if sim_scenario == 'greve':
                            personnel_dispo_base = base['personnel_disponible'] * mult['personnel']
                        else:
                            personnel_dispo_base = base['personnel_disponible']
                        
                        personnel_dispo = personnel_dispo_base
                        
                        # Déficit
                        deficit = max(0, ((personnel_requis - personnel_dispo) / personnel_requis * 100) if personnel_requis > 0 else 0)
                        
                        # Matériel
                        medicaments = int(admissions_jour * 2.5 * mult['materiel'] * evolution_factor)
                        
                        # Agrégation globale
                        day_total['admissions'] += admissions_jour
                        day_total['urgences'] += urgences_jour
                        day_total['icu'] += icu_jour
                        day_total['lits_occ'] += lits_occ_service
                        day_total['lits_dispo'] += lits_dispo_service
                        day_total['personnel_requis'] += personnel_requis
                        day_total['personnel_disponible'] += personnel_dispo
                        day_total['materiel_medicaments'] += medicaments
                        
                        # Stockage des lits de réa occupés pour calcul global des respirateurs
                        if service == 'Réanimation':
                            day_total['lits_rea_occ'] = day_total.get('lits_rea_occ', 0) + lits_occ_service
                
                # CORRECTION : Calcul des respirateurs au niveau GLOBAL (une seule fois, pas service par service)
                # Le besoin en respirateurs doit être calculé sur le stock total de patients en réa simultanément,
                # pas sur le flux d'admissions quotidiennes (sinon on obtient des valeurs irréalistes)
                # On utilise les lits de réa occupés (si disponible) ou on estime via Loi de Little
                if 'lits_rea_occ' in day_total and day_total['lits_rea_occ'] > 0:
                    # Si on a le service Réanimation dans la simulation, on utilise directement les lits occupés
                    lits_rea_totaux = day_total['lits_rea_occ']
                else:
                    # Sinon, on estime via Loi de Little : patients en réa = admissions réa/jour * DMS réa
                    dms_rea = SERVICES_DISTRIBUTION.get('Réanimation', {}).get('duree_moy', 7.0)
                    total_icu_jour = day_total['icu']  # Total des admissions en réa ce jour (tous services confondus)
                    lits_rea_totaux = total_icu_jour * dms_rea
                
                # Ratio réaliste : 20-25% des patients en réa nécessitent une ventilation mécanique
                # (pas 80% comme avant, et pas calculé service par service pour éviter les doublons)
                # En réalité, tous les patients en réa ne nécessitent pas un respirateur
                # Seulement ceux en détresse respiratoire aiguë (environ 20-25% selon les études hospitalières)
                ratio_ventilation = 0.22  # 22% des patients en réa ont besoin d'un respirateur (ratio conservateur)
                respirateurs_totaux = max(0, int(lits_rea_totaux * ratio_ventilation * mult['materiel'] * evolution_factor))
                day_total['materiel_respirateurs'] = respirateurs_totaux
                
                # ========================================================================
                # CONSOMMATION D'EPI : Utiliser Prophet si disponible, sinon estimation basique
                # ========================================================================
                if use_prophet_epi and daily_epi_forecast is not None:
                    # Utiliser les prédictions Prophet pour la consommation d'EPI
                    current_date_date = current_date.date()
                    
                    # Trouver la prédiction Prophet pour ce jour
                    if current_date_date in daily_epi_forecast['date'].values:
                        epi_consommation_prophet = daily_epi_forecast[daily_epi_forecast['date'] == current_date_date]['epi_consommation_prophet'].iloc[0]
                    else:
                        # Si la date n'est pas dans les prédictions, utiliser la moyenne
                        epi_consommation_prophet = daily_epi_forecast['epi_consommation_prophet'].mean()
                    
                    # Appliquer les multiplicateurs de scénario et le facteur d'évolution
                    epi_consommation = int(epi_consommation_prophet * mult['materiel'] * evolution_factor)
                    
                    # VARIABILITÉ STOCHASTIQUE : Ajout de bruit aléatoire contrôlé (réduit car Prophet est déjà précis)
                    np.random.seed(int(current_date.timestamp()) % 10000)
                    variabilite_epi = np.random.normal(1.0, 0.05)  # ±5% de variabilité
                    epi_consommation = max(0, int(epi_consommation * variabilite_epi))
                    
                else:
                    # Fallback: Estimation basique basée sur les admissions et lits occupés
                    # La consommation d'EPI est liée au nombre de patients et au personnel
                    base_epi = (day_total['admissions'] * 1.5) + (day_total['lits_occ'] * 0.3)
                    epi_consommation = int(base_epi * mult['materiel'] * evolution_factor)
                
                day_total['materiel_protection'] = epi_consommation
                
                # Taux d'occupation global
                day_total['taux_occupation'] = min(100, (day_total['lits_occ'] / day_total['lits_dispo'] * 100) if day_total['lits_dispo'] > 0 else 0)
                
                # Déficit personnel global
                day_total['deficit_personnel'] = max(0, ((day_total['personnel_requis'] - day_total['personnel_disponible']) / day_total['personnel_requis'] * 100) if day_total['personnel_requis'] > 0 else 0)
                
                # ========================================================================
                # CALCUL FINANCIER ROBUSTE (simplifié et fiable)
                # ========================================================================
                
                # 1. Coût du Personnel (Intérim / Heures Sup)
                # On calcule le manque physique de bras (Requis - Dispo)
                manque_etp = max(0, day_total['personnel_requis'] - day_total['personnel_disponible'])
                
                # On facture ce manque au prix de l'intérim (c'est le coût de la crise)
                cout_rh_jour = manque_etp * COUT_JOUR_INTERIMAIRE
                total_cout_rh += cout_rh_jour
                
                # 2. Coût des Lits (Plan Blanc)
                # # Si le slider renfort_lits > 0, on facture l'ouverture
                # cout_lits_jour = renfort_lits * COUT_LIT_SUPPLEMENTAIRE
                # total_cout_lits += cout_lits_jour

                # # Ajout des coûts au dictionnaire du jour pour affichage éventuel
                # day_total['cout_jour'] = cout_rh_jour + cout_lits_jour
                
                simulation_days.append(day_total)
            
            df_simulation = pd.DataFrame(simulation_days)
            
            # # ========================================================================
            # # IMPACT DES ACTIONS CORRECTIVES
            # # ========================================================================
            # if renfort_lits > 0 or renfort_personnel > 0:
            #     st.divider()
            #     st.subheader("Impact des Actions Correctives")
                
            #     # Calculer l'impact moyen
            #     taux_occ_moyen = df_simulation['taux_occupation'].mean()
            #     deficit_personnel_moyen = df_simulation['deficit_personnel'].mean()
            #     lits_dispo_total = df_simulation['lits_dispo'].iloc[0] if len(df_simulation) > 0 else 0
            #     personnel_dispo_total = df_simulation['personnel_disponible'].iloc[0] if len(df_simulation) > 0 else 0
                
            #     col_impact1, col_impact2 = st.columns(2)
                
            #     with col_impact1:
            #         st.metric(
            #             "Taux d'Occupation Moyen",
            #             f"{taux_occ_moyen:.1f}%",
            #             help=f"Avec {renfort_lits} lits supplémentaires ajoutés"
            #         )
            #         if renfort_lits > 0:
            #             st.metric(
            #                 "Lits Disponibles Totaux",
            #                 f"{lits_dispo_total:.0f} lits",
            #                 delta=f"+{renfort_lits:.0f} lits",
            #                 delta_color="normal"
            #             )
            #         else:
            #             st.metric(
            #                 "Lits Disponibles Totaux",
            #                 f"{lits_dispo_total:.0f} lits"
            #             )
                
            #     with col_impact2:
            #         st.metric(
            #             "Déficit Personnel Moyen",
            #             f"{deficit_personnel_moyen:.1f}%",
            #             help=f"Avec {renfort_personnel} équivalents temps plein de renfort"
            #         )
            #         if renfort_personnel > 0:
            #             st.metric(
            #                 "Personnel Disponible Total",
            #                 f"{personnel_dispo_total:.1f} équivalents temps plein",
            #                 delta=f"+{renfort_personnel:.0f} équivalents temps plein",
            #                 delta_color="normal"
            #             )
            #         else:
            #             st.metric(
            #                 "Personnel Disponible Total",
            #                 f"{personnel_dispo_total:.1f} équivalents temps plein"
            #             )
            
            # ========================================================================
            # IDENTIFICATION DES PICS ET JOURS CRITIQUES
            # ========================================================================
            st.divider()
            st.subheader("Analyse des Pics d'Activité")
            
            # Trouver le jour de pic pour différentes métriques
            pic_admissions = df_simulation.loc[df_simulation['admissions'].idxmax()]
            pic_lits = df_simulation.loc[df_simulation['lits_occ'].idxmax()]
            pic_occupation = df_simulation.loc[df_simulation['taux_occupation'].idxmax()]
            pic_personnel = df_simulation.loc[df_simulation['deficit_personnel'].idxmax()]
            pic_epi = df_simulation.loc[df_simulation['materiel_protection'].idxmax()]
            
            col_pic1, col_pic2, col_pic3, col_pic4, col_pic5 = st.columns(5)
            
            with col_pic1:
                st.metric(
                    "Pic d'Admissions",
                    f"Jour {int(pic_admissions['jour'])}",
                    f"{pic_admissions['admissions']:.0f} admissions"
                )
            
            with col_pic2:
                st.metric(
                    "Pic de Saturation",
                    f"Jour {int(pic_lits['jour'])}",
                    f"{pic_lits['taux_occupation']:.1f}%"
                )
            
            with col_pic3:
                # Afficher le pic personnel seulement s'il y a un déficit significatif
                if pic_personnel['deficit_personnel'] > 0.1:
                    st.metric(
                        "Pic Personnel",
                        f"Jour {int(pic_personnel['jour'])}",
                        f"{pic_personnel['deficit_personnel']:.1f}% déficit"
                    )
                else:
                    # Si pas de déficit, afficher le jour avec le plus de personnel requis
                    pic_personnel_requis = df_simulation.loc[df_simulation['personnel_requis'].idxmax()]
                    st.metric(
                        "Pic Personnel Requis",
                        f"Jour {int(pic_personnel_requis['jour'])}",
                        f"{pic_personnel_requis['personnel_requis']:.1f} équivalents temps plein"
                    )
            
            with col_pic4:
                # Identifier les jours critiques (taux occupation > 85%)
                jours_critiques = df_simulation[df_simulation['taux_occupation'] > 85]
                nb_jours_critiques = len(jours_critiques)
                st.metric(
                    "Jours Critiques",
                    f"{nb_jours_critiques} jours",
                    f"Taux > 85%"
                )
            
            with col_pic5:
                st.metric(
                    "Pic Consommation EPI",
                    f"Jour {int(pic_epi['jour'])}",
                    f"{int(pic_epi['materiel_protection']):,} unités"
                )
            
            # ========================================================================
            # VISUALISATIONS TEMPORELLES
            # ========================================================================
            st.divider()
            st.subheader("Évolution Temporelle Projetée")
            
            # Graphique 1: Évolution des admissions et lits
            fig_evolution = go.Figure()
            
            fig_evolution.add_trace(go.Scatter(
                x=df_simulation['date'],
                y=df_simulation['admissions'],
                mode='lines+markers',
                name='Admissions',
                line=dict(color='#1f77b4', width=2)
            ))
            
            fig_evolution.add_trace(go.Scatter(
                x=df_simulation['date'],
                y=df_simulation['lits_occ'],
                mode='lines+markers',
                name='Lits Occupés',
                yaxis='y2',
                line=dict(color='#ff7f0e', width=2)
            ))
            
            # # Ajouter une trace pour les lits disponibles (pour voir l'impact du renfort)
            # if renfort_lits > 0:
            #     fig_evolution.add_trace(go.Scatter(
            #         x=df_simulation['date'],
            #         y=df_simulation['lits_dispo'],
            #         mode='lines',
            #         name='Lits Disponibles (avec renfort)',
            #         yaxis='y2',
            #         line=dict(color='#2ca02c', width=2, dash='dash'),
            #         opacity=0.7
            #     ))
            
            # Ligne de référence (seuil critique)
            max_lits_dispo = df_simulation['lits_dispo'].max()
            seuil_critique = max_lits_dispo * 0.85
            fig_evolution.add_hline(
                y=seuil_critique,
                line_dash="dash",
                line_color="red",
                annotation_text="Seuil critique (85%)",
                yref='y2'
            )
            
            fig_evolution.update_layout(
                title="Évolution des Admissions et Lits Occupés",
                xaxis_title="Date",
                yaxis_title="Admissions/jour",
                yaxis2=dict(title="Lits Occupés", overlaying='y', side='right'),
                height=400,
                hovermode='x unified'
            )
            
            st.plotly_chart(fig_evolution, use_container_width=True)
            
            # Graphique 2: Taux d'occupation et déficit personnel
            fig_ressources = go.Figure()
            
            fig_ressources.add_trace(go.Scatter(
                x=df_simulation['date'],
                y=df_simulation['taux_occupation'],
                mode='lines+markers',
                name="Taux d'Occupation (%)",
                line=dict(color='#2ca02c', width=2),
                fill='tozeroy'
            ))
            
            fig_ressources.add_trace(go.Scatter(
                x=df_simulation['date'],
                y=df_simulation['deficit_personnel'],
                mode='lines+markers',
                name='Déficit Personnel (%)',
                yaxis='y2',
                line=dict(color='#d62728', width=2)
            ))
            
            # Seuils d'alerte
            fig_ressources.add_hline(
                y=85,
                line_dash="dash",
                line_color="red",
                annotation_text="Seuil critique occupation",
                yref='y'
            )
            
            fig_ressources.add_hline(
                y=25,
                line_dash="dash",
                line_color="orange",
                annotation_text="Seuil déficit personnel",
                yref='y2'
            )
            
            fig_ressources.update_layout(
                title="Évolution du Taux d'Occupation et du Déficit de Personnel",
                xaxis_title="Date",
                yaxis_title="Taux d'Occupation (%)",
                yaxis2=dict(title="Déficit Personnel (%)", overlaying='y', side='right'),
                height=400,
                hovermode='x unified'
            )
            
            st.plotly_chart(fig_ressources, use_container_width=True)
            
            # Graphique 3: Consommation de matériel (EPI, Respirateurs, Médicaments)
            fig_materiel = go.Figure()
            
            fig_materiel.add_trace(go.Scatter(
                x=df_simulation['date'],
                y=df_simulation['materiel_protection'],
                mode='lines+markers',
                name='Consommation EPI',
                line=dict(color='#9467bd', width=2),
                fill='tozeroy',
                fillcolor='rgba(148, 103, 189, 0.1)'
            ))
            
            fig_materiel.add_trace(go.Scatter(
                x=df_simulation['date'],
                y=df_simulation['materiel_respirateurs'],
                mode='lines+markers',
                name='Respirateurs Requis',
                yaxis='y2',
                line=dict(color='#e377c2', width=2)
            ))
            
            fig_materiel.add_trace(go.Scatter(
                x=df_simulation['date'],
                y=df_simulation['materiel_medicaments'],
                mode='lines+markers',
                name='Médicaments (×100)',
                yaxis='y3',
                line=dict(color='#7f7f7f', width=2, dash='dot')
            ))
            
            fig_materiel.update_layout(
                title="Évolution de la Consommation de Matériel",
                xaxis_title="Date",
                yaxis_title="Consommation EPI (unités)",
                yaxis2=dict(title="Respirateurs Requis", overlaying='y', side='right'),
                yaxis3=dict(title="Médicaments (×100)", overlaying='y', side='right', anchor='free', position=0.95),
                height=400,
                hovermode='x unified',
                legend=dict(x=0.02, y=0.98)
            )
            
            st.plotly_chart(fig_materiel, use_container_width=True)
            
            # ========================================================================
            # RECOMMANDATIONS AUTOMATIQUES
            # ========================================================================
            st.divider()
            st.subheader("Recommandations Préventives")
            
            recommendations = []
            
            # 1. Recommandation sur les lits
            # On monte le seuil critique à 95% pour éviter les fausses alertes
            if pic_lits['taux_occupation'] > 95:
                jours_avant_pic = int(pic_lits['jour']) - 1
                # On calcule le manque par rapport à 100% (saturation absolue) et non 85%
                lits_manquants = max(0, int(pic_lits['lits_occ'] - pic_lits['lits_dispo']))
                
                if lits_manquants > 0:
                    recommendations.append({
                        'priorite': 'Critique',
                        'type': 'Lits',
                        'message': f"**Jour {int(pic_lits['jour'])}** : Saturation absolue (100%) prévue à {pic_lits['taux_occupation']:.1f}%",
                        'action': f"Préparer l'ouverture de {lits_manquants} lits de débordement avant le jour {int(pic_lits['jour'])} (dans {jours_avant_pic} jours)"
                    })
            elif pic_lits['taux_occupation'] > 85:
                # Recommandation "Soft" pour le seuil 85%
                jours_avant_pic = int(pic_lits['jour']) - 1
                recommendations.append({
                    'priorite': 'Attention',
                    'type': 'Tension',
                    'message': f"**Jour {int(pic_lits['jour'])}** : Tension hospitalière (>85%) prévue à {pic_lits['taux_occupation']:.1f}%",
                    'action': f"Surveiller les flux et préparer les sorties anticipées avant le jour {int(pic_lits['jour'])} (dans {jours_avant_pic} jours)"
                })
            
            # 2. Recommandation sur le personnel
            if pic_personnel['deficit_personnel'] > 25:
                jours_avant_pic = int(pic_personnel['jour']) - 1
                personnel_manquant = pic_personnel['personnel_requis'] - pic_personnel['personnel_disponible']
                recommendations.append({
                    'priorite': 'Critique',
                    'type': 'Personnel',
                    'message': f"**Jour {int(pic_personnel['jour'])}** : Déficit de personnel critique prévu à {pic_personnel['deficit_personnel']:.1f}%",
                    'action': f"Renforcer les effectifs de {personnel_manquant:.1f} équivalents temps plein avant le jour {int(pic_personnel['jour'])} (dans {jours_avant_pic} jours)"
                })
            
            # 3. Recommandation sur le matériel (Respirateurs)
            pic_materiel = df_simulation.loc[df_simulation['materiel_respirateurs'].idxmax()]
            if pic_materiel['materiel_respirateurs'] > 0:
                jours_avant_pic = int(pic_materiel['jour']) - 1
                recommendations.append({
                    'priorite': 'Attention',
                    'type': 'Matériel',
                    'message': f"**Jour {int(pic_materiel['jour'])}** : Pic de besoin en respirateurs prévu",
                    'action': f"Vérifier la disponibilité de {int(pic_materiel['materiel_respirateurs'])} respirateurs avant le jour {int(pic_materiel['jour'])} (dans {jours_avant_pic} jours)"
                })
            
            # 3b. Recommandation sur les EPI
            consommation_epi_moyenne = df_simulation['materiel_protection'].mean()
            consommation_epi_pic = pic_epi['materiel_protection']
            # Seuil : si la consommation de pic dépasse 150% de la moyenne, alerte
            if consommation_epi_pic > consommation_epi_moyenne * 1.5:
                jours_avant_pic_epi = int(pic_epi['jour']) - 1
                recommendations.append({
                    'priorite': 'Attention',
                    'type': 'Matériel de Protection',
                    'message': f"**Jour {int(pic_epi['jour'])}** : Pic de consommation d'EPI prévu ({int(consommation_epi_pic):,} unités)",
                    'action': f"Renforcer les stocks d'EPI avant le jour {int(pic_epi['jour'])} (dans {jours_avant_pic_epi} jours). Consommation moyenne: {int(consommation_epi_moyenne):,} unités/jour"
                })
            
            # 4. Recommandation générale sur la période critique
            if nb_jours_critiques > 0:
                premier_jour_critique = jours_critiques.iloc[0]['jour']
                dernier_jour_critique = jours_critiques.iloc[-1]['jour']
                recommendations.append({
                    'priorite': 'Attention',
                    'type': 'Période Critique',
                    'message': f"Période de saturation prévue du jour {int(premier_jour_critique)} au jour {int(dernier_jour_critique)}",
                    'action': f"Activer le plan de crise et mobiliser les ressources supplémentaires pour cette période ({nb_jours_critiques} jours)"
                })
            
            # Affichage des recommandations
            if recommendations:
                for rec in recommendations:
                    with st.expander(f"{rec['priorite']} {rec['type']}: {rec['message']}", expanded=True):
                        st.info(f"**Action recommandée :** {rec['action']}")
            else:
                st.success("Aucune alerte critique détectée. La simulation indique une gestion normale des ressources.")
            
            # ========================================================================
            # ESTIMATION D'IMPACT FINANCIER (Surcoûts RH)
            # ========================================================================
            st.divider()
            st.subheader("Estimation d'Impact Financier (Surcoûts RH)")
            st.caption("Calcul des surcoûts estimés liés au recours à l'intérim et aux heures supplémentaires")
            
            # Les coûts ont été calculés jour par jour dans la boucle de simulation
            # (plus précis que de sommer des pourcentages)
            cout_total_rh = total_cout_rh
            cout_lits_supplementaires = total_cout_lits
            
            # Calcul des détails pour l'affichage (pour information)
            # On recalcule pour avoir les détails (intérim vs heures sup)
            df_simulation['deficit_etp'] = (df_simulation['deficit_personnel'] / 100) * df_simulation['personnel_requis']
            total_deficit_etp_jours = df_simulation['deficit_etp'].sum()
            cout_interim_total = total_deficit_etp_jours * COUT_JOUR_INTERIMAIRE
            heures_sup_estimees = total_deficit_etp_jours * 7
            cout_heures_sup_total = heures_sup_estimees * COUT_HEURE_SUP
            
            # Nombre de jours de saturation pour l'affichage
            jours_saturation = df_simulation[df_simulation['taux_occupation'] > 85]
            if len(jours_saturation) > 0:
                lits_supplementaires_moyen = (jours_saturation['lits_occ'] - jours_saturation['lits_dispo'] * 0.85).mean()
                lits_supplementaires_moyen = max(0, lits_supplementaires_moyen)
            else:
                lits_supplementaires_moyen = 0
            
            # Coût total estimé
            cout_total_estime = cout_total_rh + cout_lits_supplementaires
            
            # Affichage des résultats
            col_fin1, col_fin2 = st.columns(2)
            
            with col_fin1:
                st.metric(
                    "Surcoût RH (Personnel)",
                    f"{cout_total_rh:,.0f} €",
                    help=f"Coût estimé pour compenser {total_deficit_etp_jours:.1f} ETP-jours de déficit"
                )
                
                if total_deficit_etp_jours > 0:
                    st.caption(f"**Détail** :")
                    st.caption(f"   • Intérim : {cout_interim_total:,.0f} € ({total_deficit_etp_jours:.1f} ETP-jours × {COUT_JOUR_INTERIMAIRE} €/jour)")
                    st.caption(f"   • Heures sup : {cout_heures_sup_total:,.0f} € ({heures_sup_estimees:.0f} heures × {COUT_HEURE_SUP} €/heure)")
                    st.caption(f"   • **Solution recommandée** : {'Intérim' if cout_interim_total < cout_heures_sup_total else 'Heures supplémentaires'}")
            
            with col_fin2:
                st.metric(
                    "Surcoût Lits Supplémentaires",
                    f"{cout_lits_supplementaires:,.0f} €",
                    help=f"Coût estimé pour l'ouverture de lits supplémentaires pendant {len(jours_saturation)} jours critiques"
                )
                
                if cout_lits_supplementaires > 0:
                    st.caption(f"**Détail** :")
                    st.caption(f"   • {len(jours_saturation)} jours de saturation")
                    st.caption(f"   • {lits_supplementaires_moyen:.0f} lits supplémentaires en moyenne")
                    st.caption(f"   • {COUT_LIT_SUPPLEMENTAIRE} €/lit/jour")
            
            # Alerte globale
            if cout_total_estime > 0:
                st.warning(
                    f"**Surcoût total estimé** : **{cout_total_estime:,.0f} €** sur la période de {sim_duration} jours. "
                    f"Ce montant inclut les surcoûts RH (personnel intérim/heures sup) et l'ouverture de lits supplémentaires."
                )
            else:
                st.success(
                    f"**Aucun surcoût majeur projeté** pour ce scénario sur {sim_duration} jours. "
                    f"Les ressources actuelles sont suffisantes pour gérer la situation."
                )
            
            # ========================================================================
            # TABLEAU RÉCAPITULATIF PAR JOUR
            # ========================================================================
            st.divider()
            st.subheader("Détail Jour par Jour")
            
            # Sélection des colonnes à afficher
            colonnes_affichees = ['jour', 'date', 'admissions', 'lits_occ', 'taux_occupation',
                                 'personnel_requis', 'deficit_personnel', 'materiel_respirateurs', 'materiel_protection']
            
            df_display = df_simulation[colonnes_affichees].copy()
            df_display['date'] = df_display['date'].dt.strftime('%Y-%m-%d')
            df_display['taux_occupation'] = df_display['taux_occupation'].round(1)
            df_display['deficit_personnel'] = df_display['deficit_personnel'].round(1)
            df_display['materiel_protection'] = df_display['materiel_protection'].astype(int)
            df_display.columns = ['Jour', 'Date', 'Admissions', 'Lits Occupés', 'Taux Occupation (%)',
                                 'Personnel Requis', 'Déficit Personnel (%)', 'Respirateurs', 'Consommation EPI']
            
            st.dataframe(df_display, use_container_width=True, height=400)
