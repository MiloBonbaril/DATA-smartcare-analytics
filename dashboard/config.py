# ============================================================================
# CONFIGURATION - Basée sur chiffres clés Pitié-Salpêtrière 2015
# ============================================================================

# Volumes annuels de référence (2015)
ANNUAL_URGENCES = 121721
ANNUAL_HOSPITALISATIONS = 168394
TOTAL_MEDECINS = 1621
TOTAL_PARAMEDICAUX = 7850

# Distribution par service (poids + caractéristiques)
SERVICES_DISTRIBUTION = {
    'Urgences': {
        'weight': 0.25,
        'lits_base': 50,
        'duree_moy': 1.5,
        'icu_ratio': 0.05  # 5% des admissions vont en ICU
    },
    'Réanimation': {
        'weight': 0.08,
        'lits_base': 150,  # Augmenté de 92 à 150 pour éviter la saturation à 100% après correction Loi de Little
        'duree_moy': 7.0,
        'icu_ratio': 1.0
    },
    'Cardiologie': {
        'weight': 0.15,
        'lits_base': 120,
        'duree_moy': 4.5,
        'icu_ratio': 0.15
    },
    'Neurologie': {
        'weight': 0.12,
        'lits_base': 100,
        'duree_moy': 5.0,
        'icu_ratio': 0.20
    },
    'Infectiologie': {
        'weight': 0.10,
        'lits_base': 80,
        'duree_moy': 6.0,
        'icu_ratio': 0.25
    },
    'Gériatrie': {
        'weight': 0.15,
        'lits_base': 150,
        'duree_moy': 12.0,
        'icu_ratio': 0.02
    },
    'Chirurgie': {
        'weight': 0.15,
        'lits_base': 200,
        'duree_moy': 3.5,
        'icu_ratio': 0.10
    }
}

# TOTAL_LITS calculé dynamiquement à partir de la somme des lits_base des services
# (pour éviter l'incohérence entre le total global et la somme des services)
# CORRECTION : Calculé après la définition de SERVICES_DISTRIBUTION
TOTAL_LITS = sum(service['lits_base'] for service in SERVICES_DISTRIBUTION.values())

# Multiplicateurs par scénario
SCENARIO_MULTIPLIERS = {
    'normal': {
        'admissions': 1.0,
        'urgences': 1.0,
        'icu': 1.0,
        'personnel': 1.0,
        'materiel': 1.0,
        'dms': 1.0  # Durée moyenne de séjour normale
    },
    'epidemie': {
        'admissions': 1.4,  # +40% admissions
        'urgences': 1.6,    # +60% urgences
        'icu': 2.0,         # +100% réanimation
        'personnel': 1.3,   # +30% besoin personnel
        'materiel': 1.8,   # +80% matériel
        'dms': 2.0  # +100% DMS (cas plus graves, séjours plus longs)
    },
    'canicule': {
        'admissions': 1.2,
        'urgences': 1.4,
        'icu': 1.3,
        'personnel': 1.1,
        'materiel': 1.2,
        'dms': 1.1  # +10% DMS (déshydratation, complications)
    },
    'greve': {
        'admissions': 0.9,   # -10% (moins d'admissions programmées)
        'urgences': 1.1,     # +10% (report sur urgences)
        'icu': 1.0,
        'personnel': 0.6,    # -40% disponibilité
        'materiel': 1.0,
        'dms': 1.0  # DMS inchangée
    },
    'accident': {
        'admissions': 1.8,   # Pic brutal
        'urgences': 2.5,
        'icu': 2.2,
        'personnel': 1.5,
        'materiel': 2.0,
        'dms': 1.5  # +50% DMS (traumatismes, soins intensifs)
    }
}

# ============================================================================
# BENCHMARKS NATIONAUX - Valeurs de référence pour comparaison
# Basées sur les statistiques ATIH et DREES (France)
# ============================================================================

# Benchmarks nationaux moyens (hôpitaux français de taille similaire)
NATIONAL_BENCHMARKS = {
    'taux_occupation_moyen': 0.72,  # 72% - Taux d'occupation moyen national
    'dms_moyenne': 5.2,  # 5.2 jours - Durée moyenne de séjour nationale
    'taux_rotation_lits': 68,  # 68 rotations/an - Rotation moyenne des lits
    'ratio_personnel_lits': 3.5,  # 3.5 ETP pour 10 lits - Ratio moyen national
    'taux_urgences_admissions': 0.35,  # 35% - Proportion urgences/admissions
    'taux_icu_admissions': 0.08  # 8% - Proportion réanimation/admissions
}

# Seuils d'alerte (basés sur les recommandations hospitalières)
ALERT_THRESHOLDS = {
    'taux_occupation_critique': 0.85,  # 85% - Seuil critique
    'taux_occupation_alerte': 0.75,  # 75% - Seuil d'alerte
    'dms_elevee': 7.0,  # 7 jours - DMS considérée comme élevée
    'deficit_personnel': 0.25  # 25% - Déficit de personnel acceptable (seuil critique)
}

# ============================================================================
# COÛTS FINANCIERS - Estimation des surcoûts RH en cas de crise
# Basés sur les tarifs moyens du secteur hospitalier français
# ============================================================================

# Coûts unitaires (en euros)
COUT_JOUR_INTERIMAIRE = 450  # Coût journalier d'un ETP intérimaire (infirmier/IDE)
COUT_HEURE_SUP = 50  # Coût d'une heure supplémentaire (majorée)
COUT_LIT_SUPPLEMENTAIRE = 200  # Coût journalier d'ouverture d'un lit supplémentaire (nettoyage, matériel)
