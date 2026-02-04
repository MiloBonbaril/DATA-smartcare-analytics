
import joblib
import pandas as pd
from prophet import Prophet
import os
import numpy as np

def verify():
    print("--- Verifying Models & Predictions ---")
    
    # 1. Load Models
    models = {}
    files = {
        'admissions': 'admissions_prophet_model.pkl',
        'beds': 'beds_prophet_model.pkl',
        'epi': 'epi_prophet_model.pkl'
    }
    
    for name, f in files.items():
        if os.path.exists(f):
            try:
                models[name] = joblib.load(f)
                print(f"[OK] Loaded {name} model from {f}")
            except Exception as e:
                print(f"[FAIL] Could not load {name} model from {f}: {e}")
        else:
            print(f"[FAIL] File not found: {f}")

    if len(models) < 3:
        print("\n[WARNING] Not all models loaded. Proceeding with available ones...")

    # 2. Create Dummy Future Data (24 hours)
    future_dates = pd.date_range(start='2025-01-01', periods=24, freq='H')
    
    # Base features
    base_data = {
        'ds': future_dates,
        'Indicateur_Epidemie': 0,
        'Indicateur_Canicule': 0,
        'Indicateur_Greve': 0,
        'gravite': 2.5,
        'duree_sejour_estimee': 5.0
    }
    future_df = pd.DataFrame(base_data)
    
    # 3. Test Predictions
    
    # Admissions
    admissions_forecast = None
    if 'admissions' in models:
        try:
            print("\nTesting Admissions Prediction...")
            forecast = models['admissions'].predict(future_df)
            admissions_forecast = forecast['yhat'].values
            print("[OK] Admissions prediction successful.")
        except Exception as e:
            print(f"[FAIL] Admissions prediction failed: {e}")
            # print regressor info if available
            if hasattr(models['admissions'], 'extra_regressors'):
                print(f"  Expected regressors: {list(models['admissions'].extra_regressors.keys())}")

    # Beds
    beds_forecast = None
    if 'beds' in models:
        try:
            print("\nTesting Beds Prediction...")
            if admissions_forecast is None:
                print("  (Using dummy admissions data for test)")
                admissions_val = np.full(24, 50) # dummy value
            else:
                admissions_val = admissions_forecast
                
            future_df_beds = future_df.copy()
            future_df_beds['Nombre_Admissions'] = admissions_val
            
            # Ensure gravite/duree are present (already in base_data)
            
            forecast = models['beds'].predict(future_df_beds)
            beds_forecast = forecast['yhat'].values
            print("[OK] Beds prediction successful.")
        except Exception as e:
            print(f"[FAIL] Beds prediction failed: {e}")
            if hasattr(models['beds'], 'extra_regressors'):
                print(f"  Expected regressors: {list(models['beds'].extra_regressors.keys())}")

    # EPI
    if 'epi' in models:
        try:
            print("\nTesting EPI Prediction...")
            future_df_epi = future_df.copy()
            
            # Add required features for EPI
            if admissions_forecast is None: admissions_val = np.full(24, 50)
            else: admissions_val = admissions_forecast
            
            if beds_forecast is None: lits_val = np.full(24, 1500)
            else: lits_val = beds_forecast
            
            future_df_epi['Nombre_Admissions'] = admissions_val
            future_df_epi['Lits_Occupes'] = lits_val
            future_df_epi['Personnel_Present'] = lits_val * 0.6 # Logic from dashboard
            
            forecast = models['epi'].predict(future_df_epi)
            print("[OK] EPI prediction successful.")
        except Exception as e:
            print(f"[FAIL] EPI prediction failed: {e}")
            if hasattr(models['epi'], 'extra_regressors'):
                print(f"  Expected regressors: {list(models['epi'].extra_regressors.keys())}")

if __name__ == "__main__":
    verify()
