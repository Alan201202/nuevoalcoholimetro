# simulation_test.py
import numpy as np
import pandas as pd
import time
from src.engines.ml_engine import MLEngine
from src.engines.feature_extractor import TemporalFeatureExtractor

def run_biometric_simulation(num_samples=1000):
    print("=================================================================")
    print(f"🧪 INICIANDO SIMULACIÓN DE ESTRÉS BIOMÉTRICO ({num_samples} PRUEBAS)")
    print("=================================================================\n")

    ml_engine = MLEngine()
    results = []

    # Generación de perfiles sintéticos aleatorizados según rangos fisiológicos
    np.random.seed(42) # Semilla reproducible

    for i in range(num_samples):
        # Seleccionar perfil sintético al azar
        profile_type = np.random.choice(['SOBRIO', 'DETERIORO_LEVE', 'DETERIORO_SEVERO', 'CASO_LIMITE'], p=[0.4, 0.25, 0.25, 0.1])

        if profile_type == 'SOBRIO':
            ear = np.random.uniform(0.25, 0.35)
            gaze_jitter = np.random.uniform(0.0005, 0.0030)
            asymmetry = np.random.uniform(0.0, 0.03)
            blink_dur = np.random.uniform(0.10, 0.30)
            pitch = np.random.uniform(-2.0, 2.0)
            head_instability = np.random.uniform(0.1, 2.0)
            perclos = np.random.uniform(2.0, 12.0)
            bpm = np.random.uniform(12.0, 22.0)
            pitch_std = np.random.uniform(0.5, 2.5)

        elif profile_type == 'DETERIORO_LEVE':
            ear = np.random.uniform(0.20, 0.26)
            gaze_jitter = np.random.uniform(0.0032, 0.0055)
            asymmetry = np.random.uniform(0.03, 0.07)
            blink_dur = np.random.uniform(0.35, 0.60)
            pitch = np.random.uniform(-6.0, 6.0)
            head_instability = np.random.uniform(2.5, 4.5)
            perclos = np.random.uniform(14.0, 25.0)
            bpm = np.random.uniform(8.0, 11.0)
            pitch_std = np.random.uniform(3.0, 5.0)

        elif profile_type == 'DETERIORO_SEVERO':
            ear = np.random.uniform(0.12, 0.20)
            gaze_jitter = np.random.uniform(0.0060, 0.0150)
            asymmetry = np.random.uniform(0.08, 0.18)
            blink_dur = np.random.uniform(0.70, 2.00) # Eventos de microsueño
            pitch = np.random.uniform(-15.0, 15.0)
            head_instability = np.random.uniform(5.0, 12.0)
            perclos = np.random.uniform(28.0, 60.0)
            bpm = np.random.uniform(4.0, 8.0)
            pitch_std = np.random.uniform(5.5, 12.0)

        else: # CASO_LIMITE (Ruido o disociación entre módulos)
            ear = np.random.uniform(0.22, 0.28)
            gaze_jitter = np.random.uniform(0.0050, 0.0080) # Nistagmo alto pero ojos abiertos
            asymmetry = np.random.uniform(0.05, 0.10)
            blink_dur = np.random.uniform(0.20, 0.40)
            pitch = np.random.uniform(-3.0, 3.0)
            head_instability = np.random.uniform(1.0, 3.0)
            perclos = np.random.uniform(10.0, 18.0)
            bpm = np.random.uniform(14.0, 20.0)
            pitch_std = np.random.uniform(2.0, 4.0)

        mean_ear = ear

        # Evaluar en el Motor Híbrido (Reglas + ML)
        prob_final, prob_ml, prob_heur = ml_engine.predict_impairment_probability(
            ear, gaze_jitter, asymmetry, blink_dur, 
            pitch, head_instability, perclos, bpm, mean_ear, pitch_std
        )

        # Clasificación según umbrales de severidad
        if prob_final >= 0.70:
            pred_class = "ALTO RIESGO / DETERIORO SEVERO"
        elif prob_final >= 0.35:
            pred_class = "PRECAUCIÓN / ANOMALÍA LEVE"
        else:
            pred_class = "SOBRIO / NORMAL"

        results.append({
            "Perfil_Esperado": profile_type,
            "EAR": ear,
            "Gaze_Jitter": gaze_jitter,
            "PERCLOS": perclos,
            "BPM": bpm,
            "Pitch_STD": pitch_std,
            "Prob_Heuristica": prob_heur * 100,
            "Prob_ML": prob_ml * 100,
            "Prob_Final": prob_final * 100,
            "Clasificacion": pred_class
        })

    df_res = pd.DataFrame(results)

    # --- REPORTE ESTADÍSTICO DE LA SIMULACIÓN ---
    print("📊 RESULTADOS CONSOLIDADOS DE LA SIMULACIÓN DE 1000 MUESTRAS:\n")
    
    summary = df_res.groupby('Perfil_Esperado')['Clasificacion'].value_counts(normalize=True).unstack().fillna(0) * 100
    print(summary.round(2).to_string())

    print("\n-----------------------------------------------------------------")
    print("📈 ESTADÍSTICAS COMPARATIVAS POR MOTORES (Promedio %):")
    stats_engines = df_res.groupby('Perfil_Esperado')[['Prob_Heuristica', 'Prob_ML', 'Prob_Final']].mean()
    print(stats_engines.round(2).to_string())

    # Detección de posibles falsos positivos / anomalías en Casos Límite
    limites = df_res[df_res['Perfil_Esperado'] == 'CASO_LIMITE']
    print("\n🔍 ANÁLISIS DE CASOS LÍMITE (Conflicto entre Módulos):")
    print(f"Promedio Prob. Reglas: {limites['Prob_Heuristica'].mean():.2f}%")
    print(f"Promedio Prob. ML:     {limites['Prob_ML'].mean():.2f}%")
    print(f"Promedio Final:        {limites['Prob_Final'].mean():.2f}%")
    print("=================================================================\n")

    # Guardar resultados en CSV para validación
    df_res.to_csv("data/simulacion_estres_biometrico.csv", index=False)
    print("✅ Resultados guardados en 'data/simulacion_estres_biometrico.csv'")

if __name__ == "__main__":
    run_biometric_simulation()