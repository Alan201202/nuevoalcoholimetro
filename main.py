# main.py
import cv2
import time
import urllib.request
import os
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from src.config import (
    LEFT_EYE, 
    RIGHT_EYE, 
    CALIBRATION_TIME_SEC, 
    EAR_THRESHOLD_FACTOR
)
from src.modules.eye_dynamics import calculate_ear
from src.utils.calibration import Calibrator
from src.modules.gaze_tracker import GazeTracker
from src.modules.head_pose import HeadPoseEstimator
from src.engines.ml_engine import MLEngine
from src.engines.feature_extractor import TemporalFeatureExtractor
from src.utils.data_logger import DataLogger

MODEL_PATH = "face_landmarker.task"
TEST_DURATION_SEC = 30 

if not os.path.exists(MODEL_PATH):
    print("Descargando modelo 'face_landmarker.task'...")
    url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/float16/1/face_landmarker.task"
    urllib.request.urlretrieve(url, MODEL_PATH)

def get_hgn_target_position(elapsed, w, h):
    margin = int(w * 0.15)
    min_x = margin
    max_x = w - margin
    range_x = max_x - min_x
    center_y = int(h / 2)

    cycle_time = elapsed % 10.0
    
    if cycle_time < 3.0:
        ratio = cycle_time / 3.0
        cx = int(min_x + ratio * range_x)
        stage = "Persecución Suave (Der)"
    elif cycle_time < 5.0:
        cx = max_x
        stage = "Fijación Extrema (Der)"
    elif cycle_time < 8.0:
        ratio = (cycle_time - 5.0) / 3.0
        cx = int(max_x - ratio * range_x)
        stage = "Persecución Suave (Izq)"
    else:
        cx = min_x
        stage = "Fijación Extrema (Izq)"

    return cx, center_y, stage

def main():
    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
        num_faces=1
    )
    detector = vision.FaceLandmarker.create_from_options(options)
    cap = cv2.VideoCapture(1)

    calibrator = Calibrator(calibration_time=CALIBRATION_TIME_SEC)
    calibrator.start()
    
    gaze_tracker = GazeTracker(buffer_size=30)
    head_pose = HeadPoseEstimator(buffer_size=30)
    ml_engine = MLEngine()
    
    temporal_extractor = TemporalFeatureExtractor(window_seconds=15.0, fps=30.0)
    logger = DataLogger(filename="data/dataset_features.csv")

    eye_closed = False
    closed_start_time = 0.0
    last_blink_duration = 0.0
    last_log_time = time.time()

    history_ml = []
    history_heur = []
    history_final = []
    history_jitter = []
    history_perclos = []
    history_bpm = []
    history_pitch_std = []

    test_started = False
    test_start_time = 0.0

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        detection_result = detector.detect(mp_image)

        if detection_result.face_landmarks:
            landmarks = detection_result.face_landmarks[0]

            left_ear = calculate_ear(LEFT_EYE, landmarks, w, h)
            right_ear = calculate_ear(RIGHT_EYE, landmarks, w, h)
            current_ear = (left_ear + right_ear) / 2.0

            gaze_jitter, asymmetry, (l_center, r_center) = gaze_tracker.process(landmarks, w, h, left_ear, right_ear)
            pitch, yaw, roll, head_instability = head_pose.process(landmarks, w, h)

            is_calibrated, calib_val = calibrator.process(current_ear)

            if not is_calibrated:
                remaining_calib = int(CALIBRATION_TIME_SEC - calib_val) + 1
                status_text = f"CALIBRANDO BASE... ({remaining_calib}s)"
                color = (0, 255, 255)
            else:
                if not test_started:
                    test_started = True
                    test_start_time = time.time()

                elapsed_test = time.time() - test_start_time
                remaining_test = max(0, int(TEST_DURATION_SEC - elapsed_test))

                ear_base = calib_val
                threshold_ear = ear_base * EAR_THRESHOLD_FACTOR

                is_blink_event = False
                if current_ear < threshold_ear:
                    if not eye_closed:
                        eye_closed = True
                        closed_start_time = time.time()
                else:
                    if eye_closed:
                        eye_closed = False
                        last_blink_duration = time.time() - closed_start_time
                        is_blink_event = True

                curr_closed_dur = (time.time() - closed_start_time) if eye_closed else 0.0

                temporal_extractor.update(current_ear, gaze_jitter, pitch, is_blink_event)
                feats = temporal_extractor.get_feature_vector(threshold_ear)

                prob_deterioro, p_ml, p_heur = ml_engine.predict_impairment_probability(
                    current_ear, gaze_jitter, asymmetry, max(curr_closed_dur, last_blink_duration), 
                    pitch, head_instability, feats['PERCLOS'], feats['BPM'], feats['Mean_EAR'], feats['Pitch_STD']
                )

                if not eye_closed:
                    history_jitter.append(gaze_jitter)
                
                history_perclos.append(feats['PERCLOS'])
                history_bpm.append(feats['BPM'])
                history_pitch_std.append(feats['Pitch_STD'])

                history_ml.append(p_ml)
                history_heur.append(p_heur)
                history_final.append(prob_deterioro)

                if time.time() - last_log_time >= 1.0:
                    logger.log(
                        current_ear, feats['PERCLOS'], feats['BPM'], feats['Mean_EAR'], 
                        gaze_jitter, pitch, feats['Pitch_STD'], prob_deterioro
                    )
                    last_log_time = time.time()

                cx, cy, stage_name = get_hgn_target_position(elapsed_test, w, h)

                if prob_deterioro >= 0.70:
                    status_text = f"ESTADO: DETERIORO SEVERO ({prob_deterioro*100:.0f}%)"
                    color = (0, 0, 255)
                elif prob_deterioro >= 0.35:
                    status_text = f"ESTADO: ANOMALIA LEVE ({prob_deterioro*100:.0f}%)"
                    color = (0, 165, 255)
                else:
                    status_text = f"ESTADO: NORMAL ({prob_deterioro*100:.0f}%)"
                    color = (0, 255, 0)

                # Despliegue en pantalla
                cv2.putText(frame, f"EVALUACION OCULOMOTORA | Tiempo: {remaining_test}s", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 2)
                cv2.putText(frame, f"Fase Actual: {stage_name}", (30, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                cv2.putText(frame, f"Indice Biomecanico: {p_heur*100:.1f}%", (30, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
                cv2.putText(frame, f"Indice Estadistico (ML): {p_ml*100:.1f}%", (30, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
                cv2.putText(frame, f"NIVEL DE DETERIORO INTEGRADO: {prob_deterioro*100:.1f}%", (30, 135), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

                cv2.circle(frame, (cx, cy), 12, (0, 0, 255), -1)
                cv2.circle(frame, (cx, cy), 4, (255, 255, 255), -1)

                if elapsed_test >= TEST_DURATION_SEC:
                    break

            cv2.rectangle(frame, (20, h - 50), (w - 20, h - 10), (0, 0, 0), -1)
            cv2.putText(frame, status_text, (30, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        cv2.imshow("Evaluacion Biometrica Oculomotora", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    # --- REPORTE CON TABLA COMPARATIVA E ÍNDICES POR MOTOR ---
    if history_final:
        fps_approx = 30
        sec_heur = [np.mean(history_heur[i:i+fps_approx]) for i in range(0, len(history_heur), fps_approx)]
        sec_ml = [np.mean(history_ml[i:i+fps_approx]) for i in range(0, len(history_ml), fps_approx)]
        sec_final = [np.mean(history_final[i:i+fps_approx]) for i in range(0, len(history_final), fps_approx)]

        avg_heur = np.mean(sec_heur) * 100
        avg_ml = np.mean(sec_ml) * 100
        avg_final = np.mean(sec_final) * 100

        peak_p85 = np.percentile(sec_final, 85) * 100
        seconds_severe = sum(1 for s in sec_final if s >= 0.70)
        seconds_warning = sum(1 for s in sec_final if 0.35 <= s < 0.70)

        mean_jitter = np.mean(history_jitter) if history_jitter else 0.0
        mean_perclos = np.mean(history_perclos) if history_perclos else 0.0
        mean_bpm = np.mean(history_bpm) if history_bpm else 0.0
        mean_pitch_std = np.mean(history_pitch_std) if history_pitch_std else 0.0

        def get_eval_tag(val, norm_max, warn_max):
            if val <= norm_max:
                return "OK (Normal)"
            elif val <= warn_max:
                return "PRECAUCION"
            else:
                return "ALTERADO"

        print("\n==========================================================================================")
        print("                 TABLA COMPARATIVA DE INDICADORES BIOMÉTRICOS Y ESTADOS                  ")
        print("==========================================================================================")
        print(f"{'Métrica Oculomotora':<26} | {'Medido (Sujeto)':<16} | {'Rango NORMAL':<16} | {'Estado Detectado'}")
        print("------------------------------------------------------------------------------------------")
        print(f"{'1. Nistagmo (Gaze Jitter)':<26} | {mean_jitter:<16.5f} | {'0.0001 - 0.0035':<16} | {get_eval_tag(mean_jitter, 0.0035, 0.0065)}")
        print(f"{'2. PERCLOS (%)':<26} | {mean_perclos:<16.1f} | {'0.0% - 15.0%':<16} | {get_eval_tag(mean_perclos, 15.0, 30.0)}")
        print(f"{'3. Frec. Parpadeo (BPM)':<26} | {mean_bpm:<16.1f} | {'10 - 25 BPM':<16} | {'OK (Normal)' if 10 <= mean_bpm <= 25 else 'FUERA DE RANGO'}")
        print(f"{'4. Inestabilidad Cefálica':<26} | {mean_pitch_std:<16.2f} | {'0.00 - 3.00':<16} | {get_eval_tag(mean_pitch_std, 3.0, 6.0)}")
        print("==========================================================================================\n")

        print("SUMMARY DE ÍNDICES DE EVALUACIÓN:")
        print(f" • ÍNDICE BIOMECÁNICO POR REGLAS:      {avg_heur:.1f}%")
        print(f" • ÍNDICE ESTADÍSTICO (MACHINE LEARNING): {avg_ml:.1f}%")
        print(f" • ÍNDICE FINAL INTEGRADO (PROMEDIO):   {avg_final:.1f}%")
        print(f" • SEVERIDAD SOSTENIDA (PERCENTIL 85):  {peak_p85:.1f}%")
        print(f" • Tiempo en Deterioro Severo (>=70%): {seconds_severe} seg")
        print(f" • Tiempo en Anomalía Leve (35-69%):    {seconds_warning} seg\n")

        print("------------------------------------------------------------------------------------------")
        print("CLASIFICACIÓN FINAL DEL DESEMPEÑO:")
        if peak_p85 >= 65.0 or seconds_severe >= 3:
            print(" [ ESTADO: ALTO RIESGO / DETERIORO SEVERO ]")
            print(" Fundamento: Nistagmo persistente o desaceleración oculomotora fuera de rangos normales.")
        elif peak_p85 >= 35.0 or seconds_warning >= 5:
            print(" [ ESTADO: PRECAUCIÓN / ANOMALÍA LEVE ]")
            print(" Fundamento: Ligeras imprecisiones en la fijación ocular o variaciones palpebrales.")
        else:
            print(" [ ESTADO: SOBRIO / DESEMPEÑO NORMAL ]")
            print(" Fundamento: Todas las métricas se mantuvieron dentro de los rangos fisiológicos normales.")
        print("==========================================================================================\n")

if __name__ == "__main__":
    main()