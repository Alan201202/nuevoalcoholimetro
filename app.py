# app.py
import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import av
import cv2
import time
import urllib.request
import os
import json
import threading
import numpy as np
import pandas as pd
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Importaciones de los módulos del backend
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

# Configuración de página de Streamlit
st.set_page_config(
    page_title="Sistema Biométrico de Sobriedad - IPN ESIME",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS Personalizado
st.markdown("""
<style>
    .block-container {
        padding-top: 1rem !important;
        padding-bottom: 1rem !important;
        padding-left: 2rem !important;
        padding-right: 2rem !important;
        max-width: 100% !important;
    }
    .stApp {
        background-color: #090d11;
        color: #d1d5db;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    }
    section[data-testid="stSidebar"] {
        background-color: #111822 !important;
        border-right: 1px solid #1f293d;
    }
    .bio-panel {
        background-color: #111822;
        border: 1px solid #1e293b;
        border-radius: 12px;
        padding: 16px;
        margin-bottom: 12px;
    }
    .bio-label {
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #64748b;
        margin-bottom: 4px;
        font-weight: 600;
    }
    .bio-val {
        font-size: 1.5rem;
        font-weight: 700;
        color: #f8fafc;
    }
    .tag-ok {
        background-color: rgba(16, 185, 129, 0.1);
        color: #10b981;
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 0.78rem;
        font-weight: 600;
        border: 1px solid rgba(16, 185, 129, 0.25);
    }
    .tag-warn {
        background-color: rgba(245, 158, 11, 0.1);
        color: #f59e0b;
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 0.78rem;
        font-weight: 600;
        border: 1px solid rgba(245, 158, 11, 0.25);
    }
    .tag-danger {
        background-color: rgba(239, 68, 68, 0.1);
        color: #ef4444;
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 0.78rem;
        font-weight: 600;
        border: 1px solid rgba(239, 68, 68, 0.25);
    }
    .title-header {
        font-size: 1.35rem;
        font-weight: 600;
        color: #f8fafc;
        margin-bottom: 0px;
    }
    .subtitle-header {
        font-size: 0.85rem;
        color: #64748b;
        margin-bottom: 12px;
    }
</style>
""", unsafe_allow_html=True)

MODEL_PATH = "face_landmarker.task"
TEST_DURATION_SEC = 30

@st.cache_resource
def load_face_detector():
    if not os.path.exists(MODEL_PATH):
        url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
        urllib.request.urlretrieve(url, MODEL_PATH)
    base_options = python.BaseOptions(
        model_asset_path=MODEL_PATH,
        delegate=python.BaseOptions.Delegate.CPU
    )
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
        num_faces=1
    )
    return vision.FaceLandmarker.create_from_options(options)

detector = load_face_detector()

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
        stage = "Persecucion Suave (Der)"
    elif cycle_time < 5.0:
        cx = max_x
        stage = "Fijacion Extrema (Der)"
    elif cycle_time < 8.0:
        ratio = (cycle_time - 5.0) / 3.0
        cx = int(max_x - ratio * range_x)
        stage = "Persecucion Suave (Izq)"
    else:
        cx = min_x
        stage = "Fijacion Extrema (Izq)"

    return cx, center_y, stage

def get_tag_html(val, norm_max, warn_max):
    if val <= norm_max:
        return '<span class="tag-ok">NORMAL</span>'
    elif val <= warn_max:
        return '<span class="tag-warn">PRECAUCIÓN</span>'
    else:
        return '<span class="tag-danger">ALTERADO</span>'

# --- PROCESADOR DE VIDEO WEBRTC ---
class BiometricVideoProcessor:
    def __init__(self):
        self.lock = threading.Lock()
        self.calibrator = Calibrator(calibration_time=CALIBRATION_TIME_SEC)
        self.calibrator.start()
        self.gaze_tracker = GazeTracker(buffer_size=30)
        self.head_pose = HeadPoseEstimator(buffer_size=30)
        self.ml_engine = MLEngine()
        self.temporal_extractor = TemporalFeatureExtractor(window_seconds=15.0, fps=30.0)

        self.eye_closed = False
        self.closed_start_time = 0.0
        self.last_blink_duration = 0.0

        self.history_ml = []
        self.history_heur = []
        self.history_final = []
        self.history_jitter = []
        self.history_perclos = []
        self.history_bpm = []
        self.history_pitch_std = []

        self.test_started = False
        self.test_start_time = 0.0
        self.test_finished = False
        self.remaining_test = TEST_DURATION_SEC
        self.current_prob = 0.0
        self.current_stage = "Calibrando..."

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)
        h, w, _ = img.shape

        if self.test_finished:
            cv2.rectangle(img, (0, 0), (w, 80), (16, 185, 129), -1)
            cv2.putText(img, "PRUEBA COMPLETADA - PRESIONE 'GENERAR REPORTE'", (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
            return av.VideoFrame.from_ndarray(img, format="bgr24")

        rgb_frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        detection_result = detector.detect(mp_image)

        if detection_result.face_landmarks:
            landmarks = detection_result.face_landmarks[0]

            left_ear = calculate_ear(LEFT_EYE, landmarks, w, h)
            right_ear = calculate_ear(RIGHT_EYE, landmarks, w, h)
            current_ear = (left_ear + right_ear) / 2.0

            gaze_jitter, asymmetry, _ = self.gaze_tracker.process(landmarks, w, h, left_ear, right_ear)
            pitch, yaw, roll, head_instability = self.head_pose.process(landmarks, w, h)

            is_calibrated, calib_val = self.calibrator.process(current_ear)

            if not is_calibrated:
                remaining_calib = int(CALIBRATION_TIME_SEC - calib_val) + 1
                cv2.rectangle(img, (0, 0), (w, 60), (17, 24, 34), -1)
                cv2.putText(img, f"FASE 0: CALIBRANDO LINEA BASE ({remaining_calib}s) - MIRE AL CENTRO", (20, 38),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 229, 255), 2)
            else:
                if not self.test_started:
                    self.test_started = True
                    self.test_start_time = time.time()

                elapsed_test = time.time() - self.test_start_time
                rem_test = max(0, int(TEST_DURATION_SEC - elapsed_test))

                ear_base = calib_val
                threshold_ear = ear_base * EAR_THRESHOLD_FACTOR

                is_blink_event = False
                if current_ear < threshold_ear:
                    if not self.eye_closed:
                        self.eye_closed = True
                        self.closed_start_time = time.time()
                else:
                    if self.eye_closed:
                        self.eye_closed = False
                        self.last_blink_duration = time.time() - self.closed_start_time
                        is_blink_event = True

                curr_closed_dur = (time.time() - self.closed_start_time) if self.eye_closed else 0.0

                self.temporal_extractor.update(current_ear, gaze_jitter, pitch, is_blink_event)
                feats = self.temporal_extractor.get_feature_vector(threshold_ear)

                prob_deterioro, p_ml, p_heur = self.ml_engine.predict_impairment_probability(
                    current_ear, gaze_jitter, asymmetry, max(curr_closed_dur, self.last_blink_duration),
                    pitch, head_instability, feats['PERCLOS'], feats['BPM'], feats['Mean_EAR'], feats['Pitch_STD']
                )

                with self.lock:
                    if not self.eye_closed:
                        self.history_jitter.append(gaze_jitter)
                    self.history_perclos.append(feats['PERCLOS'])
                    self.history_bpm.append(feats['BPM'])
                    self.history_pitch_std.append(feats['Pitch_STD'])
                    self.history_ml.append(p_ml)
                    self.history_heur.append(p_heur)
                    self.history_final.append(prob_deterioro)
                    self.remaining_test = rem_test
                    self.current_prob = prob_deterioro

                cx, cy, stage_name = get_hgn_target_position(elapsed_test, w, h)
                self.current_stage = stage_name

                # Dibujar estímulo HGN y telemetría en el video
                cv2.circle(img, (cx, cy), 14, (0, 0, 238), -1)
                cv2.circle(img, (cx, cy), 4, (255, 255, 255), -1)

                cv2.rectangle(img, (0, 0), (w, 75), (17, 24, 34), -1)
                cv2.putText(img, f"HGN: {stage_name} | Tiempo: {rem_test}s | Riesgo: {prob_deterioro*100:.1f}%",
                            (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
                cv2.putText(img, f"PERCLOS: {feats['PERCLOS']:.1f}% | BPM: {feats['BPM']:.1f} | Jitter: {gaze_jitter:.4f}",
                            (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (16, 185, 129), 2)

                if elapsed_test >= TEST_DURATION_SEC:
                    self.test_finished = True
        else:
            cv2.putText(img, "ROSTRO NO DETECTADO - CENTRESE EN LA CAMARA", (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# --- ENCABEZADO COMPACTO DE TESIS ---
st.markdown('<div class="title-header">Prototipo para Evaluación de Sobriedad Usando un Método No Invasivo</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle-header">Plataforma de Análisis Biométrico Oculomotor y Postural en Tiempo Real &nbsp;|&nbsp; IPN - ESIME Zacatenco</div>', unsafe_allow_html=True)

st.sidebar.markdown("### Control del Protocolo")
st.sidebar.caption("Prueba de Estímulo HGN (30 Segundos)")
st.sidebar.info("1. Presione **START** en el reproductor de video para encender su cámara web.\n2. Al finalizar los 30s, presione **📊 Generar Reporte Biométrico**.")
st.sidebar.markdown("---")
st.sidebar.markdown("""
**Información de Autoría:**
* **Autores:** Alan Zaid Murillo Luna & Jesica Noemi Serralde Huertas
* **Asesor:** M. en C. Armando Martínez Ríos
* **Institución:** IPN - ESIME Zacatenco
""")

col_cam, col_info = st.columns([2.2, 1.0])

RTC_CONFIG = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302", "stun:stun1.l.google.com:19302"]}]}
)

with col_cam:
    ctx = webrtc_streamer(
        key="evaluacion-sobriedad",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIG,
        video_processor_factory=BiometricVideoProcessor,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

with col_info:
    st.markdown("""
    <div class="bio-panel">
        <div class="bio-label">Instrucciones del Protocolo</div>
        <div style="font-size:0.9rem; color:#f8fafc;">
        1. Haga clic en <b>START</b> para activar su cámara en el navegador.<br>
        2. Mire fijo al centro durante los <b>5s de calibración</b>.<br>
        3. Siga la <b>esfera roja</b> con la mirada sin mover la cabeza durante <b>30s</b>.
        </div>
    </div>
    """, unsafe_allow_html=True)

    report_btn = st.button("📊 Generar / Actualizar Reporte Biométrico", type="primary", use_container_width=True)

if report_btn and ctx.video_processor:
    vp = ctx.video_processor
    with vp.lock:
        history_final = list(vp.history_final)
        history_heur = list(vp.history_heur)
        history_ml = list(vp.history_ml)
        history_jitter = list(vp.history_jitter)
        history_perclos = list(vp.history_perclos)
        history_bpm = list(vp.history_bpm)
        history_pitch_std = list(vp.history_pitch_std)

    if history_final:
        st.session_state["report_data"] = {
            "history_final": history_final,
            "history_heur": history_heur,
            "history_ml": history_ml,
            "history_jitter": history_jitter,
            "history_perclos": history_perclos,
            "history_bpm": history_bpm,
            "history_pitch_std": history_pitch_std,
        }
    else:
        st.warning("Aún no hay datos suficientes. Inicia la cámara con START y espera a que concluya la calibración de 5 segundos.")

# --- REPORTE FINAL DESPLEGABLE ---
if "report_data" in st.session_state:
    data = st.session_state["report_data"]
    history_final = data["history_final"]
    history_heur = data["history_heur"]
    history_ml = data["history_ml"]
    history_jitter = data["history_jitter"]
    history_perclos = data["history_perclos"]
    history_bpm = data["history_bpm"]
    history_pitch_std = data["history_pitch_std"]

    fps_approx = 30
    sec_heur = [np.mean(history_heur[i:i+fps_approx]) for i in range(0, len(history_heur), fps_approx)]
    sec_ml = [np.mean(history_ml[i:i+fps_approx]) for i in range(0, len(history_ml), fps_approx)]
    sec_final = [np.mean(history_final[i:i+fps_approx]) for i in range(0, len(history_final), fps_approx)]

    avg_heur = np.mean(sec_heur) * 100
    avg_ml = np.mean(sec_ml) * 100
    avg_final = np.mean(sec_final) * 100

    peak_p85 = np.percentile(sec_final, 85) * 100
    seconds_severe = sum(1 for s in sec_final if s >= 0.50)
    seconds_warning = sum(1 for s in sec_final if 0.20 <= s < 0.50)

    mean_jitter = np.mean(history_jitter) if history_jitter else 0.0
    mean_perclos = np.mean(history_perclos) if history_perclos else 0.0
    mean_bpm = np.mean(history_bpm) if history_bpm else 0.0
    mean_pitch_std = np.mean(history_pitch_std) if history_pitch_std else 0.0

    st.markdown("---")
    st.subheader("📊 Reporte de Indicadores Biométricos Consolidados")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""
        <div class="bio-panel">
            <div class="bio-label">[MOD-B] Nistagmo (Jitter)</div>
            <div class="bio-val">{mean_jitter:.4f}</div>
            <div style="margin-top:8px;">{get_tag_html(mean_jitter, 0.0035, 0.0065)}</div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="bio-panel">
            <div class="bio-label">[MOD-A] Oclusión PERCLOS</div>
            <div class="bio-val">{mean_perclos:.1f}%</div>
            <div style="margin-top:8px;">{get_tag_html(mean_perclos, 15.0, 30.0)}</div>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class="bio-panel">
            <div class="bio-label">[MOD-A] Parpadeo (BPM)</div>
            <div class="bio-val">{mean_bpm:.1f}</div>
            <div style="margin-top:8px;">{"<span class='tag-ok'>NORMAL</span>" if 10 <= mean_bpm <= 25 else "<span class='tag-warn'>FUERA DE RANGO</span>"}</div>
        </div>
        """, unsafe_allow_html=True)
    with c4:
        st.markdown(f"""
        <div class="bio-panel">
            <div class="bio-label">[MOD-C] Estabilidad Pitch</div>
            <div class="bio-val">{mean_pitch_std:.2f}°</div>
            <div style="margin-top:8px;">{get_tag_html(mean_pitch_std, 3.0, 6.0)}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("### 📈 Evaluación Comparativa de Inferencia")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Reglas Biomecánicas (65%)", f"{avg_heur:.1f}%")
    m2.metric("Random Forest ML (35%)", f"{avg_ml:.1f}%")
    m3.metric("Índice Integrado Promedio", f"{avg_final:.1f}%")
    m4.metric("Severidad Sostenida (P85)", f"{peak_p85:.1f}%")

    st.markdown("### 📉 Curva Temporal de Coordinación Visual")
    df_chart = pd.DataFrame({
        "Segundo": np.arange(1, len(sec_final) + 1),
        "Nivel de Deterioro (%)": np.array(sec_final) * 100
    }).set_index("Segundo")
    st.line_chart(df_chart)

    st.markdown("### 📋 Dictamen Técnico Final")
    if peak_p85 >= 50.0 or seconds_severe >= 3:
        st.error("🔴 **[ DICTAMEN: ALTO RIESGO / DETERIORO SEVERO ]**\n\n**Fundamento Fisiológico:** Alteración en el seguimiento nistágmico HGN o desaceleración palpebral fuera de norma.")
    elif peak_p85 >= 20.0 or seconds_warning >= 5:
        st.warning("🟠 **[ DICTAMEN: PRECAUCIÓN / ANOMALÍA LEVE ]**\n\n**Fundamento Fisiológico:** Inestabilidad moderada en la fijación visual o ligeras variaciones en la postura.")
    else:
        st.success("🟢 **[ DICTAMEN: SOBRIO / DESEMPEÑO NORMAL ]**\n\n**Fundamento Fisiológico:** Todas las métricas de coordinación ocular y estabilidad se mantuvieron en parámetros fisiológicos normales.")

    report_json = {
        "titulo_tesis": "Prototipo para evaluación de sobriedad usando un método no invasivo",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "metricas": {
            "gaze_jitter": round(mean_jitter, 5),
            "perclos": round(mean_perclos, 2),
            "bpm": round(mean_bpm, 1),
            "pitch_std": round(mean_pitch_std, 2)
        },
        "puntuaciones": {
            "score_reglas": round(avg_heur, 2),
            "score_ml": round(avg_ml, 2),
            "score_final": round(avg_final, 2),
            "severidad_p85": round(peak_p85, 2)
        }
    }
    st.download_button(
        label="📥 Descargar Reporte Biométrico (JSON)",
        data=json.dumps(report_json, indent=4),
        file_name=f"reporte_sobriedad_{int(time.time())}.json",
        mime="application/json"
    )
