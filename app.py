import streamlit as st

# 1. ESTA DEBE SER LA PRIMERA LÍNEA
st.set_page_config(page_title="Movie AI Analysis Pro", layout="wide")

import cv2
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
from insightface.app import FaceAnalysis
from scipy.spatial.distance import cosine
import math
import json
import networkx as nx
import matplotlib.pyplot as plt
from collections import deque, Counter
import tempfile
import os

# ==========================================
# 2. CARGA DE MODELOS
# ==========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@st.cache_resource
def load_emotion_model(model_path):
    if not os.path.exists(model_path): return None, "Error"
    checkpoint = torch.load(model_path, map_location=device)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        checkpoint = checkpoint['model_state_dict']
    
    # Detección de arquitectura
    if any(k.startswith('features') for k in checkpoint.keys()):
        model = models.efficientnet_b0()
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.4), nn.Linear(in_features, 512),
            nn.ReLU(), nn.BatchNorm1d(512), nn.Dropout(p=0.2), nn.Linear(512, 8)
        )
        name = "EfficientNet-B0 (EMOTIC)"
    else:
        model = models.resnet18()
        num_ftrs = model.fc.in_features
        model.fc = nn.Sequential(
            nn.BatchNorm1d(num_ftrs), nn.Linear(num_ftrs, 512),
            nn.ReLU(), nn.Dropout(0.5), nn.Linear(512, 8)
        )
        name = "ResNet18 (AffectNet)"
    
    model.load_state_dict(checkpoint, strict=False)
    model.to(device).eval()
    return model, name

@st.cache_resource
def load_face_app():
    face_app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
    face_app.prepare(ctx_id=0, det_size=(640, 640))
    return face_app

# ==========================================
# 3. LÓGICA DE INTERACCIÓN Y GESTIÓN
# ==========================================

def get_gaze_vector(kps):
    eye_span = abs(kps[1][0] - kps[0][0])
    nose_x = kps[2][0]
    ratio_x = (nose_x - kps[0][0]) / (eye_span + 1e-6)
    dx = (ratio_x - 0.5) * 150 
    eyes_center_y = (kps[0][1] + kps[1][1]) / 2
    mouth_center_y = (kps[3][1] + kps[4][1]) / 2
    ratio_y = (kps[2][1] - eyes_center_y) / (mouth_center_y - eyes_center_y + 1e-6)
    dy = (ratio_y - 0.38) * 80 
    return dx, dy

def check_interaction(p1, p2, is_video=True):
    vx, vy = p1["gaze_vector"]
    nx, ny = p1["nose"]
    # Filtro paralelo
    vx2, vy2 = p2["gaze_vector"]
    m1, m2 = math.hypot(vx, vy), math.hypot(vx2, vy2)
    if m1 > 0 and m2 > 0:
        if (vx*vx2 + vy*vy2) / (m1*m2) > 0.85: return False
    # Hitbox
    tx1, ty1, tx2, ty2 = p2["bbox"]
    w, h = tx2 - tx1, ty2 - ty1
    tol = 0.15 if is_video else 0.25
    hitbox = (tx1 + w*tol, ty1 + h*tol, tx2 - w*tol, ty2 - h*tol)
    # Ray casting
    for t in np.linspace(0, 25, 50):
        px, py = nx + t*vx, ny + t*vy
        if hitbox[0] <= px <= hitbox[2] and hitbox[1] <= py <= hitbox[3]:
            return True
    return False

class IdentityManager:
    def __init__(self, threshold=0.48):
        self.personajes, self.candidatos, self.id_counter = [], [], 0
        self.threshold = threshold

    def get_identity(self, current_emb):
        if current_emb is None: return "Unknown"
        for p in self.personajes:
            for saved_emb in p["gallery"]:
                if cosine(saved_emb, current_emb) < self.threshold:
                    if len(p["gallery"]) < 10: p["gallery"].append(current_emb)
                    return p["id"]
        for idx, cand in enumerate(self.candidatos):
            for saved_emb in cand["gallery"]:
                if cosine(saved_emb, current_emb) < self.threshold:
                    cand["hits"] += 1
                    if cand["hits"] >= 10:
                        new_id = f"Personaje_{self.id_counter}"
                        self.personajes.append({"id": new_id, "gallery": cand["gallery"]})
                        self.id_counter += 1
                        self.candidatos.pop(idx)
                        return new_id
                    return f"Candidato_{idx}"
        self.candidatos.append({"gallery": [current_emb], "hits": 1})
        return f"Candidato_{len(self.candidatos)-1}"

# ==========================================
# 4. INTERFAZ Y PROCESAMIENTO
# ==========================================

st.sidebar.title("🎬 Movie AI Pro")
app_mode = st.sidebar.selectbox("Modo", ["Video", "Foto", "📊 Métricas"])
model_file = st.sidebar.file_uploader("Modelo (.pth)", type=['pth'])

face_app = load_face_app()
emotion_net = None

if model_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pth") as tmp:
        tmp.write(model_file.read())
        emotion_net, arch = load_emotion_model(tmp.name)
        st.sidebar.success(f"Modelo: {arch}")

def predict_emo(face_crop, kps=None):
    if face_crop.size == 0 or emotion_net is None: return "neutral"
    t = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), 
                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    img = t(Image.fromarray(cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB))).unsqueeze(0).to(device)
    with torch.no_grad():
        out = emotion_net(img)
        out[0][4] += 1.4 # Boost Happy
        probs = torch.softmax(out, dim=1)
        _, pred = torch.max(probs, 1)
    labels = {0:'anger', 1:'contempt', 2:'disgust', 3:'fear', 4:'happy', 5:'neutral', 6:'sad', 7:'surprise'}
    res = labels[pred.item()]
    if kps is not None:
        if (np.linalg.norm(kps[3]-kps[4])/np.linalg.norm(kps[0]-kps[1])) > 0.83: return "happy"
    return res

# --- MODO VIDEO ---
if app_mode == "Video":
    v_file = st.file_uploader("Subir Video", type=['mp4'])
    if v_file and emotion_net:
        if st.button("🚀 Iniciar Análisis"):
            tvideo = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            tvideo.write(v_file.read())
            cap = cv2.VideoCapture(tvideo.name)
            id_m, history, buffers = IdentityManager(), {"personajes": {}, "interacciones": []}, {}
            st_frame = st.empty()
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret: break
                if int(cap.get(cv2.CAP_PROP_POS_FRAMES)) % 6 == 0:
                    faces = face_app.get(frame)
                    current_data = []
                    for f in faces:
                        cid = id_m.get_identity(f.normed_embedding)
                        bbox = f.bbox.astype(int)
                        dx, dy = get_gaze_vector(f.kps)
                        emo_st = "..."
                        if "Personaje" in cid:
                            raw = predict_emo(frame[max(0,bbox[1]):bbox[3], max(0,bbox[0]):bbox[2]], f.kps)
                            if cid not in buffers: buffers[cid] = deque(maxlen=12)
                            buffers[cid].append(raw)
                            emo_st = Counter(buffers[cid]).most_common(1)[0][0]
                            if cid not in history["personajes"]: history["personajes"][cid] = {"conteo": {}}
                            history["personajes"][cid]["conteo"][emo_st] = history["personajes"][cid]["conteo"].get(emo_st, 0) + 1
                        
                        current_data.append({"id": cid, "bbox": bbox, "center": ((bbox[0]+bbox[2])//2, (bbox[1]+bbox[3])//2),
                                             "nose": (int(f.kps[2][0]), int(f.kps[2][1])), "gaze_vector": (dx, dy), "emotion": emo_st})

                    for p1 in current_data:
                        is_p = "Personaje" in p1["id"]
                        color = (0, 255, 0) if is_p else (0, 255, 255)
                        cv2.rectangle(frame, (p1["bbox"][0], p1["bbox"][1]), (p1["bbox"][2], p1["bbox"][3]), color, 2)
                        cv2.putText(frame, f"{p1['id']}: {p1['emotion']}", (p1["bbox"][0], p1["bbox"][1]-10), 0, 0.5, color, 1)
                        for p2 in current_data:
                            if p1["id"] != p2["id"]:
                                if check_interaction(p1, p2, is_video=True):
                                    if is_p and "Personaje" in p2["id"]:
                                        history["interacciones"].append({"de": p1["id"], "a": p2["id"], "emo": p1["emotion"]})
                                    cv2.arrowedLine(frame, p1["nose"], p2["center"], (0, 255, 255), 2)
                    st_frame.image(frame, channels="BGR", use_column_width=True)
            cap.release()
            st.session_state["history"] = history # GUARDAR EN SESIÓN

    if "history" in st.session_state:
        st.divider()
        h = st.session_state["history"]
        
        # 1. RENOMBRAR PERSONAJES
        st.subheader("👤 Personalización de Personajes")
        p_keys = list(h["personajes"].keys())
        name_map = {}
        if p_keys:
            cols = st.columns(len(p_keys))
            for i, pk in enumerate(p_keys):
                with cols[i]:
                    name_map[pk] = st.text_input(f"Nombre para {pk}", pk)
        
        # 2. GRAFO Y RESUMEN
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("📊 Grafo de Relaciones")
            G = nx.DiGraph()
            for i in h["interacciones"]:
                if i["emo"] != "neutral":
                    G.add_edge(name_map[i["de"]], name_map[i["a"]], label=i["emo"])
            
            if G.nodes:
                fig, ax = plt.subplots(figsize=(5, 4))
                pos = nx.spring_layout(G)
                nx.draw(G, pos, with_labels=True, node_color='orange', ax=ax, font_size=8, node_size=800)
                edge_labels = nx.get_edge_attributes(G, 'label')
                nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=7)
                st.pyplot(fig)
            else: st.write("Sin miradas emocionales detectadas.")

        with c2:
            st.subheader("📖 Resumen del Modelo")
            for pid, info in h["personajes"].items():
                name = name_map.get(pid, pid)
                emos_vistas = [k for k, v in info["conteo"].items() if v > 5]
                st.write(f"**{name}** experimentó: {', '.join(emos_vistas) if emos_vistas else 'Calma'}")
            
            inter = [i for i in h["interacciones"] if i["emo"] != "neutral"]
            if inter:
                (de, a, emo), _ = Counter([(i['de'], i['a'], i['emo']) for i in inter]).most_common(1)[0]
                st.info(f"Narrativa: {name_map[de]} interactuó frecuentemente con {name_map[a]} bajo un sentimiento de {emo}.")

        # 3. VER JSON
        with st.expander("📂 Ver Datos Crudos (JSON)"):
            st.json(h)

# --- MODO FOTO ---
elif app_mode == "Foto" and emotion_net:
    img_f = st.file_uploader("Subir Imagen", type=["jpg", "png", "jpeg"])
    if img_f:
        frame = cv2.cvtColor(np.array(Image.open(img_f)), cv2.COLOR_RGB2BGR)
        faces = face_app.get(frame)
        data = []
        for i, f in enumerate(faces):
            bbox = f.bbox.astype(int)
            dx, dy = get_gaze_vector(f.kps)
            emo = predict_emo(frame[max(0,bbox[1]):bbox[3], max(0,bbox[0]):bbox[2]], f.kps)
            data.append({"id": f"Sujeto_{i}", "bbox": bbox, "center": ((bbox[0]+bbox[2])//2, (bbox[1]+bbox[3])//2),
                         "nose": (int(f.kps[2][0]), int(f.kps[2][1])), "gaze_vector": (dx, dy), "emotion": emo})
        for p1 in data:
            cv2.rectangle(frame, (p1["bbox"][0], p1["bbox"][1]), (p1["bbox"][2], p1["bbox"][3]), (0, 255, 0), 2)
            cv2.putText(frame, f"{p1['id']}: {p1['emotion']}", (p1["bbox"][0], p1["bbox"][1]-10), 0, 0.6, (0, 255, 0), 2)
            for p2 in data:
                if p1["id"] != p2["id"] and check_interaction(p1, p2, is_video=False):
                    cv2.arrowedLine(frame, p1["nose"], p2["center"], (0, 255, 255), 3)
        st.image(frame, channels="BGR", use_column_width=True)
        st.json(data)

# --- MODO MÉTRICAS ---
elif app_mode == "📊 Métricas":
    st.header("Análisis de Modelos")
    st.write("- **ResNet18 (Laboratory):** 70.7% Accuracy. Muy bueno en condiciones controladas.")
    st.write("- **EfficientNet-B0 (Cinema):** 41.1% Accuracy. Superior en video real y sombras cinematográficas.")