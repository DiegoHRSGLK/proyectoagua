import streamlit as st
import numpy as np
import cv2
from PIL import Image
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler

# Inicialización segura
for key in ['features_list', 'labels_list', 'model', 'scaler']:
    if key not in st.session_state:
        st.session_state[key] = [] if 'list' in key else None

REFERENCE_VALUES = [0, 10, 25, 50, 60, 80, 100, 225, 250]
ROI_COORDS = (50, 150, 100, 200)

def preprocess_image(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.equalizeHist(l)
    lab_eq = cv2.merge((l, a, b))
    img_eq = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)
    img_blur = cv2.GaussianBlur(img_eq, (5, 5), 0)
    return img_blur

def extract_features(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    y1, y2, x1, x2 = ROI_COORDS

    height, width = lab.shape[:2]
    y1 = max(0, min(height-1, y1))
    y2 = max(0, min(height, y2))
    x1 = max(0, min(width-1, x1))
    x2 = max(0, min(width, x2))

    if y2 <= y1 or x2 <= x1:
        raise ValueError("El ROI está fuera de los límites de la imagen")

    roi = lab[y1:y2, x1:x2]
    mean = roi.mean(axis=(0, 1))
    std = roi.std(axis=(0, 1))
    percentiles = np.percentile(roi, [10, 25, 50, 75, 90], axis=(0, 1))
    return np.concatenate([mean, std, percentiles.flatten()])

def train_model(X, y):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = KNeighborsRegressor(n_neighbors=min(3, len(X)), weights='distance', metric='euclidean')
    errors = []
    for i in range(len(X)):
        X_train = np.delete(X_scaled, i, axis=0)
        y_train = np.delete(y, i)
        model.fit(X_train, y_train)
        pred = model.predict([X_scaled[i]])[0]
        errors.append(abs(pred - y[i]))

    avg_error = np.mean(errors)
    accuracy = 100 - (avg_error / 250) * 100
    st.info(f"Evaluación del modelo:\nError promedio: {avg_error:.2f} mg/L\nPrecisión: {accuracy:.1f}%")

    model.fit(X_scaled, y)
    return model, scaler

def predict_with_adjustment(model, scaler, img):
    img_proc = preprocess_image(img)
    features = extract_features(img_proc)
    features_scaled = scaler.transform([features])
    raw_pred = model.predict(features_scaled)[0]
    adjusted = min(REFERENCE_VALUES, key=lambda x: abs(x - raw_pred))
    return raw_pred, adjusted

# ---------------- UI ----------------

st.title("IA para tiras reactivas - Entrenamiento y Predicción")

# --- Cargar imagen de entrenamiento ---
st.subheader("Agregar muestra de entrenamiento")
uploaded_train = st.file_uploader("Imagen de tira con valor conocido", type=["jpg", "jpeg", "png"])
valor_real = st.number_input("Valor real (mg/L)", 0.0, 500.0, step=1.0)

if uploaded_train and st.button("Agregar muestra"):
    try:
        img = Image.open(uploaded_train)
        img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        features = extract_features(preprocess_image(img_cv))
        st.session_state['features_list'].append(features)
        st.session_state['labels_list'].append(valor_real)
        st.success(f"Muestra añadida ({valor_real} mg/L)")
    except Exception as e:
        st.error(f"Error: {e}")

# Mostrar tabla de muestras
if len(st.session_state['labels_list']) > 0:
    st.subheader("Muestras cargadas")
    tabla = {
        "N°": list(range(1, len(st.session_state['labels_list']) + 1)),
        "Valor (mg/L)": st.session_state['labels_list']
    }
    st.table(tabla)

# --- Entrenamiento del modelo ---
if st.button("Entrenar modelo"):
    if len(st.session_state['features_list']) >= 3:
        model, scaler = train_model(
            np.array(st.session_state['features_list']),
            np.array(st.session_state['labels_list'])
        )
        st.session_state['model'] = model
        st.session_state['scaler'] = scaler
        st.success("Modelo entrenado con éxito.")
    else:
        st.warning("Agrega al menos 3 muestras para entrenar.")

# --- Predicción ---
st.subheader("Realizar predicción con muestra nueva")
uploaded_pred = st.file_uploader("Imagen nueva para predecir", type=["jpg", "jpeg", "png"], key="pred_img")

if uploaded_pred:
    if st.session_state['model'] is not None:
        try:
            img_pred = Image.open(uploaded_pred)
            img_cv = cv2.cvtColor(np.array(img_pred), cv2.COLOR_RGB2BGR)
            raw, adjusted = predict_with_adjustment(
                st.session_state['model'],
                st.session_state['scaler'],
                img_cv
            )
            st.image(img_pred, caption="Muestra cargada", use_container_width=True)
            st.metric("Valor estimado", f"{raw:.2f} mg/L")
            st.metric("Valor ajustado (más cercano)", f"{adjusted} mg/L")
        except Exception as e:
            st.error(f"Error al procesar imagen: {e}")
    else:
        st.warning("Primero debes entrenar el modelo.")