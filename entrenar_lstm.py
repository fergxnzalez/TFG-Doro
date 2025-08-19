import os,sys
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Masking
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint

PRUEBAS = {
    "banderas": 0,
    "objetos": 1,
    "patrones": 2,
    "diferencias": 3
}

if len(sys.argv) < 2 or sys.argv[1] not in PRUEBAS:
    print("Uso: python calcular_embeddings_entrenamiento [banderas|objetos|patrones|diferencias]")
    sys.exit(1)

cap = PRUEBAS[sys.argv[1]]
cap_str = str(cap)

input_dir = "vectores_salida"+cap_str
# Archivos .npy ordenados
archivos = sorted([f for f in os.listdir(input_dir) if f.endswith('.npy')])

X = []
y = []

for i in range(len(archivos) - 4):
    ventana = archivos[i:i+5]

    vectores = [np.load(os.path.join(input_dir, archivo)) for archivo in ventana]

    input_vectores = [vectores[0], vectores[1], vectores[3], vectores[4]]
    X.append(np.stack(input_vectores))
    y.append(vectores[2])

X = np.array(X)
y = np.array(y)

print(f"Datos preparados: X={X.shape}, y={y.shape}")

model = Sequential([
    Masking(mask_value=0.0, input_shape=(4, 2048)),
    LSTM(512, return_sequences=True),
    LSTM(512, return_sequences=False),
    Dense(2048)
])

model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')

checkpoint = ModelCheckpoint("modelo_lstm_"+cap_str+".h5", save_best_only=True, monitor='loss', mode='min')

model.fit(X, y, epochs=50, batch_size=8, callbacks=[checkpoint])

print("Modelo entrenado y guardado como 'modelo_lstm_"+cap_str+".h5'")
