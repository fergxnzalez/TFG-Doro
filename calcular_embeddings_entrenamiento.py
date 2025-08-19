import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import os, sys
import numpy as np

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
# Ruta de imágenes de entrenamiento
carpeta_imagenes = "frames"+cap_str
# Carpeta donde se guardan los vectores
carpeta_salida = "vectores_salida"+cap_str
os.makedirs(carpeta_salida, exist_ok=True)

# Cargar modelo ResNet50 preentrenado
modelo = models.resnet50(pretrained=True)
modelo.eval()
modelo = torch.nn.Sequential(*list(modelo.children())[:-1])

# Transformaciones de imagen requeridas por ResNet50
transformaciones = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# Obtener imágenes ordenadas
imagenes = [f for f in os.listdir(carpeta_imagenes) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

# Procesar cada imagen
for nombre_archivo in imagenes:
    ruta_imagen = os.path.join(carpeta_imagenes, nombre_archivo)
    imagen = Image.open(ruta_imagen).convert('RGB')
    imagen_tensor = transformaciones(imagen).unsqueeze(0)

    with torch.no_grad():
        vector = modelo(imagen_tensor)
        vector = vector.squeeze().numpy()
        
    nombre_vector = os.path.splitext(nombre_archivo)[0] + ".npy"
    ruta_salida = os.path.join(carpeta_salida, nombre_vector)
    np.save(ruta_salida, vector)

    print(f"✅ Vector guardado: {ruta_salida}")
