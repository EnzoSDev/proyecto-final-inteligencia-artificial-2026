"""Genera el notebook 02_unet_segmentacion.ipynb.

Uso (desde la raíz del proyecto):  python scripts/crear_nb_02_unet.py
"""
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parent.parent
celdas = []


def md(texto):
    celdas.append(nbf.v4.new_markdown_cell(texto.strip()))


def code(texto):
    celdas.append(nbf.v4.new_code_cell(texto.strip()))


md("""
# 02 · UNet: encontrar la hoja en la foto

La **UNet** es una red que mira una foto y responde, **píxel por píxel**, si ese píxel es
hoja o fondo. Su salida es una máscara: blanco = hoja, negro = fondo.

- **Aprende** de las máscaras automáticas (notebook 01), sin las rechazadas.
- **Se evalúa** con las 50 máscaras corregidas a mano, que nunca ve al entrenar.
- La comparamos con el **truco automático** (saturación): si la UNet saca mejor puntaje,
  aprendió algo más que copiar el truco.

**Dónde correrlo:** el entrenamiento va en **Google Colab con GPU**
(*Entorno de ejecución → Cambiar tipo de entorno → GPU T4*). En la compu solo se prueba
con `SMOKE=1`.

**Antes de abrirlo en Colab:** subir a `Mi unidad/Proyecto_Final_IA/` el archivo
`dataset_unet.zip`, que se genera con `python scripts/armar_dataset_unet.py`.
""")

md("""
## 1. Preparar el entorno
En Colab conecta Google Drive (pide permiso) y descomprime los datos en `/content`.
En la compu no hace nada especial.
""")

code("""
import sys
EN_COLAB = 'google.colab' in sys.modules
if EN_COLAB:
    from google.colab import drive
    drive.mount('/content/drive')
    sys.path.insert(0, '/content/drive/MyDrive/Proyecto_Final_IA')

import utils
entorno = utils.configurar_entorno(['dataset_unet'])
ROOT, DATA = entorno['ROOT'], entorno['DATA']
""")

code("""
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from skimage.io import imread
from tensorflow.keras.layers import (Input, Rescaling, Conv2D, Dropout, MaxPooling2D,
                                     Conv2DTranspose, concatenate)
from tensorflow.keras.models import Model

utils.set_estilo()
tf.keras.utils.set_random_seed(utils.SEED)   # mismos resultados cada vez que se corre
""")

md("""
## 2. Configuración
- `SMOKE`: modo de prueba rápida (pocas fotos, 1 época). Se activa con la variable de
  entorno `SMOKE=1`; en Colab queda apagado.
- `REENTRENAR`: si ya hay un modelo guardado y está en `False`, lo carga en vez de entrenar
  de nuevo (útil si Colab cortó la sesión después de entrenar).
- `EPOCAS` es un máximo: el entrenamiento **se corta solo** cuando deja de mejorar en
  validación (*early stopping*) y se queda con la mejor época.
""")

code("""
SMOKE = os.environ.get('SMOKE') == '1'
REENTRENAR = False

EPOCAS = 1 if SMOKE else 50
PACIENCIA = 6            # épocas sin mejorar en validación antes de cortar
BATCH = 16
LADO = utils.IMG_SEG     # 160

CARPETA = DATA / 'dataset_unet'
NOMBRE = 'unet_smoke' if SMOKE else 'unet'          # la prueba no pisa el modelo real
RUTA_MODELO = ROOT / 'models' / f'{NOMBRE}.keras'
RUTA_MODELO.parent.mkdir(exist_ok=True)
(ROOT / 'resultados').mkdir(exist_ok=True)
print('SMOKE =', SMOKE, '| modelo:', RUTA_MODELO.name)
""")

md("""
## 3. Cargar los datos
`dataset_unet/` ya tiene todo listo en 160×160: en `images/` las fotos y en `masks/` sus
máscaras, **con el mismo nombre de archivo**.

| Parte | Fotos | Máscaras | Para qué |
|---|---|---|---|
| train | 1542 | automáticas | aprender |
| val | 332 | automáticas | decidir cuándo cortar el entrenamiento |
| test | 50 | **corregidas a mano** | el examen final |

Leemos los nombres desde `lista.csv` (siempre en el mismo orden), así cada foto queda con
**su** máscara. Las máscaras pasan de 0/255 a **0/1**, que es lo que espera la red.
""")

code("""
lista = pd.read_csv(CARPETA / 'lista.csv')
if SMOKE:   # prueba: unas pocas fotos de cada clase
    lista = pd.concat([lista[lista['split'] == 'test'],
                       lista[lista['split'] != 'test'].groupby(['split', 'clase']).head(10)])


def cargar(split, carpeta_mascaras='masks'):
    nombres = sorted(lista.loc[lista['split'] == split, 'archivo'])
    X = np.asarray([imread(CARPETA / split / 'images' / n) for n in nombres], dtype=np.uint8)
    Y = np.asarray([imread(CARPETA / split / carpeta_mascaras / n) for n in nombres])
    Y = (Y > 127).astype(np.float32)[..., None]      # (N, 160, 160, 1) con 0 y 1
    return X, Y, nombres


X_train, Y_train, _ = cargar('train')
X_val, Y_val, _ = cargar('val')
X_test, Y_test, nombres_test = cargar('test')                    # NO se usa para entrenar
_, Y_test_auto, _ = cargar('test', 'masks_auto')                # el truco, para comparar

for nombre, X, Y in [('train', X_train, Y_train), ('val', X_val, Y_val), ('test', X_test, Y_test)]:
    print(f'{nombre:5s} fotos {X.shape}  máscaras {Y.shape}  hoja = {Y.mean():.0%} de los píxeles')
""")

code("""
fig, ax = plt.subplots(2, 5, figsize=(20, 8))
for i in range(5):
    k = i * len(X_train) // 5
    ax[0, i].imshow(X_train[k])
    ax[1, i].imshow(Y_train[k, :, :, 0], cmap='gray')
    ax[0, i].axis('off'); ax[1, i].axis('off')
ax[0, 0].set_title('Foto', loc='left'); ax[1, 0].set_title('Máscara (lo que debe aprender)', loc='left')
fig.tight_layout()
utils.guardar_figura(fig, 'unet_ejemplos_train')
plt.show()
""")

md("""
## 4. Aumentación de datos
Cada vez que la red ve una foto, la **espejamos al azar** (izquierda-derecha y arriba-abajo),
y a la máscara le hacemos **lo mismo**. Una hoja espejada sigue siendo una hoja, así que es
como tener más fotos distintas sin conseguir fotos nuevas. Solo se hace en train.
""")

code("""
def espejar(foto, mascara):
    if tf.random.uniform(()) > 0.5:
        foto, mascara = tf.image.flip_left_right(foto), tf.image.flip_left_right(mascara)
    if tf.random.uniform(()) > 0.5:
        foto, mascara = tf.image.flip_up_down(foto), tf.image.flip_up_down(mascara)
    return foto, mascara


datos_train = (tf.data.Dataset.from_tensor_slices((X_train, Y_train))
               .shuffle(len(X_train), seed=utils.SEED)
               .map(espejar, num_parallel_calls=tf.data.AUTOTUNE)
               .batch(BATCH).prefetch(tf.data.AUTOTUNE))
datos_val = tf.data.Dataset.from_tensor_slices((X_val, Y_val)).batch(BATCH)
""")

md("""
## 5. La UNet
Es la misma arquitectura del notebook base (`app/Unet.ipynb`), la versión **con Dropout**:

- **Camino de bajada (codificador):** convoluciones + *max pooling*. La imagen se achica
  (160 → 80 → 40 → 20 → 10) y la red aprende *qué* hay.
- **Camino de subida (decodificador):** convoluciones transpuestas que la agrandan de nuevo
  hasta 160×160, para decir *dónde* está.
- **Conexiones de salto (`concatenate`):** le pasan a la subida los detalles finos de la
  bajada, para que los bordes de la hoja salgan precisos.
- **Dropout:** apaga neuronas al azar mientras entrena, para que no memorice las fotos.
- **Salida:** un número entre 0 y 1 por píxel (`sigmoid`) = probabilidad de que sea hoja.
""")

code("""
def construir_unet():
    Image_input = Input((LADO, LADO, 3))
    inputs = Rescaling(1/255)(Image_input)   # pasa los píxeles de 0-255 a 0-1

    # camino de bajada
    conv1 = Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    conv1 = Dropout(0.1)(conv1)
    conv1 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv1)
    maxp1 = MaxPooling2D((2, 2))(conv1)

    conv2 = Conv2D(128, (3, 3), activation='relu', padding='same')(maxp1)
    conv2 = Dropout(0.1)(conv2)
    conv2 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv2)
    maxp2 = MaxPooling2D((2, 2))(conv2)

    conv3 = Conv2D(256, (3, 3), activation='relu', padding='same')(maxp2)
    conv3 = Dropout(0.2)(conv3)
    conv3 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv3)
    maxp3 = MaxPooling2D((2, 2))(conv3)

    conv4 = Conv2D(512, (3, 3), activation='relu', padding='same')(maxp3)
    conv4 = Dropout(0.2)(conv4)
    conv4 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv4)
    maxp4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(1024, (3, 3), activation='relu', padding='same')(maxp4)
    conv5 = Dropout(0.3)(conv5)
    conv5 = Conv2D(1024, (3, 3), activation='relu', padding='same')(conv5)

    # camino de subida (con conexiones de salto)
    up6 = Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same')(conv5)
    up6 = concatenate([up6, conv4])
    conv6 = Conv2D(512, (3, 3), activation='relu', padding='same')(up6)
    conv6 = Dropout(0.2)(conv6)
    conv6 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv6)

    up7 = Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv6)
    up7 = concatenate([up7, conv3])
    conv7 = Conv2D(256, (3, 3), activation='relu', padding='same')(up7)
    conv7 = Dropout(0.2)(conv7)
    conv7 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv7)

    up8 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv7)
    up8 = concatenate([up8, conv2])
    conv8 = Conv2D(128, (3, 3), activation='relu', padding='same')(up8)
    conv8 = Dropout(0.1)(conv8)
    conv8 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv8)

    up9 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv8)
    up9 = concatenate([up9, conv1], axis=3)
    conv9 = Conv2D(64, (3, 3), activation='relu', padding='same')(up9)
    conv9 = Dropout(0.1)(conv9)
    conv9 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv9)

    outputs = Conv2D(1, (1, 1), activation='sigmoid')(conv9)
    return Model(inputs=[Image_input], outputs=[outputs])
""")

md("""
## 6. Entrenar
- **Pérdida:** `binary_crossentropy`, porque cada píxel es una pregunta de sí o no (¿es hoja?).
- **Métrica:** **IoU** (*Intersection over Union*). Compara la máscara predicha con la
  correcta: la parte donde coinciden dividida por todo lo que marcó alguna de las dos.
  1 = perfecta, 0 = nada que ver. Es mejor que el *accuracy*: si la hoja ocupa poco de la
  foto, una red que dice "todo fondo" tendría mucho accuracy y un IoU de 0.
- **Callbacks:** cortar si 6 épocas seguidas no mejora en validación y guardar la mejor época.

**Ojo:** el IoU de val se calcula contra máscaras **automáticas**, que tienen errores.
El número que vale es el de test (sección 7), contra las corregidas a mano.
""")

code("""
if RUTA_MODELO.exists() and not REENTRENAR:
    print('Cargando el modelo ya entrenado:', RUTA_MODELO.name)
    model = tf.keras.models.load_model(RUTA_MODELO)
else:
    model = construir_unet()
    model.compile(optimizer='adam', loss='binary_crossentropy',
                  metrics=[tf.keras.metrics.BinaryIoU(target_class_ids=[1], threshold=0.5, name='iou')])
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=PACIENCIA, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(RUTA_MODELO, monitor='val_loss', save_best_only=True),
    ]
    historia = model.fit(datos_train, validation_data=datos_val, epochs=EPOCAS, callbacks=callbacks)
    pd.DataFrame(historia.history).to_csv(ROOT / 'resultados' / f'{NOMBRE}_historia.csv', index_label='epoca')
    model.save(RUTA_MODELO)

print(f'{model.count_params():,} parámetros')
""")

code("""
ruta_historia = ROOT / 'resultados' / f'{NOMBRE}_historia.csv'
if ruta_historia.exists():
    h = pd.read_csv(ruta_historia)
    epocas = h['epoca'] + 1
    fig, ax = plt.subplots(1, 2, figsize=(18, 6))
    ax[0].plot(epocas, h['loss'], 'o-', label='train'); ax[0].plot(epocas, h['val_loss'], 'o-', label='val')
    ax[0].set_title('Pérdida (más bajo = mejor)'); ax[0].set_xlabel('Época'); ax[0].legend()
    ax[1].plot(epocas, h['iou'], 'o-', label='train'); ax[1].plot(epocas, h['val_iou'], 'o-', label='val')
    ax[1].set_title('IoU (más alto = mejor)'); ax[1].set_xlabel('Época'); ax[1].legend()
    fig.tight_layout()
    utils.guardar_figura(fig, f'{NOMBRE}_curvas')
    plt.show()
    print('Mejor época (menor pérdida en val):', int(h['val_loss'].idxmin()) + 1, 'de', len(h))
""")

md("""
## 7. El examen: IoU en las 50 fotos de test
Ahora sí usamos test: la UNet predice las 50 fotos y comparamos con las máscaras corregidas
a mano. Para pasar de probabilidad a máscara, todo lo que da más de 0.5 cuenta como hoja.

Medimos lo mismo para el **truco automático**. Así sabemos si la UNet mejoró al método del
que aprendió.
""")

code("""
def iou(verdad, prediccion):
    verdad, prediccion = verdad > 0.5, prediccion > 0.5
    union = np.logical_or(verdad, prediccion).sum()
    return np.logical_and(verdad, prediccion).sum() / union if union else 1.0


P_test = model.predict(X_test, batch_size=BATCH, verbose=0)
clase_de = dict(zip(lista['archivo'], lista['clase']))
res = pd.DataFrame({
    'archivo': nombres_test,
    'clase': [clase_de[n] for n in nombres_test],
    'iou_unet': [iou(Y_test[i], P_test[i]) for i in range(len(X_test))],
    'iou_truco': [iou(Y_test[i], Y_test_auto[i]) for i in range(len(X_test))],
})
res.to_csv(ROOT / 'resultados' / f'{NOMBRE}_iou_test.csv', index=False)

tabla = res.groupby('clase')[['iou_unet', 'iou_truco']].mean()
tabla.loc['PROMEDIO'] = res[['iou_unet', 'iou_truco']].mean()
tabla.columns = ['UNet', 'Truco automático']
print(tabla.round(3).to_string())
print('\\nFotos donde la UNet le gana al truco:', (res['iou_unet'] > res['iou_truco']).sum(), 'de', len(res))
""")

code("""
fig, ax = plt.subplots(figsize=(12, 6))
tabla.drop('PROMEDIO').plot.bar(ax=ax, rot=0)
ax.set_ylim(0, 1); ax.set_ylabel('IoU promedio (test)'); ax.set_xlabel('')
ax.set_title('UNet vs truco automático (50 fotos de test)')
ax.legend(loc='lower right')
fig.tight_layout()
utils.guardar_figura(fig, f'{NOMBRE}_iou_por_clase')
plt.show()
""")

md("""
## 8. Mirar los resultados
Los números no alcanzan: hay que **ver** dónde acierta y dónde falla. Mostramos las 4 fotos
con mejor IoU y las 4 con peor. Contorno **verde** = máscara correcta (a mano);
contorno **rojo** = lo que predijo la UNet.
""")

code("""
import cv2


def contornos(foto, verdad, prediccion):
    img = foto.copy()
    for mascara, color in [(verdad, (0, 255, 0)), (prediccion, (255, 0, 0))]:
        bordes, _ = cv2.findContours((mascara[..., 0] > 0.5).astype(np.uint8),
                                     cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(img, bordes, -1, color, 1)
    return img


orden = res.sort_values('iou_unet').index
elegidas = list(orden[-4:][::-1]) + list(orden[:4])
fig, ax = plt.subplots(2, 4, figsize=(20, 11))
for a, i in zip(ax.flat, elegidas):
    a.imshow(contornos(X_test[i], Y_test[i], P_test[i]))
    a.set_title(f"{res.loc[i, 'clase']}\\nIoU {res.loc[i, 'iou_unet']:.2f}", fontsize=16)
    a.axis('off')
ax[0, 0].text(-20, 80, 'MEJORES', rotation=90, fontsize=20, va='center')
ax[1, 0].text(-20, 80, 'PEORES', rotation=90, fontsize=20, va='center')
fig.tight_layout()
utils.guardar_figura(fig, f'{NOMBRE}_mejores_peores')
plt.show()
""")

md("""
## 9. Conclusiones
*Completar después de entrenar en Colab, solo con los números que salieron arriba:*

- IoU promedio de la UNet en test: …
- IoU promedio del truco automático: …
- ¿En qué clases o tipos de foto falla más? (mirar la figura de peores) …

**Recordar:** la UNet aprendió de máscaras automáticas con errores, así que puede repetirlos.
Además, el test son solo 50 fotos y casi todas tienen fondo liso: no sabemos cómo segmenta
fotos sacadas en el viñedo.

El modelo quedó guardado en `models/unet.keras`. Lo usan el notebook 03 (para recortar las
hojas) y la app.
""")

nb = nbf.v4.new_notebook()
nb['cells'] = celdas
nb['metadata']['kernelspec'] = {'name': 'python3', 'display_name': 'Python 3', 'language': 'python'}
nb['metadata']['accelerator'] = 'GPU'
nbf.write(nb, ROOT / '02_unet_segmentacion.ipynb')
print('Creado 02_unet_segmentacion.ipynb')
