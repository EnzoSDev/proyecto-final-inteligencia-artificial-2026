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
from tensorflow.keras.layers import (Input, Rescaling, Conv2D, BatchNormalization,
                                     Activation, Dropout, MaxPooling2D,
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

EPOCAS_F1 = 1 if SMOKE else 10   # Fase 1: encoder MobileNetV2 congelado (calienta el decodificador)
EPOCAS_F2 = 1 if SMOKE else 40   # Fase 2: fine-tuning completo (con BN del encoder congeladas)
PACIENCIA = 8                    # épocas sin mejorar en validación antes de cortar en Fase 2
LR_F1 = 1e-3
LR_F2 = 1e-4
BATCH = 16
LADO = utils.IMG_SEG             # 160
FACTOR_MANCHA = 4                # repetir mancha_bacteriana solo en train (tiene solo 60 fotos)
FACTOR_MANUAL_TRAIN = 3          # dar más peso a fotos difíciles de train corregidas a mano (si existen)

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


X_train, Y_train, nombres_train = cargar('train')
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
## 4. Balanceo y aumentación de datos (solo en train)
Usamos exactamente la misma aumentación limpia de **v2** (sin fondos sintéticos) para medir
el impacto puro del **Transfer Learning (`MobileNetV2` preentrenado)** en el codificador:

1. **Sobremuestreo de `mancha_bacteriana`:** repetimos ×4 sus 60 fotos **solo en train**.
2. **Aumentación al azar:** espejado H/V, rotaciones de 90° (foto y máscara) y variación suave
   de brillo, contraste y saturación (solo foto).
""")

code("""
clase_de = dict(zip(lista['archivo'], lista['clase']))
origen_de = dict(zip(lista['archivo'], lista['mascara']))
idx_extra = []
for i, n in enumerate(nombres_train):
    rep = 1
    if clase_de[n] == 'mancha_bacteriana':
        rep = max(rep, FACTOR_MANCHA)
    if origen_de.get(n) == 'manual_train':
        rep = max(rep, FACTOR_MANUAL_TRAIN)
    if rep > 1:
        idx_extra.extend([i] * (rep - 1))
X_train_bal = np.concatenate([X_train, X_train[idx_extra]], axis=0) if idx_extra else X_train
Y_train_bal = np.concatenate([Y_train, Y_train[idx_extra]], axis=0) if idx_extra else Y_train
assert len(X_train_bal) == len(Y_train_bal)
n_man = sum(origen_de.get(n) == 'manual_train' for n in nombres_train)
print(f'Train balanceado: {len(X_train)} -> {len(X_train_bal)} fotos '
      f'(+{len(idx_extra)} copias; incluye {n_man} máscaras manuales de train)')


def aumentar(foto, mascara):
    foto = tf.cast(foto, tf.float32) / 255.0

    flip_h = tf.random.uniform(()) > 0.5
    foto = tf.cond(flip_h, lambda: tf.image.flip_left_right(foto), lambda: foto)
    mascara = tf.cond(flip_h, lambda: tf.image.flip_left_right(mascara), lambda: mascara)
    flip_v = tf.random.uniform(()) > 0.5
    foto = tf.cond(flip_v, lambda: tf.image.flip_up_down(foto), lambda: foto)
    mascara = tf.cond(flip_v, lambda: tf.image.flip_up_down(mascara), lambda: mascara)
    k = tf.random.uniform((), minval=0, maxval=4, dtype=tf.int32)
    foto = tf.image.rot90(foto, k)
    mascara = tf.image.rot90(mascara, k)

    # Cambios suaves de luz y color solo en la foto
    foto = tf.image.random_brightness(foto, max_delta=0.15)
    foto = tf.image.random_contrast(foto, lower=0.85, upper=1.15)
    foto = tf.image.random_saturation(foto, lower=0.75, upper=1.25)
    foto = tf.clip_by_value(foto, 0.0, 1.0) * 255.0
    return foto, mascara


datos_train = (tf.data.Dataset.from_tensor_slices((X_train_bal, Y_train_bal))
               .shuffle(len(X_train_bal), seed=utils.SEED)
               .map(aumentar, num_parallel_calls=tf.data.AUTOTUNE)
               .batch(BATCH).prefetch(tf.data.AUTOTUNE))
datos_val = tf.data.Dataset.from_tensor_slices((X_val, Y_val)).batch(BATCH)
""")

md("""
## 5. La UNet con codificador preentrenado (`MobileNetV2`)
En vez de entrenar el codificador desde cero (solo 1.542 fotos de laboratorio), usamos
**`MobileNetV2` preentrenado en ImageNet** (1,4 millones de fotos reales del mundo) como
camino de bajada y construimos el decodificador UNet con conexiones de salto:

- **Camino de bajada (codificador `MobileNetV2`):** achica la foto (`160 → 80 → 40 → 20 → 10 → 5`)
  aprovechando filtros ya aprendidos para reconocer objetos, hojas, ramas, sombras y fondos.
- **Salto de borde fino (`160×160`):** un bloque chico directo sobre la entrada conserva el
  detalle del borde dentado de la hoja en resolución completa.
- **Camino de subida (decodificador UNet):** `Conv2DTranspose` + `concatenate` con cada escala
  del codificador + `BatchNormalization` y `Dropout`.
- **Salida:** probabilidad por píxel (`sigmoid`) de que sea hoja.
""")

code("""
def bloque_dec(x, filtros, drop):
    x = Conv2D(filtros, (3, 3), padding='same', use_bias=False,
               kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(drop)(x)
    x = Conv2D(filtros, (3, 3), padding='same', use_bias=False,
               kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    return x


def construir_unet():
    Image_input = Input((LADO, LADO, 3))
    x_norm = Rescaling(1.0 / 127.5, offset=-1.0)(Image_input)   # rango [-1, 1] para MobileNetV2

    # Rama liviana en resolución completa (160x160) para bordes finos de la hoja
    s0 = Conv2D(32, (3, 3), padding='same', use_bias=False,
                kernel_initializer='he_normal')(x_norm)
    s0 = BatchNormalization()(s0)
    s0 = Activation('relu')(s0)

    # Codificador preentrenado en ImageNet
    base = tf.keras.applications.MobileNetV2(
        input_shape=(LADO, LADO, 3),
        input_tensor=x_norm,
        include_top=False,
        weights='imagenet',
    )
    s1 = base.get_layer('block_1_expand_relu').output    # 80x80
    s2 = base.get_layer('block_3_expand_relu').output    # 40x40
    s3 = base.get_layer('block_6_expand_relu').output    # 20x20
    s4 = base.get_layer('block_13_expand_relu').output   # 10x10
    cuello = base.get_layer('block_16_project').output   # 5x5

    # Decodificador UNet con conexiones de salto
    x = Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(cuello)
    x = bloque_dec(concatenate([x, s4]), 256, 0.2)       # 10x10

    x = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(x)
    x = bloque_dec(concatenate([x, s3]), 128, 0.2)       # 20x20

    x = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(x)
    x = bloque_dec(concatenate([x, s2]), 64, 0.1)        # 40x40

    x = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(x)
    x = bloque_dec(concatenate([x, s1]), 32, 0.1)        # 80x80

    x = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(x)
    x = bloque_dec(concatenate([x, s0]), 32, 0.1)        # 160x160

    outputs = Conv2D(1, (1, 1), activation='sigmoid')(x)
    return Model(inputs=[Image_input], outputs=[outputs]), base
""")

md("""
## 6. Entrenar en 2 fases (Transfer Learning + Fine-tuning)
- **Fase 1 (encoder congelado):** entrena solo el decodificador UNet y el salto de borde fino
  (`lr = 1e-3`), sin romper los pesos preentrenados de ImageNet.
- **Fase 2 (*fine-tuning*):** descongela el codificador `MobileNetV2` (manteniendo congeladas
  sus capas `BatchNormalization`, práctica estándar con batch chico) y ajusta toda la red con
  `lr = 1e-4`, `ReduceLROnPlateau` y `EarlyStopping`.
- **Pérdida (`utils.bce_dice_loss`):** combina `binary_crossentropy` + `Dice Loss`.

**Ojo:** el IoU de val se calcula contra máscaras **automáticas**, que tienen errores.
El número que vale es el de test (sección 7), contra las corregidas a mano.
""")

code("""
if RUTA_MODELO.exists() and not REENTRENAR:
    print('Cargando el modelo ya entrenado:', RUTA_MODELO.name)
    model = tf.keras.models.load_model(
        RUTA_MODELO, custom_objects={'bce_dice_loss': utils.bce_dice_loss}
    )
else:
    model, base = construir_unet()

    # Fase 1: encoder congelado
    for capa in base.layers:
        capa.trainable = False
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LR_F1),
        loss=utils.bce_dice_loss,
        metrics=[tf.keras.metrics.BinaryIoU(target_class_ids=[1], threshold=0.5, name='iou')],
    )
    print(f'--- Fase 1 (encoder congelado, máx {EPOCAS_F1} épocas) ---')
    h1 = model.fit(
        datos_train, validation_data=datos_val, epochs=EPOCAS_F1,
        callbacks=[tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=4, restore_best_weights=True, verbose=1
        )],
    )

    # Fase 2: fine-tuning (descongela el encoder excepto sus BatchNormalization)
    for capa in base.layers:
        capa.trainable = not isinstance(capa, BatchNormalization)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LR_F2),
        loss=utils.bce_dice_loss,
        metrics=[tf.keras.metrics.BinaryIoU(target_class_ids=[1], threshold=0.5, name='iou')],
    )
    print(f'--- Fase 2 (fine-tuning, máx {EPOCAS_F2} épocas) ---')
    h2 = model.fit(
        datos_train, validation_data=datos_val, epochs=EPOCAS_F2,
        callbacks=[
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1
            ),
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss', patience=PACIENCIA, restore_best_weights=True, verbose=1
            ),
            tf.keras.callbacks.ModelCheckpoint(RUTA_MODELO, monitor='val_loss', save_best_only=True),
        ],
    )

    df1 = pd.DataFrame(h1.history); df1['fase'] = 1
    df2 = pd.DataFrame(h2.history); df2['fase'] = 2
    df_hist = pd.concat([df1, df2], ignore_index=True)
    df_hist.to_csv(ROOT / 'resultados' / f'{NOMBRE}_historia.csv', index_label='epoca')
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
    if 'fase' in h.columns and (h['fase'] == 2).any():
        corte = int(h.index[h['fase'] == 2][0]) + 0.5
        for a in ax:
            a.axvline(corte, color='gray', linestyle='--', alpha=0.7)
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
Resultados reales en las 50 fotos de test (con máscaras corregidas a mano) tras las mejoras
de la v2 (filtros `32→512` con `BatchNormalization`, `BCE + Dice Loss`, aumentación de color
y rotaciones de 90°, y balanceo ×4 de `mancha_bacteriana` en train):

- **IoU promedio de la UNet en test:** **0.931** (mediana **0.968**; 38/50 fotos ≥ 0.95 y 43/50 ≥ 0.90).
  Subió desde **0.919** en la v1, mejorando en 45 de las 50 fotos y con un modelo 4× más liviano
  (7,77M parámetros / 90 MB vs 31M / 356 MB).
- **IoU promedio del truco automático:** **0.934** (mediana 0.985). La UNet empata en `mildiu`
  (0.930 vs 0.931), supera al truco en `oidio` (0.930 vs 0.893) y le gana con holgura en las
  fotos difíciles donde el truco falla fuerte (`Bacterial Leaf Spot_3`: 0.960 vs 0.327;
  `Powdery Mildew_395`: 0.603 vs 0.057; `Downy Mildew_852`: 0.674 vs 0.514).
- **Dónde falla más (las únicas 4 fotos debajo de 0.80):** una rama casi sin hoja
  (`Bacterial Leaf Spot_56`, 0.53), un fondo de viñedo verde sin ejemplos similares en train
  (`Powdery Mildew_395`, 0.60), una foto con 3 hojas separadas ya rechazada por la regla de
  una hoja por foto (`Bacterial Leaf Spot_97`, 0.62) y una mancha blanquecina grande tocando
  el borde (`Downy Mildew_852`, 0.67).

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
