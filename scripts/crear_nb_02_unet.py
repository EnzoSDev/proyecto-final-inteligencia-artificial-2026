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
`datos/dataset_unet.zip` (dentro de una carpeta `datos/`), que se genera con `python scripts/armar_dataset_unet.py`.
""")

md("""
## 1. Preparar el entorno
En Colab conecta Google Drive (pide permiso) y descomprime los datos en `/content`.
En la compu no hace nada especial.
""")

code("""
import sys
EN_COLAB = 'google.colab' in sys.modules   # True si estamos en Google Colab
if EN_COLAB:
    from google.colab import drive
    drive.mount('/content/drive')   # conecta Google Drive (pide permiso)
    sys.path.insert(0, '/content/drive/MyDrive/Proyecto_Final_IA')   # así Python encuentra utils.py

import utils
# Prepara los datos: en Colab descomprime datos/dataset_unet.zip en /content (se lee rápido).
# ROOT = dónde se guardan modelos y figuras (Drive en Colab); DATA = de dónde se leen las fotos.
entorno = utils.configurar_entorno(['datos/dataset_unet'])
ROOT, DATA = entorno['ROOT'], entorno['DATA']
""")

code("""
import os

# numpy = arrays de números | pandas = tablas | matplotlib = gráficos
# TensorFlow/Keras = la red neuronal | skimage = leer imágenes
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from skimage.io import imread
from tensorflow.keras.layers import (Input, Rescaling, Conv2D, BatchNormalization,
                                     Activation, Dropout, MaxPooling2D,
                                     Conv2DTranspose, concatenate)
from tensorflow.keras.models import Model

utils.set_estilo()   # letra grande en los gráficos (van a la presentación)
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
SMOKE = os.environ.get('SMOKE') == '1'   # prueba rápida en la compu (pocas fotos, 1 época)
REENTRENAR = False   # False: si ya hay un modelo guardado, lo carga en vez de entrenar

EPOCAS_F1 = 1 if SMOKE else 10   # Fase 1: encoder MobileNetV2 congelado (calienta el decodificador)
EPOCAS_F2 = 1 if SMOKE else 40   # Fase 2: fine-tuning completo (con BN del encoder congeladas)
PACIENCIA = 8                    # épocas sin mejorar en validación antes de cortar en Fase 2
LR_F1 = 1e-3   # learning rate de la fase 1
LR_F2 = 1e-4   # learning rate de la fase 2 (10 veces más chico: cambios suaves)
BATCH = 16   # fotos por grupo: la red corrige sus pesos después de cada grupo
LADO = utils.IMG_SEG             # 160
FACTOR_MANCHA = 4                # repetir mancha_bacteriana solo en train (tiene solo 60 fotos)
FACTOR_MANUAL_TRAIN = 3          # dar más peso a fotos difíciles de train corregidas a mano (si existen)

CARPETA = DATA / 'datos' / 'dataset_unet'   # de dónde se leen fotos y máscaras
NOMBRE = 'unet_smoke' if SMOKE else 'unet'          # la prueba no pisa el modelo real
RUTA_MODELO = ROOT / 'models' / f'{NOMBRE}.keras'   # dónde se guarda el modelo
RUTA_MODELO.parent.mkdir(exist_ok=True)
(ROOT / 'resultados').mkdir(exist_ok=True)
print('SMOKE =', SMOKE, '| modelo:', RUTA_MODELO.name)
""")

md("""
## 3. Cargar los datos
`datos/dataset_unet/` ya tiene todo listo en 160×160: en `images/` las fotos y en `masks/` sus
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
lista = pd.read_csv(CARPETA / 'lista.csv')   # tabla: split, archivo, clase, foto original y tipo de máscara
if SMOKE:   # prueba: unas pocas fotos de cada clase
    lista = pd.concat([lista[lista['split'] == 'test'],
                       lista[lista['split'] != 'test'].groupby(['split', 'clase']).head(10)])


# Lee las fotos y máscaras de un split. Usa UNA sola lista de nombres ordenada,
# así cada foto queda con SU máscara (tienen el mismo nombre de archivo).
def cargar(split, carpeta_mascaras='masks'):
    nombres = sorted(lista.loc[lista['split'] == split, 'archivo'])
    X = np.asarray([imread(CARPETA / split / 'images' / n) for n in nombres], dtype=np.uint8)   # fotos: (N, 160, 160, 3)
    Y = np.asarray([imread(CARPETA / split / carpeta_mascaras / n) for n in nombres])
    Y = (Y > 127).astype(np.float32)[..., None]      # (N, 160, 160, 1) con 0 y 1
    return X, Y, nombres


# train = aprender | val = decidir cuándo cortar | test = examen final (máscaras a mano)
X_train, Y_train, nombres_train = cargar('train')
X_val, Y_val, _ = cargar('val')
X_test, Y_test, nombres_test = cargar('test')                    # NO se usa para entrenar
_, Y_test_auto, _ = cargar('test', 'masks_auto')                # el truco, para comparar

for nombre, X, Y in [('train', X_train, Y_train), ('val', X_val, Y_val), ('test', X_test, Y_test)]:
    print(f'{nombre:5s} fotos {X.shape}  máscaras {Y.shape}  hoja = {Y.mean():.0%} de los píxeles')
""")

code("""
# Muestra 5 fotos de train con su máscara (lo que la red tiene que aprender)
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
# Balanceo: algunas fotos se repiten en train para que la red las vea más veces:
# - mancha_bacteriana x4 (tiene muy pocas fotos)
# - fotos difíciles corregidas a mano por Tato x3
clase_de = dict(zip(lista['archivo'], lista['clase']))
origen_de = dict(zip(lista['archivo'], lista['mascara']))
idx_extra = []   # índices de las fotos a repetir
for i, n in enumerate(nombres_train):
    rep = 1
    if clase_de[n] == 'mancha_bacteriana':
        rep = max(rep, FACTOR_MANCHA)
    if origen_de.get(n) == 'manual_train':
        rep = max(rep, FACTOR_MANUAL_TRAIN)
    if rep > 1:
        idx_extra.extend([i] * (rep - 1))
# Train balanceado = train original + las copias extra
X_train_bal = np.concatenate([X_train, X_train[idx_extra]], axis=0) if idx_extra else X_train
Y_train_bal = np.concatenate([Y_train, Y_train[idx_extra]], axis=0) if idx_extra else Y_train
assert len(X_train_bal) == len(Y_train_bal)
n_man = sum(origen_de.get(n) == 'manual_train' for n in nombres_train)
print(f'Train balanceado: {len(X_train)} -> {len(X_train_bal)} fotos '
      f'(+{len(idx_extra)} copias; incluye {n_man} máscaras manuales de train)')


# Aumentación: cada vez que la red ve una foto, la cambia un poco al azar.
# Espejar y rotar se aplica IGUAL a la foto y a la máscara (si no, dejarían de coincidir).
# Brillo, contraste y saturación solo a la foto (la máscara no tiene color).
def aumentar(foto, mascara):
    foto = tf.cast(foto, tf.float32) / 255.0   # pasa a 0..1 para poder cambiar el brillo

    flip_h = tf.random.uniform(()) > 0.5
    foto = tf.cond(flip_h, lambda: tf.image.flip_left_right(foto), lambda: foto)
    mascara = tf.cond(flip_h, lambda: tf.image.flip_left_right(mascara), lambda: mascara)
    flip_v = tf.random.uniform(()) > 0.5
    foto = tf.cond(flip_v, lambda: tf.image.flip_up_down(foto), lambda: foto)
    mascara = tf.cond(flip_v, lambda: tf.image.flip_up_down(mascara), lambda: mascara)
    k = tf.random.uniform((), minval=0, maxval=4, dtype=tf.int32)   # rotar k veces 90° (0, 90, 180 o 270)
    foto = tf.image.rot90(foto, k)
    mascara = tf.image.rot90(mascara, k)

    # Cambios suaves de luz y color solo en la foto
    foto = tf.image.random_brightness(foto, max_delta=0.15)
    foto = tf.image.random_contrast(foto, lower=0.85, upper=1.15)
    foto = tf.image.random_saturation(foto, lower=0.75, upper=1.25)
    foto = tf.clip_by_value(foto, 0.0, 1.0) * 255.0   # vuelve a 0..255
    return foto, mascara


# tf.data arma la cinta de datos: mezcla, aumenta, agrupa de a BATCH y prepara el próximo
# grupo mientras la GPU entrena. Val no se mezcla ni se aumenta.
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
# Bloque del decodificador: 2 convoluciones 3x3, cada una con BatchNorm (estabiliza) y ReLU.
# El Dropout del medio apaga neuronas al azar al entrenar, para que no memorice.
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


# Arma la UNet: MobileNetV2 baja (achica y entiende la foto) y el decodificador sube
# (agranda hasta 160x160 y dice qué píxel es hoja). Devuelve el modelo y la base (para congelarla).
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
        include_top=False,   # sin la capa final de 1000 clases de ImageNet
        weights='imagenet',   # con lo que ya aprendió en ImageNet
    )
    # Salidas intermedias de MobileNetV2 a distintos tamaños: son las conexiones de salto
    s1 = base.get_layer('block_1_expand_relu').output    # 80x80
    s2 = base.get_layer('block_3_expand_relu').output    # 40x40
    s3 = base.get_layer('block_6_expand_relu').output    # 20x20
    s4 = base.get_layer('block_13_expand_relu').output   # 10x10
    cuello = base.get_layer('block_16_project').output   # 5x5

    # Decodificador UNet con conexiones de salto
    # En cada paso: Conv2DTranspose duplica el tamaño, concatenate le pega el salto del mismo
    # tamaño (el detalle fino de la bajada) y bloque_dec lo procesa.
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

    outputs = Conv2D(1, (1, 1), activation='sigmoid')(x)   # 1 canal: probabilidad de hoja (0 a 1) en cada píxel
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
# Si ya hay un modelo guardado lo carga; si no, entrena en 2 fases.
if RUTA_MODELO.exists() and not REENTRENAR:
    print('Cargando el modelo ya entrenado:', RUTA_MODELO.name)
    model = tf.keras.models.load_model(
        RUTA_MODELO, custom_objects={'bce_dice_loss': utils.bce_dice_loss}
    )
else:
    model, base = construir_unet()

    # Fase 1: encoder congelado
    for capa in base.layers:
        capa.trainable = False   # congela todo MobileNetV2
    # loss: BCE (píxel a píxel) + Dice (cuánto se superponen las máscaras); métrica: IoU de la hoja
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LR_F1),
        loss=utils.bce_dice_loss,
        metrics=[tf.keras.metrics.BinaryIoU(target_class_ids=[1], threshold=0.5, name='iou')],
    )
    print(f'--- Fase 1 (encoder congelado, máx {EPOCAS_F1} épocas) ---')
    h1 = model.fit(
        datos_train, validation_data=datos_val, epochs=EPOCAS_F1,
        callbacks=[tf.keras.callbacks.EarlyStopping(   # corta si val_loss no mejora en 4 épocas
            monitor='val_loss', patience=4, restore_best_weights=True, verbose=1
        )],
    )

    # Fase 2: fine-tuning (descongela el encoder excepto sus BatchNormalization)
    for capa in base.layers:
        capa.trainable = not isinstance(capa, BatchNormalization)   # descongela todo menos las BatchNorm
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LR_F2),
        loss=utils.bce_dice_loss,
        metrics=[tf.keras.metrics.BinaryIoU(target_class_ids=[1], threshold=0.5, name='iou')],
    )
    print(f'--- Fase 2 (fine-tuning, máx {EPOCAS_F2} épocas) ---')
    h2 = model.fit(
        datos_train, validation_data=datos_val, epochs=EPOCAS_F2,
        callbacks=[
            # si se estanca 3 épocas, baja el learning rate a la mitad
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1
            ),
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss', patience=PACIENCIA, restore_best_weights=True, verbose=1
            ),
            # guarda la mejor época mientras entrena (por si Colab se corta)
            tf.keras.callbacks.ModelCheckpoint(RUTA_MODELO, monitor='val_loss', save_best_only=True),
        ],
    )

    # Guarda la historia (loss e IoU por época) para graficar las curvas
    df1 = pd.DataFrame(h1.history); df1['fase'] = 1
    df2 = pd.DataFrame(h2.history); df2['fase'] = 2
    df_hist = pd.concat([df1, df2], ignore_index=True)
    df_hist.to_csv(ROOT / 'resultados' / f'{NOMBRE}_historia.csv', index_label='epoca')
    model.save(RUTA_MODELO)

print(f'{model.count_params():,} parámetros')
""")

code("""
# Curvas de entrenamiento; la línea punteada marca el paso de la fase 1 a la fase 2
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
# IoU = píxeles donde las dos dicen hoja / píxeles donde al menos una dice hoja (1 = perfecto)
def iou(verdad, prediccion):
    verdad, prediccion = verdad > 0.5, prediccion > 0.5
    union = np.logical_or(verdad, prediccion).sum()
    return np.logical_and(verdad, prediccion).sum() / union if union else 1.0


P_test = model.predict(X_test, batch_size=BATCH, verbose=0)   # probabilidad de hoja en las 50 fotos
clase_de = dict(zip(lista['archivo'], lista['clase']))
# Una fila por foto: IoU de la UNet y del truco automático contra la máscara a mano
res = pd.DataFrame({
    'archivo': nombres_test,
    'clase': [clase_de[n] for n in nombres_test],
    'iou_unet': [iou(Y_test[i], P_test[i]) for i in range(len(X_test))],
    'iou_truco': [iou(Y_test[i], Y_test_auto[i]) for i in range(len(X_test))],
})
res.to_csv(ROOT / 'resultados' / f'{NOMBRE}_iou_test.csv', index=False)

# Promedio por clase y general
tabla = res.groupby('clase')[['iou_unet', 'iou_truco']].mean()
tabla.loc['PROMEDIO'] = res[['iou_unet', 'iou_truco']].mean()
tabla.columns = ['UNet', 'Truco automático']
print(tabla.round(3).to_string())
print('\\nFotos donde la UNet le gana al truco:', (res['iou_unet'] > res['iou_truco']).sum(), 'de', len(res))
""")

code("""
fig, ax = plt.subplots(figsize=(12, 6))
tabla.drop('PROMEDIO').plot.bar(ax=ax, rot=0)   # gráfico de barras por clase
ax.set_ylim(0, 1); ax.set_ylabel('IoU promedio (test)'); ax.set_xlabel('')
ax.set_title('UNet vs truco automático (50 fotos de test)')
ax.legend(loc='lower right')
fig.tight_layout()
utils.guardar_figura(fig, f'{NOMBRE}_iou_por_clase')
plt.show()
""")

md("""
## 7b. Aumentación al predecir (TTA)
Idea de Enzo. Le pedimos la máscara a la UNet **8 veces** por foto: normal y espejada, cada una
rotada 0°, 90°, 180° y 270°. Cada máscara se vuelve a su posición original y se **promedian**.
Si la red se equivoca en una posición pero no en las otras, el promedio corrige el error.

No hay que volver a entrenar: se usa el mismo modelo.

**Primero lo medimos en val** para decidir si sirve. Test se mira después, así no elegimos
mirando el examen.

**Resultado con la v6: no mejoró** (val 0.973 → 0.973, test 0.941 → 0.941). Ver conclusiones.
""")

code("""
def predecir_tta(X):
    suma = 0
    for espejar in [False, True]:
        for k in range(4):                                   # k = 0, 1, 2, 3 -> 0°, 90°, 180°, 270°
            Xa = X[:, :, ::-1] if espejar else X             # espejar izquierda-derecha
            Xa = np.rot90(Xa, k, axes=(1, 2))                # rotar
            P = model.predict(np.ascontiguousarray(Xa), batch_size=BATCH, verbose=0)
            P = np.rot90(P, -k, axes=(1, 2))                 # deshacer la rotación
            if espejar:
                P = P[:, :, ::-1]                            # deshacer el espejo
            suma = suma + P
    return suma / 8                                          # promedio de las 8 máscaras


# Comparamos en val: sin TTA y con TTA
P_val = model.predict(X_val, batch_size=BATCH, verbose=0)
P_val_tta = predecir_tta(X_val)
iou_val = np.mean([iou(Y_val[i], P_val[i]) for i in range(len(X_val))])
iou_val_tta = np.mean([iou(Y_val[i], P_val_tta[i]) for i in range(len(X_val))])
print(f'Val  ->  UNet: {iou_val:.3f}   UNet + TTA: {iou_val_tta:.3f}')
""")

code("""
# Lo mismo en test, por clase
P_test_tta = predecir_tta(X_test)
res['iou_tta'] = [iou(Y_test[i], P_test_tta[i]) for i in range(len(X_test))]

tabla_tta = res.groupby('clase')[['iou_unet', 'iou_tta', 'iou_truco']].mean()
tabla_tta.loc['PROMEDIO'] = res[['iou_unet', 'iou_tta', 'iou_truco']].mean()
tabla_tta.columns = ['UNet', 'UNet + TTA', 'Truco automático']
print(tabla_tta.round(3).to_string())
print('\\nFotos donde TTA mejora a la UNet:', (res['iou_tta'] > res['iou_unet']).sum(), 'de', len(res))
""")

md("""
## 7c. Limpiar la máscara y elegir el umbral
Dos ajustes que **no necesitan volver a entrenar**: se aplican a lo que ya predice la UNet.

1. **Limpiar:** como hay una sola hoja por foto, nos quedamos con la **mancha blanca más grande**
   (se borran puntitos sueltos) y **rellenamos los agujeros** que quedan adentro de la hoja.
2. **Umbral:** hasta ahora, más de 0.5 = hoja. Probamos otros valores.

Elegimos la mejor combinación **en val** y después la miramos en test.

**Ojo:** las máscaras de val son automáticas y a veces tienen agujeros en las zonas enfermas.
Por eso rellenar agujeros puede salir peor en val aunque esté bien.

**Resultado con la v6:** el mejor umbral en val es 0.5, con limpieza. En test sube de 0.941 a
0.955, pero casi todo por 2 fotos de mancha bacteriana (ver conclusiones).
""")

code("""
# utils.limpiar_mascara: deja la mancha blanca más grande y rellena sus agujeros.
# Está en utils.py porque también la usan el notebook 03 y la app.
limpiar = utils.limpiar_mascara


def iou_promedio(Y, P, umbral, con_limpieza):
    valores = []
    for i in range(len(Y)):
        m = P[i, :, :, 0] > umbral
        if con_limpieza:
            m = limpiar(m)
        valores.append(iou(Y[i], m[..., None]))
    return np.mean(valores)


# Probamos 5 umbrales, con y sin limpieza, y nos quedamos con el mejor en val
filas = []
for umbral in [0.3, 0.4, 0.5, 0.6, 0.7]:
    for con_limpieza in [False, True]:
        filas.append({'umbral': umbral, 'limpiar': con_limpieza,
                      'iou_val': iou_promedio(Y_val, P_val, umbral, con_limpieza)})
tabla_val = pd.DataFrame(filas)
print(tabla_val.round(4).to_string(index=False))

mejor = tabla_val.loc[tabla_val['iou_val'].idxmax()]
UMBRAL, LIMPIAR = float(mejor['umbral']), bool(mejor['limpiar'])
print(f'\\nMejor en val: umbral {UMBRAL}, limpiar = {LIMPIAR}')
""")

code("""
# En test: (1) solo limpieza con umbral 0.5 y (2) la combinación elegida en val
def mascara_final(p, umbral, con_limpieza):
    m = p[:, :, 0] > umbral
    return limpiar(m) if con_limpieza else m


res['iou_limpia'] = [iou(Y_test[i], mascara_final(P_test[i], 0.5, True)[..., None]) for i in range(len(X_test))]
res['iou_elegida'] = [iou(Y_test[i], mascara_final(P_test[i], UMBRAL, LIMPIAR)[..., None]) for i in range(len(X_test))]

columnas = ['iou_unet', 'iou_limpia', 'iou_elegida', 'iou_truco']
tabla_7c = res.groupby('clase')[columnas].mean()
tabla_7c.loc['PROMEDIO'] = res[columnas].mean()
tabla_7c.columns = ['UNet (0.5)', 'UNet + limpiar (0.5)', f'Elegida en val ({UMBRAL}, limpiar={LIMPIAR})', 'Truco']
print(tabla_7c.round(3).to_string())
""")

md("""
## 8. Mirar los resultados
Los números no alcanzan: hay que **ver** dónde acierta y dónde falla. Mostramos las 4 fotos
con mejor IoU y las 4 con peor. Contorno **verde** = máscara correcta (a mano);
contorno **rojo** = lo que predijo la UNet.
""")

code("""
import cv2


# Dibuja sobre la foto el borde de la máscara a mano (verde) y el de la UNet (rojo)
def contornos(foto, verdad, prediccion):
    img = foto.copy()
    for mascara, color in [(verdad, (0, 255, 0)), (prediccion, (255, 0, 0))]:
        bordes, _ = cv2.findContours((mascara[..., 0] > 0.5).astype(np.uint8),
                                     cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(img, bordes, -1, color, 1)
    return img


# Elige las 4 fotos con mejor IoU y las 4 con peor
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
Resultados reales en las 50 fotos de test (con máscaras corregidas a mano) de la **UNet v6**
(`MobileNetV2` preentrenado en ImageNet + decodificador UNet con salto en 160×160, entrenada
en 2 fases con `BCE + Dice Loss`, balanceo ×4 de `mancha_bacteriana` y **39 fotos difíciles
de `train` corregidas a mano** en `datos/masks/manual_train/` con peso ×3):

- **IoU promedio de la UNet en test:** **0.941** (mediana **0.967**; **45/50** fotos ≥ 0.90
  y **39/50** ≥ 0.95), con un modelo compacto de **5,60M parámetros / 65 MB** (vs 31M / 356 MB
  de la v1 inicial).
- **Supera al truco automático (`0.934`) en promedio general (`+0.007`)**, incluso a pesar de
  que las máscaras manuales de test partieron de las automáticas (9 sin tocar = 1.000 para el
  truco):
  - `mildiu`: **0.958** vs 0.931 (`+0.028`)
  - `oidio`: **0.938** vs 0.893 (`+0.045`)
  - `hojas_sanas`: **0.968** vs 0.983 (`-0.015`)
  - `mancha_bacteriana`: **0.896** vs 0.926 (`-0.030`)
- **Impacto comprobado de los datos sobre la arquitectura (v1–v5 → v6):** 5 iteraciones de
  arquitectura y aumentación (`v1`–`v5`) habían chocado contra un techo de `0.931`; corregir
  a mano solo 39 casos difíciles de `train` disparó `Downy Mildew_852` (`0.72 → 0.94`, truco
  `0.51`), `Downy Mildew_838` (`0.84 → 0.96`, truco `0.83`), `Powdery Mildew_395` con fondo
  verde de viñedo (`0.59 → 0.78`, truco `0.06`) y `Bacterial Leaf Spot_56` (`0.60 → 0.66`).
- **Límites restantes:** las únicas 2 fotos por debajo de `0.75` son atípicas (`Bacterial Leaf
  Spot_97`, 0.53, tiene 3 hojas separadas ya en `rechazadas.txt`, y `Bacterial Leaf Spot_56`,
  0.66, es una rama larga casi sin hoja). Las fotos de viñedo real (`395` y `8`, ambas en `0.78`)
  mejoraron fuerte, aunque para uso general en campo harían falta más fotos de viñedo etiquetadas.
- **Aumentación al predecir (TTA, sección 7b): la probamos y no mejoró.** Promediar las 8 máscaras
  (normal y espejada, rotada 0°, 90°, 180° y 270°) dio lo mismo que predecir una vez: val
  **0.973 → 0.973** y test **0.941 → 0.941**. La red ya se entrenó con espejados y rotaciones de
  90°, así que predice casi igual en cualquier posición y promediar no agrega nada.
- **Limpiar la máscara (sección 7c): sí mejoró.** Quedarse con la mancha más grande y rellenar
  agujeros sube el promedio en test de **0.941 a 0.955** (`mancha_bacteriana`: 0.896 → 0.952).
  El umbral que mejor anda en val es el mismo 0.5, y en val el cambio es mínimo (0.9726 → 0.9728).
  **Casi toda la mejora sale de 2 fotos:** `Bacterial Leaf Spot_97` (0.535 → 0.943; tiene varias
  hojas y está en `rechazadas.txt`) y `Bacterial Leaf Spot_56` (0.663 → 0.926; se borran los
  pedazos de tallo sueltos). Las otras 48 cambian como mucho 0.006.

El modelo quedó guardado en `models/unet.keras`. Lo usan el notebook 03 (para recortar las
hojas) y la app.
""")

nb = nbf.v4.new_notebook()
nb['cells'] = celdas
nb['metadata']['kernelspec'] = {'name': 'python3', 'display_name': 'Python 3', 'language': 'python'}
nb['metadata']['accelerator'] = 'GPU'
nbf.write(nb, ROOT / '02_unet_segmentacion.ipynb')
print('Creado 02_unet_segmentacion.ipynb')
