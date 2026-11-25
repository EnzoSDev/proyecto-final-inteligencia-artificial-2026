"""Genera el notebook 03_clasificacion.ipynb.

Uso (desde la raíz del proyecto):  python scripts/crear_nb_03_clasificacion.py
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
# 03 · Clasificación de enfermedades: con fondo (A) vs sin fondo (B)

Experimento central del proyecto: entrenamos **el mismo clasificador (`MobileNetV2` preentrenada)**
dos veces, con exactamente las mismas imágenes y los mismos 5 folds:

- **Modelo A (con fondo):** fotos originales de `dataset/`.
- **Modelo B (sin fondo):** las mismas fotos segmentadas por la UNet v6 (`datos/dataset_segmentado/`),
  recortadas al **cuadro ajustado de la hoja (*bounding box*)** y rellenadas con **gris neutro `128`**
  (ver `utils.recortar_a_hoja()`).

Así medimos si quitar el fondo mejora la clasificación (tanto en el test interno como en fotos
externas de campo) y, con **Grad-CAM**, comprobamos visualmente si el modelo A se guiaba por el
fondo y si el modelo B se enfoca en la hoja.

**Alcance:** el sistema solo informa la clase predicha y sus probabilidades (**no da recomendaciones**
agronómicas; las decisiones corresponden al ingeniero agrónomo). Es una aproximación: las manchas
en una hoja pueden tener otras causas (nutrientes, estrés hídrico, sol, insectos, químicos u otras
enfermedades) y la red siempre elige una de sus 4 clases.
""")

md("""
## 1. Preparar el entorno
En Colab conecta Google Drive y copia/descomprime `dataset/` y `datos/dataset_segmentado.zip`
en `/content`. En la compu local usa la carpeta del proyecto.
""")

code("""
import sys
EN_COLAB = 'google.colab' in sys.modules
if EN_COLAB:
    from google.colab import drive
    drive.mount('/content/drive')
    sys.path.insert(0, '/content/drive/MyDrive/Proyecto_Final_IA')

import utils
entorno = utils.configurar_entorno(['dataset', 'datos/dataset_segmentado'])
ROOT, DATA = entorno['ROOT'], entorno['DATA']
""")

code("""
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.layers import (BatchNormalization, Dense, Dropout,
                                     GlobalAveragePooling2D, Input, Rescaling)
from tensorflow.keras.models import Model

utils.set_estilo()
tf.keras.utils.set_random_seed(utils.SEED)
""")

md("""
## 2. Configuración
- `SMOKE`: prueba rápida (`SMOKE=1`) con pocas fotos, 2 folds y 1 época por fase.
- `REENTRENAR`: si es `False` y ya existen resultados/modelos guardados, los carga sin
  volver a entrenar.
""")

code("""
SMOKE = os.environ.get('SMOKE') == '1'
REENTRENAR = os.environ.get('REENTRENAR') == '1'

N_FOLDS = 2 if SMOKE else 5
EPOCAS_F1 = 1 if SMOKE else 10    # Fase 1: MobileNetV2 congelada (entrena la cabeza)
EPOCAS_F2 = 1 if SMOKE else 20    # Fase 2: fine-tuning de las últimas capas
CAPAS_FINETUNE = 35               # últimas capas de MobileNetV2 a descongelar (sin BN)
LR_F1 = 1e-3
LR_F2 = 1e-4
LABEL_SMOOTHING = 0.05            # evita sobreconfianza extrema fuera de dominio
FONDO_B = 128                     # gris neutro -> entra como 0.0 en MobileNetV2 [-1, +1]
BATCH = utils.BATCH               # 32
LADO = utils.IMG_CLF              # 224

SUFIJO = '_smoke' if SMOKE else ''
DIR_MOD = ROOT / 'models'
DIR_RES = ROOT / 'resultados'
DIR_MOD.mkdir(exist_ok=True)
DIR_RES.mkdir(exist_ok=True)

RUTA_CV = DIR_RES / f'clf_cv_folds{SUFIJO}.csv'
RUTA_MOD_A = DIR_MOD / f'clasificador_A{SUFIJO}.keras'
RUTA_MOD_B = DIR_MOD / f'clasificador_B{SUFIJO}.keras'
print(f'SMOKE={SMOKE} | folds={N_FOLDS} | fases={EPOCAS_F1}+{EPOCAS_F2} épocas | lado={LADO}')
""")

md("""
## 3. Datos y 5-fold por grupos (sin fuga de casi duplicados)
- De `datos/splits.csv` separamos **test fijo** (359 fotos) y lo dejamos afuera hasta el examen final.
- Del resto (`train + val`), `utils.quitar_copias_de_test()` saca las 100 fotos cuyo grupo en
  `datos/casi_duplicados.csv` tiene alguna copia en test → quedan **1.934 fotos**.
- Sobre esas 1.934 aplicamos `StratifiedGroupKFold(n_splits=5)`: todas las copias de un mismo
  grupo caen siempre del mismo lado. Los **mismos 5 folds** se usan para **A** y **B**.
""")

code("""
df = utils.cargar_splits()
df_cv = utils.quitar_copias_de_test(df)
df_test = df[df['split'] == 'test'].reset_index(drop=True)

if SMOKE:
    df_cv = df_cv.groupby('clase', group_keys=False).head(10).reset_index(drop=True)
    df_test = df_test.groupby('clase', group_keys=False).head(6).reset_index(drop=True)

grupos_cv = utils.asignar_grupos(df_cv)
grupos_test = utils.asignar_grupos(df_test)
assert not (set(grupos_cv) & set(grupos_test)), 'Fuga de grupos entre CV y test!'

sgkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=utils.SEED)
folds = list(sgkf.split(df_cv, df_cv['label'], grupos_cv))
for k, (tr, va) in enumerate(folds):
    assert not (set(grupos_cv[tr]) & set(grupos_cv[va])), f'Fuga de grupos en fold {k}!'

print(f'CV (sin copias de test): {len(df_cv)} fotos | Test fijo: {len(df_test)} fotos')
print('Distribución en CV:')
print(df_cv['clase'].value_counts().reindex(utils.CLASES).to_string())
""")

code("""
def cargar_imagenes(df_sub, modo):
    if modo == 'A':
        X = np.asarray([utils.cargar_imagen(p, LADO, base=DATA) for p in df_sub['path']], dtype=np.uint8)
    else:
        rutas = [utils.ruta_segmentada(p) for p in df_sub['path']]
        X = np.asarray(
            [utils.recortar_a_hoja(utils.cargar_imagen(p, base=DATA), margen=0.05, fondo=FONDO_B, lado=LADO)
             for p in rutas],
            dtype=np.uint8,
        )
    y = df_sub['label'].to_numpy(dtype=np.int32)
    return X, y


X_cv_A, y_cv = cargar_imagenes(df_cv, 'A')
X_cv_B, _ = cargar_imagenes(df_cv, 'B')
X_test_A, y_test = cargar_imagenes(df_test, 'A')
X_test_B, _ = cargar_imagenes(df_test, 'B')
print('Formas cargadas -> CV:', X_cv_A.shape, '| Test:', X_test_A.shape)
""")

code("""
# Muestra: 1 foto por clase con fondo (A) y sin fondo (B)
fig, ax = plt.subplots(2, 4, figsize=(18, 9))
for j, clase in enumerate(utils.CLASES):
    idx = df_cv.index[df_cv['clase'] == clase][0]
    ax[0, j].imshow(X_cv_A[idx]); ax[0, j].set_title(clase, fontsize=16); ax[0, j].axis('off')
    ax[1, j].imshow(X_cv_B[idx]); ax[1, j].axis('off')
ax[0, 0].text(-25, LADO // 2, 'A · Con fondo', rotation=90, va='center', fontsize=18)
ax[1, 0].text(-25, LADO // 2, 'B · Sin fondo', rotation=90, va='center', fontsize=18)
fig.tight_layout()
utils.guardar_figura(fig, f'clf_ejemplos_A_vs_B{SUFIJO}')
plt.show()
""")

md("""
## 4. El clasificador (`MobileNetV2` en 2 fases — receta `v4d`)
- **Sin anidar el backbone:** conectamos `Rescaling(1/127.5, -1)` directo a `MobileNetV2(input_tensor=...)`
  para acceder luego a su última capa convolucional (`out_relu`) y calcular **Grad-CAM**.
- **Preprocesado de B (`utils.recortar_a_hoja`):** recorte al cuadro ajustado de la hoja (+5% margen,
  sin deformar) sobre **fondo gris neutro `128`** (entra como `0.0` en `[-1, +1]`), eliminando el
  atajo "hoja chica / mucho negro = `mancha_bacteriana`".
- **Desbalanceo suavizado + `label_smoothing=0.05`:** usamos la raíz cuadrada de los pesos balanceados
  (`sqrt(balanced)`) en vez de `balanced` puro para dar peso extra a `mancha_bacteriana` (~×2,4 en vez
  de ×5,7) sin sobre-dispararla en fotos de campo, junto con suavizado de etiquetas (`0.05`).
- **Aumentación (solo train, idéntica para A y B):** espejado H/V, rotaciones de 90°, **recorte/zoom
  aleatorio suave (80–100%)** y cambios suaves de brillo/contraste (±15%). No tocamos tono ni
  saturación para no lavar los síntomas de mildiu (amarillo) ni oidio (blanco).
- **2 fases:**
  1. **Congelada (`lr = 1e-3`):** entrena solo la cabeza final (`Dense(4, softmax)`).
  2. **Fine-tuning (`lr = 1e-4`):** descongela las últimas 35 capas de `MobileNetV2` (dejando
     congeladas las `BatchNormalization`) para adaptar los filtros a hojas de vid.
""")

code("""
def aumentar(foto, etiqueta):
    foto = tf.cast(foto, tf.float32) / 255.0
    flip_h = tf.random.uniform(()) > 0.5
    foto = tf.cond(flip_h, lambda: tf.image.flip_left_right(foto), lambda: foto)
    flip_v = tf.random.uniform(()) > 0.5
    foto = tf.cond(flip_v, lambda: tf.image.flip_up_down(foto), lambda: foto)
    k = tf.random.uniform((), minval=0, maxval=4, dtype=tf.int32)
    foto = tf.image.rot90(foto, k)
    escala = tf.random.uniform((), minval=0.80, maxval=1.0)
    c_tam = tf.cast(tf.round(escala * LADO), tf.int32)
    foto = tf.image.random_crop(foto, size=[c_tam, c_tam, 3])
    foto = tf.image.resize(foto, [LADO, LADO])
    foto = tf.image.random_brightness(foto, max_delta=0.15)
    foto = tf.image.random_contrast(foto, lower=0.85, upper=1.15)
    foto = tf.clip_by_value(foto, 0.0, 1.0) * 255.0
    return foto, etiqueta


def armar_dataset(X, y_oh, entrenar=False):
    ds = tf.data.Dataset.from_tensor_slices((X, y_oh))
    if entrenar:
        ds = ds.shuffle(len(X), seed=utils.SEED).map(aumentar, num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(BATCH).prefetch(tf.data.AUTOTUNE)


def construir_clasificador():
    Image_input = Input((LADO, LADO, 3))
    x = Rescaling(1.0 / 127.5, offset=-1.0)(Image_input)
    base = tf.keras.applications.MobileNetV2(
        input_shape=(LADO, LADO, 3),
        input_tensor=x,
        include_top=False,
        weights='imagenet',
    )
    y = GlobalAveragePooling2D()(base.output)
    y = Dropout(0.3)(y)
    outputs = Dense(len(utils.CLASES), activation='softmax')(y)
    return Model(inputs=Image_input, outputs=outputs), base


def entrenar_una_vez(X_tr, y_tr, X_va, y_va, seed=utils.SEED, verbose=0):
    tf.keras.utils.set_random_seed(seed)
    pesos = np.sqrt(compute_class_weight('balanced', classes=np.arange(len(utils.CLASES)), y=y_tr))
    class_weight = {i: float(p) for i, p in enumerate(pesos)}

    y_tr_oh = tf.keras.utils.to_categorical(y_tr, num_classes=len(utils.CLASES))
    y_va_oh = tf.keras.utils.to_categorical(y_va, num_classes=len(utils.CLASES))
    loss_fn = tf.keras.losses.CategoricalCrossentropy(label_smoothing=LABEL_SMOOTHING)

    ds_tr = armar_dataset(X_tr, y_tr_oh, entrenar=True)
    ds_va = armar_dataset(X_va, y_va_oh, entrenar=False)

    model, base = construir_clasificador()

    # Fase 1: backbone congelado
    for capa in base.layers:
        capa.trainable = False
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LR_F1),
        loss=loss_fn,
        metrics=['accuracy'],
    )
    h1 = model.fit(
        ds_tr, validation_data=ds_va, epochs=EPOCAS_F1, class_weight=class_weight,
        callbacks=[tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=3, restore_best_weights=True
        )],
        verbose=verbose,
    )

    # Fase 2: fine-tuning de las últimas capas (sin tocar BatchNormalization)
    for capa in base.layers[-CAPAS_FINETUNE:]:
        capa.trainable = not isinstance(capa, BatchNormalization)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LR_F2),
        loss=loss_fn,
        metrics=['accuracy'],
    )
    h2 = model.fit(
        ds_tr, validation_data=ds_va, epochs=EPOCAS_F2, class_weight=class_weight,
        callbacks=[
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6, verbose=0
            ),
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss', patience=5, restore_best_weights=True
            ),
        ],
        verbose=verbose,
    )

    df1 = pd.DataFrame(h1.history); df1['fase'] = 1
    df2 = pd.DataFrame(h2.history); df2['fase'] = 2
    return model, pd.concat([df1, df2], ignore_index=True)
""")

md("""
## 5. Experimento central: 5-fold por grupos (A con fondo vs B sin fondo)
Entrenamos **A** y **B** sobre exactamente los mismos 5 folds y reportamos **promedio ± desvío**
de **Macro-F1** (la métrica principal porque las clases están desbalanceadas) y **Accuracy**.
""")

code("""
if RUTA_CV.exists() and not REENTRENAR:
    print('Cargando resultados de validación cruzada ya guardados:', RUTA_CV.name)
    res_cv = pd.read_csv(RUTA_CV)
else:
    filas = []
    for modo, X_cv in [('A_con_fondo', X_cv_A), ('B_sin_fondo', X_cv_B)]:
        print(f'=== Validación cruzada {N_FOLDS}-fold · {modo} ===')
        for k, (tr, va) in enumerate(folds):
            modelo_k, _ = entrenar_una_vez(
                X_cv[tr], y_cv[tr], X_cv[va], y_cv[va], seed=utils.SEED + k, verbose=0
            )
            preds = np.argmax(modelo_k.predict(X_cv[va], batch_size=BATCH, verbose=0), axis=1)
            acc = accuracy_score(y_cv[va], preds)
            mf1 = f1_score(y_cv[va], preds, average='macro')
            print(f'  Fold {k + 1}/{N_FOLDS} -> accuracy={acc:.4f}  macro_f1={mf1:.4f}')
            filas.append({'modelo': modo, 'fold': k + 1, 'accuracy': acc, 'macro_f1': mf1})
            tf.keras.backend.clear_session()
    res_cv = pd.DataFrame(filas)
    res_cv.to_csv(RUTA_CV, index=False)

resumen_cv = res_cv.groupby('modelo')[['macro_f1', 'accuracy']].agg(['mean', 'std'])
print('\\nResumen 5-fold (promedio ± desvío):')
for modo in ['A_con_fondo', 'B_sin_fondo']:
    f1_m, f1_s = resumen_cv.loc[modo, ('macro_f1', 'mean')], resumen_cv.loc[modo, ('macro_f1', 'std')]
    ac_m, ac_s = resumen_cv.loc[modo, ('accuracy', 'mean')], resumen_cv.loc[modo, ('accuracy', 'std')]
    print(f'  {modo:12s} | Macro-F1: {f1_m:.4f} ± {f1_s:.4f} | Accuracy: {ac_m:.4f} ± {ac_s:.4f}')
""")

code("""
fig, ax = plt.subplots(figsize=(10, 6))
etiquetas = ['Macro-F1', 'Accuracy']
x = np.arange(len(etiquetas))
w = 0.34
for offset, modo, nombre in [(-w / 2, 'A_con_fondo', 'A · Con fondo'), (w / 2, 'B_sin_fondo', 'B · Sin fondo')]:
    medias = [resumen_cv.loc[modo, ('macro_f1', 'mean')], resumen_cv.loc[modo, ('accuracy', 'mean')]]
    desvios = [resumen_cv.loc[modo, ('macro_f1', 'std')], resumen_cv.loc[modo, ('accuracy', 'std')]]
    barras = ax.bar(x + offset, medias, w, yerr=desvios, capsize=6, label=nombre)
    for b, m in zip(barras, medias):
        ax.text(b.get_x() + b.get_width() / 2, m + 0.02, f'{m:.3f}', ha='center', fontsize=15)
ax.set_xticks(x); ax.set_xticklabels(etiquetas)
ax.set_ylim(0, 1.08); ax.set_ylabel('Puntaje (promedio ± desvío)')
ax.set_title(f'Validación cruzada ({N_FOLDS}-fold por grupos): A vs B')
ax.legend(loc='lower right')
fig.tight_layout()
utils.guardar_figura(fig, f'clf_cv_comparacion{SUFIJO}')
plt.show()
""")

md("""
## 6. Modelos finales y examen en test (359 fotos)
Entrenamos un modelo final para **A** (`models/clasificador_A.keras`) y otro para **B**
(`models/clasificador_B.keras`) usando el fold 0 para *early stopping* (siempre sin las 100
copias de test), y evaluamos ambos una sola vez sobre las **359 fotos de test**.
""")

code("""
tr_fin, va_fin = folds[0]
assert not (set(grupos_cv[tr_fin]) & set(grupos_cv[va_fin]))
assert not (set(grupos_cv[tr_fin]) & set(grupos_test))

modelos = {}
for clave, ruta, X_cv in [('A', RUTA_MOD_A, X_cv_A), ('B', RUTA_MOD_B, X_cv_B)]:
    if ruta.exists() and not REENTRENAR:
        print('Cargando modelo final:', ruta.name)
        modelos[clave] = tf.keras.models.load_model(ruta)
    else:
        print(f'Entrenando modelo final {clave}...')
        m, hist = entrenar_una_vez(
            X_cv[tr_fin], y_cv[tr_fin], X_cv[va_fin], y_cv[va_fin], seed=utils.SEED, verbose=1
        )
        m.save(ruta)
        hist.to_csv(DIR_RES / f'clf_{clave}_historia{SUFIJO}.csv', index_label='epoca')
        modelos[clave] = m
        print(f'Guardado {ruta.name} ({m.count_params():,} parámetros)')
""")

code("""
probs_A = modelos['A'].predict(X_test_A, batch_size=BATCH, verbose=0)
probs_B = modelos['B'].predict(X_test_B, batch_size=BATCH, verbose=0)
pred_A = np.argmax(probs_A, axis=1)
pred_B = np.argmax(probs_B, axis=1)

pd.DataFrame({
    'path': df_test['path'],
    'real': df_test['clase'],
    'pred_A': [utils.CLASES[i] for i in pred_A],
    'conf_A': probs_A.max(axis=1),
    'pred_B': [utils.CLASES[i] for i in pred_B],
    'conf_B': probs_B.max(axis=1),
}).to_csv(DIR_RES / f'clf_test_preds{SUFIJO}.csv', index=False)

metricas_test = []
for nombre, pred in [('A_con_fondo', pred_A), ('B_sin_fondo', pred_B)]:
    acc = accuracy_score(y_test, pred)
    mf1 = f1_score(y_test, pred, average='macro')
    f1_clases = f1_score(y_test, pred, average=None, labels=range(len(utils.CLASES)), zero_division=0)
    fila = {'modelo': nombre, 'accuracy': acc, 'macro_f1': mf1}
    for c, v in zip(utils.CLASES, f1_clases):
        fila[f'f1_{c}'] = v
    metricas_test.append(fila)
    print(f'=== TEST · {nombre} (accuracy={acc:.4f}, macro_f1={mf1:.4f}) ===')
    print(classification_report(y_test, pred, target_names=utils.CLASES, digits=3, zero_division=0))

df_test_met = pd.DataFrame(metricas_test)
df_test_met.to_csv(DIR_RES / f'clf_test_metricas{SUFIJO}.csv', index=False)
print('Fotos de test donde A y B predicen distinto:', int((pred_A != pred_B).sum()), 'de', len(y_test))
""")

code("""
# Matrices de confusión lado a lado (A vs B)
fig, ax = plt.subplots(1, 2, figsize=(18, 7))
etiquetas_cortas = ['sanas', 'mancha\\nbact.', 'mildiu', 'oídio']
for a, pred, titulo in [(ax[0], pred_A, 'Modelo A · Con fondo'), (ax[1], pred_B, 'Modelo B · Sin fondo')]:
    cm = confusion_matrix(y_test, pred, labels=range(len(utils.CLASES)))
    im = a.imshow(cm, cmap='Blues')
    a.set_xticks(range(4)); a.set_xticklabels(etiquetas_cortas)
    a.set_yticks(range(4)); a.set_yticklabels(etiquetas_cortas)
    a.set_xlabel('Predicción'); a.set_ylabel('Clase real'); a.set_title(titulo)
    for i in range(4):
        for j in range(4):
            color = 'white' if cm[i, j] > cm.max() / 2 else 'black'
            a.text(j, i, str(cm[i, j]), ha='center', va='center', color=color, fontsize=18)
fig.tight_layout()
utils.guardar_figura(fig, f'clf_matrices_confusion{SUFIJO}')
plt.show()

# F1 por clase en test
fig, ax = plt.subplots(figsize=(12, 6))
t_f1 = df_test_met.set_index('modelo')[[f'f1_{c}' for c in utils.CLASES]].T
t_f1.index = utils.CLASES
t_f1.columns = ['A · Con fondo', 'B · Sin fondo']
t_f1.plot.bar(ax=ax, rot=0)
ax.set_ylim(0, 1.05); ax.set_ylabel('F1 en test'); ax.set_xlabel('')
ax.set_title('F1 por clase en test: con fondo (A) vs sin fondo (B)')
ax.legend(loc='lower right')
fig.tight_layout()
utils.guardar_figura(fig, f'clf_f1_por_clase{SUFIJO}')
plt.show()
""")

md("""
## 7. Grad-CAM: ¿mira el fondo o mira la hoja?
Calculamos **Grad-CAM** para **A (con fondo)** y **B (sin fondo)** sobre 4 fotos de test (1 por
clase, priorizando fotos donde A y B difieren si las hay). Rojo/amarillo = zona que más influyó
en la predicción; azul = zona ignorada.
""")

code("""
elegidas = []
for c_idx in range(len(utils.CLASES)):
    idxs = np.where(y_test == c_idx)[0]
    dif = [i for i in idxs if pred_A[i] != pred_B[i]]
    elegidas.append(int(dif[0]) if dif else int(idxs[0]))

fig, ax = plt.subplots(len(elegidas), 4, figsize=(20, 5 * len(elegidas)))
titulos_col = ['Foto A (con fondo)', 'Grad-CAM A', 'Foto B (sin fondo)', 'Grad-CAM B']
for fila, idx in enumerate(elegidas):
    real = utils.CLASES[y_test[idx]]
    mA = utils.grad_cam(modelos['A'], X_test_A[idx], clase_idx=int(pred_A[idx]))
    mB = utils.grad_cam(modelos['B'], X_test_B[idx], clase_idx=int(pred_B[idx]))
    imgs = [
        X_test_A[idx],
        utils.superponer_gradcam(X_test_A[idx], mA),
        X_test_B[idx],
        utils.superponer_gradcam(X_test_B[idx], mB),
    ]
    subtitulos = [
        f'Real: {real}',
        f'A: {utils.CLASES[pred_A[idx]]} ({probs_A[idx].max():.0%})',
        'Sin fondo (UNet v6)',
        f'B: {utils.CLASES[pred_B[idx]]} ({probs_B[idx].max():.0%})',
    ]
    for col in range(4):
        ax[fila, col].imshow(imgs[col])
        ax[fila, col].axis('off')
        if fila == 0:
            ax[fila, col].set_title(f'{titulos_col[col]}\\n{subtitulos[col]}', fontsize=15)
        else:
            ax[fila, col].set_title(subtitulos[col], fontsize=15)

fig.tight_layout()
utils.guardar_figura(fig, f'clf_gradcam_A_vs_B{SUFIJO}')
plt.show()
""")

md("""
## 8. Conclusiones (entrenamiento oficial receta `v4d` — Modelo A vs Modelo B)
Resultados obtenidos ejecutando exactamente la misma arquitectura (`MobileNetV2` 224×224,
2.263.108 parámetros, 2 fases: congelada + fine-tuning de las últimas 35 capas, aumentación
`zoom_suave`, pesos `sqrt(balanced)` y `label_smoothing=0.05`) sobre los mismos 5 folds por grupos
y el mismo test fijo; para **B** se aplica además `utils.recortar_a_hoja()` (cuadro ajustado +5%
margen sobre gris neutro `128`):

- **Validación cruzada 5-fold por grupos (1.934 fotos sin copias de test, promedio ± desvío):**
  - **Modelo A (con fondo):** Macro-F1 **`0.9772 ± 0.0136`** · Accuracy **`0.9845 ± 0.0066`**
  - **Modelo B (sin fondo):** Macro-F1 **`0.9706 ± 0.0174`** · Accuracy **`0.9788 ± 0.0096`**
  - En CV ambos modelos están prácticamente empatados dentro de 1 desvío estándar (`-0.0066` en Macro-F1).
- **Test interno fijo (359 fotos de Niphad):**
  - **Modelo A (con fondo):** Accuracy **`0.9582`** (15 errores) · Macro-F1 **`0.9578`**. F1 por clase: `hojas_sanas 0.956`, `mancha_bacteriana 0.968`, `mildiu 0.966`, `oidio 0.941`.
  - **Modelo B (sin fondo):** Accuracy **`0.9610`** (14 errores) · Macro-F1 **`0.9523`** (+0.020 respecto al baseline v0 `0.9320`). F1 por clase: `hojas_sanas 0.968`, `mancha_bacteriana 0.929` (100% precisión, 0 falsos positivos), `mildiu 0.962`, `oidio 0.951`.
  - Discrepancias (15/359): **B acierta 8 que A falla** (incluida `Powdery Mildew_395.png` en viñedo real) vs A acierta 7 que B falla.
- **Generalización en test externo de campo real (1.432 fotos de 4 fuentes independientes: iNaturalist, GBIF, GLVD y `j4xs3kh3fd-2`):**
  - **Modelo A:** Macro-F1 **`0.3667`** · Balanced-Acc **`43.6%`** (`sanas 70.2%`, `mildiu 45.8%`, `oidio 14.9%`).
  - **Modelo B:** Macro-F1 **`0.4313`** · Balanced-Acc **`43.8%`** (`sanas 71.0%`, `mildiu 28.5%`, `oidio 32.0%`, `campo_una_hoja 85.5%`, `lab_externo 46.0%`).
  - Frente al baseline v0 de B (`33.8%` Balanced-Acc, `37.4%` falsos `mancha_bacteriana`), la receta `v4d` subió **+10,0 pp** la Balanced-Acc en campo, redujo los falsos `mancha_bacteriana` a **`15.3%`** (-22,1 pp) y duplicó el acierto de B frente a A en `oidio` real de viñedo (`32.0%` vs `14.9%`).
- **Grad-CAM:** en las discrepancias de test (por ej. `hojas_sanas` y `oidio`), B concentra su atención en la lámina foliar sobre el fondo gris neutro, mientras A sigue siendo sensible al pecíolo/sombras y al fondo de campo.

**Recordar:** el modelo es una aproximación limitada a 4 clases (otras causas como deficiencias de
nutrientes, estrés hídrico, quemaduras de sol, insectos, agroquímicos u otras enfermedades también
producen manchas y la red siempre elige una de sus 4 clases). El sistema **no emite recomendaciones**
de manejo ni tratamiento; la decisión corresponde siempre al ingeniero agrónomo.
""")

nb = nbf.v4.new_notebook()
nb['cells'] = celdas
nb['metadata']['kernelspec'] = {'name': 'python3', 'display_name': 'Python 3', 'language': 'python'}
nb['metadata']['accelerator'] = 'GPU'
nbf.write(nb, ROOT / '03_clasificacion.ipynb')
print('Creado 03_clasificacion.ipynb')
