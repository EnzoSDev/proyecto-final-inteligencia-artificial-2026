"""Funciones compartidas por todos los notebooks del proyecto.

Los notebooks lo importan con:  import utils
"""
import hashlib
import shutil
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------
# Constantes del proyecto
# ---------------------------------------------------------------------------
SEED = 42                     # semilla: hace que la división al azar salga siempre igual
CLASES = ['hojas_sanas', 'mancha_bacteriana', 'mildiu', 'oidio']
EXT = {'.jpg', '.jpeg', '.png'}  # extensiones válidas (se comparan en minúsculas)
IMG_CLF = 224                 # tamaño de entrada del clasificador (MobileNetV2)
IMG_SEG = 160                 # tamaño de entrada de la UNet
BATCH = 32

ROOT = Path(__file__).resolve().parent   # carpeta del proyecto (donde está utils.py)
DATOS = ROOT / 'datos'                   # todo lo que armamos a partir de dataset/
MASKS = DATOS / 'masks'                  # auto/, manual/ y rechazadas.txt
RUTA_SPLITS = DATOS / 'splits.csv'
RUTA_COPIAS = DATOS / 'casi_duplicados.csv'


# ---------------------------------------------------------------------------
# Entorno: compu o Google Colab
# ---------------------------------------------------------------------------
def configurar_entorno(carpetas=('dataset',)):
    """Dice dónde están los datos y lo prepara si estamos en Colab.

    - En la compu: ROOT y DATA son la carpeta del proyecto.
    - En Colab: ROOT es la carpeta del proyecto en Drive (ahí se guardan modelos y
      figuras) y DATA es /content. Cada carpeta de `carpetas` se copia a /content,
      porque leer miles de archivos chicos desde Drive es muy lento. Si existe
      <carpeta>.zip en Drive, se descomprime (es mucho más rápido que copiar).
      Una carpeta puede tener subcarpetas, por ejemplo 'datos/dataset_unet' busca
      datos/dataset_unet.zip y lo deja en /content/datos/dataset_unet.
    """
    en_colab = 'google.colab' in sys.modules
    data = Path('/content') if en_colab else ROOT
    if en_colab:
        for carpeta in carpetas:
            destino = data / carpeta
            if destino.exists():
                continue
            if (ROOT / f'{carpeta}.zip').exists():
                print('Descomprimiendo', carpeta + '.zip ...')
                shutil.unpack_archive(ROOT / f'{carpeta}.zip', destino)
            elif (ROOT / carpeta).exists():
                print('Copiando', carpeta, '(puede tardar) ...')
                shutil.copytree(ROOT / carpeta, destino)
            else:
                print('ATENCIÓN: no encuentro', carpeta, 'ni', carpeta + '.zip', 'en', ROOT)

    import tensorflow as tf
    print('Entorno:', 'Google Colab' if en_colab else 'compu local')
    print('GPU:', tf.config.list_physical_devices('GPU') or 'no hay (CPU)')
    print('TensorFlow:', tf.__version__)
    return {'ROOT': ROOT, 'DATA': data, 'EN_COLAB': en_colab}


# ---------------------------------------------------------------------------
# Listar las imágenes del dataset
# ---------------------------------------------------------------------------
def listar_imagenes(root=ROOT):
    """Devuelve una tabla con una fila por imagen: path, clase y label.

    - path: ruta relativa a la carpeta del proyecto (ej. 'dataset/mildiu/foto.jpg').
    - label: la clase como número (0..3), en el orden de CLASES.
    Ignora lo que no sea imagen (por ejemplo desktop.ini).
    """
    root = Path(root)
    filas = []
    for label, clase in enumerate(CLASES):
        for archivo in sorted((root / 'dataset' / clase).iterdir()):
            if archivo.suffix.lower() in EXT:          # .JPG también cuenta
                filas.append({
                    'path': archivo.relative_to(root).as_posix(),
                    'clase': clase,
                    'label': label,
                })
    return pd.DataFrame(filas)


def _md5(ruta):
    """Huella del archivo: dos archivos idénticos dan el mismo MD5."""
    return hashlib.md5(Path(ruta).read_bytes()).hexdigest()


# ---------------------------------------------------------------------------
# Preprocesado: sacar fotos duplicadas
# ---------------------------------------------------------------------------
def duplicados(df, root=ROOT):
    """Devuelve las filas que tienen al menos una copia exacta en el dataset."""
    df = df.copy()
    df['md5'] = [_md5(Path(root) / p) for p in df['path']]
    return df[df.duplicated('md5', keep=False)].sort_values(['md5', 'path'])


def quitar_duplicados(df, root=ROOT):
    """Deja una sola foto por cada grupo de copias exactas (la primera por nombre).

    No borra nada de dataset/: solo la saca de la tabla.
    """
    df = df.copy()
    df['md5'] = [_md5(Path(root) / p) for p in df['path']]
    df = df.sort_values('path').drop_duplicates('md5', keep='first')
    return df.drop(columns='md5').reset_index(drop=True)


# ---------------------------------------------------------------------------
# Dividir en train / val / test (una sola vez)
# ---------------------------------------------------------------------------
def crear_splits(df, guardar=True):
    """Divide las imágenes en 70% train, 15% val y 15% test, estratificado por clase.

    Recibe la tabla ya sin duplicados (ver quitar_duplicados).
    Si guardar=True, escribe splits.csv. NO llamar si splits.csv ya existe:
    usar cargar_splits().
    """
    # 1) 70% train / 30% resto   2) el resto se parte en 15% val / 15% test.
    train, resto = train_test_split(
        df, test_size=0.30, stratify=df['label'], random_state=SEED)
    val, test = train_test_split(
        resto, test_size=0.50, stratify=resto['label'], random_state=SEED)

    df = df.copy()
    df.loc[train.index, 'split'] = 'train'
    df.loc[val.index, 'split'] = 'val'
    df.loc[test.index, 'split'] = 'test'

    if guardar:
        df.to_csv(RUTA_SPLITS, index=False)
    return df


def cargar_splits():
    """Lee splits.csv (la división única del proyecto)."""
    return pd.read_csv(RUTA_SPLITS)


def quitar_copias_de_test(df):
    """Devuelve solo train y val, sin las fotos que son copia de alguna de test.

    casi_duplicados.csv dice qué fotos son copias entre sí (mismo 'grupo').
    Si un grupo tiene una foto en test, sus otras fotos no se usan para entrenar.
    No modifica splits.csv ni dataset/.
    """
    copias = pd.read_csv(RUTA_COPIAS)
    grupos_test = copias.loc[copias['split'] == 'test', 'grupo']
    prohibidas = copias.loc[copias['grupo'].isin(grupos_test), 'path']
    return df[(df['split'] != 'test') & ~df['path'].isin(prohibidas)].reset_index(drop=True)


def asignar_grupos(df):
    """Devuelve el ID de grupo de cada foto según datos/casi_duplicados.csv.

    Las fotos que son casi copias comparten el mismo grupo; las demás son su propio grupo.
    Sirve para que StratifiedGroupKFold no parta copias entre train y val.
    """
    copias = pd.read_csv(RUTA_COPIAS)
    mapa = dict(zip(copias['path'], 'g_' + copias['grupo'].astype(str)))
    return np.array([mapa.get(p, p) for p in df['path']])


def ruta_segmentada(path):
    """dataset/<clase>/foto.jpg -> datos/dataset_segmentado/<clase>/foto.png"""
    p = Path(path)
    return (Path('datos') / 'dataset_segmentado' / p.parent.name / f'{p.stem}.png').as_posix()


def cargar_rechazadas():
    """Lee datos/masks/rechazadas.txt: fotos con la máscara mal (no entrenan la UNet)."""
    texto = (MASKS / 'rechazadas.txt').read_text(encoding='utf-8')
    return [linea.strip() for linea in texto.splitlines() if linea.strip()]


# ---------------------------------------------------------------------------
# Cargar imágenes y figuras
# ---------------------------------------------------------------------------
def cargar_imagen(path, size=None, base=ROOT):
    """Abre una foto y la devuelve como array RGB uint8 de 3 canales.

    Convierte los PNG con transparencia y las fotos en escala de grises.
    Si se pasa size, la redimensiona a size x size.
    """
    img = Image.open(Path(base) / path).convert('RGB')
    if size is not None:
        img = img.resize((size, size), Image.BILINEAR)
    return np.array(img)


def set_estilo():
    """Letra grande en los gráficos, porque van a la presentación."""
    plt.rcParams.update({'font.size': 16, 'axes.titlesize': 20,
                         'figure.titlesize': 22})


def guardar_figura(fig, nombre):
    """Guarda la figura en figuras/<nombre>.png."""
    ruta = ROOT / 'figuras' / f'{nombre}.png'
    ruta.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(ruta, dpi=150, bbox_inches='tight')
    return ruta


# ---------------------------------------------------------------------------
# Máscara automática (hoja = blanco, fondo = negro)
# ---------------------------------------------------------------------------
def mascara_auto(img, kernel=7, pasos=False):
    """Calcula la máscara de la hoja con un truco de color, sin red neuronal.

    La hoja tiene colores más intensos (saturados) que el fondo, que suele
    ser blanco o gris. Pasos:
      1. RGB -> HSV y quedarse con el canal S (saturación).
      2. Umbral de Otsu: elige solo el corte entre "saturado" y "apagado".
      3. Apertura: borra puntitos blancos sueltos en el fondo.
      4. Cierre: tapa agujeritos negros dentro de la hoja.
      5. Quedarse con la mancha blanca más grande (la hoja).
      6. Rellenar huecos: lesiones marrones (poco saturadas) que quedaron
         como agujeros dentro de la hoja.

    Devuelve la máscara uint8 (0 = fondo, 255 = hoja).
    Con pasos=True devuelve además un dict con el resultado de cada paso.
    """
    # 1. Canal de saturación
    s = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)[:, :, 1]

    # 2. Umbral de Otsu
    _, m_otsu = cv2.threshold(s, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 3 y 4. Apertura y cierre con un "pincel" circular
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel, kernel))
    m_limpia = cv2.morphologyEx(m_otsu, cv2.MORPH_OPEN, k)
    m_limpia = cv2.morphologyEx(m_limpia, cv2.MORPH_CLOSE, k)

    # 5. Mancha más grande (la etiqueta 0 es el fondo, no se cuenta)
    n, etiquetas, stats, _ = cv2.connectedComponentsWithStats(m_limpia)
    if n > 1:
        mayor = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        m_mayor = np.where(etiquetas == mayor, 255, 0).astype(np.uint8)
    else:
        m_mayor = m_limpia.copy()

    # 6. Rellenar huecos: se pinta el fondo desde afuera; lo que quedó
    #    negro sin pintar es un hueco encerrado por la hoja.
    borde = cv2.copyMakeBorder(m_mayor, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    pintado = borde.copy()
    relleno = np.zeros((borde.shape[0] + 2, borde.shape[1] + 2), np.uint8)
    cv2.floodFill(pintado, relleno, (0, 0), 255)
    huecos = cv2.bitwise_not(pintado)[1:-1, 1:-1]
    m_final = cv2.bitwise_or(m_mayor, huecos)

    if pasos:
        return m_final, {'S (saturación)': s, 'Otsu': m_otsu,
                         'Apertura + cierre': m_limpia,
                         'Mancha más grande': m_mayor, 'Huecos rellenos': m_final}
    return m_final


def es_sospechosa(mascara, minimo=0.02, maximo=0.95):
    """True si la máscara ocupa menos del 2% o más del 95% de la foto."""
    area = np.mean(mascara > 0)
    return area < minimo or area > maximo


def limpiar_mascara(m):
    """Limpia la máscara que predice la UNet (True = hoja).

    Como hay una sola hoja por foto, deja solo la mancha blanca más grande (borra
    pedacitos sueltos) y rellena los agujeros de adentro. Se usa siempre después de
    la UNet: en el notebook 02, al recortar las fotos y en la app.
    """
    from scipy import ndimage
    etiquetas, n = ndimage.label(m)                  # numera cada mancha blanca separada
    if n > 1:
        tamanios = ndimage.sum(m, etiquetas, range(1, n + 1))
        m = etiquetas == (np.argmax(tamanios) + 1)   # deja solo la más grande
    return ndimage.binary_fill_holes(m)              # rellena los agujeros de adentro


def segmentar_hoja(model, foto):
    """Usa la UNet para encontrar la hoja. Devuelve la máscara (True = hoja) del tamaño de la foto.

    1. Achica la foto a 160x160 (lo que espera la UNet).
    2. La UNet da, para cada píxel, la probabilidad de ser hoja.
    3. Agranda esas probabilidades al tamaño original, corta en 0.5 y limpia la máscara.
    """
    chica = np.array(Image.fromarray(foto).resize((IMG_SEG, IMG_SEG), Image.BILINEAR))
    prob = model(chica[None], training=False).numpy()[0, :, :, 0]
    prob = np.array(Image.fromarray(prob).resize((foto.shape[1], foto.shape[0]), Image.BILINEAR))
    return limpiar_mascara(prob > 0.5)


def quitar_fondo(foto, mascara):
    """Pinta de negro todo lo que no es hoja."""
    return foto * mascara[..., None].astype(np.uint8)


# ---------------------------------------------------------------------------
# Interpretabilidad: Grad-CAM para el clasificador (MobileNetV2 sin anidar)
# ---------------------------------------------------------------------------
def grad_cam(model, foto, clase_idx=None, capa='out_relu'):
    """Mapa de calor Grad-CAM (0..1) del tamaño de `foto`.

    Indica qué zonas de la imagen pesaron más al predecir `clase_idx`
    (si es None, usa la clase con mayor probabilidad).
    """
    import tensorflow as tf
    h, w = foto.shape[:2]
    entrada = np.array(Image.fromarray(foto).resize((IMG_CLF, IMG_CLF), Image.BILINEAR), dtype=np.float32)[None]
    extractor = tf.keras.Model(model.inputs[0], [model.get_layer(capa).output, model.outputs[0]])
    with tf.GradientTape() as tape:
        conv_out, preds = extractor(entrada, training=False)
        if clase_idx is None:
            clase_idx = int(tf.argmax(preds[0]))
        puntaje = preds[:, clase_idx]
    grads = tape.gradient(puntaje, conv_out)
    pesos = tf.reduce_mean(grads, axis=(0, 1, 2))
    mapa = tf.reduce_sum(conv_out[0] * pesos, axis=-1)
    mapa = tf.maximum(mapa, 0)
    maximo = tf.reduce_max(mapa)
    if maximo > 0:
        mapa = mapa / maximo
    mapa = cv2.resize(mapa.numpy(), (w, h), interpolation=cv2.INTER_LINEAR)
    return np.clip(mapa, 0.0, 1.0)


def superponer_gradcam(foto, mapa, alpha=0.45):
    """Superpone el mapa de calor Grad-CAM (rojo = mayor atención) sobre `foto` RGB."""
    calor = cv2.applyColorMap(np.uint8(255 * mapa), cv2.COLORMAP_JET)[:, :, ::-1]
    return np.uint8(np.clip((1.0 - alpha) * foto + alpha * calor, 0, 255))


# ---------------------------------------------------------------------------
# Pérdida de segmentación (UNet)
# ---------------------------------------------------------------------------
def bce_dice_loss(y_true, y_pred):
    """Combina entropía cruzada binaria (píxel a píxel) con Dice (solapamiento)."""
    import tensorflow as tf
    y_true = tf.cast(y_true, tf.float32)
    bce = tf.reduce_mean(tf.keras.losses.binary_crossentropy(y_true, y_pred))
    inter = tf.reduce_sum(y_true * y_pred, axis=(1, 2, 3))
    denom = tf.reduce_sum(y_true + y_pred, axis=(1, 2, 3))
    dice = tf.reduce_mean((2.0 * inter + 1.0) / (denom + 1.0))
    return bce + (1.0 - dice)


if __name__ == '__main__':
    # python utils.py  -> genera splits.csv si todavía no existe
    if RUTA_SPLITS.exists():
        print('splits.csv ya existe: no se regenera.')
    else:
        todas = listar_imagenes()
        unicas = quitar_duplicados(todas)
        print('fotos:', len(todas), '| sin duplicados:', len(unicas))
        df = crear_splits(unicas)
        print('splits.csv creado con', len(df), 'filas')
