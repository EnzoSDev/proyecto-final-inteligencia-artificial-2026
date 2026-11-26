"""App de demo: enfermedades en hojas de vid.

Uso (desde la carpeta del proyecto, con el entorno virtual activado):
    python app/app.py
y abrir en el navegador: http://127.0.0.1:7860
"""
import inspect
import sys
from pathlib import Path

import gradio as gr
import numpy as np
import tensorflow as tf
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))   # así Python encuentra utils.py
import utils

# Nombres lindos para mostrar en pantalla
NOMBRES = {
    'hojas_sanas': 'Sana',
    'mancha_bacteriana': 'Mancha bacteriana',
    'mildiu': 'Mildiu',
    'oidio': 'Oídio',
}

# ---------------------------------------------------------------------------
# Apariencia: los mismos colores y fuentes que la presentación
# ---------------------------------------------------------------------------
VID, HOJA, BROTE, TRIGO, PAJA = '#1f3a26', '#4e7d3a', '#a3c06e', '#f4ecd8', '#e3c47f'

COLORES = dict(
    body_background_fill=TRIGO,
    body_text_color=VID,
    body_text_color_subdued='#6e4b31',
    background_fill_primary='#fbf7ec',
    background_fill_secondary=TRIGO,
    block_background_fill='#fbf7ec',
    block_border_color='#e3d7bb',
    border_color_primary='#e3d7bb',
    block_label_background_fill=BROTE,
    block_label_text_color=VID,
    block_title_text_color=VID,
    input_background_fill='#fbf7ec',
    button_primary_background_fill=HOJA,
    button_primary_background_fill_hover=VID,
    button_primary_text_color=TRIGO,
    color_accent=HOJA,
    color_accent_soft=BROTE,
)
# Los mismos colores también en modo oscuro (si la compu está en modo oscuro, Gradio usa los "_dark").
# Solo se agregan los "_dark" que existen en Gradio.
_ACEPTA = set(inspect.signature(gr.themes.Base.set).parameters)
TEMA = gr.themes.Base(font=[gr.themes.GoogleFont('Work Sans'), 'sans-serif']).set(
    **COLORES, **{k + '_dark': v for k, v in COLORES.items() if k + '_dark' in _ACEPTA}
)

HEAD = ('<link href="https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,800'
        '&family=Work+Sans:wght@400;500;700&display=swap" rel="stylesheet">'
        # Siempre en modo claro, aunque la compu esté en modo oscuro (Gradio lo elige con ?__theme=light)
        '<script>if (!new URLSearchParams(location.search).has("__theme")) {'
        ' const u = new URL(location); u.searchParams.set("__theme", "light"); location.replace(u); }</script>')

CSS = f'''
.encabezado {{ background-color: {VID}; color: {TRIGO}; border-radius: 18px; padding: 28px 36px;
  background-image: repeating-linear-gradient(115deg, rgba(255,255,255,.035) 0 2px, transparent 2px 38px); }}
.encabezado .sobre {{ color: {PAJA}; letter-spacing: .2em; text-transform: uppercase; font-size: 15px; margin: 0; }}
.encabezado h1 {{ font-family: 'Fraunces', Georgia, serif; font-weight: 800; font-size: 44px; line-height: 1.05;
  margin: 10px 0 8px; color: {TRIGO}; }}
.encabezado .bajo {{ color: {BROTE}; font-size: 20px; margin: 0; }}
.encabezado .bajo b {{ color: {TRIGO}; }}
.aviso {{ background: {VID}; color: {TRIGO}; border-radius: 14px; padding: 16px 22px; font-size: 16px; line-height: 1.5; }}
.aviso b {{ color: {PAJA}; }}
.explica {{ font-size: 18px; margin: 6px 0; }}
footer {{ display: none !important; }}
'''

ENCABEZADO = '''<div class="encabezado">
  <p class="sobre">Inteligencia Artificial · Proyecto final · 2026</p>
  <h1>Segmentación y detección de enfermedades en hojas de vid</h1>
  <p class="bajo">Subí una foto de <b>una sola hoja</b>: la UNet le saca el fondo y la CNN dice qué tiene.</p>
</div>'''

AVISO = '''<div class="aviso"><b>Es una aproximación.</b> La red siempre elige una de sus 4 clases, aunque la hoja
tenga otra cosa (falta de nutrientes o de agua, sol, insectos, químicos u otras enfermedades).
<b>No da recomendaciones:</b> la decisión es del ingeniero agrónomo.</div>'''

# Los modelos se cargan una sola vez, al abrir la app
unet = tf.keras.models.load_model(ROOT / 'models' / 'unet.keras',
                                  custom_objects={'bce_dice_loss': utils.bce_dice_loss})
modelos_A = [tf.keras.models.load_model(ROOT / 'models' / f'clasificador_A_fold{k}.keras', compile=False)
             for k in range(1, 6)]
modelos_B = [tf.keras.models.load_model(ROOT / 'models' / f'clasificador_B_fold{k}.keras', compile=False)
             for k in range(1, 6)]

# Fotos de ejemplo (de test) para la demo
EJEMPLOS = sorted(str(p) for p in (ROOT / 'app' / 'ejemplos').glob('*.jpg'))


def a_porcentajes(probs):
    """Probabilidades de la red -> diccionario {nombre de la clase: probabilidad} para mostrar."""
    return {NOMBRES[c]: float(p) for c, p in zip(utils.CLASES, probs)}


def entrada_A(foto):
    """Lo que mira el modelo A: la foto entera achicada a 224x224 (con fondo)."""
    return np.array(Image.fromarray(foto).resize((utils.IMG_CLF, utils.IMG_CLF), Image.BILINEAR))


def entrada_B(foto):
    """Lo que mira el modelo B: la UNet saca el fondo y se recorta la hoja sobre gris (224x224).

    min_area=0: siempre recorta la hoja, igual que en el notebook 03 (decisión de Enzo).
    """
    mascara = utils.segmentar_hoja(unet, foto)          # UNet + limpieza de la máscara
    return utils.recortar_a_hoja(foto, mascara, min_area=0)


def mapa_de_calor(modelos, img, probs):
    """Grad-CAM promediado de los 5 modelos, sobre la clase elegida. Rojo = lo que más pesó."""
    clase = int(np.argmax(probs))
    mapa = np.mean([utils.grad_cam(m, img, clase_idx=clase) for m in modelos], axis=0)
    if mapa.max() > 0:
        mapa = mapa / mapa.max()
    return utils.superponer_gradcam(img, mapa)


def diagnosticar(foto):
    """Pestaña 1: foto -> UNet saca el fondo -> modelo B (5 modelos promediados) -> probabilidades."""
    if foto is None:
        return None, None
    hoja = entrada_B(foto)
    probs = utils.predecir_ensamble(modelos_B, hoja)    # 5 modelos x 4 posiciones, promediado
    return hoja, a_porcentajes(probs)


def comparar(foto):
    """Pestaña 2: la misma foto en el modelo A (con fondo) y en el B (sin fondo), con su Grad-CAM."""
    if foto is None:
        return None, None, None, None
    img_A, img_B = entrada_A(foto), entrada_B(foto)
    probs_A = utils.predecir_ensamble(modelos_A, img_A)
    probs_B = utils.predecir_ensamble(modelos_B, img_B)
    return (mapa_de_calor(modelos_A, img_A, probs_A), a_porcentajes(probs_A),
            mapa_de_calor(modelos_B, img_B, probs_B), a_porcentajes(probs_B))


with gr.Blocks(title='Hojas de vid') as app:
    gr.HTML(ENCABEZADO)

    with gr.Tab('Diagnosticar'):
        with gr.Row():
            entrada = gr.Image(label='Foto de la hoja', type='numpy')
            with gr.Column():
                salida_hoja = gr.Image(label='Hoja sin fondo (lo que mira la red)')
                salida_probs = gr.Label(label='Probabilidad de cada clase', num_top_classes=4)
        boton = gr.Button('Diagnosticar', variant='primary')
        gr.Examples(EJEMPLOS, inputs=entrada, label='Fotos de ejemplo (de test)')
        gr.HTML(AVISO)
    boton.click(diagnosticar, inputs=entrada, outputs=[salida_hoja, salida_probs])

    with gr.Tab('Con fondo vs sin fondo'):
        gr.HTML('<p class="explica">La misma foto en las dos redes: <b>A</b> se entrenó con fondo y '
                '<b>B</b> sin fondo. El mapa de calor muestra dónde miró cada una '
                '(<b>rojo</b> = lo que más pesó para decidir).</p>')
        with gr.Row():
            entrada2 = gr.Image(label='Foto de la hoja', type='numpy')
            with gr.Column():
                calor_A = gr.Image(label='A · con fondo: dónde mira')
                probs_A = gr.Label(label='A · probabilidades', num_top_classes=4)
            with gr.Column():
                calor_B = gr.Image(label='B · sin fondo: dónde mira')
                probs_B = gr.Label(label='B · probabilidades', num_top_classes=4)
        boton2 = gr.Button('Comparar', variant='primary')
        gr.Examples(EJEMPLOS, inputs=entrada2, label='Fotos de ejemplo (de test)')
        gr.HTML(AVISO)
    boton2.click(comparar, inputs=entrada2, outputs=[calor_A, probs_A, calor_B, probs_B])


if __name__ == '__main__':
    app.launch(server_name='127.0.0.1', server_port=7860, theme=TEMA, css=CSS, head=HEAD,
               allowed_paths=[str(ROOT / 'app' / 'ejemplos')])
