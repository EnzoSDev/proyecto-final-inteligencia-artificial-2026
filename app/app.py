"""App de demo: enfermedades en hojas de vid.

Uso (desde la carpeta del proyecto, con el entorno virtual activado):
    python app/app.py
y abrir en el navegador: http://127.0.0.1:7860
"""
import sys
from pathlib import Path

import gradio as gr
import tensorflow as tf

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

AVISO = (
    '**Es una aproximación.** La red siempre elige una de sus 4 clases, aunque la hoja tenga otra '
    'cosa (falta de nutrientes o de agua, sol, insectos, químicos u otras enfermedades). '
    '**No da recomendaciones:** la decisión es del ingeniero agrónomo.'
)

# Los modelos se cargan una sola vez, al abrir la app
unet = tf.keras.models.load_model(ROOT / 'models' / 'unet.keras',
                                  custom_objects={'bce_dice_loss': utils.bce_dice_loss})
modelos_B = [tf.keras.models.load_model(ROOT / 'models' / f'clasificador_B_fold{k}.keras', compile=False)
             for k in range(1, 6)]


def diagnosticar(foto):
    """Foto -> UNet saca el fondo -> modelo B (5 modelos promediados) -> probabilidades."""
    if foto is None:
        return None, None
    mascara = utils.segmentar_hoja(unet, foto)         # UNet + limpieza de la máscara
    # Lo que mira el modelo B: hoja recortada, 224x224, fondo gris.
    # min_area=0: siempre recorta la hoja, igual que en el notebook 03 (decisión de Enzo).
    hoja = utils.recortar_a_hoja(foto, mascara, min_area=0)
    probs = utils.predecir_ensamble(modelos_B, hoja)    # 5 modelos x 4 posiciones, promediado
    resultado = {NOMBRES[c]: float(p) for c, p in zip(utils.CLASES, probs)}
    return hoja, resultado


with gr.Blocks(title='Hojas de vid') as app:
    gr.Markdown('# Enfermedades en hojas de vid\nSubí una foto de **una sola hoja**.')

    with gr.Tab('Diagnosticar'):
        with gr.Row():
            entrada = gr.Image(label='Foto de la hoja', type='numpy')
            with gr.Column():
                salida_hoja = gr.Image(label='Hoja sin fondo (lo que mira la red)')
                salida_probs = gr.Label(label='Probabilidad de cada clase', num_top_classes=4)
        boton = gr.Button('Diagnosticar', variant='primary')
        gr.Markdown(AVISO)

    boton.click(diagnosticar, inputs=entrada, outputs=[salida_hoja, salida_probs])


if __name__ == '__main__':
    app.launch(server_name='127.0.0.1', server_port=7860)
