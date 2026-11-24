"""Recorta todas las hojas sin fondo con la UNet y las guarda en datos/dataset_segmentado/.

Uso (desde la carpeta del proyecto):
    python scripts/armar_dataset_segmentado.py

- Usa la UNet guardada en models/unet.keras (la v6) y utils.segmentar_hoja(), que ya
  incluye la limpieza de la máscara.
- Pasa las 2393 fotos de datos/splits.csv (las mismas que usa el clasificador A, con fondo).
- Guarda cada hoja con el fondo en negro, del mismo tamaño que la original (256x256), en
  datos/dataset_segmentado/<clase>/<nombre>.png. Además arma datos/dataset_segmentado.zip
  para subir a Colab.
- No toca dataset/: solo lee.
"""
import shutil
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import utils

DESTINO = utils.DATOS / 'dataset_segmentado'


def main():
    model = tf.keras.models.load_model(ROOT / 'models' / 'unet.keras',
                                       custom_objects={'bce_dice_loss': utils.bce_dice_loss})
    df = utils.cargar_splits()

    if DESTINO.exists():
        shutil.rmtree(DESTINO)

    sospechosas = []
    for i, path in enumerate(df['path']):
        foto = utils.cargar_imagen(path)
        mascara = utils.segmentar_hoja(model, foto)
        destino = DESTINO / Path(path).parent.name / f'{Path(path).stem}.png'
        destino.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(utils.quitar_fondo(foto, mascara)).save(destino)
        if utils.es_sospechosa(mascara):            # la hoja ocupa menos del 2% o más del 95%
            sospechosas.append((path, round(float(mascara.mean()), 3)))
        if (i + 1) % 200 == 0:
            print(i + 1, 'de', len(df))

    print('Fotos guardadas:', sum(1 for _ in DESTINO.rglob('*.png')), 'de', len(df))
    print('Máscaras sospechosas (hoja < 2% o > 95% de la foto):', len(sospechosas))
    for path, area in sospechosas:
        print('  ', path, area)
    shutil.make_archive(str(DESTINO), 'zip', DESTINO)    # datos/dataset_segmentado.zip, para Colab
    print('Listo:', DESTINO, 'y', DESTINO.name + '.zip')


if __name__ == '__main__':
    main()
