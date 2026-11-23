"""Junta las fotos y sus máscaras en dataset_unet/, listo para entrenar la UNet.

Uso (desde la carpeta del proyecto):
    python scripts/armar_dataset_unet.py

Arma esta estructura, con todo ya en 160x160 (utils.IMG_SEG):

    dataset_unet/
      train/images/<foto>.png   train/masks/<foto>.png   <- máscaras automáticas
      val/images/...            val/masks/...            <- máscaras automáticas
      test/images/...           test/masks/...           <- las 50 corregidas a mano
      test/masks_auto/...       las automáticas de esas 50 (para comparar con la UNet)
      lista.csv                 de qué foto original sale cada archivo

Además guarda dataset_unet.zip, que es lo que se sube a Drive para Colab.

- train y val: sin las máscaras rechazadas ni las copias de fotos de test.
- test: solo las 50 fotos con máscara corregida a mano (el "examen" de la UNet).
- No toca dataset/ ni masks/: solo lee. Si dataset_unet/ ya existe, lo borra y lo arma de nuevo.
"""
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import utils

DESTINO = ROOT / 'dataset_unet'


def ruta_mascara(path, carpeta):
    """dataset/<clase>/foto.jpg -> masks/<carpeta>/<clase>/foto.png"""
    p = Path(path)
    return ROOT / 'masks' / carpeta / p.parent.name / f'{p.stem}.png'


def guardar_mascara(path, carpeta_mascara, destino):
    # Achicada con "vecino más cercano" para que siga siendo solo 0 y 255.
    mascara = np.array(Image.open(ruta_mascara(path, carpeta_mascara)).convert('L'))
    mascara = cv2.resize(mascara, (utils.IMG_SEG, utils.IMG_SEG), interpolation=cv2.INTER_NEAREST)
    mascara = np.where(mascara > 127, 255, 0).astype(np.uint8)
    Image.fromarray(mascara).save(destino)


def guardar_par(path, split, carpeta_mascara):
    nombre = f'{Path(path).stem}.png'
    # Foto: RGB de 3 canales, achicada con interpolación suave.
    foto = utils.cargar_imagen(path, utils.IMG_SEG)
    Image.fromarray(foto).save(DESTINO / split / 'images' / nombre)
    guardar_mascara(path, carpeta_mascara, DESTINO / split / 'masks' / nombre)
    return nombre


def main():
    if not (ROOT / 'masks' / 'auto').exists():
        sys.exit('Falta masks/auto/: primero ejecutá 01_mascaras.ipynb.')

    df = utils.cargar_splits()
    rechazadas = utils.cargar_rechazadas()

    # train y val: sin copias de test y sin máscaras rechazadas.
    entrenar = utils.quitar_copias_de_test(df)
    entrenar = entrenar[~entrenar['path'].isin(rechazadas)]

    # test: las 50 de masks/manual/LISTA.txt.
    lista = (ROOT / 'masks' / 'manual' / 'LISTA.txt').read_text(encoding='utf-8').splitlines()
    lista = [x.strip() for x in lista if x.strip()]
    examen = df[df['path'].isin(lista)]
    assert len(examen) == len(lista) and (examen['split'] == 'test').all()

    if DESTINO.exists():
        shutil.rmtree(DESTINO)
    for split in ('train', 'val', 'test'):
        (DESTINO / split / 'images').mkdir(parents=True)
        (DESTINO / split / 'masks').mkdir(parents=True)
    (DESTINO / 'test' / 'masks_auto').mkdir()

    filas = []
    for _, fila in entrenar.iterrows():
        nombre = guardar_par(fila['path'], fila['split'], 'auto')
        filas.append({'split': fila['split'], 'archivo': nombre, 'clase': fila['clase'],
                      'original': fila['path'], 'mascara': 'auto'})
    for _, fila in examen.iterrows():
        nombre = guardar_par(fila['path'], 'test', 'manual')
        guardar_mascara(fila['path'], 'auto', DESTINO / 'test' / 'masks_auto' / nombre)
        filas.append({'split': 'test', 'archivo': nombre, 'clase': fila['clase'],
                      'original': fila['path'], 'mascara': 'manual'})

    tabla = pd.DataFrame(filas)
    tabla.to_csv(DESTINO / 'lista.csv', index=False)
    print(pd.crosstab(tabla['clase'], tabla['split'], margins=True, margins_name='total'))
    shutil.make_archive(str(DESTINO), 'zip', DESTINO)      # dataset_unet.zip, para Colab
    print('Listo:', DESTINO, 'y', DESTINO.name + '.zip')


if __name__ == '__main__':
    main()
