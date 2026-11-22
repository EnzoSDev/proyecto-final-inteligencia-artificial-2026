"""Corregir a mano las 50 máscaras de masks/manual/ pintando sobre la foto.

Uso (desde la carpeta del proyecto, con el entorno virtual activado):
    python scripts/corregir_mascaras.py enzo     -> le tocan 25 fotos (las impares)
    python scripts/corregir_mascaras.py tato     -> le tocan las otras 25 (las pares)

Lo que queda en color es HOJA; lo oscuro es FONDO. La línea roja es el borde.

    clic izquierdo   agrega hoja
    clic derecho     borra (lo manda a fondo)
    rueda  o  + -    agranda / achica el pincel
    Z                deshace el último trazo
    R                vuelve a la máscara automática
    F                muestra la foto sola (y de nuevo para volver)
    ENTER o ESPACIO  guarda, marca como revisada y pasa a la siguiente
    A / D            va a la anterior / siguiente sin marcarla como revisada
    Q o ESC          sale

Al cambiar de foto o salir, lo pintado se guarda solo. Si la máscara ya está bien,
alcanza con apretar ENTER.
"""
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import utils

MANUAL = ROOT / 'masks' / 'manual'
LADO = 600        # tamaño de la foto en pantalla (la máscara sigue siendo de 256x256)
ALTO_TEXTO = 60
VENTANA = 'Corregir mascaras'


def ruta_mascara(path, carpeta):
    """dataset/<clase>/foto.jpg -> masks/<carpeta>/<clase>/foto.png"""
    p = Path(path)
    return ROOT / 'masks' / carpeta / p.parent.name / f'{p.stem}.png'


def leer_mascara(ruta):
    # np.fromfile + imdecode: funciona aunque la ruta tenga tildes (cv2.imread no).
    m = cv2.imdecode(np.fromfile(str(ruta), np.uint8), cv2.IMREAD_GRAYSCALE)
    return np.where(m > 127, 255, 0).astype(np.uint8)


def escribir_mascara(ruta, mascara):
    ok, datos = cv2.imencode('.png', mascara)
    datos.tofile(str(ruta))


def repartir(lista, quien):
    """Enzo: posiciones 1, 3, 5... / Tato: 2, 4, 6... Así a los dos les tocan todas las clases."""
    return lista[0::2] if quien == 'enzo' else lista[1::2]


class Editor:
    def __init__(self, fotos, quien):
        self.fotos = fotos
        self.ruta_revisadas = MANUAL / f'revisadas_{quien}.txt'
        self.revisadas = set()
        if self.ruta_revisadas.exists():
            self.revisadas = set(self.ruta_revisadas.read_text(encoding='utf-8').splitlines())
        self.radio = 6            # radio del pincel, en píxeles de la foto de 256
        self.pintando = None      # 255 = agregando hoja, 0 = borrando, None = nada
        self.anterior = None
        self.cursor = None
        self.solo_foto = False
        # Arranca en la primera que falte revisar.
        faltan = [k for k, f in enumerate(fotos) if f not in self.revisadas]
        self.abrir(faltan[0] if faltan else 0)

    def abrir(self, k):
        self.k = k
        self.path = self.fotos[k]
        self.foto = utils.cargar_imagen(self.path, 256)[:, :, ::-1]   # RGB -> BGR (OpenCV)
        self.mascara = leer_mascara(ruta_mascara(self.path, 'manual'))
        self.original = self.mascara.copy()
        self.historial = []
        self.pintando = self.anterior = None

    def guardar(self):
        """Solo escribe el archivo si se cambió algo."""
        if not np.array_equal(self.mascara, self.original):
            escribir_mascara(ruta_mascara(self.path, 'manual'), self.mascara)
            self.original = self.mascara.copy()

    def marcar_revisada(self):
        self.revisadas.add(self.path)
        orden = [f for f in self.fotos if f in self.revisadas]
        self.ruta_revisadas.write_text('\n'.join(orden) + '\n', encoding='utf-8')

    def mover(self, paso):
        self.guardar()
        self.abrir(min(max(self.k + paso, 0), len(self.fotos) - 1))

    # ----- pintar -----
    def trazo(self, x, y):
        punto = (int(x * 256 / LADO), int(y * 256 / LADO))
        cv2.line(self.mascara, self.anterior or punto, punto, self.pintando, 2 * self.radio)
        cv2.circle(self.mascara, punto, self.radio, self.pintando, -1)
        self.anterior = punto

    def cambiar_radio(self, paso):
        self.radio = min(max(self.radio + paso, 1), 40)

    def mouse(self, evento, x, y, flags, _):
        y -= ALTO_TEXTO
        self.cursor = (x, y)
        if evento in (cv2.EVENT_LBUTTONDOWN, cv2.EVENT_RBUTTONDOWN):
            self.historial.append(self.mascara.copy())
            self.pintando = 255 if evento == cv2.EVENT_LBUTTONDOWN else 0
            self.trazo(x, y)
        elif evento == cv2.EVENT_MOUSEMOVE and self.pintando is not None:
            if flags & (cv2.EVENT_FLAG_LBUTTON | cv2.EVENT_FLAG_RBUTTON):
                self.trazo(x, y)
            else:                          # se soltó el botón fuera de la ventana
                self.pintando = self.anterior = None
        elif evento in (cv2.EVENT_LBUTTONUP, cv2.EVENT_RBUTTONUP):
            self.pintando = self.anterior = None
        elif evento == cv2.EVENT_MOUSEWHEEL:
            self.cambiar_radio(1 if flags > 0 else -1)

    # ----- mostrar -----
    def dibujar(self):
        vista = cv2.resize(self.foto, (LADO, LADO), interpolation=cv2.INTER_CUBIC)
        if not self.solo_foto:
            m = cv2.resize(self.mascara, (LADO, LADO), interpolation=cv2.INTER_NEAREST)
            oscura = (vista * 0.35).astype(np.uint8)
            vista = np.where(m[:, :, None] > 0, vista, oscura)
            bordes, _ = cv2.findContours(m, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(vista, bordes, -1, (0, 0, 255), 1)
        if self.cursor is not None:
            cv2.circle(vista, self.cursor, max(1, int(self.radio * LADO / 256)), (255, 255, 255), 1)

        texto = np.zeros((ALTO_TEXTO, LADO, 3), np.uint8)
        hechas = sum(f in self.revisadas for f in self.fotos)
        estado = 'REVISADA' if self.path in self.revisadas else 'sin revisar'
        clase = Path(self.path).parent.name
        for y, t, color in (
            (24, f'{self.k + 1}/{len(self.fotos)}  {clase}  {Path(self.path).stem}  [{estado}]', (255, 255, 255)),
            (48, f'hechas {hechas}/{len(self.fotos)}   pincel {self.radio}   ENTER=lista  Z=deshacer  R=reiniciar  Q=salir',
             (170, 170, 170)),
        ):
            cv2.putText(texto, t, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
        return np.vstack([texto, vista])

    # ----- teclado: devuelve False para salir -----
    def tecla(self, t):
        c = chr(t).lower() if 0 < t < 128 else ''
        if c in ('\r', '\n', ' '):
            self.guardar()
            self.marcar_revisada()
            self.mover(1)
        elif c == 'a':
            self.mover(-1)
        elif c == 'd':
            self.mover(1)
        elif c == 'z' and self.historial:
            self.mascara = self.historial.pop()
        elif c == 'r':
            self.historial.append(self.mascara.copy())
            auto = ruta_mascara(self.path, 'auto')
            # masks/auto/ no está en git: si no se generó, vuelve a como estaba al abrir.
            self.mascara = leer_mascara(auto) if auto.exists() else self.original.copy()
        elif c == 'f':
            self.solo_foto = not self.solo_foto
        elif c in ('+', '='):
            self.cambiar_radio(1)
        elif c == '-':
            self.cambiar_radio(-1)
        elif c in ('q', '\x1b'):
            self.guardar()
            return False
        return True

    def correr(self):
        cv2.namedWindow(VENTANA, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(VENTANA, self.mouse)
        while True:
            cv2.imshow(VENTANA, self.dibujar())
            t = cv2.waitKey(15)
            if t != -1 and not self.tecla(t & 0xFF):
                break
            if cv2.getWindowProperty(VENTANA, cv2.WND_PROP_VISIBLE) < 1:   # cerró con la X
                self.guardar()
                break
        cv2.destroyAllWindows()
        hechas = sum(f in self.revisadas for f in self.fotos)
        print(f'Revisadas: {hechas} de {len(self.fotos)}')


if __name__ == '__main__':
    quien = sys.argv[1].lower() if len(sys.argv) == 2 else ''
    if quien not in ('enzo', 'tato'):
        sys.exit('Uso: python scripts/corregir_mascaras.py enzo   (o tato)')
    lista = (MANUAL / 'LISTA.txt').read_text(encoding='utf-8').split('\n')
    lista = [x.strip() for x in lista if x.strip()]
    Editor(repartir(lista, quien), quien).correr()
