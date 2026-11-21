# PLAN DE EJECUCIÓN PARA AGENTE — Segmentación (UNet) + Clasificación (CNN) de enfermedades de vid

> Documento para un agente de IA. La versión para humanos está en `README.md`. Este archivo reúne las **reglas técnicas** del proyecto; **no hay lista de tareas**: los humanos (Enzo y Tato) dicen qué se hace en cada sesión.

## 0. Contexto y objetivo
Proyecto final de IA (equipo: Enzo y Tato). Pipeline de 2 etapas sugerido por el profesor:
`imagen → UNet (máscara de hoja) → recorte + fondo neutro → CNN preentrenada (MobileNetV2) → clase`.
Experimento central: comparar clasificador entrenado sobre **A = imágenes originales** vs **B = imágenes segmentadas**, para medir si el fondo sesga la clasificación. Entregable: notebooks Jupyter ejecutados + figuras grandes para la presentación. Ver `docs/Consigna.txt` (exige análisis de resultados, matriz de confusión, métricas).

## 1. Hechos verificados del entorno
- Raíz: `c:\Users\enzos\OneDrive\Desktop\Archivos\Facultad\Inteligencia Artificial\Proyecto Final`
- Python 3.11.9, TensorFlow 2.21, scikit-learn, matplotlib, jupyter instalados. **Sin GPU** (CPU). torch NO instalado (no usar). OpenCV/gradio: verificar e instalar vía `requirements.txt`.
- `dataset/<clase>/` con 4 clases, imágenes 256×256: `hojas_sanas` 1254, `mancha_bacteriana` 100, `mildiu` 966, `oidio` 406 → total **2726**. **Preprocesado:** hay 333 copias exactas (MD5), casi todas de `hojas_sanas`; `utils.quitar_duplicados()` deja una por grupo → **2393** fotos (sanas 927, mancha 100, mildiu 966, oidio 400). `splits.csv` tiene esas 2393; las copias siguen en `dataset/` pero no se usan. Extensiones mixtas `.jpg/.JPG/.png` (algunas PNG pueden tener alfa). Existe `dataset/mildiu/desktop.ini` → ignorar.
- Fondos: mayoría blanco/gris liso con sombras; algunas (sobre todo mancha_bacteriana) con ramas, fondo gris oscuro u hoja pequeña en el encuadre. No hay máscaras.
- git: rama `main`, 1 commit. `.gitignore` vacío.

## 2. Reglas para el agente
- NUNCA modificar ni borrar nada en `dataset/` ni `docs/`.
- No hacer commits salvo pedido explícito.
- Crear notebooks generándolos con un script Python usando `nbformat` (no escribir JSON a mano). Cada notebook: celdas markdown explicativas en español (cortas, aptas para defender oralmente) + código.
- Lógica reutilizable va en `utils.py`; los notebooks la importan. No duplicar código entre notebooks.
- Constantes globales (en `utils.py`): `SEED=42`, `CLASES=['hojas_sanas','mancha_bacteriana','mildiu','oidio']`, `EXT={'.jpg','.jpeg','.png'}` (comparar en minúsculas), `IMG_CLF=224`, `IMG_SEG=160`, `BATCH=32`.
- Fijar semillas (`random`, `numpy`, `tf.keras.utils.set_random_seed`).
- Figuras: `utils.set_estilo()` (font.size ≥ 16, títulos ≥ 20) y `utils.guardar_figura(fig, nombre)` → `figuras/<nombre>.png` dpi 150.
- Ejecutar notebooks con `jupyter nbconvert --to notebook --execute --inplace <nb> --ExecutePreprocessor.timeout=-1`. Tareas largas: ejecutar en background.

## 2a. Registro de avances (obligatorio, todo el proyecto)
- Al terminar cada tarea, cada cambio de plan y cada decisión, registrar en `avances/AAAA-MM-DD.md` (fecha del día). Si existe, **agregar** al final; nunca borrar entradas previas.
- Usar la plantilla de `avances/README.md` (qué se hizo, decisiones y por qué, archivos, resultados con números reales, problemas, pendiente) y actualizar su tabla **Índice**.
- Registrar también los fallos y resultados malos. No inventar números: solo métricas realmente obtenidas.
- Cuando los humanos traigan resultados de Colab, registrarlos en el día correspondiente.

## 2b. Flujo de trabajo con Google Colab (obligatorio)
El agente escribe y verifica el código **en local**; los humanos ejecutan el **entrenamiento completo en Colab (GPU T4)**. El agente no puede ejecutar Colab.
- Carpeta en Drive: `/content/drive/MyDrive/Proyecto_Final_IA/` (copia del proyecto completo).
- `utils.configurar_entorno() -> dict(ROOT, DATA, EN_COLAB)`; **primera celda de cada notebook**:
  - `EN_COLAB = 'google.colab' in sys.modules`.
  - Colab: `drive.mount('/content/drive')`; `ROOT` = carpeta de Drive; `sys.path.insert(0, ROOT)`; `pip install -q` solo de lo que falte (p. ej. `gradio`), sin reinstalar TensorFlow; copiar `dataset/` (y `dataset_segmentado/` si existe) a `/content/` y usar esa copia como `DATA` (leer miles de archivos chicos desde Drive es muy lento).
  - Local: `ROOT = DATA = carpeta del proyecto`.
  - Imprimir: entorno, GPU detectada (`tf.config.list_physical_devices('GPU')`), versión de TF.
- Todas las rutas se construyen desde `ROOT`/`DATA` (nunca rutas absolutas de Windows ni `/content` fijos fuera de `configurar_entorno`).
- Salidas persistentes (`models/`, `figuras/`, `resultados/`, `masks/`, `splits.csv`) se escriben en `ROOT` (Drive). `dataset_segmentado/` se genera en `DATA` y además se guarda como `ROOT/dataset_segmentado.zip` para la sesión siguiente (descomprimir al inicio si existe).
- Notebooks reanudables: si un modelo ya existe en `models/`, ofrecer variable `REENTRENAR = False` para cargarlo en vez de entrenar de nuevo.
- Modo de prueba local: si la variable de entorno `SMOKE=1`, usar ~40 imágenes por clase y 1 época por fase. Sirve para que el agente verifique que los notebooks corren en CPU sin esperar el entrenamiento real.
- `requirements.txt`: sin versiones fijas de TensorFlow (Colab trae la suya). Modelos en formato `.keras`.


## 3. Estructura objetivo
```
README.md                     (humanos, ya existe)
requirements.txt              tensorflow, scikit-learn, opencv-python, matplotlib, pandas, pillow, jupyter, nbformat, gradio
utils.py
01_mascaras.ipynb
02_unet_segmentacion.ipynb
03_clasificacion.ipynb
app/app.py                    (interfaz Gradio local)
app/ejemplos/                 4 fotos de test, 1 por clase
splits.csv                    path,clase,label,split   (fuente única de verdad)
masks/auto/<clase>/<stem>.png       0/255, 256×256
masks/manual/<clase>/<stem>.png     corregidas a mano (test UNet)
masks/rechazadas.txt                una ruta relativa por línea
dataset_segmentado/<clase>/<stem>.png
models/  resultados/  figuras/
```
`.gitignore` agregar: `models/ masks/auto/ dataset_segmentado/ figuras/ resultados/ .ipynb_checkpoints/ __pycache__/` (NO ignorar `masks/manual/` ni `masks/rechazadas.txt`, son trabajo humano).
