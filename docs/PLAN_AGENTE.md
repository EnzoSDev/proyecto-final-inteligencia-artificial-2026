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

## 2.1. Una hoja por foto (decisión 2026-11-22)
- **Limpieza de la máscara (decisión 2026-11-24, Enzo):** toda máscara que predice la UNet pasa por `utils.limpiar_mascara()` (mancha más grande + rellenar agujeros), con umbral 0.5: en el notebook 02, al generar `datos/dataset_segmentado/` y en la app. Elegido en val; en test sube el IoU de 0.941 a 0.955.
- El sistema asume **una sola hoja principal por foto** (la salida es una clase por imagen). La app (Gradio) debe indicarlo al pedir la foto. No segmentar varias hojas por separado.
- `datos/masks/rechazadas.txt`: se rechaza **por la máscara, no por la foto** (p. ej. máscaras que unen varias hojas o pierden parte de la hoja). Fondo complejo con buena máscara → se conserva.
- Las rechazadas se excluyen **solo** del entrenamiento de la UNet (train/val); siguen en `splits.csv` para los clasificadores. Antes de entrenar, comprobar que todas las rutas de `rechazadas.txt` existan en `splits.csv`.

## 2.2. Casi duplicados y 5-fold por grupos (decisión 2026-11-22)
- `splits.csv` conserva 324 copias recomprimidas o con otro fondo que el MD5 no detecta; 79 de 359 fotos de test tienen copia en train. Lista: `datos/casi_duplicados.csv` (columnas `grupo,clase,path,split`). **No modificar `splits.csv`.**
- Grupo = columna `grupo`; cada foto que no figura en la lista es su propio grupo.
- **Clasificadores A y B:** test fijo y afuera. De train+val, excluir las fotos cuyo grupo tenga alguna foto en test (100 → quedan 1934). Sobre el resto, `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)` con los mismos folds para A y B; reportar promedio ± desvío. Evaluar al final en test, entrenando sin esas 100 fotos.
- Verificar con un assert que no haya grupos compartidos entre entrenamiento, validación y test.
- UNet: también excluye esas 100 fotos. Entrena con `quitar_copias_de_test(cargar_splits())` menos `cargar_rechazadas()` (ambas en `utils.py`).

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
  - Colab: `drive.mount('/content/drive')`; `ROOT` = carpeta de Drive; `sys.path.insert(0, ROOT)`; `pip install -q` solo de lo que falte (p. ej. `gradio`), sin reinstalar TensorFlow; copiar `dataset/` (y `datos/dataset_segmentado/` si existe) a `/content/` y usar esa copia como `DATA` (leer miles de archivos chicos desde Drive es muy lento).
  - Local: `ROOT = DATA = carpeta del proyecto`.
  - Imprimir: entorno, GPU detectada (`tf.config.list_physical_devices('GPU')`), versión de TF.
- Todas las rutas se construyen desde `ROOT`/`DATA` (nunca rutas absolutas de Windows ni `/content` fijos fuera de `configurar_entorno`).
- Salidas persistentes (`models/`, `figuras/`, `resultados/`, `datos/masks/`, `datos/splits.csv`) se escriben en `ROOT` (Drive). `datos/dataset_segmentado/` se genera en `DATA` y además se guarda como `ROOT/datos/dataset_segmentado.zip` para la sesión siguiente (descomprimir al inicio si existe).
- Notebooks reanudables: si un modelo ya existe en `models/`, ofrecer variable `REENTRENAR = False` para cargarlo en vez de entrenar de nuevo.
- **Explicar todo (pedido de Enzo, 2026-11-23):** Enzo y Tato tienen que poder explicar cada línea. El código nuevo va lo más simple posible, y cada cambio se explica en pasos chicos antes de seguir.
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
app/app.py                    (interfaz Gradio local; B con utils.recortar_a_hoja(..., min_area=0): sin la regla del 15%, igual que el notebook 03 — decisión de Enzo 2026-11-26)
app/ejemplos/                 4 fotos de test, 1 por clase
docs/referencias/             notebooks de referencia (unet_original.ipynb, unet_pruebas_enzo.ipynb, unet_simple_enzo.ipynb); no son parte del informe
datos/                        todo lo derivado de dataset/ (utils.DATOS; máscaras en utils.MASKS)
datos/splits.csv              path,clase,label,split   (fuente única de verdad; path relativo a ROOT)
datos/casi_duplicados.csv     grupo,clase,path,split
datos/masks/auto/<clase>/<stem>.png       0/255, 256×256
datos/masks/manual/<clase>/<stem>.png     corregidas a mano (test UNet)
datos/masks/manual_train/<clase>/<stem>.png  39 fotos difíciles de train corregidas a mano (Tato); se usan con peso ×3
datos/masks/rechazadas.txt                una ruta relativa por línea
datos/dataset_segmentado/<clase>/<stem>.png   256×256, fondo negro; generado por scripts/armar_dataset_segmentado.py (UNet v6 + limpieza, utils.segmentar_hoja/quitar_fondo); también datos/dataset_segmentado.zip
datos/dataset_unet/{train,val,test}/{images,masks}/<stem>.png   160×160, generado por scripts/armar_dataset_unet.py (no va a git); también test/masks_auto/ y datos/dataset_unet.zip (para Colab: configurar_entorno(['datos/dataset_unet']))
.venv/                        entorno virtual local (no va a git)
models/  resultados/  figuras/
```
`.gitignore` agregar: `models/ datos/masks/auto/ datos/dataset_segmentado/ datos/dataset_unet/ figuras/ resultados/ .venv/ .ipynb_checkpoints/ __pycache__/` (NO ignorar `datos/masks/manual/`, `datos/masks/manual_train/` ni `datos/masks/rechazadas.txt`, son trabajo humano).
