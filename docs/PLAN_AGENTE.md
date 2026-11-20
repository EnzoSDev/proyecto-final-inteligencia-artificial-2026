# PLAN DE EJECUCIÓN PARA AGENTE — Segmentación (UNet) + Clasificación (CNN) de enfermedades de vid

> Documento para un agente de IA. La versión para humanos está en `README.md` (ya creado; solo completar la sección Resultados en T5). Ejecutar las tareas T1→T6 en orden.

## 0. Contexto y objetivo
Proyecto final de IA (equipo: Enzo y Tato). Pipeline de 2 etapas sugerido por el profesor:
`imagen → UNet (máscara de hoja) → recorte + fondo neutro → CNN preentrenada (MobileNetV2) → clase`.
Experimento central: comparar clasificador entrenado sobre **A = imágenes originales** vs **B = imágenes segmentadas**, para medir si el fondo sesga la clasificación. Entregable: notebooks Jupyter ejecutados + figuras grandes para la presentación. Ver `docs/Consigna.txt` (exige análisis de resultados, matriz de confusión, métricas).

## 1. Hechos verificados del entorno
- Raíz: `c:\Users\enzos\OneDrive\Desktop\Archivos\Facultad\Inteligencia Artificial\Proyecto Final`
- Python 3.11.9, TensorFlow 2.21, scikit-learn, matplotlib, jupyter instalados. **Sin GPU** (CPU). torch NO instalado (no usar). OpenCV/gradio: verificar e instalar vía `requirements.txt`.
- `dataset/<clase>/` con 4 clases, imágenes 256×256: `hojas_sanas` 1254, `mancha_bacteriana` 100, `mildiu` 966, `oidio` 406 → total **2726**. Extensiones mixtas `.jpg/.JPG/.png` (algunas PNG pueden tener alfa). Existe `dataset/mildiu/desktop.ini` → ignorar.
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
- **PUNTOS DE PARADA** marcados con ⛔: detenerse y pedir acción humana; no continuar hasta confirmación.

## 2a. Registro de avances (obligatorio, todo el proyecto)
- Al terminar cada tarea (T1…T6), cada cambio de plan, cada decisión y cada parada ⛔, registrar en `avances/AAAA-MM-DD.md` (fecha del día). Si existe, **agregar** al final; nunca borrar entradas previas.
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

⛔ **PARADA 2 (después de T4 escrito y verificado con SMOKE=1):** pedir a los humanos que suban el proyecto a Drive, ejecuten 01→02→03 en Colab con GPU (respetando la PARADA 1) y bajen `resultados/` y `figuras/` a la carpeta local. Recién entonces hacer T5 con las métricas reales.

## 3. Estructura objetivo
```
README.md                     (humanos, ya existe)
requirements.txt              tensorflow, scikit-learn, opencv-python, matplotlib, pandas, pillow, jupyter, nbformat, gradio
utils.py
01_mascaras.ipynb
02_unet_segmentacion.ipynb
03_clasificacion.ipynb
app.py                        (opcional, T6)
splits.csv                    path,clase,label,split   (fuente única de verdad)
masks/auto/<clase>/<stem>.png       0/255, 256×256
masks/manual/<clase>/<stem>.png     corregidas a mano (test UNet)
masks/rechazadas.txt                una ruta relativa por línea
dataset_segmentado/<clase>/<stem>.png
models/  resultados/  figuras/
```
`.gitignore` agregar: `models/ masks/auto/ dataset_segmentado/ figuras/ resultados/ .ipynb_checkpoints/ __pycache__/` (NO ignorar `masks/manual/` ni `masks/rechazadas.txt`, son trabajo humano).

## 4. Tareas

### T1 — Base (`requirements.txt`, `.gitignore`, `utils.py`, `splits.csv`)
`utils.py` debe exponer:
- `listar_imagenes(root='dataset') -> pd.DataFrame[path, clase, label]` (filtra por `EXT`, ordenado).
- `crear_splits(df) -> df con columna split` 70/15/15 estratificado (`train_test_split` dos veces, `SEED`); además chequear duplicados exactos por MD5 y, si los hay, asignar todo el grupo al mismo split; guardar `splits.csv`. Si `splits.csv` existe, `cargar_splits()` lo lee (no regenerar).
- `cargar_imagen(path, size) -> np.uint8 RGB` (maneja alfa y escala de grises).
- `set_estilo()`, `guardar_figura(fig, nombre)`.
- `dataset_tf(df, root_col, size, augment, preprocess) -> tf.data.Dataset` para clasificación.
**Aceptación:** `splits.csv` tiene 2726 filas; proporciones por clase ≈70/15/15; ninguna ruta `.ini`.

### T2 — `01_mascaras.ipynb` (máscaras semiautomáticas)
1. Exploración: gráfico de barras por clase (`figuras/distribucion_clases`), grilla de ejemplos (`figuras/ejemplos_clases`).
2. `mascara_auto(img)` (poner en `utils.py`): HSV → canal S → umbral Otsu → apertura+cierre (kernel elíptico 5–7) → componente conexa mayor → relleno de huecos (floodfill) para incluir lesiones marrones/amarillas. Si el área < 2% o > 95% de la imagen, marcar como sospechosa.
3. Generar `masks/auto/` para las 2726 imágenes. Guardar grillas de revisión (imagen + contorno de máscara) en `figuras/revision_mascaras/<clase>_<n>.png`, 24 por grilla, con índice visible; listar sospechosas automáticamente en `masks/rechazadas.txt` (prellenado).
4. Elegir 50 imágenes del split **test** (estratificadas; incluir todas las sospechosas del test hasta completar) y copiar su máscara auto a `masks/manual/` como punto de partida; escribir la lista en `masks/manual/LISTA.txt`.

⛔ **PARADA 1:** pedir a los humanos: (a) revisar grillas y agregar rutas malas a `masks/rechazadas.txt`; (b) corregir a mano las 50 máscaras de `masks/manual/` (blanco=hoja, negro=fondo; herramientas sugeridas: CVAT, Label Studio, GIMP). Si deciden saltear (b), continuar marcando la evaluación de la UNet como **provisional** (contra máscaras auto) en el notebook y el README.

### T3 — `02_unet_segmentacion.ipynb`
- Datos: imágenes con split train/val **no** rechazadas → pares (imagen, máscara auto) a `IMG_SEG`. Test = las 50 de `masks/manual/`. (La UNet nunca ve imágenes del split test del clasificador.)
- Aumento sincronizado imagen/máscara: flips, rotación 90°, zoom leve; brillo/contraste solo a la imagen.
- Modelo: encoder `MobileNetV2(input_shape=(160,160,3), include_top=False, weights='imagenet')` congelado, skips en `block_1_expand_relu`, `block_3_expand_relu`, `block_6_expand_relu`, `block_13_expand_relu`, bottleneck `block_16_project`; decoder con `Conv2DTranspose` + concat + Conv-BN-ReLU; salida `Conv2DTranspose(1, sigmoid)` a 160×160.
- Pérdida BCE + Dice; Adam 1e-3; EarlyStopping(val_loss, patience 5, restore_best); máx 30 épocas. Opcional fase 2: descongelar encoder con lr 1e-5.
- Métricas en test: IoU y Dice (global y por clase), UNet vs `mascara_auto` → tabla `resultados/segmentacion.csv`.
- Figuras: `curvas_unet`, `unet_ejemplos` (original | real | predicha), `unet_peores` (5 peores IoU), `unet_vs_umbral` (casos donde difieren más).
- `segmentar_y_recortar(img, unet)` (en `utils.py`): predecir a 160, reescalar máscara a 256, umbral 0.5, bbox con margen 10%, fondo fuera de máscara = gris (128,128,128), recorte cuadrado (pad), resize 256. Generar `dataset_segmentado/` para las 2726.
- Guardar `models/unet.keras`.
**Aceptación:** IoU test UNet ≥ 0.90 (si no, reportarlo y analizar, no forzar); `dataset_segmentado/` con 2726 archivos.

### T4 — `03_clasificacion.ipynb`
- Usa `splits.csv` idéntico para A (`dataset/`) y B (`dataset_segmentado/`, mismo stem `.png`).
- Modelo **sin anidar** el backbone (construir con `base.input`/`base.output`) para facilitar Grad-CAM: `MobileNetV2(224, imagenet, include_top=False)` → GAP → Dropout 0.3 → Dense(4, softmax). Preproceso `mobilenet_v2.preprocess_input`.
- Aumento (solo train): flip H/V, rotación 0.15, zoom 0.15, contraste 0.2.
- `class_weight` = `compute_class_weight('balanced')` sobre train.
- Fase 1: backbone congelado, Adam 1e-3, ≤15 épocas. Fase 2: descongelar desde la capa `block_13_expand` en adelante (dejar BatchNorm congeladas), Adam 1e-5, ≤15 épocas. EarlyStopping patience 4 + ModelCheckpoint en `val_loss`.
- Función `entrenar_y_evaluar(tag in {'A','B'})` → `models/clf_<tag>.keras`, `resultados/historial_<tag>.json`, `resultados/metricas_<tag>.json` (accuracy, macro-F1, `classification_report` dict, matriz de confusión).
- Figuras por tag: `curvas_<tag>` (línea vertical en inicio fase 2), `confusion_<tag>` (normalizada, anotaciones grandes), `errores_<tag>` (hasta 12 mal clasificadas con prob.). Comparativas: `comparacion_AB` (barras F1 por clase A vs B), `resultados/comparacion.csv`.
- Grad-CAM (capa `out_relu`): mismas 8 imágenes (2 por clase, del test) para A y B → `gradcam_AB`.
- `diagnosticar(ruta) -> dict(clase, probs, mascara, gradcam)` (en `utils.py`) usando UNet + clf_B; demo en 4 imágenes (1 por clase).
  - **NO generar recomendaciones ni acciones a seguir** (ni umbrales de decisión). El sistema solo informa la clase más probable y las probabilidades de las 4 clases; la decisión corresponde al ingeniero agrónomo.
- Análisis de confianza (solo informativo): distribución de probabilidades máximas en test (aciertos vs errores) y diagrama de confiabilidad (calibración), para mostrar qué tan confiables son las probabilidades que reporta la red.
- **Limitación obligatoria a documentar** (celda markdown y análisis): el modelo es una aproximación. Las manchas pueden tener otras causas (nutrición, falta de agua, sol, insectos, químicos, enfermedades no incluidas) y la red siempre elige una de las 4 clases (conjunto cerrado). El resultado es un insumo para el ingeniero agrónomo, no un diagnóstico.
- Celda markdown final "Análisis" con placeholders basados en los números reales (no inventar conclusiones: redactar a partir de las métricas obtenidas).
**Aceptación:** ambos modelos entrenados y evaluados sobre el mismo test; todas las figuras existen.

### T5 — Cierre
- Completar sección "Resultados" del `README.md` con las métricas reales (tabla A vs B, IoU UNet) y links a figuras.
- Reportar a los humanos: métricas clave, tiempos de entrenamiento, problemas encontrados.

### T6 — Opcional (solo si los humanos lo piden)
- `app.py` Gradio: subir imagen → mostrar máscara, recorte, clase+probabilidades y Grad-CAM, con un aviso fijo: "Resultado aproximado. La interpretación y las acciones a seguir corresponden al ingeniero agrónomo." Sin recomendaciones. Ejecutar con `python app.py`.
- CNN propia desde cero (comparación TL vs ad-hoc, tema de nota 9) sobre B.

## 5. Verificación final
- Agente (local): los 3 notebooks se ejecutan de punta a punta con nbconvert y `SMOKE=1` sin errores.
- Humanos (Colab): ejecución completa con GPU sin errores; la celda inicial detecta GPU.
- `splits.csv`: 2726 filas; splits idénticos usados en A y B.
- `figuras/` contiene todas las figuras listadas; texto legible.
- `diagnosticar()` funciona con una imagen por clase y con una imagen fuera del dataset.
