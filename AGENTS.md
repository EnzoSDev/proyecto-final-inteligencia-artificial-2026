# AGENTS.md: instrucciones para agentes de IA

Leé este archivo completo antes de hacer cualquier cosa en el proyecto.

## Qué es el proyecto
Proyecto final de Inteligencia Artificial de dos estudiantes, **Enzo** y **Tato**. El sistema clasifica enfermedades en hojas de vid con un pipeline de 2 etapas:

`foto → UNet (segmenta la hoja) → hoja recortada sin fondo → CNN preentrenada MobileNetV2 → clase + probabilidades`

Hay 4 clases: `hojas_sanas`, `mancha_bacteriana`, `mildiu` y `oidio`. Se entrega en notebooks de Jupyter y se expone oralmente: los estudiantes **tienen que poder explicar todo** lo que el agente produzca.

## 1. Qué leer, en este orden
1. **`avances/README.md`** y **el archivo más reciente de `avances/`** (`AAAA-MM-DD.md`). Ahí está el **estado actual**: qué se hizo, qué se decidió y qué quedó **pendiente**.
2. **`docs/PLAN_AGENTE.md`**: las reglas técnicas (rutas, tamaños, splits, Colab, estructura de carpetas). **No tiene lista de tareas.**
3. **`README.md`**: la explicación para humanos. Leelo para entender el porqué de las decisiones.
4. **`docs/Consigna.txt`**: la consigna de la materia y los criterios de evaluación.
5. El paper del dataset (`docs/*.pdf`) solo si hace falta.

No empieces a escribir código sin haber leído 1 y 2.

## 2. Flujo de trabajo en cada sesión
1. Hacé `git pull`, porque el otro integrante pudo haber subido cambios.
2. Leé lo indicado en la sección 1.
3. **El humano dice qué se hace en cada sesión.** No elijas ni empieces tareas por tu cuenta; si no está claro, preguntá.
4. Hacé lo pedido respetando las reglas técnicas de `docs/PLAN_AGENTE.md`.
5. Verificá que funciona. En local, los notebooks se prueban con `SMOKE=1`.
6. **Registrá en `avances/AAAA-MM-DD.md`** lo que hiciste (ver sección 4).
7. Informá al humano: qué hiciste, resultados, problemas y lo próximo. **Preguntá antes de hacer commit o push.**

## 3. Reglas que no se pueden romper
- **NO modificar ni borrar `dataset/`.** Las imágenes originales son sagradas.
- **NO modificar `docs/Consigna.txt` ni el PDF.**
- **El sistema NO da recomendaciones** ("llevar al laboratorio", "aplicar tratamiento", umbrales de decisión, etc.). Solo informa la clase y las probabilidades. **Las decisiones corresponden al ingeniero agrónomo.**
- **Siempre aclarar que el modelo es una aproximación:** las manchas pueden tener otras causas (nutrientes, falta de agua, sol, insectos, químicos, otras enfermedades) y la red siempre elige una de sus 4 clases.
- **NO inventar números ni conclusiones.** Las métricas y el análisis salen solo de resultados realmente obtenidos.
- **NO hacer commit ni push sin que el humano lo pida.**
- **Registrar todo en `avances/`**, incluidos los errores y los resultados malos.
- Todo texto para humanos (markdown de notebooks, README, avances, commits) va **en español**, simple y explicable en una exposición.

## 4. Registro de avances (obligatorio)
- Archivo del día: `avances/AAAA-MM-DD.md`. Si ya existe, **agregá al final** y no borres nada.
- Usá la plantilla de `avances/README.md` y actualizá su tabla **Índice**.
- En **Quién**, anotá qué humano estaba trabajando (Enzo o Tato) y el agente. Si no sabés quién es, preguntá.

## 5. Decisiones ya tomadas (no re-discutir)
| Decisión | Motivo |
|----------|--------|
| Clasificar la hoja entera, sin cajas (YOLO) | El dataset no tiene cajas marcadas |
| UNet + CNN preentrenada (MobileNetV2) | Sugerido por el profesor; cubre 2 temas de la consigna |
| Comparar la CNN con fondo (A) y sin fondo (B) | Experimento central: ver si la red se guía por el fondo |
| Máscaras semiautomáticas por saturación + ~50 corregidas a mano | No existen máscaras y dibujarlas todas es inviable |
| Notebooks de Jupyter | Sirven como informe |
| Entrenar en Google Colab (GPU) y escribir código en local | No hay GPU local |
| Sin recomendaciones en la salida | Decide el ingeniero agrónomo |
| App en Gradio, solo local, con 2 pestañas (Diagnosticar; Con fondo vs sin fondo) | La consigna pide una aplicación; la demo se hace desde la compu |

Si el humano pide cambiar una decisión, actualizá **los tres lugares**: `docs/PLAN_AGENTE.md` (técnico), `README.md` (humano) y `avances/` (registro, con el porqué).

## 6. Errores comunes a evitar
- **Archivos que no son imágenes:** puede aparecer `desktop.ini` en `dataset/`. Filtrá siempre por extensión (`.jpg`, `.jpeg`, `.png`) y **en minúsculas**, porque hay archivos `.JPG`.
- **PNG con transparencia:** convertí siempre a RGB de 3 canales al cargar.
- **No regenerar `datos/splits.csv`** si ya existe. Es la división única de los datos y tiene que ser **idéntica** para A, B y la UNet. Regenerarlo invalida las comparaciones.
- **Fuga de datos:** la UNet no se entrena con imágenes del split *test* del clasificador, y las métricas se reportan solo sobre *test*.
- **Rutas:** nada de rutas absolutas de Windows ni `/content/...` fijas. Todo sale de `utils.configurar_entorno()` (`ROOT`/`DATA`), para que funcione igual en local y en Colab.
- **Colab:** no reinstalar TensorFlow. Copiar el dataset a `/content` para leerlo rápido y guardar los resultados en Drive.
- **Local:** no hay GPU, así que no lances entrenamientos completos. Usá `SMOKE=1` para verificar y dejá el entrenamiento real para Colab.
- **Notebooks:** generalos con `nbformat` desde un script; no escribas el JSON a mano. El código reutilizable va en `utils.py`, no duplicado entre notebooks.
- **Grad-CAM:** construí el clasificador sin anidar el backbone (con `base.input`/`base.output`).
- **Figuras:** letra grande (≥ 16), porque van a la presentación. Guardalas con `utils.guardar_figura()` en `figuras/`.
- **Archivos generados** (`models/`, `figuras/`, `resultados/`, `datos/masks/auto/`, `datos/dataset_unet/`, `datos/dataset_segmentado/`) están en `.gitignore`: no los fuerces al repo.

## 7. Entorno
- **Local:** Windows 11, Python 3.11, TensorFlow 2.21, sin GPU. OpenCV y Gradio pueden no estar instalados: se instalan con `pip install -r requirements.txt` (si el archivo ya existe).
- **Colab:** carpeta del proyecto en `Mi unidad/Proyecto_Final_IA/`, con GPU T4.
- **Repo:** GitHub, rama `main`.
