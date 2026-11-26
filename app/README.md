# Cómo abrir la app

La app corre **en la compu**, sin internet y sin placa de video. Se abre en el navegador.

## Antes de la primera vez

1. **Entorno virtual** (una sola vez). En una terminal, parados en la carpeta del proyecto:

   ```
   python -m venv .venv
   .venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. **Modelos** en la carpeta `models/` (no van a git: se pasan por Drive). Tienen que estar estos 11:

   ```
   models/unet.keras
   models/clasificador_A_fold1.keras  ...  models/clasificador_A_fold5.keras
   models/clasificador_B_fold1.keras  ...  models/clasificador_B_fold5.keras
   ```

   Si al bajarlos de Drive aparecen como `.keras.zip` o como una carpeta con `config.json`,
   `metadata.json` y `model.weights.h5` adentro: el `.keras` **es** ese zip. Hay que dejarlo como
   un solo archivo con terminación `.keras` (sacarle el `.zip`, o volver a comprimir esos 3 archivos).

## Cada vez que la quieran abrir

1. Abrir una terminal en la carpeta del proyecto (en VSCode: *Terminal → Nueva terminal*).
2. Activar el entorno virtual:

   ```
   .venv\Scripts\activate
   ```

3. Lanzar la app:

   ```
   python app/app.py
   ```

4. **Esperar** a que en la terminal aparezca esta línea (tarda 1 o 2 minutos, porque carga los 11 modelos):

   ```
   * Running on local URL:  http://127.0.0.1:7860
   ```

5. Abrir en el navegador **http://127.0.0.1:7860**. También se abre con el botón
   **"Abrir la aplicación"** de la presentación (diapositiva 20).

Para cerrarla: en la terminal, **Ctrl + C**.

## Para el oral

- **Lanzarla antes de empezar a exponer**, porque tarda en cargar. Dejarla abierta en otra pestaña.
- Probar una vez cada pestaña antes de entrar: la primera foto tarda más (unos 30 s en "Comparar");
  después, unos 2 s en "Diagnosticar" y unos 10 s en "Comparar".
- Hay **fotos de ejemplo** abajo de cada pestaña: se hace clic en una y después en el botón.
  `oidio_A_se_equivoca` es la que muestra la diferencia: el modelo A dice "Sana" con 89% y el B
  acierta "Oídio".

## Qué hace cada pestaña

- **Diagnosticar:** la UNet saca el fondo, se recorta la hoja y los 5 modelos B (sin fondo)
  dan la probabilidad de cada clase. Muestra la hoja tal como la mira la red.
- **Con fondo vs sin fondo:** la misma foto en el modelo A (con fondo) y en el B (sin fondo),
  con el mapa de calor (Grad-CAM) de dónde miró cada uno.

Siempre muestra el aviso: es una aproximación, no da recomendaciones, decide el ingeniero agrónomo.

## Si algo falla

| Problema | Qué hacer |
|---|---|
| El navegador dice que no puede conectarse | Todavía está cargando: esperar la línea `Running on…` y recargar |
| `Address already in use` / puerto 7860 ocupado | Ya hay una app abierta: usar esa, o cerrarla (Ctrl + C en su terminal) |
| `No such file` con un `.keras` | Falta un modelo en `models/` o tiene mal el nombre (ver arriba) |
| `No module named ...` | No se activó el entorno virtual (`.venv\Scripts\activate`) |
