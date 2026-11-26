# Explicación de la clasificación (notebook 03)

Guía para entender y explicar en la exposición cómo clasificamos las enfermedades.
Todos los números salen de resultados reales (ver `avances/2026-11-24.md` y `avances/2026-11-25.md`).

---

## 1. Qué hace el clasificador

Recibe la foto de una hoja y devuelve **una de 4 clases con su probabilidad**:
`hojas_sanas`, `mancha_bacteriana`, `mildiu` u `oidio`.

Entrenamos **dos versiones** con la misma red y las mismas fotos:

- **A (con fondo):** la foto original.
- **B (sin fondo):** la hoja recortada por la UNet, sin el fondo.

**Para qué:** ver si la red se guía por la hoja o hace "trampa" mirando el fondo.
Es el **experimento central** del proyecto.

---

## 2. MobileNetV2

- Es una **red convolucional (CNN)** que hizo Google en 2018.
- **Ya viene entrenada** con ImageNet (más de un millón de fotos de 1000 cosas distintas).
  Ya sabe reconocer bordes, colores, texturas y formas.
- Es **liviana** ("Mobile": pensada para celulares). Entrena rápido y la app corre sin placa de video.

**Transfer learning:** reusamos lo que MobileNetV2 ya aprendió y solo le enseñamos nuestras 4 clases.
Lo necesitamos porque tenemos pocas fotos (2393, y solo 100 de mancha bacteriana).

---

## 3. Cómo está armada la red

```
foto 224×224  →  MobileNetV2 (sin su última capa)  →  promedio  →  Dropout 30%  →  capa de 4 salidas
```

| Parte | Qué hace | Parámetros |
|---|---|---|
| MobileNetV2 | "Mira" la foto y saca sus características | 2.257.984 |
| Promedio (*GlobalAveragePooling*) | Resume todo en una lista de 1280 números | 0 |
| Dropout 30% | Apaga neuronas al azar al entrenar para que no memorice | 0 |
| Capa final (4 salidas, *softmax*) | Da la probabilidad de cada clase (suman 100%) | 5.124 |
| **Total** | | **2.263.108** |

Casi todo el "saber ver" viene de Google; nosotros agregamos una capa chiquita.

### Global Average Pooling (GAP) en vez de Flatten

Con una foto de 224×224, MobileNetV2 entrega un bloque de **7 × 7 × 1280**: una grilla de 7×7
zonas de la foto y, en cada zona, 1280 números que dicen qué detectó (texturas, manchas, bordes).
La capa final necesita una **lista** de números. Hay dos formas de armarla:

| | Flatten | Global Average Pooling |
|---|---|---|
| Qué hace | Pone todos los números en fila | Promedia las 7×7 zonas de cada uno de los 1280 canales |
| Tamaño de la lista | 62.720 números | **1280 números** |
| Parámetros de la capa de 4 clases | 250.884 | **5.124** |
| ¿Importa *dónde* está la mancha? | Sí | No, solo *si* está |

**Por qué GAP:** la capa final tiene unas 50 veces menos parámetros, así que con pocas fotos
memoriza menos. Además no depende de la posición: la enfermedad es la misma esté arriba o abajo.

**Esa lista de 1280 números es un *embedding*:** un resumen numérico de la foto. Dos hojas
parecidas dan listas parecidas. Tato lo usó así para limpiar las fotos de iNaturalist: comparando
embeddings descartó las que no eran hojas (racimos, tallos, microscopio).

---

## 4. Qué le pasamos a cada modelo

- **A:** la foto original achicada a 224×224.
- **B:** la hoja sin fondo, con dos ajustes (función `utils.recortar_a_hoja`):
  1. **Recorte a la hoja:** se hace "zoom" a la hoja (su cuadro + 5% de margen, sin deformarla).
     Así todas las hojas llenan el cuadro por igual.
  2. **Fondo gris** en vez de negro. Para la red, el negro es una señal muy fuerte
     (entra como -1); el gris es neutro (entra como 0).

**Por qué esos ajustes:** sin ellos, B aprendió un atajo: "hoja chica con mucho negro alrededor
= mancha bacteriana". Con el recorte y el fondo gris ese atajo casi desapareció.

### Por qué achicamos las fotos de 256 a 224

- **MobileNetV2 se preentrenó con fotos de 224×224.** Los pesos de Keras vienen para pocos
  tamaños fijos (96, 128, 160, 192 y 224), y 224 es el de ImageNet. Con ese tamaño, las manchas y
  texturas le llegan a la escala que ya sabe reconocer.
- **Se pierde poco:** es achicar un 12% por lado.
- **A y B usan el mismo tamaño**, así lo único que cambia entre los dos es el fondo.

(La UNet trabaja en 160×160: es otra red y se eligió ese tamaño al empezar el proyecto, para que
entrene más rápido.)

---

## 5. Entrenamiento en 2 fases

La capa nueva arranca sin saber nada. Si entrenáramos todo junto desde el principio, sus errores
grandes "romperían" lo que MobileNetV2 ya sabía.

| Fase | Qué se entrena | Learning rate | Máx. épocas |
|---|---|---|---|
| **1. Congelada** | Solo la capa nueva; MobileNetV2 queda fija | 0.001 | 10 |
| **2. Ajuste fino** | Además, las **últimas 35 capas** de MobileNetV2 | 0.0001 (10 veces más chico) | 20 |

**¿Cómo se congela una capa?** En Keras, cada capa tiene una propiedad `trainable`. Si la ponemos
en `False`, sus pesos no se tocan durante el entrenamiento: la imagen pasa por ellos (siguen
calculando), pero el optimizador no los modifica. En la Fase 1 hacemos `trainable = False` en
todas las capas de MobileNetV2 y solo queda entrenable la capa nueva de 4 salidas.

En la fase 2 se ajusta MobileNetV2 a las hojas de vid de a poco, sin olvidar lo que sabía.

**¿Por qué las últimas 35 capas?** MobileNetV2 tiene 155 capas en 17 bloques: los primeros
detectan bordes y colores (sirven para cualquier foto) y los últimos, patrones complejos. Las
últimas 35 son los **bloques 13 a 16** más la convolución final (tienen el 71% de los parámetros).
Se adaptan esas y las primeras quedan fijas. El número 35 es un punto de partida razonable, no
sale de un cálculo; Tato probó con 15 y 20 y no mejoró. Las capas **BatchNorm** quedan congeladas
porque guardan promedios de ImageNet que se desordenarían con nuestros grupos chicos de fotos.
En las dos fases, **EarlyStopping** corta solo cuando deja de mejorar en validación.

---

## 6. Ayudas para que aprenda mejor

- **Aumentación (solo al entrenar, igual para A y B):** espejar, rotar 90°, un zoom suave
  (80–100%) y cambios suaves de brillo y contraste. **No** se toca el color, para no borrar el
  amarillo del mildiu ni el blanco del oídio.
- **Pesos de clase:** mancha bacteriana tiene muy pocas fotos, así que sus errores "cuentan más".
  Se usó una versión suave (raíz cuadrada): pesa unas 2,4 veces más, no 5,7, para que la red
  no la elija de más.
- **Label smoothing (0.05):** en vez de enseñarle "100% seguro", le enseñamos "95% seguro".
  Así no se vuelve sobreconfiada.

---

## 7. Validación cruzada con 5 folds por grupos

- Las **359 fotos de test** se separan al principio y no se tocan hasta el final.
- Se sacan 100 fotos que son casi copias de fotos de test. Quedan **1934** para entrenar.
- Esas 1934 se dividen en **5 partes (folds)**. Se entrena 5 veces: cada vez una parte es
  validación y las otras 4 son entrenamiento.
- **"Por grupos":** las fotos que son casi copias entre sí (otro recorte, otra compresión) quedan
  siempre del mismo lado. Si una copia quedara en entrenamiento y la otra en validación,
  la red "ya la habría visto" y el resultado saldría mejor de lo real.
- A y B usan **exactamente los mismos 5 folds**, así la comparación es justa.

---

## 8. Ensamble y TTA

- **Ensamble:** en vez de quedarnos con un solo modelo, usamos **los 5** (uno por fold) y
  **promediamos** sus probabilidades. Así no dependemos de la suerte de un solo entrenamiento.
- **TTA 4×:** cada foto se predice 4 veces (original, espejada horizontal, espejada vertical y
  rotada 90°) y se promedia. En el clasificador sí ayudó (en la UNet no había servido).

Cada foto de test se predice **20 veces** (5 modelos × 4 posiciones) y se promedia.

---

## 9. Cómo medimos

- **Accuracy:** porcentaje de fotos bien clasificadas.
- **F1 de una clase:** combina "de las que dije X, cuántas eran X" (precisión) y "de las X,
  cuántas encontré" (recall).
- **Macro-F1:** el promedio del F1 de las 4 clases, **todas con el mismo peso**.

**Por qué Macro-F1:** con clases desbalanceadas, el accuracy engaña. Hay solo 15 fotos de mancha
bacteriana en test: una red que nunca la elige igual tendría accuracy alto. El Macro-F1 la castiga.

---

## 10. Resultados

**Validación cruzada (5 folds, promedio ± desvío):**

| Modelo | Macro-F1 | Accuracy |
|---|---|---|
| A (con fondo) | 0.973 ± 0.011 | 0.982 ± 0.005 |
| B (sin fondo) | 0.962 ± 0.016 | 0.975 ± 0.009 |

**Test (359 fotos, ensamble 5 folds + TTA):**

| Modelo | Accuracy | Errores | Macro-F1 | Sana | Mancha bact. | Mildiu | Oídio |
|---|---|---|---|---|---|---|---|
| A (con fondo) | 0.969 | 11 | 0.966 | 0.971 | 0.968 | 0.976 | 0.950 |
| B (sin fondo) | 0.969 | 11 | **0.974** | 0.975 | **1.000** | 0.969 | 0.950 |

**Cómo leerlo:** con fotos de laboratorio, A y B andan **prácticamente igual** (los dos cerca de 97%).
B acierta las 15 de mancha bacteriana.

---

## 11. Grad-CAM: dónde mira la red

Grad-CAM es un **mapa de calor** sobre la foto: lo rojo es lo que más pesó para decidir.

- En algunas fotos, **A mira el fondo blanco** además de la hoja.
- **B mira siempre adentro de la hoja**, porque no tiene fondo para mirar.

Figura: `figuras/clf_gradcam_A_vs_B.png`. Muestra fotos donde A y B no coinciden, así que en
algunas se equivoca A y en otras B: sirve para ver **dónde mira** cada uno, no para decir que
uno siempre acierta.

---

## 12. Prueba con fotos de campo real (lo más interesante)

Tato juntó **1432 fotos de viñedo real** de otras fuentes (iNaturalist, GBIF y otros datasets),
que la red nunca vio. Solo se usaron para medir, nunca para entrenar.

| | A (con fondo) | B (sin fondo) |
|---|---|---|
| Macro-F1 en campo real (1432 fotos) | 0.370 | **0.472** |
| Macro-F1 cuando la UNet encuentra bien la hoja (1084 fotos) | 0.402 | **0.516** |

- **A tenía un atajo:** en campo dijo "sana" en el 57% de las fotos (versión inicial). Aprendió
  "fondo verde de campo = hoja sana" y por eso falla casi todo el oídio real.
- **B gana en campo**, porque mira la hoja y no el fondo.
- **Pero los dos andan mal en campo** (alrededor de 0.47). La red solo vio fotos de laboratorio.
  Además, en campo el mildiu suele fotografiarse **por abajo** de la hoja (blanco algodonoso),
  y la red lo confunde con oídio.

**Conclusión del experimento:** en laboratorio, A y B empatan; en campo real, **B generaliza mejor**.
Sacar el fondo obliga a la red a mirar la hoja.

---

## 13. Limitaciones (decirlas en la exposición)

- **El modelo es una aproximación.** Siempre elige una de sus 4 clases, aunque la hoja tenga otra
  cosa: falta de nutrientes o de agua, sol, insectos, químicos u otras enfermedades. Con 3
  enfermedades que no conoce (black rot, black measles, leaf blight), respondió con 81–89% de
  confianza igual.
- **No da recomendaciones.** Solo informa la clase y las probabilidades; **decide el ingeniero agrónomo**.
- **En campo real anda mucho peor** que en laboratorio (0.47 contra 0.97). Para mejorarlo harían
  falta fotos de campo en el entrenamiento (trabajo futuro).
- **Las mejoras de B se eligieron mirando el test** (el interno y el de campo). Lo ideal es elegir
  con validación y usar el test una sola vez al final, así que los números pueden estar un poco
  inflados.
- Fotos con **varias hojas** (rompen la regla de "una hoja por foto") pueden recortarse mal.

---

## 14. Preguntas que nos pueden hacer

**¿Por qué MobileNetV2 y no una red propia?**
Porque tenemos pocas fotos. Una red desde cero memorizaría; MobileNetV2 ya sabe ver y solo la ajustamos.

**¿Por qué Global Average Pooling y no Flatten?**
Porque resume la foto en un embedding de 1280 números: la capa final queda con 5.124 parámetros
en vez de 250.884 (memoriza menos) y no depende de dónde está la mancha.

**¿Por qué las fotos van en 224×224 si son de 256×256?**
Porque MobileNetV2 se preentrenó con 224×224. Se pierde poco (12% por lado) y A y B usan el
mismo tamaño, así solo cambia el fondo.

**¿Por qué 2 fases?**
Para no romper lo que MobileNetV2 ya sabía mientras la capa nueva todavía no sabe nada.

**¿Por qué 5 folds y no un solo split?**
Para que el resultado no dependa de qué fotos cayeron en validación por suerte. Reportamos promedio ± desvío.

**¿Por qué "por grupos"?**
Porque hay fotos casi repetidas. Si una copia queda en entrenamiento y otra en validación, la red hace trampa.

**¿Por qué Macro-F1 y no accuracy?**
Porque hay muy pocas fotos de mancha bacteriana; el accuracy no se da cuenta si la red la ignora.

**¿Entonces conviene sacar el fondo?**
En laboratorio da casi lo mismo. En campo real, B anda mejor (0.47 contra 0.37), y Grad-CAM muestra que B mira la hoja.

**¿Se puede usar en el campo?**
Como apoyo, no como diagnóstico: en campo acierta mucho menos y siempre elige una de 4 clases.
La decisión es del ingeniero agrónomo.
