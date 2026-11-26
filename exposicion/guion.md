# Guion del oral

Lo que dice cada uno en cada diapositiva. No es para leer: es para practicar y no olvidarse de nada.
Duración total: unos **15 minutos**.

| Parte | Diapositivas | Quién | Tiempo aprox. |
|---|---|---|---|
| Introducción y UNet | 1 a 12 | Enzo | 7 min 30 s |
| Clasificador y conclusiones | 13 a 19 | Martin | 6 min |
| Demo y preguntas | 20 y 21 | Los dos | 1 min 30 s |

---

## Enzo · Introducción y UNet

### 1. Portada (≈ 20 s)
> Buenas. Somos Enzo Sorrenti y Martin Bellone. Nuestro proyecto es la segmentación y detección
> de enfermedades en hojas de vid, usando dos redes: una UNet y una red convolucional.

### 2. El problema (≈ 40 s)
> ¿Cómo surgió esta idea? Yo tengo un hermano mellizo, Matías, que estudia Ingeniería Agronómica
> en la FCA. Desde que entramos a nuestras carreras soñamos con hacer algún proyecto juntos.
> Tuvimos algunas ideas, pero esta materia logró unir las dos carreras.
>
> En una de nuestras charlas, él me comentó que en Mar del Plata y en la zona de Balcarce se están
> plantando muchos viñedos, y que el objetivo es que la zona sea una de las más importantes del
> país. Por eso buscamos ideas que ayudaran a los productores, y de ahí surgió la detección de
> enfermedades en las hojas.
>
> La idea es simple: a partir de una foto de una hoja de vid, saber si está sana o si tiene alguna
> de estas tres enfermedades: mildiu, oídio o mancha bacteriana.
>
> Algo importante: el sistema **no da recomendaciones**. Solo dice qué clase cree que es y con qué
> probabilidad. La decisión la toma el ingeniero agrónomo: el sistema es una ayuda que después
> el experto interpreta.

### 3. Los datos (≈ 50 s)
> Usamos un dataset público de una universidad de la India, con 2726 fotos. Lo elegimos porque
> tiene dos de las enfermedades de la hoja más importantes en Argentina, el mildiu y el oídio.
> La mancha bacteriana, en cambio, no está presente en el país, pero la dejamos a propósito: la
> idea es que, si más adelante probamos con fotos de la zona y el modelo da mancha bacteriana con
> un porcentaje muy alto, eso nos avise de que algo está fallando.
>
> Lo primero que hicimos fue analizar el dataset, y encontramos que **333 fotos eran copias
> exactas** de otras: archivos idénticos, que detectamos con una "huella" del archivo llamada MD5.
> Las sacamos y nos quedamos con **2393 fotos distintas**.
>
> Las clases están muy desbalanceadas: hay casi mil de mildiu y de hojas sanas, pero solo **100 de
> mancha bacteriana**.
>
> Además encontramos otras **324 fotos casi iguales**: la misma foto vuelta a comprimir o con otro
> color de fondo. Como los archivos no son idénticos, el MD5 no las detecta. Para encontrarlas,
> achicamos cada foto y dejamos solo la textura fina, las nervaduras y las manchas, así el color del
> fondo no importa. Después comparamos todas contra todas, y las que se parecían mucho las
> revisamos a ojo. Si una copia queda en entrenamiento y la otra en prueba, la
> red "ya la vio" y los resultados salen mejores de lo real. Por eso las mantenemos siempre del
> mismo lado al dividir los datos.

### 4. La idea: 2 redes (≈ 40 s)
> Entonces, ¿cuál es la idea? Si usábamos una sola red para clasificar, podía aprender del fondo
> en vez de la hoja. Además, no sabíamos cómo iban a ser las fotos en la app: si la persona iba a
> sacar la foto sobre un fondo liso o directamente en la planta. Al final decidimos que sea con fondo lizo, igual esto no quitaba tanto el problema solo lo simplificab
>
> Por eso armamos **dos versiones del dataset: una con fondo y otra sin fondo**.
>
> Nuestro sistema tiene dos pasos. Primero, una **UNet** encuentra la hoja en la foto y le saca el
> fondo. Después, una **red convolucional**, MobileNetV2, mira solo la hoja y dice qué enfermedad tiene.
>
> Así podemos responder una pregunta: **¿la red mira la hoja, o hace trampa mirando el fondo?**
> Eso es lo que va a contar Martin en la segunda parte.

### 5. No había máscaras: las hicimos (≈ 50 s)
> Para entrenar una UNet hacen falta máscaras: imágenes en blanco y negro que dicen qué píxel es
> hoja y cuál es fondo. El dataset no las traía.
>
> Así que las hicimos en cuatro pasos:
> - Primero, un **método automático** que separa la hoja usando la saturación del color, porque la
>   hoja es mucho más colorida que el fondo. Así sacamos las 2393 máscaras.
> - Después **revisamos a ojo las 2393**, en grillas. Las que tenían la máscara mal (agarraba
>   varias hojas, el tallo o una mano, o perdía parte de la hoja) las **descartamos** del
>   entrenamiento de la UNet: fueron 69.
> - **Corregimos 50 a mano**, pintando sobre la foto. Esas son el test: la UNet nunca las ve al
>   entrenar.
> - Y al final corregimos a mano **39 fotos difíciles** del entrenamiento, donde el método
>   automático fallaba. Varias eran de las descartadas, así que las recuperamos.

### 6. La UNet original (≈ 40 s)
> Esta es la **UNet original**, del paper de Ronneberger y otros, de 2015, que se creó para
> segmentar imágenes médicas. Tiene forma de U. Del lado izquierdo, el **encoder**, la imagen
> **se achica** y la red entiende qué hay en la foto. Del lado derecho, el **decoder**, la imagen
> **se vuelve a agrandar** hasta el tamaño original, y en cada píxel la red dice qué probabilidad
> hay de que sea hoja.
>
> Las flechas grises son las **conexiones de salto**: le pasan a la subida los detalles finos de
> la bajada, y gracias a eso los bordes de la hoja salen prolijos.

### 7. Primer resultado (≈ 45 s)
> Para medir qué tan bien anda usamos el **IoU**: la parte donde la máscara de la red coincide con
> la correcta, dividida por todo lo que marcó alguna de las dos. Uno es perfecto.
>
> Abajo está cómo dividimos las fotos: con **1542 entrena**, con **332 de validación** decide
> cuándo dejar de entrenar, y el **test son las 50 corregidas a mano**, que nunca ve al entrenar.
> Ninguna foto de test del clasificador se usa para entrenar la UNet.
>
> Con la UNet original sacamos **0.919** en test. Todavía no le ganaba al método automático, que
> da 0.934. Solo le ganaba en oídio, y donde peor andaba era en mancha bacteriana.

### 8. ¿Qué pasó al entrenar? (≈ 40 s)
> Este es el gráfico de nuestro primer entrenamiento, en Colab. A la izquierda está la pérdida y a
> la derecha el IoU, en train y en validación. Hasta la época 12 todo va bien, pero en la 13 el
> entrenamiento **se rompe**: la pérdida se dispara y el IoU se cae.
>
> La causa: la red era muy grande, **31 millones de parámetros** para unas 1500 fotos, y no tenía
> BatchNorm.

### 9. La solución: BatchNorm (≈ 45 s)
> **BatchNorm** es una capa que ordena los números que pasan de una capa a otra, para que no se
> desordenen. Con eso el entrenamiento queda estable.
>
> Además achicamos la red cuatro veces, de 31 a 7,8 millones de parámetros; usamos una pérdida que
> combina **BCE con Dice**, que premia que las máscaras se superpongan; agregamos más aumentación,
> y repetimos cuatro veces las fotos de mancha bacteriana, que eran pocas.
>
> Con todo eso subió de **0.919 a 0.931**, y ya **sin picos** en el entrenamiento.

### 10. Cambiamos el encoder (≈ 45 s)
> Después **cambiamos el encoder por MobileNetV2**, una red ya entrenada con más de un millón de
> fotos. Así no arrancamos de cero, y eso nos sirve porque tenemos pocas fotos. El decoder sigue
> siendo el de la UNet.
>
> Pero el resultado fue el mismo: **0.931**. Probamos varias redes distintas y todas chocaban con
> el mismo techo. Ahí nos dimos cuenta de que **el problema no era la red, eran los datos**.

### 11. Resumen: qué mejoró la UNet (≈ 45 s)
> Acá está todo junto. Lo que realmente la mejoró fue **corregir a mano 39 fotos difíciles** del
> entrenamiento: subió a 0.941. Y después, como hay una sola hoja por foto, **limpiamos la
> máscara**: nos quedamos con la mancha más grande y rellenamos los agujeros. Eso la llevó a
> **0.955**, por encima del método automático.
>
> La conclusión es que **mejorar los datos sirvió más que cambiar la red**.

### 12. 2393 hojas sin fondo (≈ 40 s)
> Con la UNet final recortamos las 2393 hojas. Arriba están las fotos originales y abajo, cómo
> quedan sin fondo.
>
> Y abajo, una foto de un viñedo real, que la red nunca vio: aunque el fondo son otras hojas
> verdes, separa bastante bien la hoja principal.
>
> Con estas hojas sin fondo entrenamos el clasificador. Le paso la palabra a Martin.

---

## Martin · Clasificador y conclusiones

### 13. MobileNetV2 + transfer learning
> *(a completar por Martin)*

### 14. El experimento
> *(a completar por Martin)*

### 15. En laboratorio: empate
> *(a completar por Martin)*

### 16. ¿Dónde mira la red?
> *(a completar por Martin)*

### 17. La prueba de fuego: el campo
> *(a completar por Martin)*

### 18. ¿Por qué?
> *(a completar por Martin)*

### 19. Es una aproximación
> *(a completar por Martin)*

---

## Los dos · Demo y preguntas

### 20. Probémoslo
> *(a definir cuando esté la app)*

### 21. ¿Preguntas?
> Gracias. ¿Tienen alguna pregunta?
