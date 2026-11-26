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

### 13. MobileNetV2 + transfer learning (≈ 50 s)
> Gracias Enzo. Ahora pasamos a la segunda etapa: con la hoja ya separada del fondo, necesitamos
> una red que la mire y diga a cuál de las cuatro clases pertenece.
>
> Como tenemos unas dos mil fotos, entrenar una red convolucional desde cero haría que memorice el
> dataset. Por eso usamos **transfer learning** con **MobileNetV2**, que ya viene preentrenada con
> más de un millón de imágenes de ImageNet y tiene unos **2,26 millones de parámetros** que ya
> saben detectar bordes, texturas y colores.
>
> Encima le agregamos una capa nueva de **4 neuronas**, una por clase, con apenas **5.124
> parámetros nuevos**. La entrenamos en **dos fases**: en la **Fase 1** congelamos MobileNetV2 para
> que aprenda solo la capa nueva sin romper lo que la red ya sabía; y en la **Fase 2**
> descongelamos las **últimas 35 capas** con una tasa de aprendizaje chica, para ajustar esos
> filtros finos a las manchas de la vid.

### 14. El experimento (≈ 55 s)
> Con esa arquitectura armamos el experimento central: **la misma red, las mismas fotos, y solo
> cambia el fondo**.
>
> El **modelo A** recibe la foto original con fondo. El **modelo B** recibe la hoja segmentada por
> la UNet *(deslizar sobre la foto para mostrar el antes y después)*. Acá hicimos dos ajustes
> importantes que mejoraron mucho a B: la **recortamos al cuadro de la hoja** para que todas llenen
> la imagen por igual —así no asocia "hoja chica" con mancha bacteriana— y cambiamos el fondo negro
> por **gris neutro**, porque al entrar a MobileNetV2 el gris vale cero y no crea bordes artificiales
> tan fuertes.
>
> Para evaluar sin trampa dividimos en **5 folds por grupos**, manteniendo las fotos casi iguales
> siempre del mismo lado, y dejamos un **test fijo de 359 fotos** afuera hasta el final. En lugar
> de quedarnos con un solo modelo, promediamos los **5 folds** junto con **TTA** (cuatro
> orientaciones de la foto), y medimos con **Macro-F1**, porque como mancha bacteriana tiene muy
> pocas fotos, el accuracy solo nos engañaría.

### 15. En laboratorio: empate (≈ 50 s)
> Estos son los resultados en las **359 fotos de test**. En laboratorio, **A y B prácticamente
> empatan**: A saca **0.966** de Macro-F1 y B saca **0.974**, los dos con **11 errores** de 359
> fotos, casi un 97% de acierto.
>
> Si miramos las matrices de confusión abajo, las dos redes encuentran las **15 fotos de mancha
> bacteriana**. La pequeña diferencia es que A confunde una hoja sana con mancha bacteriana,
> mientras que B logra 15 de 15 sin falsos positivos.
>
> A primera vista parecería que todo el trabajo de la UNet no cambió casi nada. Pero en fotos de
> laboratorio, todas parecidas y con fondo liso, todavía no sabemos **en qué se está fijando cada
> red para decidir**.

### 16. ¿Dónde mira la red? (≈ 45 s)
> Para responder eso usamos **Grad-CAM**, que dibuja un mapa de calor sobre la imagen: **lo rojo
> es la zona que más pesó** en la decisión de la red y lo azul lo que ignoró.
>
> Estas cuatro fotos son casos del test donde A y B no coincidieron —en algunas acierta A y en
> otras B— y nos sirven para comparar **dónde mira** cada uno.
>
> Lo que vemos es que el **modelo A a veces derrama atención sobre el fondo claro**, afuera de la
> hoja, sobre todo cuando la hoja es chica, como en la segunda fila. El **modelo B**, en cambio,
> mira siempre **adentro de la hoja**, porque directamente no tiene fondo del que agarrarse.

### 17. La prueba de fuego: el campo (≈ 55 s)
> Entonces nos hicimos la pregunta que de verdad importa: ¿qué pasa si sacamos las dos redes del
> laboratorio? Juntamos **1.432 fotos de viñedo real** de fuentes externas —iNaturalist, GBIF y
> otros datasets— que **nunca se usaron para entrenar**, solo para medir.
>
> Los números caen fuerte en los dos, pero acá sí aparece la ventaja de quitar el fondo: **A cae a
> 0.37** de Macro-F1 y **B logra 0.47** —y sube a **0.52** cuando la UNet recorta bien una sola
> hoja.
>
> ¿Por qué se desplomó A? Porque aprendió un atajo: **"fondo verde de campo = hoja sana"**. De
> hecho, en oídio real falla casi todo y lo manda a hoja sana, como en esta foto de abajo, donde
> **A se equivoca y B acierta** al sacar el fondo y mirar las manchas.

### 18. ¿Por qué? (≈ 50 s)
> Para resumir, acá están los dos lados de la moneda.
>
> **¿Por qué anduvo tan bien en laboratorio?** Porque las fotos son muy parecidas entre sí, el
> transfer learning ya traía filtros buenos para ver formas y texturas, y quitar el fondo obligó a
> la red B a mirar la hoja.
>
> **¿Por qué anduvo mal en el campo?** Por tres motivos concretos: primero, nunca vio fotos de
> campo al entrenar; segundo, en campo muchas fotos de **mildiu** se sacan **por el envés** de la
> hoja, donde se ve un moho blanco algodonoso que la red confunde con **oídio**; y tercero, cuando
> hay **varias hojas encimadas**, la UNet —que entrenó con una hoja por foto— recorta mal.
>
> La conclusión del experimento es clara: **quitar el fondo no mejora en laboratorio, pero hace que
> la red generalice mejor afuera**.

### 19. Es una aproximación (≈ 50 s)
> Para cerrar, es fundamental dejar claras las **limitaciones** del sistema.
>
> Primero, el modelo es una **aproximación** y **siempre va a elegir una de sus cuatro clases**,
> aunque la hoja tenga otra cosa: falta de nutrientes o de agua, quemadura de sol, insectos,
> agroquímicos u otra enfermedad. Acá a la derecha se ve un ejemplo real: una hoja con **black
> rot**, una enfermedad que la red no conoce, y sin embargo responde una de las cuatro clases con
> mucha confianza.
>
> También somos transparentes en lo metodológico: fuimos eligiendo las mejoras de B mirando los
> resultados en test, así que esos números pueden estar un poco inflados.
>
> Por todo esto, el sistema **no da recomendaciones** ni tratamientos: solo informa una clase y sus
> probabilidades. **La decisión siempre corresponde al ingeniero agrónomo.** Como **trabajo
> futuro**, el paso natural es incorporar fotos reales de campo —y del envés de las hojas— al
> entrenamiento.

---

## Los dos · Demo y preguntas

### 20. Probémoslo (≈ 1 min 15 s)
> *(Hacer clic en **Abrir la aplicación** → `http://127.0.0.1:7860`, ya levantada con `python app/app.py`.)*
>
> - **Enzo (pestaña "Diagnosticar"):** subimos uno de los ejemplos (por ej. `mildiu` o `sana`) y
>   mostramos el flujo completo: la UNet recorta la hoja sobre gris, el ensamble de 5 modelos B da
>   las probabilidades de las cuatro clases y abajo aparece siempre el aviso de que decide el
>   agrónomo.
> - **Martin (pestaña "Con fondo vs sin fondo"):** elegimos el ejemplo `oidio_A_se_equivoca` para
>   ver en vivo a los dos ensambles sobre la misma foto —**A dice Sana y B acierta Oídio**— junto
>   con sus mapas **Grad-CAM** lado a lado.

### 21. ¿Preguntas?
> Gracias. ¿Tienen alguna pregunta?
