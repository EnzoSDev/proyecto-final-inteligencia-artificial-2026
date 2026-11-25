# Registro de avances

Acá anotamos **todo lo que hacemos y cambiamos** durante el proyecto: qué hicimos,
qué decidimos y por qué, qué problemas aparecieron y qué falta.

Sirve para no perder el hilo entre sesiones y para poder explicar cada decisión en la exposición.

## Cómo se usa
- **Un archivo por día de trabajo:** `AAAA-MM-DD.md` (ej. `2026-11-20.md`).
- Si ese día ya tiene archivo, se **agrega** abajo; no se borra lo anterior.
- Se anota **siempre**, aunque el cambio sea chico o el resultado sea malo.

## Plantilla
    # AAAA-MM-DD

    **Quién:** Enzo / Tato / Claude

    ## Qué hicimos
    -

    ## Decisiones (y por qué)
    -

    ## Archivos creados o modificados
    -

    ## Resultados / números
    -

    ## Problemas
    -

    ## Pendiente
    -

## Índice
| Fecha | Resumen |
|-------|---------|
| [2026-11-20](2026-11-20.md) | Definición del proyecto, plan, README y flujo con Colab |
| [2026-11-21](2026-11-21.md) | Entorno virtual, `requirements.txt`, UNet base, fase 1 y `splits.csv` (con duplicados detectados) |
| [2026-11-22](2026-11-22.md) | Revisión de máscaras, decisión "una hoja por foto", casi duplicados y corrección manual de 25/50 máscaras (Tato) |
| [2026-11-23](2026-11-23.md) | Paso 4 (50 máscaras test); paso 5: `dataset_unet/`, GPU WSL2, barrido UNet v1–v5 (0.931) y preparación `masks/manual_train/` (40 fotos train) · Enzo: orden en `datos/`, `.venv`, notebook 02 simple y entrenamiento en Colab (0.923) |
| [2026-11-24](2026-11-24.md) | UNet v6 (`0.955`) + `dataset_segmentado` + test externo 1.432 fotos + barrido `v0–v5` + **NB03 oficial `v4d`** (test B `96.1%`, con TTA 4× **`97.8%`**; campo externo B `0.431–0.499`, `0.539` con UNet ok) |
