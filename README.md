# INF8239_U03 — Ejercicio 06: Autoencoder y VAE con Fashion-MNIST

Comparación reproducible de un autoencoder (AE) y un autoencoder variacional (VAE) con 2 y 16 dimensiones latentes, frente a una línea base lineal (PCA), con métricas numéricas de reconstrucción, medidas de diversidad y de cercanía al entrenamiento, costo, pruebas, Model Card y una aplicación local.

**Pregunta:** reconstruir bien una imagen, ¿significa saber generar imágenes nuevas?

## Datos
Fashion-MNIST (Zalando Research, licencia MIT): 60,000 imágenes de entrenamiento y 10,000 de prueba, 28×28 en grises. Se descarga con Keras al entrenar y **no se guarda en el repositorio**.

## Estructura
```
scripts/   check_runtime.py · train_ae_vae.py
src/inf8239_u03_gen/   data.py · models.py · metrics.py
tests/     pruebas de datos, modelos y métricas
notebooks/ 06_entrega_generativos.ipynb  (lee reports/, no reentrena)
app/       streamlit_app.py
models/    modelos entrenados (.keras)
reports/   métricas, figuras y captura de la aplicación
MODEL_CARD.md · AI_USE_DECLARATION.md · requirements.txt
```

## Cómo reproducir
Requiere [uv](https://docs.astral.sh/uv/). Usa CPU; TensorFlow queda aislado en este proyecto.
```
uv sync --extra cpu
uv run python scripts/check_runtime.py
uv run pytest -q                              # esperado: 19 passed
uv run ruff check .
uv run python scripts/train_ae_vae.py         # corrida completa (varios minutos en CPU)
uv run streamlit run app/streamlit_app.py     # aplicación local
```
Prueba rápida del flujo: `uv run python scripts/train_ae_vae.py --quick`. Después, abrir y ejecutar `notebooks/06_entrega_generativos.ipynb`.

## Resultados principales (semilla 42, una corrida, prueba de 10,000 imágenes)
| Modelo | MSE | Clases al generar (de 10) | Diversidad / real | Muestras muy cercanas al entrenamiento | Entrenamiento |
|---|---|---|---|---|---|
| AE-2 | 0.0585 | 1 | 0.17 | 0.0 % | 31.8 s |
| VAE-2 | 0.0330 | 9 | 0.54 | 13.7 % | 133.6 s |
| PCA-2 | 0.0456 | — | — | — | 9.6 s |
| AE-16 | 0.0203 | 2 | 0.47 | 0.0 % | 44.0 s |
| VAE-16 | **0.0165** | **10** | **0.68** | 0.1 % | 122.0 s |
| PCA-16 | 0.0198 | — | — | — | 7.4 s |

**Lectura honesta**
- El autoencoder reconstruye bien pero **no sabe generar**: al decodificar ruido, casi todas las muestras caen en una sola clase ("Bag").
- El VAE genera las 10 clases (con 16 dimensiones), pero las imágenes son borrosas (confianza del clasificador 0.65 frente a 0.85 en imágenes reales).
- No hay evidencia de memorización en VAE-16. En VAE-2 la cifra de muestras "muy cercanas" se debe a imágenes oscuras y es una limitación de la métrica.
- La comparación AE–VAE **no es de igual capacidad** (el VAE tiene una capa oculta más), y el AE no terminó de converger en 6 épocas.
- PCA-16 tiene ≈20 % más error que el VAE-16, pero entrena en 7 s.

**Decisión de uso:** VAE-16 como demostración educativa del espacio latente. No usar para producir imágenes que se presenten como reales ni afirmar originalidad o privacidad.

## Limitaciones
Una sola semilla; entrenamiento corto; fidelidad baja; métrica de vecino cercano sensible al contraste. Detalle completo en `MODEL_CARD.md`.

## Aplicación
`reports/app_captura.png` muestra la aplicación con el punto latente (2.30, 2.30), que decodifica una bota. Una muestra sintética no garantiza originalidad ni privacidad.

## Uso de IA
Declaración completa, con prompts, verificaciones y correcciones, en `AI_USE_DECLARATION.md`.

## Licencias y atribución
Código de uso académico. Fashion-MNIST: Xiao, H., Rasul, K. y Vollgraf, R. (2017), *Fashion-MNIST: a Novel Image Dataset for Benchmarking Machine Learning Algorithms*, licencia MIT.
