# Model Card · Autoencoder y VAE (Fashion-MNIST)

## Modelos y versiones
Proyecto `inf8239-u03-generative` v0.1.0. Se comparan cuatro modelos neuronales y una línea base lineal, todos entrenados con la semilla 42:

| Modelo | Descripción | Archivo guardado |
|---|---|---|
| AE-2, AE-16 | Autoencoder denso con 2 y 16 dimensiones latentes | `models/autoencoder.keras` (AE-16) |
| VAE-2, VAE-16 | Autoencoder variacional con 2 y 16 dimensiones latentes | `models/encoder.keras`, `models/decoder.keras` (VAE-2); `models/decoder_vae16.keras` |
| PCA-2, PCA-16 | Línea base lineal (no genera muestras) | no se guarda |

Entorno: Python 3.12.14, TensorFlow 2.21.0 (CPU), Keras 3.15.1, Windows 11.

## Uso previsto y usuarios
Demostración académica para estudiantes y docentes: explorar cómo se comprime una imagen, cómo se organiza un espacio latente y qué diferencia hay entre reconstruir y generar. La aplicación local (`app/streamlit_app.py`) decodifica un punto de 2 dimensiones con el decodificador del VAE-2.

## Usos fuera de alcance
- Producir imágenes que se presenten como reales o como fotografías de productos.
- Afirmar que una muestra sintética es original, privada o libre de sesgo.
- Aplicar los modelos a otros dominios o a imágenes de mayor resolución sin una nueva evaluación.
- Tomar decisiones sobre personas.

## Dataset, licencia y particiones
Fashion-MNIST (Xiao, Rasul y Vollgraf, 2017; Zalando Research), licencia MIT: 60,000 imágenes de entrenamiento y 10,000 de prueba, de 28×28 píxeles en escala de grises, normalizadas a [0, 1]. Se descarga con Keras y **no se guarda en el repositorio**.
- Todas las métricas se calculan sobre la **prueba** (10,000 imágenes), que no se usa para entrenar.
- El autoencoder reserva un 10 % del entrenamiento para validación; el VAE entrena con las 60,000 imágenes y no usa validación.
- Los paneles usan 16 imágenes de prueba elegidas con la semilla 42 (índices en `reports/lab10_metrics.json`), no seleccionadas a mano.

## Arquitectura y espacio latente
- **AE:** `Flatten → Dense(z, ReLU) → Dense(784, sigmoide) → Reshape`. Pérdida: entropía cruzada binaria. Hasta 6 épocas con parada anticipada (paciencia 2).
- **VAE:** `Flatten → Dense(128, ReLU) → (z_mean, z_log_var) → reparametrización → Dense(128, ReLU) → Dense(784, sigmoide)`. Pérdida: reconstrucción (entropía cruzada binaria por imagen) + KL con peso 1. 8 épocas.
- Optimizador Adam, lote de 256, operaciones deterministas.
- **Comparabilidad:** el VAE tiene una capa oculta de 128 unidades y el AE no, por lo que el VAE tiene mucha más capacidad (203 mil frente a 4 mil parámetros con 2 dimensiones). Las diferencias AE–VAE no se atribuyen solo al carácter variacional.
- El espacio latente del AE no está regularizado (activación ReLU, valores ≥ 0, escalas de hasta 60); el del VAE se agrupa cerca del origen (aprox. entre −3 y 4 con 2 dimensiones).

## Métricas de reconstrucción y generación
**Reconstrucción (prueba, 10,000 imágenes, una semilla)**

| Modelo | MSE | PSNR (dB) |
|---|---|---|
| AE-2 | 0.0585 | 12.33 |
| VAE-2 | 0.0330 | 14.82 |
| PCA-2 | 0.0456 | 13.41 |
| AE-16 | 0.0203 | 16.93 |
| VAE-16 | 0.0165 | 17.82 |
| PCA-16 | 0.0198 | 17.03 |

**Generación desde z ~ N(0, I), 1,000 muestras por modelo**

| Modelo | Clases presentes | Diversidad / real | Entropía de clases (máx. 2.30) | Confianza del clasificador |
|---|---|---|---|---|
| Imágenes reales (referencia) | 10 | 1.00 | 2.28 | 0.85 |
| AE-2 | 1 (todas "Bag") | 0.17 | 0.00 | 1.00 |
| AE-16 | 2 (999 "Bag") | 0.47 | 0.01 | 0.98 |
| VAE-2 | 9 (falta "Coat") | 0.54 | 2.11 | 0.53 |
| VAE-16 | 10 | 0.68 | 2.23 | 0.65 |

El clasificador (exactitud 0.854 en prueba) es solo una regla de medir. Su confianza **no** es una medida de calidad: da casi 1.0 a las muestras colapsadas del AE.

## Diversidad y vecinos cercanos
Para cada muestra se busca la imagen de entrenamiento más cercana (distancia euclídea). Como referencia, el 5 % de las imágenes reales nuevas más cercanas a su vecino está por debajo de 1.94.

| Modelo | Distancia mediana al vecino (real: 3.40) | Muestras por debajo de 1.94 |
|---|---|---|
| AE-2 | 5.68 | 0.0 % |
| AE-16 | 6.32 | 0.0 % |
| VAE-2 | 2.61 | 13.7 % |
| VAE-16 | 3.77 | 0.1 % |

Lectura: no hay evidencia de copia en VAE-16. La cifra de VAE-2 se debe a muestras oscuras y de bajo contraste, que están cerca de casi cualquier imagen oscura; es un límite de la métrica y no prueba memorización. Revisión visual en `reports/nearest_neighbors.png`.

## Hardware, tiempo y tamaño
CPU Intel64 Family 6 Model 122, sin GPU.

| Modelo | Entrenamiento (s) | Parámetros | Memoria de parámetros (KB) | Decodificar (ms/imagen) |
|---|---|---|---|---|
| AE-2 | 31.8 | 3,922 | 15 | 0.20 |
| VAE-2 | 133.6 | 202,516 | 791 | 0.36 |
| PCA-2 | 9.6 | 2,352 | 9 | — |
| AE-16 | 44.0 | 25,888 | 101 | 0.26 |
| VAE-16 | 122.0 | 207,920 | 812 | 0.23 |
| PCA-16 | 7.4 | 13,328 | 52 | — |

## Limitaciones, riesgos y supervisión
- Una sola semilla: no se estima variabilidad entre corridas.
- El AE seguía mejorando al terminar las 6 épocas (no convergió); su comparación con PCA y VAE es provisoria.
- Capacidad desigual entre AE y VAE.
- Muestras borrosas y de baja fidelidad (confianza 0.53–0.65 frente a 0.85 en imágenes reales).
- La métrica de vecino cercano es sensible a imágenes oscuras y no demuestra privacidad ni originalidad.
- Fashion-MNIST es un conjunto pequeño, balanceado y de 10 clases de prendas; no representa imágenes de producto reales.
- **Supervisión:** cualquier uso fuera de la demostración académica exige repetir la evaluación (varias semillas, más épocas, modelos convolucionales) y una revisión humana de las muestras.
