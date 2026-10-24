# Declaración de uso de inteligencia artificial

## Herramientas utilizadas
Claude (Anthropic), usado como asistente paso a paso durante todo el ejercicio.

## Propósito de cada uso
- Revisar el proyecto base contra la rúbrica e identificar lo que le faltaba.
- Escribir o ampliar el código: `scripts/train_ae_vae.py`, `src/inf8239_u03_gen/models.py` y `metrics.py`, y las pruebas `tests/test_metrics.py` y `tests/test_models.py`.
- Redactar `notebooks/06_entrega_generativos.ipynb`, `MODEL_CARD.md`, este documento, `README.md` y el PDF de entrega, usando los resultados reales de mi corrida.
- Explicarme cada paso (instalación, ejecución, Git) y ayudarme a interpretar los resultados.

## Prompts o decisiones relevantes
- Pegué el mandato y la rúbrica del Ejercicio 06 y el manual del LAB10, y pedí avanzar paso a paso, mostrando un checklist, como si yo no supiera nada.
- Pedí revisar el proyecto base contra la rúbrica antes de modificarlo. La revisión detectó: sin métrica numérica de reconstrucción; sin evaluación de fidelidad, diversidad ni memoria; panel con las primeras 12 imágenes en vez de 16 elegidas con semilla; sin curvas por componente del VAE; AE de 16 dimensiones y VAE de 2 no comparables sin línea base; y un `.gitignore` que excluía modelos, imágenes y JSON.
- Pegué la salida real de cada comando que ejecuté y envié mis figuras para interpretarlas.
- Decisión de diseño: comparar AE y VAE con 2 y 16 dimensiones y añadir PCA como línea base.

## Elementos verificados por el equipo
Marcar solo lo que se hizo de verdad:
- [x] Ejecuté yo el entrenamiento completo en mi computadora y los números de este repositorio salen de esa corrida.
- [x] Ejecuté `uv run pytest -q` (19 pruebas aprobadas) y `uv run ruff check .`.
- [x] Abrí la aplicación local y tomé la captura `reports/app_captura.png`.
- [x] Revisé las figuras de `reports/` (reconstrucción, muestras, interpolación, vecinos y espacio latente).
- [x] Ejecuté el notebook y las salidas quedaron guardadas.
- [ ] Puedo explicar la diferencia entre reconstruir y generar, y qué hace el término KL.
- [ ] Puedo explicar por qué la comparación AE–VAE no es de igual capacidad.

Verificaciones hechas por la IA: ejecutó el código y las pruebas en su propio entorno con Fashion-MNIST real; mi corrida reprodujo sus cifras con diferencias menores al milésimo, normales entre computadoras.

## Cambios realizados sobre resultados generados
- Se corrigió un error de la leyenda del gráfico del espacio latente durante las pruebas.
- Se retiraron comentarios de lint innecesarios.
- Se corrigieron dos frases del notebook que daban razones de costo imprecisas (cuántas veces más tiempo y más parámetros usa el VAE).
- Se declararon explícitamente las limitaciones que la IA detectó al interpretar los resultados: capacidad desigual entre AE y VAE, AE sin converger, una sola semilla y la sensibilidad de la métrica de vecino a imágenes oscuras.
- Hipótesis no verificada: se atribuye el mal resultado de AE-2 frente a PCA-2 a la activación ReLU y a las pocas épocas; no se probó.

## Responsabilidad asumida
Soy responsable del código, los resultados, las referencias y las conclusiones entregadas, y debo poder explicarlos. No se inventaron resultados ni bibliografía. Ejecuté el entrenamiento, las pruebas, el notebook y la aplicación en mi computadora, revisé las figuras y comprobé que las cifras de los documentos coinciden con reports/. El código fue redactado con asistencia de IA. Quedan pendientes de verificación independiente: la implementación interna del término KL y la reparametrización del VAE, y dos hipótesis de la IA que no probé (que el AE-2 salió peor que PCA-2 por la activación ReLU y las pocas épocas, y que las muestras "muy cercanas" de VAE-2 se deben a imágenes oscuras). Los resultados son de una sola semilla.