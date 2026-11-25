# Vision-IA

Detector propio VisionLocaliza-D3 de una clase («residuo»), CNN y Transformer propios para seis materiales, OpenCV y Streamlit. Entrenamiento desde cero, sin pesos preentrenados ni servicios externos de detección. D2 y los experimentos A/B/C1b/C3 se conservan.

## Iniciar en Windows

Desde la raíz del proyecto:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app/main.py --server.address 127.0.0.1
```

Abre http://127.0.0.1:8501. Reinicia la terminal si había una versión anterior abierta. Dependencias: `.\.venv\Scripts\python.exe -m pip install -r requirements-app.txt`.

## Probar un residuo

1. Abre **Cámara en vivo** y pulsa **Activar cámara**.
2. Muestra el residuo completo. Verás la imagen, el material propuesto y dónde depositarlo.
3. Pulsa **Detener** para liberar la cámara. No es necesario pausar para ver la imagen ni calibrar un fondo.

La aplicación reconoce **un residuo a la vez**, tanto en video como en fotografías. En video mantiene el objeto anterior si sigue detectado y su puntuación está a menos de 0,10 de la mejor. El material aparece cuando coinciden tres de las últimas cuatro lecturas. Una vez confirmado, otro material necesita cuatro de las últimas cinco para reemplazarlo.

La cámara mantiene un recuadro punteado como guía durante la búsqueda. Si faltan detecciones claras, permite propuestas de menor confianza dentro de la guía y revisa una vista ampliada del centro. Una propuesta débil debe coincidir durante tres lecturas para iniciar el seguimiento. La clasificación usa el recorte del fotograma original.

También puedes **Tomar foto** o **Subir imagen**. En fotografías, **Ajustar el recorte si no lo reconoce** conserva el respaldo manual. La pantalla no muestra configuración de modelos, métricas, diagnóstico ni registro de fotografías. Sus herramientas y resultados anteriores se conservan fuera del flujo normal. El video usa la cámara del navegador: autoriza su acceso y elige la webcam externa o la cámara del móvil en el selector.

El componente `app/camera_component/` envía JPEG de hasta 640 px a Streamlit, con una solicitud pendiente y hasta cuatro envíos por segundo. Entre respuestas, el navegador sigue la posición y el tamaño del residuo en cada fotograma nuevo. La caja se mantiene durante las omisiones breves del detector mientras haya evidencia visual. Si se pierde esa referencia o vence la última detección, vuelve la guía.

## Desplegar en Streamlit Cloud

Selecciona la rama que contenga la versión preparada y `app/main.py` como archivo de entrada. `requirements.txt` instala el runtime con OpenCV headless y PyTorch CPU. La revisión debe incluir los pesos de `models/runtime/` que indica `config/app.json` y los archivos de `app/camera_component/`. La cámara usa el HTTPS de Community Cloud o localhost durante el desarrollo.

## Detector propio D2

**VisionLocaliza-D2**, 630.370 parámetros, RGB 512 × 512. Diseñado en `training/detector_own.py` únicamente con convoluciones, BatchNorm, ReLU y pooling de PyTorch. Cinco etapas de canales 16/32/64/96/128 producen mapas a pasos 8/16/32. Un resumen global de 16 canales aporta contexto a tres cabezas independientes: cada una predice presencia de centro, dos desplazamientos, anchura/altura y calidad de localización. No hay backbone, neck ni cabeza importada de una arquitectura oficial. Es un diseño propio basado en conceptos generales de detección; no se reclama novedad científica.

La supervisión usa centros y cajas humanas, pérdida focal de presencia, GIoU, Smooth L1 y calidad IoU. Las cajas finales usan NMS. B conserva la clasificación del material y sus pesos no se usan para entrenar D2. Configuración: `config/detector_D2.json`. Entrenamiento: `scripts/train_detector_own.py`. Una iteración real en CPU y T4 comprobó gradientes, actualización de pesos y recarga exacta. Una comprobación de 100 pasos localizó los tres objetos de dos imágenes train a IoU ≥ 0,5; esto verifica capacidad de aprendizaje, no generalización.

Desde la época 76 se continúa el mismo entrenamiento con vistas cercanas de objetos anotados y recortes de contexto que no intersectan cajas humanas. Todas las cajas visibles se transforman con el recorte; no se etiqueta automáticamente la fotografía entera como un objeto. El checkpoint, configuración y código anteriores se conservan en `before_closeups/`, `source/` y `source_closeups/` dentro de la ejecución. No se reutilizan pesos de D1 ni de los clasificadores.

**D1 era la arquitectura oficial YOLOv8 nano** de Ultralytics y no cumple la condición de arquitectura propia. Se detuvo tras 45 épocas. Se eliminaron del proyecto local su entrenador, configuración, notebook y dependencia Ultralytics. La aplicación carga las arquitecturas propias registradas. Las anotaciones, manifiestos y métricas se conservan. Los dos checkpoints de `runs/detector_D1/archived_epoch45/` siguen pendientes de eliminación porque la revisión automática bloqueó su borrado. No borrar la carpeta remota completa: contiene el archivo TACO compartido que utiliza el notebook propio.

**Resultado original D2 (checkpoint anterior):** terminó en la época 185 y se seleccionó la 145, aprendida con TACO (1.106 train / 197 val / 197 test). Con confianza 0,25 y NMS 0,45 obtuvo en la única prueba final: Precision **42,82 %**, Recall **41,23 %**, mAP50 **36,31 %** y mAP50:95 **17,90 %**. Son cifras históricas de ese checkpoint, no del refinamiento actual. Métricas: `runs/detector_D2/D2_20261123_222215/test_final/metrics.json`.

**Refinamiento de los pesos D2:** misma arquitectura y origen aleatorio. Desde nuestra selección conservadora anterior se continuó en T4 entrenando únicamente cajas y calidad, con extractores, presencia y estadísticas BatchNorm fijos. Se penalizan propuestas parciales según su IoU con la caja humana completa, y algunos aumentos suavizan estampados interiores sin cambiar los bordes. Se seleccionó la época 2. Configuración de entrenamiento en `config/detector_D2_box_only.json` y `scripts/train_detector_refine.py`. La política de inferencia está en `config/app.json`.

Selección por validación: TACO conserva 197 imágenes y MJU 36 imágenes / 42 objetos, con polígonos humanos contrastados con máscaras y auditoría de similitud. No se abre test en este refinamiento. Con confianza 0,25 / NMS 0,45, TACO obtiene Precision **53,89 %**, Recall **29,13 %**, mAP50 **31,27 %**, mAP50:95 **16,15 %**; MJU obtiene **20,97 %**, **30,95 %**, **19,21 %**, **10,17 %**, respectivamente. AP coincide con pycocotools; mediana CPU **38,5 ms**, 2 hilos. Entre las 75 botellas transparentes de val, las propuestas asociadas que cubren ≥85 % de la caja humana y tienen IoU ≥0,5 pasan de **17 a 21**; en 56 empaques impresos, de **16 a 18**. Son diagnósticos de cobertura, no aciertos de material ni garantía de resolver cada etiqueta. Procedencia: `models/runtime/manifest.json`; resultados: `runs/detector_D2_shape/`.

La referencia histórica de material combinaba la CNN refinada en recortes TACO con B cuando la refinada quedaba por debajo de 0,40 de confianza. Ambas CNN tenían 89.686 parámetros. En las 884 imágenes de validación original, la política histórica pasó de 74,55 % / 76,10 % a **77,38 % / 78,58 %** de accuracy / Macro F1. En 63 recortes de botellas transparentes pasó de **7 a 10** aciertos y, entre los objetos asociados por IoU ≥0,5 con material conocido, de **15,0 % a 22,86 %**. El entrenador de la CNN refinada está en `scripts/train_material_candidate.py`.

La aplicación fusiona dos vistas de VisionLocaliza-D3 a 512 px. La CNN refinada y la residual aportan el 55 % y el 45 % de la predicción inicial de material. **VisionMaterial-Transformer** refina esa predicción con cuatro bloques de atención, cuatro cabezas y dos tokens de lectura sobre recortes de 160 px. La fusión conserva las predicciones de vidrio de las CNN y las de papel o cartón cuando ambas CNN coinciden. En la misma validación de 409 objetos, el recall conjunto pasó de **23,72 % a 28,36 %**. El entrenamiento y el ajuste de fusión usan datos de entrenamiento, incluidos 847 recortes de D3 emparejados con anotaciones humanas de TACO train. La procedencia se guarda en `models/runtime/manifest.json`.

Persisten omisiones, cajas parciales y falsos positivos. En las cuatro fotos propias detecta los cuatro objetos y conserva los tres materiales correctos de B; la botella transparente grande de una foto sigue confundiéndose con cartón. La mejora es moderada y no resuelve todas las etiquetas ni los reflejos. El video físico se verificó previamente en local; la precisión con la cámara del usuario requiere comprobarse en sus escenas. La candidata, los clasificadores originales, los checkpoints descartados y los resultados se conservan. Entrenador: `scripts/train_material_candidate.py`; procedencia y política: `models/runtime/manifest.json`. La sincronización actual usa Drive G: y Colab cuenta 0.

Fuente: [TACO oficial](https://github.com/pedropro/TACO), Proença y Simões (2020), 1.500 imágenes y 4.784 anotaciones humanas COCO. No se convierten fotos de TrashNet/RealWaste en cajas. Se verifican dimensiones/orientación EXIF y se conservan coordenadas, URL y licencia originales en el manifiesto. Anotaciones CC BY 4.0; imágenes con licencias específicas registradas, incluidas entradas `CC` sin variante y ODbL de OpenLitterMap. El [mantenedor de ExtendedTACO](https://huggingface.co/datasets/roboticslaburjc/ExtendedTACO-reviewed) reproduce los términos: si falta la licencia, el valor por defecto es CC BY 4.0. La licencia MIT del repositorio corresponde al código. No publicar las imágenes como si todas tuvieran la misma licencia.

Desde la época 158 se añaden 600 RGB de **train oficial de [MJU-Waste](https://github.com/realwecan/mju-waste)** (Wang et al., 2020), con licencia MIT del repositorio original. Se usan cajas calculadas de sus polígonos humanos, contrastando cada máscara de instancia con la máscara suministrada (IoU ≥ 0,95). Se descarta el campo bbox oficial por un error de escala; no se utilizan anotadores neuronales. `scripts/prepare_mju.py` conserva la anotación original y audita duplicados. Train pasa de 1.106 a 1.706 imágenes; val/test TACO permanecen en 197/197. En esa primera continuación no se abrió val/test MJU; el refinamiento actual prepara únicamente las 36 imágenes val auditadas mediante `scripts/prepare_refinement_data.py`. Test MJU sigue sin abrirse. Se conservan `before_mju/` y `source_mju/`; todos los pesos proceden de nuestro entrenamiento.

Desde la época 170 se usa precisión mixta en T4, tras comprobar una iteración con pesos aleatorios y otra con los pesos propios actuales. Se conservan `before_amp/` y `source_amp/`; las pérdidas y la inferencia CPU se calculan en float32.

La partición mantiene lotes de adquisición completos y agrupa coincidencias SHA256, píxeles, URL y pHash (distancia ≤ 6). Es una auditoría heurística de similitud, no una garantía de detectar todos los duplicados posibles. `data/detection/taco/audit.json` conserva conteos y los siete ajustes de borde de menos de dos píxeles. El YAML de desarrollo solo contiene train/val; test requiere evaluación final explícita con checkpoint y umbral congelados. AP50 corresponde a la única clase «residuo»; los materiales se evalúan aparte mediante la CNN, sin atribuirle clases que el detector no aprendió.

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-detector.txt
.\.venv\Scripts\python.exe -m scripts.import_detector
```

La importación usa `H:\Mi unidad\Vision-IA\detector_D2\runs\selected.json`, conserva versiones distintas y actualiza únicamente la ruta del detector en `config/app.json`. El entrenamiento y sus checkpoints se guardan separados de A/B/C en Drive. `notebooks/train_detector_own_drive.ipynb` reproduce el flujo con T4. Las métricas y verificaciones se guardan en `runs/detector_D2/`.

Para registrar un error, congela el fotograma y abre **Registrar esta fotografía real**. Confirma su categoría verdadera. Guarda original, recorte y entrada CNN en `data/field_samples/`, separados de los datasets oficiales. No reentrena automáticamente.

## Modelos comprobados

El checkpoint activo está en `config/app.json`; admite también `VISION_IA_CHECKPOINT`. Clases, resolución y normalización se leen del checkpoint. Modelo en caché, inferencia CPU, confirmación temporal en video y guía de reciclaje por reglas.

| Modelo | Accuracy val | Macro F1 val | Imágenes val |
|---|---:|---:|---:|
| A, 89.686 parámetros | 71,73 % | 0,722538 | 421 |
| B, 89.686 parámetros | 74,55 % | 0,761014 | 884 |
| C1b, residual 1.240.222 parámetros | 84,05 % | 0,851690 | Las mismas 884 de B |
| C2, residual 2.800.678 parámetros, margen + espejo | 81,56 % | 0,828856 | Las mismas 884 de B |
| C3, residual 1.240.222 parámetros, espejo | 78,73 % | 0,798613 | Las mismas 884 de B; confirmado en CPU |

**Activo: B, conservado por su mejor resultado en las fotos disponibles.** C1b completó 70 épocas, seleccionado en la 66. C2 se detuvo tras 50 épocas, seleccionado en la 48. C3 completó sus 80 épocas, seleccionado en la 75: se recuperó desde el estado de la 67 en la nueva cuenta de Colab con T4. Modelos, optimizadores e historiales permanecen guardados, sin sobrescribir A/B. C1b y C3 se evaluaron también sobre las 884 imágenes en CPU; sus resultados coinciden con Colab.

En las cuatro fotos humanas disponibles, B acierta 3/4 imágenes completas y 3/4 regiones manuales ampliadas a cuadrado; C1b, 1/4 y 2/4; C2, 1/4 y 1/4; C3 seleccionado, 0/4 y 1/4. C3 sin espejo acierta 2/4 imágenes completas, pero tampoco supera a B. Estas regiones fueron elegidas para diagnóstico: el automático sin referencia de fondo rechaza las cuatro escenas ocupadas. **La clasificación real todavía no está resuelta y ningún modelo superó simultáneamente el 85 % de accuracy y Macro F1.** Comparación técnica y fotos: `runs/candidate_comparison/summary.csv`. La prueba física con residuos quedó pendiente porque el usuario no pudo realizarla.

Se comprobaron las clases y la normalización originales, igualdad exacta de tensores entre aplicación y evaluación, y cero discrepancias de etiquetas en las 60 imágenes de regresión estratificada de C3 frente a Colab. Pasaron 48 pruebas de aplicación y compatibilidad. Las pruebas de balance de color y recalibración de BatchNorm no aportaron una mejora práctica suficiente; no se activaron en la aplicación.

La CNN residual propia tiene cuatro etapas de dos bloques con BatchNorm, ReLU y conexiones residuales. C1b/C3 usan canales 24/48/96/160, pooling global medio, Dropout y RGB 192×192: 1.240.222 parámetros. C2 amplía canales a 32/64/128/256 y combina pooling medio y máximo. C3 evita el margen obligatorio de C2 durante las aumentaciones. AdamW, semillas, scheduler, aumentaciones y estado de reanudación quedan guardados en cada ejecución. Cada experimento comenzó aleatoriamente; ninguno reutiliza pesos de A/B ni de otro experimento C.

A tiene una validación diferente. B/C comparten las 884 imágenes originales, usadas para seleccionar variantes; sus métricas no son una estimación final independiente. **Test sigue reservado y no se extrajo en Colab.**

## Datos y entrenamiento

B conserva 4.112 train, 884 val y 879 test. C2/C3 añaden solo a train 300 imágenes FMD: 100 vidrio, 100 metal y 100 plástico. Se excluye Paper de FMD por posible mezcla de papel/cartón. Auditoría de SHA256, píxeles y hashes perceptuales contra todo el registro B y dentro del suplemento: 0 coincidencias/candidatos. Registro y ROI originales en `data/supplemental/fmd/provenance.json`. Esta ampliación no ha demostrado mejoras en las cuatro fotos disponibles.

Fuente: [Flickr Material Database, MIT](https://people.csail.mit.edu/lavanya/fmd.html), Sharan, Rosenholtz y Adelson (2014), *Journal of Vision* 14(9):12, CC BY 2.0 para las categorías importadas. No se usa Foliage. Descarga selectiva de imágenes y máscaras originales, sin etiquetado con IA. Atribuciones TrashNet/RealWaste: [DATASETS.md](docs/DATASETS.md).

```powershell
.\.venv\Scripts\python.exe -m scripts.export_compact --config config/train_C.json
```

El segundo comando **solo prepara el paquete**, sin entrenar. Drive para escritorio debe estar activo, con la carpeta configurada `H:\Mi unidad\Vision-IA`. Paquetes y resultados C están separados de A/B.

El ejemplo prepara la configuración de C1b, el mejor candidato por validación. C2/C3 se conservan como experimentos que no demostraron mejora en las fotos reales. Abre en Colab el notebook que imprime el exportador, selecciona GPU, autoriza Drive y ejecuta sus celdas. Reutiliza el paquete B existente y extrae solo train/val. `RESUME_RUN` permite recuperar una ejecución interrumpida de la misma configuración; en blanco crea una ejecución nueva desde cero. C3 ya está terminado: no requiere reanudación. No entrenar prolongadamente en la CPU de Windows.

Solo para reproducir C2/C3, prepara su suplemento con `python -m scripts.prepare_materials` y selecciona `--config config/train_C2.json` o `config/train_C3.json` al exportar. No se mezclan automáticamente los experimentos.

La selección activa se define en `config/app.json`. El checkpoint C1b sigue disponible en `runs/synced/C/C1b_20261121_residual192_seed42/best_model.pth` para comparación. Las métricas de validación corresponden a imágenes guardadas.

```powershell
.\.venv\Scripts\python.exe -m scripts.import_compact --run-name C1b_20261121_residual192_seed42
.\.venv\Scripts\python.exe -m scripts.evaluate_compact --checkpoint runs/synced/C/C1b_20261121_residual192_seed42/best_model.pth --output-dir runs/C1b_val_nueva
```

La importación verifica hashes y no sobrescribe versiones distintas. La evaluación abre únicamente las 884 imágenes de validación B. Flujo histórico de preparación y Drive: [SYNC_TRAINING.md](docs/SYNC_TRAINING.md), [DATASETS.md](docs/DATASETS.md) y notebooks existentes.

## Límites

La localización automática utiliza el detector neuronal. TACO contiene principalmente residuos en exteriores: manos, envases transparentes, escenas interiores y objetos poco representados pueden fallar. Se conserva la alternativa manual. Un cuadro es una hipótesis de localización, no una prueba del material. La prueba física de webcam con el detector nuevo requiere verificación presencial; las comprobaciones simuladas solo verifican el flujo de código.

Softmax no es certeza calibrada, incluso con confianza alta. Cuatro fotos no estiman precisión general. Se comprobaron físicamente la captura de webcam y el rechazo de una escena calibrada vacía; el seguimiento desplazado tiene pruebas técnicas y necesita además comprobarse con residuos reales. La guía no identifica contaminación, peligrosidad ni composición de envases multicapa.
