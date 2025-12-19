Bitacora Dia 1
Alexander Vacaflores y Jhessael Sixto

Durante el dia 1 se configuro el entorno de analisis multimodal. Se extrajeron frames de video y se aplico análisis de emociones faciales con DeepFace, incorporando mejoras de iluminación y agregación temporal. Paralelamente, se extrajo el audio del video y se realizo la transcripcion automatica con Whisper. Finalmente, se diseño e implementó una estructura de datos multimodal que integra emocion facial y contenido textual sincronizado temporalmente.

Durante el primer dia de trabajo se realizo la organizacion inicial del proyecto y la configuracion del entorno de desarrollo. Se decidio la estructura de las carpetas para separar los videos originales, los datos procesados, los resultados obtenidos y el codigo fuente, facilitando el orden y la comprensión del proyecto.

Inicialmente se realizaron pruebas de extracción de frames y analisis de emociones faciales. En los primeros resultados se observo que el sistema clasificaba erroneamente algunas expresiones como “tristes” observando un poco habia frames donde si parecia triste realmente, pero la mayoria era sonriendo por lo que nos parecio extraño. Esto permitio identificar que factores como la iluminación, el movimiento al hablar y los primeros segundos del video afectaban el analisis.

A partir de esta observacion, se realizaron mejoras en el proceso, como descartar los primeros segundos del video y aplicar un preprocesamiento de imagen para mejorar la iluminacion. Tras estos cambios, los resultados se volvieron mas coherentes, predominando emociones como “neutral” y “happy”, acordes al contexto real del video.

Posteriormente, se extrajo el audio del video y se genero una transcripción automatica en español. Finalmente, se integraron los resultados de emociones faciales con el texto transcrito, creando una estructura multimodal que relaciona lo que se dice con la emoción detectada por el sistema en cada instante.

El dia 1 concluyo con un sistema funcional y resultados iniciales positivos, sentando las bases para analisis mas profundos en las siguientes etapas.

Bitacora Dia 2

Alexander Vacaflores y Jhessael Sixto

Durante el dia 2 se consolidaron los modulos principales para que funcionen de forma independiente. En el módulo facial se mejoró el análisis frame por frame, incorporando una mejora simple de iluminación para reducir errores por condiciones de luz y se agregó tolerancia a fallos (si un frame no se puede procesar, el pipeline continúa sin detenerse). Esto permitió obtener una serie temporal más estable y útil para integración posterior.

En paralelo, se avanzó el módulo de audio y texto. Se automatizó la transcripción del audio en español utilizando un modelo preentrenado, generando segmentos con tiempos de inicio y fin. A partir de esa transcripción, se implementó un análisis emocional del texto por segmento usando Transformers, dejando los resultados listos en formato JSON para poder integrarlos con las emociones faciales.

Finalmente, se implementaron utilidades comunes (lectura/escritura JSON, normalización de timestamps, logging) para mantener el proyecto ordenado y reproducible. El dia 2 cerro con los tres modulos funcionando por separado, listos para sincronización e integración en el siguiente dia.

Bitacora Dia 3

Alexander Vacaflores y Jhessael Sixto

Durante el dia 3 se realizo la integración del sistema multimodal, enfocándose en alinear correctamente el tiempo entre lo que ocurre en el video (frames) y lo que se transcribe del audio (segmentos). Se definio una regla simple y robusta: para cada frame con timestamp t, se busca el segmento de texto cuyo intervalo cumple start ≤ t ≤ end. Esto permitió asociar el contenido hablado a cada instante analizado.

Con la sincronización lista, se construyó una estructura multimodal final más clara y organizada, separando la información en dos bloques: emociones faciales (dominante y puntajes) y texto (contenido, rango del segmento y emoción del texto si existe). Además, se incorporó un runner que ejecuta el pipeline end-to-end, y una validación rápida para confirmar que el formato final cumple con la estructura esperada.

El dia 3 concluyo con un pipeline integrado funcional que produce un JSON multimodal por video, dejando el proyecto preparado para analisis temporal mas avanzado y generacion de metricas.

Bitacora Dia 4
Alexander Vacaflores y Jhessael Sixto

Durante el dia 4 se implemento el analisis temporal sobre los datos multimodales ya integrados. Primero, se desarrolló un detector de cambios emocionales, capaz de identificar transiciones en la emoción facial (y también en emoción de texto cuando existe) a lo largo del tiempo. Esto permitió pasar de resultados aislados por frame a una lectura más “dinámica” del comportamiento emocional.

Luego, se añadieron métricas de congruencia. Por un lado, se midió qué tan seguido coincide la emoción detectada en el rostro con la emoción inferida desde el texto. Por otro lado, se implemento una comparación contra etiquetas manuales (ground truth) para evaluar el acuerdo del sistema con segmentos previamente anotados. Con estas métricas, el sistema ya no solo “predice”, sino que puede justificarse con números.

Finalmente, se generaron insights automáticos que resumen el comportamiento del video (variación emocional, congruencia cara-texto y acuerdo con etiquetas). Todo se empaquetó en un reporte JSON por video, listo para ser mostrado en el informe y para preparar resultados comparativos entre varios videos.

Incluso se intento mejorar el reconocimiento de algunos videos y se logro:
Antes: la red facial confundía anger como fear (probable “expresión tensa” interpretada como miedo).
Después: se aplicó una heurística basada en scores para re-clasificar casos donde fear domina pero hay señales fuertes de enojo (angry y disgust presentes), aumentando el acuerdo con labels.

Bitacora Dia 5

Documentamos y mejoramos algunas funciones, volvimos a revisar los vídeos y sacar la información tanto de gráficos faltantes como de tablitas, fue un buen trabajo y estamos contentos con el resultado, también se grabó el video.