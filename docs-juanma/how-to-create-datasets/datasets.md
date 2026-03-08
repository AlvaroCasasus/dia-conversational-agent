## Assessing RAG: A Comprehensive Review of Evaluation Frameworks

[Paper link](https://doi.org/10.1109/ICCIKE67021.2025.11318213)

El artículo señala que tradicionalmente la evaluación de RAG dependía de bases de datos etiquetadas manualmente por humanos, lo cual es costoso y lento. Para solucionar esto, destaca cómo varias herramientas modernas abordan la generación de datasets sintéticos o artificiales:
- DeepEval: este marco de trabajo facilita la creación de datasets sintéticos. Esto permite a los usuarios establecer criterios de evaluación únicos, ofreciendo mucha flexibilidad para aplicaciones basadas en dominios específicos. [GitHub de DeepEval](https://github.com/confident-ai/deepeval).
- RAGEval: introduce la generación de datasets de evaluación que son específicos para cada escenario (scenario-specific). [GitHub de RAGEval](https://github.com/OpenBMB/RAGEval).
- ARES: esta herramienta emplea un pipeline de múltiples etapas que entrena a jueces evaluadores utilizando tripletes de "question-passage-answer" generados artificialmente. [GitHub de ARES](https://github.com/stanford-futuredata/ARES)

Esta revisión concluye con que el ecosistema actual se está moviendo hacia la automatización y la creación de datos sintéticos. El objetivo de estas herramientas es reducir drásticamente la dependencia de los lentos y costosos procesos de anotación manual, permitiendo evaluaciones más rápidas, escalables y adaptadas al dominio particular de cada empresa o proyecto.

---

## Know Your RAG: Dataset Taxonomy and Generation Strategies for Evaluating RAG Systems

[Paper link](https://arxiv.org/html/2411.19710v1)

Los autores (investigadores de IBM) proponen dos estrategias principales para generar datasets equilibrados y diversos:
1. Estrategia de extracción de declaraciones (statement extraction). En lugar de pedir directamente una pregunta y una respuesta, invierten el proceso. Primero aseguran la respuesta y luego generan la pregunta. El proceso sigue estos pasos:
    - Resumen del contexto en una sola oración temática.
    - Extraen declaraciones puramente factuales del texto.
    - Para generar preguntas de resumen, fusionan varios de esos hechos en tres declaraciones resumidas. Para generar preguntas de razonamiento, derivan tres conclusiones lógicas a partir de los hechos.
    - Eligen al azar una de esas declaraciones (factual, resumen o conclusión) y le piden a un modelo (utilizaron Llama-3-70b) que genere una pregunta que sea respondida inequívocamente por esa declaración específica.
2. Ajuste fino de modelos pequeños (fine-tuning). Dado que usar modelos gigantes para generar y validar múltiples datos es muy costoso e ineficiente, los autores proponen entrenar un modelo más pequeño para que haga el mismo trabajo.
    - Tomaron un modelo de lenguaje más pequeño llamado Flan-T5-large y lo ajustaron usando la técnica LoRA.
    - Este modelo más pequeño logró generar pares de preguntas y respuestas de alta calidad, siendo capaz de abarcar los distintos tipos de preguntas (resumen, razonamiento, hechos) a un costo computacional drásticamente menor y de manera mucho más rápida.

---

## Generating Q&A Benchmarks for RAG Evaluation in Enterprise Settings

[Paper link](https://aclanthology.org/2025.acl-industry.33/)

Se introduce una herramienta denominada *DataMorgana*. La creación del dataset se divide metodológicamente en dos etapas:

- Etapa de configuración: el usuario define las categorías de las preguntas y establece una distribución de probabilidad sobre ellas para determinar con qué frecuencia aparecerán en el dataset generado.
- Etapa de generación: el dataset se construye de forma incremental (pregunta por pregunta) mediante un proceso muy ligero:
    - Se toma un documento de la base de datos.
    - Se selecciona una categoría al azar para cada categorización, respetando la probabilidad asignada.
    - Se envía un prompt al LLM pidiéndole que genere una pregunta basada en ese documento y que cumpla estrictamente con las características de las categorías seleccionadas.

Según los autores, en los resultados de los experimentos realizados, los datasets creados con este nivel de control granular logran una diversidad léxica, sintáctica y semántica superior a las herramientas existentes, manteniendo en todo momento la calidad.

---

## Generating Diverse Q&A Benchmarks for RAG Evaluation with DataMorgana

[Paper link](https://arxiv.org/abs/2501.12789)

Este paper también usa *DataMorgana* pero con mayor profundidad.

La generación ocurre en dos grandes fases:

- Etapa de configuración: 
    - Todo el proceso se controla desde un único archivo JSON.

    - En él se definen diferentes categorías para las preguntas y para los usuarios.

    - A cada categoría se le asigna una descripción en lenguaje natural y una probabilidad específica para definir con qué frecuencia aparecerá en el dataset final.

- En la etapa de generación, el dataset se construye creando un par de pregunta y respuesta a la vez, repitiendo este ciclo exacto:
    - Selección: se escoge aleatoriamente una categoría de usuario y una de pregunta, respetando siempre las probabilidades configuradas previamente.
    - Muestreo: se extrae un documento al azar del corpus (la base de datos) del sistema RAG.
    - Invocación del LLM: se envía un prompt al LLM que incluye el documento seleccionado, junto con las descripciones del tipo de pregunta y de usuario; el modelo genera entonces 3 pares candidatos de preguntas y respuestas.
    - Filtrado: los pares generados pasan por una revisión para asegurar que cumplen las restricciones (como ser fieles al texto original y no requerir contexto externo), seleccionando finalmente uno de los candidatos válidos para añadirlo al dataset.

La conclusión a la que se llega es que DataMorgana mejora claramente los resultados frente a dejar al LLM generar preguntas sin restricciones y también supera a herramientas como Know Your RAG y DeepEval, especialmente en diversidad sintáctica y semántica. Un estudio de ablación muestra que esta mejora proviene principalmente de las categorizaciones de las preguntas, mientras que las categorías de usuario tienen un impacto mínimo. Además, aunque la diversidad tiende a disminuir cuando se generan muchas preguntas sobre un mismo documento, DataMorgana mantiene una ventaja notable frente a otras herramientas en estos escenarios.

---

## Automatic Dataset Generation for Knowledge Intensive Question Answering Tasks

[Paper link](https://arxiv.org/abs/2505.14212)

En este paper se utiliza un modelo base específico (*Mistral-7b-instruct-v0.3*) y un proceso riguroso de limpieza posterior:

1. Generación de preguntas: se introduce un documento (como documentación técnica) y se le pide al LLM que genere un máximo de 10 preguntas factuales sobre ese contenido.
2. Generación de respuestas: mediante otro prompt, se solicita al modelo que responda a esas preguntas basándose estrictamente en el documento. Se le da permiso explícito para responder que no hay respuestas factuales posibles si considera que la pregunta no se puede contestar con ese texto.
3. Post-procesamiento (limpieza): esta es una fase crítica donde usan modelos especializados más pequeños (basados en BERT y RoBERTa) para limpiar los datos generados. En esta fase:
    - Eliminan las preguntas que el LLM evadió o no respondió.
    - Etiquetan (pero no eliminan) las respuestas que quedaron truncadas o incompletas.
    - Eliminan los pares de preguntas y respuestas que no guardan relación semántica con el documento original.

El paper concluye que los datasets humanos funcionan mejor cuando el modelo responde con el documento de referencia delante, ya que aprende a extraer la información correcta del contexto proporcionado. En cambio, cuando el modelo debe responder sin contexto, los datasets sintéticos obtienen mejores resultados, porque permiten que el modelo retenga mejor el conocimiento técnico. Por tanto, los datos humanos son ideales para tareas basadas en recuperación de documentos, mientras que los datos sintéticos masivos son más efectivos para que el LLM interiorice conocimiento de un dominio.

---

## RAGEval: Scenario Specific RAG Evaluation Dataset Generation Framework

[Paper link](https://aclanthology.org/2025.acl-long.418/)

[RAGEval GitHub](https://github.com/OpenBMB/RAGEval)

En este paper se realiza un enfoque que utiliza una cadena de generación basada en esquemas (schemas) para asegurar que los datos sean precisos, controlables y libres de alucinaciones.

El proceso fluye de manera secuencial a través de los siguientes pasos:
1. Creación del esquema (schema summary): a partir de un grupo de documentos reales (semilla), utilizan un modelo de lenguaje (GPT) para extraer una estructura abstracta o esquema. Este esquema define qué elementos clave son esenciales para un dominio específico, sin contener los datos reales en sí (por ejemplo, en un entorno legal, el esquema dictará que debe existir un caso, un estatuto y un fallo del tribunal).
2. Generación de configuración y documentos: el sistema llena este esquema vacío con datos específicos, utilizando una mezcla de reglas y modelos de lenguaje, para crear lo que llaman una configuración. Una vez que tienen esta configuración estructurada, utilizan GPT-4o para redactar un documento narrativo coherente y realista que contenga toda esa información factual.
3. Generación de Preguntas, Respuestas y Referencias (QRA): utilizando la configuración como guía, el modelo genera 7 tipos diferentes de preguntas (como factuales, de razonamiento, resúmenes, etc.) junto con respuestas iniciales. Para asegurar que la respuesta sea rastreable y no inventada, extraen fragmentos exactos del documento generado y refinan la respuesta final para que coincida perfectamente con esos fragmentos.
4. Extracción de puntos clave (keypoints): como paso final, el sistema resume cada respuesta generada en 3 a 5 puntos clave que contienen la información factual esencial.

RAGEval demuestra que su método genera datasets de mucha mayor calidad que los enfoques tradicionales, logrando mejores resultados en claridad, seguridad y realismo según evaluaciones humanas. También muestra que las métricas clásicas como BLEU o ROUGE no evalúan bien los sistemas RAG, mientras que sus nuevas métricas (completitud, alucinación e irrelevancia) se alinean mucho mejor con el juicio humano. En las pruebas, GPT-4o fue el modelo con mejor rendimiento, aunque algunos modelos open-source pequeños también fueron competitivos. Además, concluyen que la calidad de la recuperación de información es crucial para la respuesta final y que parámetros como el número de documentos recuperados deben ajustarse según el tipo de pregunta, ya que no existe una configuración universal.

---

## Diverse And Private Synthetic Datasets Generation for RAG evaluation: A multi-agent framework

[Paper link](https://doi.org/10.48550/arXiv.2508.18929)

Los autores proponen crear conjuntos de datos sintéticos utilizando un sistema compuesto por tres agentes de Inteligencia Artificial distintos que trabajan en cadena (pipeline) para asegurar que el dataset final sea diverso, pero sin filtrar información sensible (como datos médicos o personales).
Así es como funciona su proceso de creación (detallado en el Algoritmo 1):
1. Agente de diversidad: toma la base de datos original y utiliza técnicas de agrupamiento (algoritmo k-means sobre embeddings) para dividir los textos en clústeres temáticos. Luego, selecciona muestras representativas de cada grupo para garantizar que el dataset final cubra todos los temas posibles y tenga variabilidad semántica.
2. Agente de privacidad: toma las muestras seleccionadas por el primer agente y se encarga de detectar y enmascarar (seudonimizar) cualquier información confidencial. Escanea el texto en busca de nombres, datos médicos, salarios o teléfonos, y los reemplaza de forma coherente con el contexto para crear una versión privada del texto.
3. Agente curador de QA: este agente toma los documentos que ya son diversos y que han sido limpiados de datos privados, y genera a partir de ellos los pares de preguntas y respuestas (QA) sintéticos. Este es el conjunto de datos definitivo que servirá para evaluar el sistema RAG.

Los autores destacan que en sectores como la medicina o las finanzas, evaluar un sistema RAG con datos reales podría exponer información confidencial de usuarios o pacientes. Sus experimentos demostraron que este enfoque de múltiples agentes logra ocultar los datos privados con gran precisión (identificando correctamente entre el 75% y el 91% de los datos sensibles según el sector) y, además, genera preguntas más diversas.

---


