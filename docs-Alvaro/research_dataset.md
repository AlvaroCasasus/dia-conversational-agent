# Generación del Dataset de Evaluación para el Sistema RAG

En este documento se presenta un análisis de las estrategias descritas en: Know Your RAG: Dataset Taxonomy and Generation Strategies for Evaluating RAG Systems (https://arxiv.org/abs/2411.19710), para la generación del dataset de evaluación de nuestro sistema RAG del departamento de inteligencia artificial.

---

# Métodos de generación del dataset

Se contemplaron dos métodos de generación:

- **Statement Extraction Strategy**
- **Fine-tuning**

## Método Statement Extraction

Consiste en extraer un *statement* del documento y después generar una pregunta cuya respuesta sea esa afirmación. Este enfoque también se conoce como **answer-first generation**.

El proceso se divide en los siguientes pasos:

1. **Extraer statements del texto**.


2. **Generar preguntas** a partir de esos statements.

   Ejemplo de pregunta factual:  
   *¿Quién imparte la asignatura de Machine Learning en el Máster de Inteligencia Artificial?*

3. **Crear la tripleta**: question, answer, context


### Ventajas

- Garantiza que la respuesta esté en el documento, ya que el *statement* proviene directamente del texto.
- Permite que el **retriever** encuentre la información sin que el LLM tenga que inventarla.
- Reduce significativamente las **alucinaciones** del modelo.
- El paper muestra que con este método se obtiene un **dataset más balanceado** que con generación directa.
- Permite **controlar el tipo de preguntas** generadas mediante la taxonomía de preguntas.

### Desventajas

- El **pipeline es más largo**, debido al proceso de extracción de statements.
- Puede implicar un **mayor coste computacional**.

---

## Método Fine-Tuning

El método propone entrenar un modelo pequeño de generación de preguntas mediante *fine-tuning*.  

Para ello, primero se construye un dataset sintético de pares *(contexto, pregunta, respuesta)* a partir de los documentos, utilizando un pipeline basado en **statement extraction** del texto.  

El modelo aprende a generar preguntas y respuestas condicionadas al contexto y al tipo de pregunta.  

Una vez entrenado, el modelo puede aplicarse a nuevos documentos para generar automáticamente datasets de evaluación para sistemas **RAG**.

### Ventajas
- Menor coste de generación. El modelo fine-tuneado permite generar preguntas y respuestas **sin depender continuamente de modelos grandes**, reduciendo el coste computacional.

### Desventajas
- Riesgo de alucinaciones. En este enfoque, el modelo genera la respuesta **a partir del contexto proporcionado**, lo que puede producir errores o información inventada.


---

## Validación del dataset

Ambas estrategias requieren un **segundo paso de validación y limpieza** del dataset generado para garantizar su calidad.

---

# Tipos de preguntas del dataset

El paper contempla **tres categorías principales de preguntas**:

## 1. Factual

Requieren recuperar un **hecho específico** del corpus.

Ejemplo: ¿Quién imparte la asignatura de Machine Learning en el Máster de Inteligencia Artificial?


## 2. Summarization

Implican **sintetizar información** contenida en uno o varios fragmentos del documento.

## 3. Reasoning

Requieren **combinar múltiples afirmaciones del corpus** para inferir una respuesta.

---

# Diversidad del dataset

Es importante que el dataset **no esté dominado por un único tipo de preguntas**, ya que perderíamos la capacidad de evaluar distintas habilidades del sistema RAG.

Por ello se busca:

- Mantener **diversidad en el dataset**
- Incluir **distintos niveles de complejidad**
- Evaluar tanto recuperación directa como capacidades de síntesis y razonamiento

---

# Distribución de preguntas en nuestro proyecto

En nuestro proyecto:

- Las preguntas más comunes serán de tipo **factual**, debido a que el RAG contiene principalmente **información estructurada** sobre:
  - asignaturas
  - profesores
  - investigación
  - guías docentes

- Las preguntas de tipo **reasoning** también estarán presentes, pero **no dominarán el dataset**, ya que suelen requerir recuperar información de **múltiples fragmentos del corpus**, lo que aumenta su complejidad.

Se incluirá un número **reducido pero representativo** de estas preguntas para evaluar el comportamiento del sistema en escenarios más complejos.

### Ejemplo de distribución para nuestro proyecto 
- 50% factual
- 30% summarization
- 20% reasoning


---

# Evaluación del sistema RAG

Para analizar el rendimiento del sistema se puede observar **cómo responden el RAG y el LLM a los distintos tipos de preguntas**.

Por ejemplo:

- Si el sistema responde bien a preguntas **factual**
- Pero obtiene **bajo accuracy en preguntas reasoning**

Esto puede indicar problemas en:

- la **recuperación de chunks**
- el **razonamiento multifragmento**

El sistema RAG se podrá evaluar utilizando **RAGAS**, que permite medir el rendimiento del sistema a partir del dataset generado.


