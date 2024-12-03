# GPT Multi-Shield

GPT Multi-Shield es una herramienta diseñada para analizar textos y detectar posibles contenidos generados por inteligencia artificial. Utiliza modelos de lenguaje preentrenados, técnicas de procesamiento de lenguaje natural (NLP) y medidas estadísticas para evaluar la "humanidad" del texto.

## Características

- **Cálculo de Perplejidad:** Mide la previsibilidad de las secuencias de texto.
- **Análisis de Burstiness:** Evalúa la variabilidad en las frecuencias de palabras, un indicador típico del texto humano.
- **Longitud Media de las Frases:** Determina la complejidad de las frases en el texto.
- **Porcentaje de Palabras Comunes:** Calcula el porcentaje de palabras frecuentes para identificar patrones en el lenguaje.
- **Análisis Gráfico:** Visualización de las palabras más repetidas en el texto.
- **Extracción de Textos de PDFs:** Procesamiento directo de archivos PDF para análisis.
- **Identificación de Prompts Potenciales:** Sugerencias de prompts relacionados con el contenido.

## Requisitos

Asegúrate de tener instalados los siguientes requisitos antes de ejecutar el proyecto:

- Python 3.8 o superior
- Bibliotecas Python:
  - `streamlit`
  - `transformers`
  - `torch`
  - `nltk`
  - `plotly`
  - `PyPDF2`
  - `pandas`
  - `requests`
  - `beautifulsoup4`

Puedes instalar las dependencias ejecutando:

```bash
pip install -r requirements.txt
```
## Ejecución en CMD
```bash
streamlit run GPT0_Model.py
