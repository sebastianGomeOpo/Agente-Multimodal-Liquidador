# 🤖 MultiDoc-Agent: Agente Multimodal para Procesamiento de Documentos

## 📋 Descripción del Proyecto

**MultiDoc-Agent** es un sistema inteligente que procesa y analiza documentos multimodales (Excel visual + PDF de liquidación) utilizando:

- **CLIP**: Para crear embeddings en un espacio vectorial compartido (imagen + texto)
- **LangGraph**: Para definir el flujo del agente
- **ChromaDB**: Para almacenar y recuperar documentos relevantes
- **OCR**: Para extraer texto de imágenes (Landing AI o DeepSeek)
- **LLM**: Para razonar sobre los documentos recuperados

### Aplicación de Conceptos de Clase 17

Este proyecto implementa el **"Enfoque 3: RAG Multimodal Verdadero"** con:
- Espacio vectorial compartido CLIP para imágenes y texto
- Recuperación de documentos por similitud multimodal
- Integración completa de LangGraph como orquestador

---

## 🏗️ Arquitectura de 3 MODOS

### MODO 1: ENTRADA (Preprocesamiento)
```
Excel (visual) → Imagen PNG
PDF (liquidación) → Imágenes PNG
```

### MODO 2: PROCESO (Transformación)
```
Imágenes → OCR → Texto
Texto → Parser → Estructura JSON
Texto + Imagen → CLIP → Embeddings compartidos
Embeddings → ChromaDB → Indexado
```

### MODO 3: SALIDA (Consulta)
```
Query Usuario → Embedding CLIP → Busca ChromaDB
→ LLM Razona → Respuesta Estructurada
```

---

## 🌟 Enfoque Multimodal

### ¿Qué lo hace diferente?

**Espacio Vectorial Compartido CLIP:**
- Las imágenes de Excel se convierten a embeddings
- Los textos de liquidación se convierten a embeddings
- **Ambos están en el MISMO espacio vectorial** (512 dimensiones)
- Puedes buscar con texto y encontrar imágenes relevantes (y viceversa)

```python
# Ejemplo:
imagen_embedding = CLIP.encode_image("excel.png")  # [512 dimensiones]
texto_embedding = CLIP.encode_text("¿Cuál es el total?")  # [512 dimensiones]
similitud = cosine(imagen_embedding, texto_embedding)  # ~0.87
```

---

## 📁 Estructura del Proyecto

```
multidoc-agent/
├── data/
│   ├── input/          # Archivos de entrada (Excel, PDF)
│   ├── images/         # Imágenes generadas
│   ├── processed/      # Texto y tablas extraídas
│   └── embeddings/     # Vectores CLIP guardados
├── src/
│   ├── utils/          # Configuración y logging
│   ├── preprocessors/  # Excel→PNG, PDF→PNG
│   ├── extractors/     # OCR, Parser
│   ├── embeddings/     # CLIP encoder
│   ├── vectorstore/    # ChromaDB manager
│   └── agent/          # LangGraph + Nodos
├── notebooks/          # Experimentación
├── main.py             # Punto de entrada
├── requirements.txt    # Dependencias
└── README.md          # Este archivo
```

---

## 🚀 Instalación

### Requisitos
- Python 3.10+
- pip
- CUDA (opcional, para GPU)

### Pasos

1. **Clonar/descargar el proyecto**
```bash
cd multidoc-agent
```

2. **Crear entorno virtual**
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# o
venv\Scripts\activate  # Windows
```

3. **Instalar dependencias**
```bash
pip install -r requirements.txt
```

4. **Configurar variables de entorno**
```bash
cp .env.example .env
# Editar .env con tus API keys
```

5. **Preparar datos**
- Colocar Excel en: `data/input/excel/`
- Colocar PDF en: `data/input/pdf/`

---

## 💡 Uso

### Modo Indexación (Poblar ChromaDB)
```bash
python main.py indexar
```

Esto:
1. Convierte Excel a imágenes PNG
2. Convierte PDF a imágenes PNG
3. Extrae texto con OCR (Landing AI o DeepSeek)
4. Estructura el texto con parser
5. Genera embeddings CLIP (imagen + texto)
6. Indexa todo en ChromaDB

### Modo Consulta
```bash
python main.py consultar "¿Cuál es el total de la liquidación?"
```

Proceso:
1. Tu pregunta se convierte a embedding CLIP
2. Se busca en ChromaDB (espacio multimodal)
3. Se recuperan documentos relevantes (pueden ser imágenes o texto)
4. LLM analiza los documentos
5. Se retorna respuesta estructurada

### Modo Interactivo
```bash
python main.py interactivo
```

Permite:
- `indexar` - Indexar documentos
- `consultar` - Hacer preguntas
- `stats` - Ver estadísticas
- `salir` - Terminar

---

## 📊 Ejemplos de Queries

```python
# Ejemplo 1: Pregunta sobre el total
"¿Cuál es el monto total de la liquidación?"

# Ejemplo 2: Comparar valores
"¿Cuánto es la diferencia entre el salario base y las gratificaciones?"

# Ejemplo 3: Buscar por concepto
"¿Qué descuentos se aplicaron?"

# Ejemplo 4: Validar fechas
"¿Cuál es la fecha de pago?"
```

---

## 🔑 Conceptos Clave de Clase 17

### 1. Multimodalidad
- Procesamiento simultáneo de **imágenes** y **texto**
- No es solo OCR → es integración multimodal

### 2. CLIP y Espacio Vectorial Compartido
- **CLIP** = modelo pre-entrenado de OpenAI
- **Ventaja**: Entiende tanto imágenes como texto
- **Espacio compartido**: Ambos tipos de datos caben en 512 dimensiones
- **Resultado**: Similitud entre imagen y descripción de texto

### 3. RAG Multimodal Verdadero
- **R**etrieval: Busca en ChromaDB (imagen + texto)
- **A**ugmented: Aumenta el prompt del LLM con documentos
- **G**eneration: LLM genera respuesta final

### 4. LangGraph
- Define el flujo del agente con **nodos** y **edges**
- Cada nodo es una función (query, retrieve, reason, format)
- Las transiciones son determinísticas
- Facilita debugging y testing

---

## 📝 Estructura de Nodos (LangGraph)

```
START
  ↓
[query_node] - Procesa query del usuario
  ↓
[retrieve_node] - Busca en ChromaDB usando embedding CLIP
  ↓
[reason_node] - LLM analiza documentos recuperados
  ↓
[format_node] - Estructura la respuesta final
  ↓
END
```

---

## 🛠️ Módulos Principales

### src/utils/
- `config.py` - Configuración centralizada
- `logger.py` - Sistema de logging

### src/preprocessors/
- `excel_to_image.py` - Convierte rango Excel a PNG
- `pdf_to_image.py` - Convierte PDF a PNGs

### src/extractors/
- `ocr_extractor.py` - Extrae texto (Landing AI/DeepSeek)
- `structure_parser.py` - Parsea y estructura texto

### src/embeddings/
- `clip_encoder.py` - Genera embeddings multimodales CLIP

### src/vectorstore/
- `chroma_manager.py` - Gestor de ChromaDB
- `multimodal_indexer.py` - Indexador multimodal

### src/agent/
- `graph_agent.py` - Definición del grafo LangGraph
- `nodes.py` - Implementación de nodos
- `tools.py` - Herramientas disponibles para el agente

---

## 📈 Pipeline Completo

### Fase 1: INDEXACIÓN (Una vez)
```
Excel + PDF
   ↓
[preprocessors] → Imágenes
   ↓
[extractors] → Texto + Estructura
   ↓
[embeddings] → CLIP Encoding
   ↓
[vectorstore] → ChromaDB
```

### Fase 2: CONSULTA (Cada pregunta)
```
Query Usuario
   ↓
[embeddings] → Embedding CLIP
   ↓
[vectorstore] → Búsqueda
   ↓
[agent] → Grafo LangGraph
   ↓
[reason_node] → LLM
   ↓
[format_node] → Respuesta
```

---

## ⚙️ Configuración

Editar `.env`:

```env
# API Keys
OPENAI_API_KEY=sk-...
LANDING_AI_API_KEY=...

# Modelos
CLIP_MODEL_NAME=openai/clip-vit-base-patch32
LLM_MODEL=gpt-4

# ChromaDB
CHROMA_COLLECTION_NAME=multidoc_collection
CHROMA_DISTANCE_METRIC=cosine

# Agente
RETRIEVE_TOP_K=5
SIMILARITY_THRESHOLD=0.5
```

---

## 🧪 Testing

```python
# En Python interactivo o notebook:
from src.embeddings import CLIPEncoder

encoder = CLIPEncoder()

# Verificar espacio compartido
result = encoder.verify_shared_space("image.png", "texto descriptivo")
print(f"Similitud: {result['cosine_similarity']}")
```

---

## 📚 Dependencias Principales

- **langchain** - Framework para LLMs
- **langgraph** - Orquestación de agentes
- **chromadb** - Base de datos vectorial
- **transformers** - CLIP pre-entrenado
- **torch** - Computación tensor
- **pillow** - Procesamiento de imágenes
- **pdf2image** - Conversión PDF

---

## 🤝 Contribución

Este proyecto es educativo y forma parte del curso Clase 17 sobre RAG Multimodal.

---

## 📞 Soporte

Para errores o preguntas:
1. Revisar logs en `logs/multidoc_agent.log`
2. Verificar configuración en `.env`
3. Asegurar que API keys son válidas

---

## 📄 Licencia

Proyecto educativo - Libre para uso académico

---

**¡Listo para procesar documentos multimodales! 🚀**