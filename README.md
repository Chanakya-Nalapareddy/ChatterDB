# ChatterDB
ChatterDB — From data to dialogue. A metadata-driven system for querying databases using natural language.

# 📊 ChatterDB – Metadata Catalog, Analytics & NLQ Platform : From data to dialogue. A metadata-driven system for querying databases using natural language.

ChatterDB is an **end-to-end, metadata-driven data exploration and analytics platform** that combines:

- **Metadata cataloging**
- **Interactive Power BI dashboards**
- **Retrieval-Augmented Generation (RAG)**
- **Natural Language → SQL → Results → Natural Language**
- **Automatic plots and visualizations**

The goal of this project is to make **databases explorable and queryable using natural language**, while remaining **transparent, safe, auditable, and developer-friendly**.

This README intentionally contains **both functional and deep technical explanations** so the project can be understood by:
- Data engineers
- Analytics engineers
- ML / LLM engineers
- BI developers
- Reviewers and interviewers

---

## 🧠 Problem Statement

Modern data platforms suffer from:
- Poor schema discoverability
- Tribal knowledge of tables and joins
- SQL complexity for non-technical users
- LLM hallucinations when querying databases
- Lack of trust and auditability in NLQ systems

**ChatterDB solves this by grounding LLMs in metadata, semantics, and validation.**

---

## 🏗 High-Level Architecture

```
User
 ↓ Natural Language Question
Streamlit UI
 ↓
Conversation Context (SQLite)
 ↓
Semantic + Metadata Retrieval (RAG)
 ↓
LLM (Intent + SQL Generation)
 ↓
SQL Intent Validation
 ↓
DuckDB Execution Engine
 ↓
Result DataFrame
 ↓
Auto Chart Selection
 ↓
LLM (Result → Natural Language Explanation)
 ↓
Answer + Charts + SQL
```

---

## 🚀 Core Components (Detailed)

---

## 1️⃣ Metadata Foundation (Data Catalog)

The system begins by extracting **database metadata** directly from DuckDB using `information_schema`.

### MetaDataMaster Table
One row per column, containing:

- database_name
- table_schema
- table_name
- table_type
- ordinal_position
- column_name
- data_type
- max_length_bytes
- precision
- scale
- is_nullable
- obj_order

This table is **authoritative** and used everywhere:
- Power BI dashboards
- Semantic layer grounding
- RAG retrieval
- SQL validation

> This ensures the LLM never operates blindly.

---

## 2️⃣ Power BI – Visual Metadata Explorer

Power BI provides a **human-first schema discovery layer**.

### Capabilities
- Filter by database / schema / table
- Keyword search across all metadata fields
- Drill-down from schema → table → column
- Acts as a visual data dictionary

Power BI is **not optional UI fluff** — it directly improves NLQ accuracy by helping users ask better questions.

---

## 3️⃣ Semantic Layer (YAML)

Instead of letting the LLM infer meaning from raw schemas, the project introduces a **semantic abstraction layer** defined in YAML.

### What the Semantic Layer Does
- Maps business concepts → physical tables
- Defines metrics, dimensions, and relationships
- Restricts which columns are allowed in queries
- Provides business-friendly naming

Example (conceptual):
```yaml
entities:
  sales:
    table: fact_sales
    measures:
      - total_revenue
      - total_orders
    dimensions:
      - country
      - order_date
```

This is the **first safety barrier**.

---

## 4️⃣ Retrieval-Augmented Generation (RAG) – Technical Details

ChatterDB uses **RAG to ground LLM behavior**.

### What Is Retrieved
- Relevant tables from MetaDataMaster
- Column descriptions and data types
- Semantic layer definitions
- Prior conversation turns

### Retrieval Sources
- **Vector database (LanceDB)** for embeddings
- **Structured metadata tables**
- **Conversation history (SQLite)**

### Why RAG Is Critical
Without RAG:
- LLM guesses schema
- Hallucinates joins
- Generates invalid SQL

With RAG:
- LLM sees only relevant schema slices
- Context window stays small
- Accuracy improves dramatically

> RAG is used for **context injection**, not answer generation.

---

## 5️⃣ Vector Database (LanceDB)

LanceDB is used as the **vector store** for:
- Table names
- Column names
- Semantic descriptions
- Example queries

### Embedding Strategy
- Text chunks are embedded using OpenAI embeddings
- Stored locally in `.lancedb/`
- Queried using semantic similarity at runtime

This allows questions like:
> “customer revenue by region”

to retrieve:
- `customers.country`
- `sales.revenue`
- `orders.customer_id`

even if those words don’t exactly match.

---

## 6️⃣ Natural Language → SQL Generation

### Process
1. User asks a question
2. Relevant metadata is retrieved (RAG)
3. Semantic layer constraints are applied
4. LLM generates SQL
5. SQL is checked for:
   - Table validity
   - Column validity
   - Aggregation correctness
   - Intent alignment

### SQL Validation
A custom **SQL intent validator** ensures:
- No hallucinated columns
- No invalid aggregations
- No schema violations

Invalid SQL is rejected and regenerated.

---

## 7️⃣ DuckDB Execution Engine

Once validated:
- SQL runs directly on DuckDB
- Results are returned as Pandas DataFrames
- Supports joins, aggregations, filters, windows

DuckDB is chosen because it is:
- Fast
- Embedded
- SQL-compliant
- Analytics-optimized

---

## 8️⃣ Automatic Charts & Visualizations

Based on result shape:
- Time-based → line charts
- Categorical aggregates → bar charts
- Distributions → histograms
- Small results → tables

Charts are rendered using:
- Matplotlib (backend)
- Streamlit (frontend)

This is **data-driven visualization**, not hardcoded charts.

---

## 9️⃣ Results → Natural Language Explanation

After execution:
- Results summary is passed back to the LLM
- The LLM explains:
  - Trends
  - Outliers
  - Comparisons

Example:
> “Revenue peaked in Q3, driven primarily by the US market.”

This completes the **NL → SQL → NL loop**.

---

## 🔁 Conversation Memory

- Chat history stored in SQLite
- Enables follow-up questions
- Maintains conversational context
- Thread-based conversations

Example:
> “What about only last quarter?”

---

## 🗂 Project Structure

```
chatterdb/
├── src/
│   └── rag_semantic/
│       ├── rag_app.py
│       ├── sql_generator_gpt.py
│       ├── semantic_model.py
│       ├── sql_intent_validator.py
│       └── thread_store_sqlite.py
│
├── data/
│   ├── warehouse/
│   │   └── chatterdb.duckdb
│   └── chat_threads.sqlite
│
├── powerbi/
│   ├── Metadata_Catalog.pbix
│   └── MetaDataMaster.csv
│
├── streamlit_rag_app.py
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

---

## 🛠 Running the Project

```bash
pip install -r requirements.txt
cp .env.example .env
streamlit run streamlit_rag_app.py
```

---

## 🧪 Typical User Flow

1. Explore schema in Power BI
2. Discover relevant tables/columns
3. Ask a natural language question
4. Review generated SQL (optional)
5. View charts + explanation
6. Ask follow-up questions

---

## 📌 Design Principles

- Metadata-first
- Semantic grounding
- Retrieval before generation
- Validation before execution
- Explainability by default

---

## 📜 License

MIT License

---

## 🙌 Acknowledgements

DuckDB • Streamlit • Power BI • LanceDB • LangChain • Azure OpenAI
