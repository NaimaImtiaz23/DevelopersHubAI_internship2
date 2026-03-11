# Task 4: Context-Aware Chatbot Using LangChain and RAG

**DevelopersHub Corporation — AI/ML Engineering Internship**
**Submitted by:** [Your Name]
**Deadline:** 27th March 2026

---

## Objective

Build a conversational chatbot that remembers context across multiple turns and retrieves relevant answers from an external knowledge base. The system uses Retrieval-Augmented Generation (RAG) via LangChain to ground every response in factual source documents rather than relying on model weights alone.

---

## Dataset / Knowledge Base

A custom AI/ML reference corpus was built covering six topic areas:

- Machine Learning Overview (supervised, unsupervised, reinforcement learning)
- Deep Learning and Neural Networks (CNNs, RNNs, Transformers)
- Natural Language Processing (embeddings, LLMs, prompt engineering)
- Retrieval-Augmented Generation (indexing, retrieval, generation pipeline)
- Model Evaluation Metrics (accuracy, F1, ROC-AUC, MAE, RMSE)
- MLOps and Production ML (feature stores, drift monitoring, CI/CD for ML)

In a real deployment this corpus can be replaced with internal documents, PDFs, or Wikipedia articles without changing any pipeline code.

---

## Methodology and Approach

### Step 1 — Document Chunking

The corpus is split using `RecursiveCharacterTextSplitter` with chunk size 400 and overlap 60 tokens. The overlap preserves semantic continuity at chunk boundaries so answers that span multiple paragraphs are not fragmented.

### Step 2 — Embedding and Indexing

Each chunk is embedded using `sentence-transformers/all-MiniLM-L6-v2` from the `langchain-huggingface` package. This model is lightweight, CPU-friendly, and produces high-quality dense embeddings for question-answer retrieval. Embeddings are stored in a FAISS vector index and saved locally for reuse.

### Step 3 — Retrieval

At query time the user question is embedded and the top 2 most similar chunks are retrieved from FAISS using cosine similarity. Retrieved chunks are formatted with their source title before being injected into the prompt.

### Step 4 — Context-Aware Generation

The chain is built using LangChain's modern LCEL (LangChain Expression Language) pattern:

```
trim_history
    |
RunnablePassthrough.assign(context = retriever)
    |
ChatPromptTemplate  <-- system prompt + chat_history + question
    |
HuggingFacePipeline (TinyLlama-1.1B)
    |
StrOutputParser
```

`RunnableWithMessageHistory` wraps the chain and injects per-session `ChatMessageHistory` automatically. A custom `trim_history` step keeps only the last 4 messages (2 exchange pairs) before building the prompt, preventing token overflow on TinyLlama's 2048-token context window.

### Step 5 — Deployment

A Streamlit application (`app.py`) provides a clean browser-based chat interface. Each response includes source attribution showing which knowledge base documents were used to generate the answer.

---

## Dataset Loading and Preprocessing

```python
# Documents converted to LangChain Document objects
documents = [
    Document(page_content=entry['content'], metadata={'title': entry['title']})
    for entry in CORPUS
]

# Chunked with overlap to preserve cross-boundary context
splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=60)
chunks = splitter.split_documents(documents)

# Embedded and indexed
embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
vector_store = FAISS.from_documents(chunks, embeddings)
```

---

## Model Development

```python
# LCEL chain with history trimming and retrieval
rag_chain = (
    RunnableLambda(trim_history)
    | RunnablePassthrough.assign(
        context=RunnableLambda(lambda x: format_docs(retriever.invoke(x['question'])))
    )
    | RAG_PROMPT
    | llm
    | StrOutputParser()
)

# Wrapped with per-session message history
chain_with_history = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key='question',
    history_messages_key='chat_history'
)
```

---

## Evaluation

| Aspect | Observation |
|---|---|
| Retrieval relevance | Correct source retrieved for all 4 test questions |
| Context resolution | Follow-up questions resolved correctly using prior turn |
| Token safety | History trimming prevents overflow across long conversations |
| Source attribution | Title of retrieved document displayed after every answer |

Qualitative evaluation was used since this is an open-domain QA system without fixed ground-truth answers. In a production setting, RAG evaluation frameworks such as RAGAS (faithfulness, answer relevancy, context precision) would be applied.

---

## Key Results and Observations

- The RAG approach produces answers grounded in the knowledge base, significantly reducing hallucination compared to vanilla generation.
- Chunking with overlap of 60 tokens reduced cases where answers were split across chunk boundaries.
- RunnableWithMessageHistory correctly maintained context across turns — pronoun references in follow-up questions resolved to the correct topic from prior exchanges.
- Trimming chat history to the last 4 messages was necessary to keep inputs within TinyLlama's 2048-token context window without sacrificing recent context.
- Source attribution after each answer increased response transparency. Users can identify which document each answer came from.
- Swapping TinyLlama for a larger instruction-tuned model such as Mistral-7B or LLaMA-3-8B would substantially improve answer fluency with no changes to the pipeline.

---

## How to Run

```bash
# Install dependencies
pip install langchain langchain-community langchain-core langchain-text-splitters \
    langchain-huggingface faiss-cpu sentence-transformers transformers \
    accelerate torch streamlit

# Step 1 — Run the notebook to build the FAISS index and test the chain
jupyter notebook task4_rag_chatbot.ipynb

# Step 2 — Launch the Streamlit chat interface
streamlit run app.py
```

---

## Output Files

| File | Description |
|---|---|
| `task4_rag_chatbot.ipynb` | Main notebook with full pipeline |
| `faiss_index/` | Saved FAISS vector store (index + docstore) |
| `app.py` | Streamlit deployment script |

---

## Skills Demonstrated

- Document chunking and vector indexing with FAISS
- Dense retrieval using sentence-transformer embeddings
- LangChain LCEL chain composition with RunnableWithMessageHistory
- Context-aware multi-turn conversation with automatic history management
- Token budget management to prevent context window overflow
- Source attribution and answer transparency in RAG pipelines
- Streamlit deployment of a production-style conversational AI system
