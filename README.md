# 🚀 AI-Powered RAG Chatbot using Endee Vector Database

## 📌 Project Overview

This project implements a **Retrieval-Augmented Generation (RAG)** system using **Endee** as the vector database. It enables users to ask questions from custom documents and receive **accurate, context-aware answers** powered by semantic search and large language models.

---

## 🎯 Problem Statement

Traditional chatbots rely only on pre-trained knowledge and often fail to provide accurate answers from **private or domain-specific data**.

This project solves that problem by:

* Storing document knowledge as **vector embeddings**
* Retrieving relevant context using **semantic similarity search**
* Generating intelligent answers using an **LLM**

---

## 🧠 Key Features

* 🔍 Semantic Search using vector embeddings
* 🤖 RAG-based intelligent Q&A system
* ⚡ Fast similarity search using Endee
* 💻 Multiple interfaces:

  * CLI (Command Line)
  * Streamlit UI
* 🐳 Docker support for easy setup
* 📦 Modular and scalable architecture

---

## 🏗️ System Architecture

```
User Query
    ↓
Embedding Model (Sentence Transformers)
    ↓
Endee Vector Database
    ↓
Top-K Similar Documents Retrieval
    ↓
LLM (Claude / OpenAI)
    ↓
Final Answer (Context-aware)
```

---

## ⚙️ Technical Approach

### 1. Data Processing

* Documents are split into smaller chunks
* Each chunk is converted into embeddings using:

  * `sentence-transformers (all-MiniLM-L6-v2)`

### 2. Vector Storage (Endee)

* Embeddings are stored in Endee
* Each vector is associated with metadata (text content)

### 3. Query Processing

* User query is converted into embedding
* Endee performs **similarity search**
* Top-K relevant chunks are retrieved

### 4. Response Generation

* Retrieved context is passed to LLM
* LLM generates a **context-aware answer**

---

## 🗄️ Why Endee?

Endee is used as the core vector database because it provides:

* ⚡ High-performance vector search
* 📈 Scalability (handles large datasets efficiently)
* 🔍 Accurate similarity matching
* 🧩 Easy integration with AI pipelines

In this project, Endee is responsible for:

* Storing embeddings
* Performing semantic search
* Retrieving relevant documents

---

## 📂 Project Structure

```
endee/
│── app/
│   ├── rag_engine.py      # Core RAG logic
│   ├── app.py             # Streamlit UI
│   ├── cli.py             # CLI interface
│   ├── demo.py            # Demo script
│   └── data/              # Sample documents
│
│── Dockerfile
│── docker-compose.yml
│── requirements.txt
│── README.md
```

---

## 🚀 Setup Instructions

### 🔹 1. Clone Repository

```
git clone https://github.com/YOUR_USERNAME/endee.git
cd endee
```

---

### 🔹 2. Install Dependencies

```
python -m venv venv
venv\Scripts\activate   # Windows

pip install -r requirements.txt
pip install sentence-transformers fastapi uvicorn streamlit python-dotenv
```

---

### 🔹 3. Setup Environment Variables

Create a `.env` file:

```
OPENAI_API_KEY=your_api_key
ANTHROPIC_API_KEY=your_api_key
```

---

### 🔹 4. Run Endee (Docker)

```
docker-compose up
```

---

### 🔹 5. Run Application

#### ▶️ Streamlit UI

```
streamlit run app/app.py
```

#### ▶️ CLI Mode

```
python app/cli.py
```

---

## 🧪 Example Query

```
What is the difference between stack and queue?
```

### ✅ Output:

* Retrieves relevant documents
* Generates accurate, context-based answer

---

## 📸 Screenshots

### 🔹 Streamlit UI  
User asking a question and receiving an AI-generated response  

![Streamlit UI](screenshots/ui.png)

---

### 🔹 Upload Interface  
Adding new documents to the knowledge base  

![Upload](screenshots/upload.png)

---

### 🔹 Document Index View  
Displaying all indexed documents stored in Endee  

![Documents](screenshots/documents.png)


---

## 🔮 Future Improvements

* 📄 PDF / Document Upload Feature
* 💬 Chat history & memory
* 🌐 Web deployment (Render / Railway)
* 🧠 Multi-document support
* 📊 Analytics dashboard

---

## 🏁 Conclusion

This project demonstrates how **vector databases + LLMs** can be combined to build **real-world AI systems**.

It highlights:

* Practical use of RAG
* Efficient semantic search with Endee
* Scalable AI application design

---

## 👨‍💻 Author

**Anurag Kumar**
MCA Student | Aspiring Data Scientist

---

## ⭐ Acknowledgment

* Endee for vector database
* Sentence Transformers for embeddings
* OpenAI / Anthropic for LLM capabilities

---
