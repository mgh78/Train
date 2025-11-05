# Document Q&A Chatbot - RAG System

A Retrieval-Augmented Generation (RAG) system that allows users to query PDF documents using natural language questions. The system uses vector embeddings for semantic search and OpenAI for answer generation, with the ability to filter documents by publication year.

## 🚀 Quick Start

Follow these steps to set up and run the application:

### Step 1: Install Ollama

Install Ollama on your system:

**macOS/Linux:**
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

**Windows:**
Download and install from [https://ollama.com/download](https://ollama.com/download)

After installation, pull the required model:
```bash
ollama pull bge-m3
```

### Step 2: Install Dependencies

Install the required Python dependencies from `requirements.txt`:

```bash
pip install -r requirements.txt
```

### Step 3: Set Up API Keys

Create a `.env` file in the project root directory with your API keys:

```bash
PINECONE_API_KEY="your_pinecone_api_key_here"
OPENAI_API_KEY="your_openai_api_key_here"
```

### Step 4: Start the Application

Run the Flask application from the terminal:

```bash
python app_rag.py
```

### Step 5: Access the Application

Open your browser and navigate to:
```
http://localhost:5004
```

---

## 📋 Prerequisites

- **Python** (version 3.8+)
- **Ollama** installed and running locally with the bge-m3 model pulled
- **Pinecone Account** with an API key
- **OpenAI Account** with an API key


## 🏗️ Architecture

1. **Query Processing**: User questions are converted to embeddings using Ollama's bge-m3 model
2. **Vector Search**: Similar document chunks are retrieved from Pinecone
3. **Filtering**: Optional year-based filtering using Pinecone metadata filters
4. **Answer Generation**: Retrieved chunks are sent to OpenAI for answer synthesis
5. **Caching**: Results are cached using LRU cache to avoid redundant API calls

## 📄 Document Indexing

The document indexing process (parsing PDFs, chunking, vectorizing, and storing in Pinecone) is implemented in `test.ipynb`. This notebook contains the complete pipeline for preparing documents for the RAG system:

1. **PDF Parsing**: PDFs are loaded and parsed using `PyPDFDirectoryLoader`
2. **Metadata Cleaning**: Extra metadata (author, creator, etc.) is dropped
3. **Publication Year Mapping**: Each PDF is mapped to its publication year from a mapping file
4. **Chunking**: Documents are split into chunks using `RecursiveCharacterTextSplitter` (chunk_size=2000, chunk_overlap=200)
5. **Vectorization**: Text chunks are converted to embeddings using Ollama's bge-m3 model
6. **Storage**: Vectors are stored in Pinecone with metadata including text, page number, and publication year

**Note**: Run `test.ipynb` to index your PDF documents before using the RAG system. The indexed documents will be available for querying through the Flask application.

## ⚙️ Configuration


### Pinecone Configuration

- **Index Name**: `"know"` (must exist in your Pinecone account)
- **Namespace**: `"knows"` (default namespace for queries)
- **Vector Dimension**: 1024 (for bge-m3 embeddings)

### OpenAI Configuration

- **Model**: `gpt-4o-mini`
- **Temperature**: `0.2`
- **Max Tokens**: `900`

## 📡 API Endpoints

### POST `/api/query`

Query the RAG system with a question.

**Request Body:**
```json
{
  "query": "Your question here",
  "top_k": 5,
  "year": 2020,
  "namespace": "knows"
}
```

**Parameters:**
- `query` (required): The question to ask
- `top_k` (optional): Number of document chunks to retrieve (default: 5)
- `year` (optional): Filter by publication year. Use `null` or `"all"` for no filter
- `namespace` (optional): Pinecone namespace (default: "knows")

### GET `/api/years`

Get list of available publication years (2013-2025).

## 📁 Project Structure

```
Document_rag/
├── app_rag.py          # Main Flask application
├── test.ipynb          # Jupyter notebook for PDF parsing, chunking, and indexing
├── index.html          # Frontend HTML
├── script.js           # Frontend JavaScript
├── style.css           # Frontend styles
├── requirements.txt    # Python dependencies
├── .env                # Environment variables (API keys) - create this file
├── .gitignore         # Git ignore rules
└── README.md          # This file
```


### Issue: Ollama model not found
```bash
# Pull the model
ollama pull bge-m3
# Verify: ollama list
```


### Issue: Invalid API Key
- Verify API keys are set correctly in your `.env` file
- Check Pinecone API key starts with `pcsk_`
- Check OpenAI API key starts with `sk-`
- Ensure no extra spaces in the API key strings
- Make sure the `.env` file is in the project root directory

## 📝 Notes

- The application uses Ollama embeddings (bge-m3) which run locally
- Results are cached using LRU cache (128 entries) for performance
- The system does not implement rate limiting or authentication
- For production, consider adding security measures

