from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from pinecone import Pinecone
from langchain_ollama import OllamaEmbeddings
from openai import OpenAI
from functools import lru_cache
import json

app = Flask(__name__)
CORS(app)

import os
from dotenv import load_dotenv
load_dotenv() 

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OPENAI_API_KEY   = os.getenv("OPENAI_API_KEY")



# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index("know")

# Initialize OpenAI
try:
    openai_client = OpenAI(api_key=OPENAI_API_KEY)
    print("✓ OpenAI client initialized successfully")
except Exception as e:
    print(f"✗ Error initializing OpenAI client: {e}")
    openai_client = None

def get_embeddings(text: str):
    """Get embeddings using OllamaEmbeddings with bge-m3 model."""
    embeddings = OllamaEmbeddings(
        model="bge-m3"
    )
    return embeddings.embed_query(text)

def search(query: str, top_k: int = 5, namespace: str = "knows", filter_dict: dict = None):
    """
    Search Pinecone using the query and filter parameters.
    Returns Pinecone query results.
    """
    query_params = {
        "namespace": namespace,
        "vector": get_embeddings(query), 
        "top_k": top_k,
        "include_metadata": True,
        "include_values": False
    }
    
    # Add filter if provided
    if filter_dict is not None:
        query_params["filter"] = filter_dict
    
    results = index.query(**query_params)
    return results

@lru_cache(maxsize=128)
def cached_search(query: str, top_k: int, namespace: str, filter_str: str):
    """
    Cached version of search function.
    Uses LRU cache to avoid redundant Pinecone queries.
    """
    filter_dict = json.loads(filter_str) if filter_str != "null" else None
    return search(query, top_k, namespace, filter_dict)

def synthesize_answer(query: str, context: str):
   
    if not openai_client:
        raise ValueError("OpenAI client not initialized")
    
    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant that answers questions based only on the provided context. "
                    "If the context is insufficient, say that clearly."
                ),
            },
            {
                "role": "user",
                "content": f"Question: {query}\n\nContext:\n{context}\n\nAnswer clearly and concisely."
            }
        ],
        temperature=0.2,
        max_tokens=900
    )
    
    return response.choices[0].message.content

@app.route('/')
def home():
    return send_from_directory('.', 'index.html')

@app.route('/api/query', methods=['POST'])
def query():
    try:
        data = request.json
        if not data:
            return jsonify({'error': 'No data provided'}), 400
            
        query_text = data.get('query', '').strip()
        year_filter = data.get('year', None)
        top_k = int(data.get('top_k', 5))
        namespace = data.get('namespace', 'knows')
        
        if not query_text:
            return jsonify({'error': 'Query is required'}), 400
        
        # Check if OpenAI client is initialized
        if not openai_client:
            return jsonify({'error': 'OpenAI client not initialized'}), 500
        
        # Build filter if year is provided (same format as test.ipynb)
        filter_dict = None
        if year_filter is not None and year_filter != "" and year_filter != "null" and year_filter != "all":
            try:
                year_int = int(year_filter)
                # Use "year" field name as in test.ipynb: filter={"year":{"$eq": 2014}}
                filter_dict = {"year": {"$eq": year_int}}
                print(f"Year filter applied: {year_int}")
            except (ValueError, TypeError):
                filter_dict = None
        
        print(f"Query: {query_text}")
        print(f"Parameters: top_k={top_k}, namespace={namespace}, year_filter={year_filter}, filter={filter_dict}")
        
        # Search Pinecone using cached search
        try:
            print("Searching Pinecone...")
            # Convert filter_dict to JSON string for caching
            filter_str = json.dumps(filter_dict) if filter_dict else "null"
            results = cached_search(query_text, top_k, namespace, filter_str)
            print(f"Pinecone returned {len(results.matches) if results.matches else 0} matches")
        except Exception as e:
            import traceback
            print(f"Error searching Pinecone: {e}")
            print(f"Traceback: {traceback.format_exc()}")
            return jsonify({'error': f'Search failed: {str(e)}'}), 500
        
        # Extract chunks from matches (same as test.ipynb Cell 7)
        matches = results.get("matches", [])
        chunks = []
        match_details = []
        years_found = set()
        
        for m in matches:
            if "metadata" in m and "text" in m["metadata"]:
                chunks.append(m["metadata"]["text"])
                metadata = m.get("metadata", {})
                match_details.append({
                    'id': m.get('id', ''),
                    'score': m.get('score', 0),
                    'text': m["metadata"]["text"][:200],  # Preview
                    'metadata': metadata
                })
                # Extract year from metadata (using "year" field name as in test.ipynb)
                year_value = metadata.get('year') or metadata.get('publication_year')
                if year_value is not None:
                    try:
                        years_found.add(int(year_value))
                    except (ValueError, TypeError):
                        years_found.add(year_value)
        
        if not chunks:
            if year_filter and year_filter != "all":
                answer_msg = f"I couldn't find any information from documents published in {year_filter} to answer your question. Please try selecting a different year or 'All Years'."
            else:
                answer_msg = "I couldn't find any relevant information to answer your question."
            
            return jsonify({
                'answer': answer_msg,
                'matches': [],
                'query': query_text,
                'year_filter': year_filter,
                'years_in_results': [],
                'match_count': 0
            })
        
        # Combine chunks into context 
        context = "\n\n".join(
            f"[Chunk {i+1}]\n{chunk}"
            for i, chunk in enumerate(chunks[:top_k])
        )
        
        print(f"Context created with {len(chunks)} chunks, total length: {len(context)} characters")
        
        # Synthesize answer using OpenAI 
        try:
            print("Synthesizing answer with OpenAI...")
            answer = synthesize_answer(query_text, context)
            print(f"Answer generated (length: {len(answer)})")
        except Exception as e:
            import traceback
            print(f"Error synthesizing answer: {e}")
            print(f"Traceback: {traceback.format_exc()}")
            return jsonify({'error': f'Answer synthesis failed: {str(e)}'}), 500
        
        return jsonify({
            'answer': answer,
            'matches': match_details,
            'query': query_text,
            'year_filter': year_filter,
            'years_in_results': sorted(list(years_found)) if years_found else [],
            'match_count': len(chunks),
            'top_k': top_k,
            'namespace': namespace
        })
        
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print("=" * 60)
        print(f"ERROR in query endpoint: {e}")
        print("=" * 60)
        print(f"Traceback: {error_trace}")
        print("=" * 60)
        
        # Return more detailed error message
        error_message = str(e)
        if "authentication" in error_message.lower() or "invalid_api_key" in error_message.lower():
            error_message = "OpenAI API key invalid. Please check your API key in app_rag.py."
        elif "rate_limit" in error_message.lower():
            error_message = "OpenAI API rate limit exceeded. Please try again later."
        elif "insufficient_quota" in error_message.lower():
            error_message = "OpenAI API quota exceeded. Please check your account billing."
        
        return jsonify({'error': error_message}), 500

@app.route('/api/years', methods=['GET'])
def get_years():
    """Get list of available publication years from 2013 to 2025."""
    try:
        years = list(range(2013, 2026))  # 2013 to 2025
        return jsonify({'years': years})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/<path:path>')
def serve_static(path):
    """Serve static files (CSS, JS, etc.)"""
    if path.startswith('api/'):
        return jsonify({'error': 'Not found'}), 404
    return send_from_directory('.', path)

if __name__ == '__main__':
    port = 5004
    app.run(debug=True, port=port, host="127.0.0.1")

