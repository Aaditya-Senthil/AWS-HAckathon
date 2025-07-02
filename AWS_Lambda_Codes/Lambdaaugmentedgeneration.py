import json
import boto3
import os
import requests
import math
import time
import re
from boto3.dynamodb.conditions import Key

dynamodb = boto3.resource('dynamodb')
table = dynamodb.Table(os.environ['DYNAMODB_TABLE'])

GROQ_API_KEY = os.environ['GROQ_API_KEY']
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')

def lambda_handler(event, context):
    try:
        if 'body' in event:
            body = json.loads(event['body'])
        else:
            body = event
            
        query = body.get('query', '')
        user_id = body.get('user_id', '')
        
        if not query:
            return {
                'statusCode': 400,
                'headers': {'Content-Type': 'application/json', 'Access-Control-Allow-Origin': '*'},
                'body': json.dumps({'error': 'Query is required'})
            }
        
        search_results = enhanced_search(query, user_id)
        context_text = prepare_enhanced_context(search_results)
        groq_response = generate_groq_response(query, context_text)
        
        final_response = refine_response(groq_response)
        
        return {
            'statusCode': 200,
            'headers': {'Content-Type': 'application/json', 'Access-Control-Allow-Origin': '*'},
            'body': json.dumps({
                'response': final_response,
                'sources': [{'filename': chunk['filename'], 'score': round(chunk['final_score'], 3)} for chunk in search_results]
            })
        }
    except Exception as e:
        print(f"Error: {str(e)}")
        return {
            'statusCode': 500,
            'headers': {'Content-Type': 'application/json', 'Access-Control-Allow-Origin': '*'},
            'body': json.dumps({'error': str(e)})
        }

def refine_response(groq_response):
    if not OPENAI_API_KEY:
        return groq_response
    
    try:
        headers = {
            'Authorization': f'Bearer {OPENAI_API_KEY}',
            'Content-Type': 'application/json'
        }
        
        prompt = f"""Make this response more precise and sharp while keeping all the information:

{groq_response}

Requirements:
- Keep it concise and clear
- Remove unnecessary words  
- Make it more professional
- Don't add new information"""

        payload = {
            'model': 'gpt-3.5-turbo',
            'messages': [{'role': 'user', 'content': prompt}],
            'max_tokens': 500,
            'temperature': 0.2
        }
        
        response = requests.post('https://api.openai.com/v1/chat/completions', 
                               headers=headers, json=payload, timeout=20)
        
        if response.status_code == 200:
            result = response.json()
            return result['choices'][0]['message']['content']
        else:
            return groq_response
            
    except Exception as e:
        print(f"Refinement failed: {str(e)}")
        return groq_response

def enhanced_search(query, user_id, top_k=5):
    query_embedding = generate_embedding(query)
    keywords = extract_keywords(query)
    documents = get_all_documents(user_id)
    scored_documents = []
    
    for doc in documents:
        semantic_scores = []
        
        if 'extracted_data_embedding' in doc and doc['extracted_data_embedding']:
            embedding = [float(x) for x in doc['extracted_data_embedding']]
            semantic_score = cosine_similarity(query_embedding, embedding)
            content = doc.get('extracted_data', '')
            semantic_scores.append({
                'score': semantic_score,
                'content': content,
                'type': 'extracted'
            })
        
        if 'parsed_content_embedding' in doc and doc['parsed_content_embedding']:
            embedding = [float(x) for x in doc['parsed_content_embedding']]
            semantic_score = cosine_similarity(query_embedding, embedding)
            content = doc.get('parsed_content', '')
            semantic_scores.append({
                'score': semantic_score,
                'content': content,
                'type': 'parsed'
            })
        
        for sem_result in semantic_scores:
            content = sem_result['content']
            keyword_score = calculate_keyword_score(content, keywords)
            length_score = calculate_length_score(content)
            position_score = calculate_position_score(content, keywords)
            exact_match_score = calculate_exact_match_score(content, query)
            
            final_score = (
                sem_result['score'] * 0.4 +
                keyword_score * 0.25 +
                exact_match_score * 0.20 +
                position_score * 0.10 +
                length_score * 0.05
            )
            
            scored_documents.append({
                'job_id': doc['job_id'],
                'filename': doc.get('filename', 'Unknown'),
                'content': content,
                'final_score': final_score,
                'semantic_score': sem_result['score'],
                'keyword_score': keyword_score,
                'exact_match_score': exact_match_score,
                'type': sem_result['type']
            })
    
    scored_documents.sort(key=lambda x: x['final_score'], reverse=True)
    
    seen_content = set()
    unique_results = []
    
    for doc in scored_documents:
        content_hash = hash(doc['content'][:100])
        if content_hash not in seen_content:
            seen_content.add(content_hash)
            unique_results.append(doc)
            if len(unique_results) >= top_k:
                break
    
    return unique_results

def extract_keywords(query):
    stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'what', 'how', 'when', 'where', 'why', 'who'}
    words = re.findall(r'\b\w+\b', query.lower())
    keywords = [word for word in words if word not in stop_words and len(word) > 2]
    
    return keywords

def calculate_keyword_score(content, keywords):
    if not keywords or not content:
        return 0.0
    
    content_lower = content.lower()
    total_score = 0.0
    
    for keyword in keywords:
        count = content_lower.count(keyword)
        if count > 0:
            keyword_score = min(count * 0.1, 1.0)
            total_score += keyword_score
    
    return min(total_score / len(keywords), 1.0)

def calculate_exact_match_score(content, query):
    if not query or not content:
        return 0.0
    
    content_lower = content.lower()
    query_lower = query.lower()
    
    if query_lower in content_lower:
        return 1.0
    
    query_words = query_lower.split()
    if len(query_words) >= 3:
        for i in range(len(query_words) - 2):
            phrase = ' '.join(query_words[i:i+3])
            if phrase in content_lower:
                return 0.7
    
    return 0.0

def calculate_position_score(content, keywords):
    if not keywords or not content:
        return 0.0
    
    content_lower = content.lower()
    position_scores = []
    
    for keyword in keywords:
        pos = content_lower.find(keyword)
        if pos != -1:
            position_score = max(0, 1.0 - (pos / len(content_lower)))
            position_scores.append(position_score)
    
    return sum(position_scores) / len(keywords) if position_scores else 0.0

def calculate_length_score(content):
    if not content:
        return 0.0
    
    length = len(content)
    
    if 100 <= length <= 2000:
        return 1.0
    elif length < 100:
        return length / 100.0
    else:
        return max(0.5, 2000 / length)

def get_all_documents(user_id):
    try:
        scan_kwargs = {}
        if user_id:
            scan_kwargs['FilterExpression'] = Key('user_id').eq(user_id)
        scan_kwargs['ProjectionExpression'] = 'job_id, filename, extracted_data, parsed_content, extracted_data_embedding, parsed_content_embedding'
        
        response = table.scan(**scan_kwargs)
        return response['Items']
    except Exception as e:
        print(f"Error getting documents: {str(e)}")
        return []

def prepare_enhanced_context(chunks):
    context_parts = []
    
    for i, chunk in enumerate(chunks, 1):
        content = chunk['content'][:1500]
        relevance_info = f"[Relevance: {chunk['final_score']:.2f}]"
        
        context_parts.append(
            f"Document {i} ({chunk['filename']}) {relevance_info}:\n{content}"
        )
    
    return "\n\n".join(context_parts)

def generate_embedding(text, max_retries=3):
    url = "https://api-inference.huggingface.co/pipeline/feature-extraction/sentence-transformers/all-MiniLM-L6-v2"
    
    payload = {"inputs": text}
    
    for attempt in range(max_retries):
        try:
            response = requests.post(url, json=payload, timeout=30)
            
            if response.status_code == 503:
                print(f"Model loading, waiting 10 seconds... (attempt {attempt + 1})")
                time.sleep(10)
                continue
                
            response.raise_for_status()
            embedding = response.json()
            
            if isinstance(embedding, list):
                if len(embedding) > 0 and isinstance(embedding[0], list):
                    return embedding[0]
                else:
                    return embedding
            else:
                raise ValueError("Unexpected embedding format")
                
        except requests.exceptions.RequestException as e:
            print(f"Attempt {attempt + 1} failed: {str(e)}")
            if attempt == max_retries - 1:
                print("All attempts failed, using simple fallback embedding")
                return generate_simple_embedding(text)
            time.sleep(2 ** attempt)
    
    return generate_simple_embedding(text)

def generate_simple_embedding(text, size=384):
    words = text.lower().split()
    embedding = [0.0] * size
    
    for word in words:
        word = ''.join(c for c in word if c.isalnum())
        if len(word) > 2:
            for i, char in enumerate(word[:10]):
                idx = (hash(word + str(i)) % size)
                embedding[idx] += 1.0
    
    magnitude = math.sqrt(sum(x * x for x in embedding))
    if magnitude > 0:
        embedding = [x / magnitude for x in embedding]
    
    return embedding

def cosine_similarity(vec1, vec2):
    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    magnitude1 = math.sqrt(sum(a * a for a in vec1))
    magnitude2 = math.sqrt(sum(b * b for b in vec2))
    if magnitude1 == 0 or magnitude2 == 0:
        return 0.0
    return dot_product / (magnitude1 * magnitude2)

def generate_groq_response(query, context):
    try:
        headers = {
            'Authorization': f'Bearer {GROQ_API_KEY}',
            'Content-Type': 'application/json'
        }
        
        prompt = f"""You are a helpful assistant that provides precise, accurate answers based on the provided context documents.

IMPORTANT INSTRUCTIONS:
1. Answer ONLY based on the information in the context documents below
2. If the context doesn't contain enough information, clearly state this
3. Cite specific document numbers when referencing information
4. Be concise but comprehensive
5. If multiple documents have conflicting information, mention this

Context Documents (ordered by relevance):
{context}

User Question: {query}

Provide a detailed answer based on the context above. If the information is not available in the context, say so clearly."""
        
        payload = {
            'model': 'llama3-8b-8192',
            'messages': [{'role': 'user', 'content': prompt}],
            'max_tokens': 1200,
            'temperature': 0.3
        }
        
        response = requests.post(GROQ_API_URL, headers=headers, json=payload)
        response.raise_for_status()
        
        result = response.json()
        return result['choices'][0]['message']['content']
    except Exception as e:
        print(f"Error with Groq API: {str(e)}")
        raise e
