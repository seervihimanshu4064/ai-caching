from flask import Flask, request
from collections import OrderedDict
import time, hashlib, re, numpy as np
import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'  # Warnings off
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
model = SentenceTransformer('all-MiniLM-L6-v2')

# Globals
total_requests = hits = processed_tokens = 0
exact_cache = OrderedDict()
semantic_cache = []
MAX_SIZE = 1500
TTL = 86400

def normalize(q):
    return re.sub(r'[^\w\s]', '', q.lower().strip())

@app.route('/', methods=['POST'])  # << YE LINE CHECK! methods=['POST']
def query():
    global total_requests, hits, processed_tokens
    start = time.time()
    data = request.json
    query = data.get('query', '')
    if not query:
        return {'error': 'No query'}, 400
    
    norm_q = normalize(query)
    key = hashlib.md5(norm_q.encode()).hexdigest()
    cached = False
    
    # Exact
    if key in exact_cache and time.time() - exact_cache[key]['ts'] < TTL:
        exact_cache.move_to_end(key)
        cached = True
    else:
        # Semantic
        q_emb = model.encode([norm_q])[0]
        for entry in semantic_cache:
            if time.time() - entry['ts'] < TTL:
                sim = cosine_similarity([q_emb], [entry['emb']])[0][0]
                if sim > 0.95:
                    exact_cache[key] = {'resp': entry['resp'], 'ts': time.time()}
                    exact_cache.move_to_end(key)
                    cached = True
                    break
    
    if cached:
        answer = exact_cache[key]['resp']
    else:
        time.sleep(1.5)  # Fake LLM
        answer = f"AI answer for: {query}"
        ts = time.time()
        exact_cache[key] = {'resp': answer, 'ts': ts}
        q_emb = model.encode([norm_q])[0]
        semantic_cache.append({'emb': q_emb, 'resp': answer, 'ts': ts})
        
        # Evict
        if len(exact_cache) > MAX_SIZE:
            exact_cache.popitem(last=False)
        if len(semantic_cache) > MAX_SIZE:
            semantic_cache.pop(0)
    
    total_requests += 1
    if cached:
        hits += 1
    else:
        processed_tokens += 800
    
    latency = int((time.time() - start) * 1000)
    return {
        'answer': answer,
        'cached': cached,
        'latency': latency,
        'cacheKey': key
    }

@app.route('/analytics')  # << YE GET ke liye
def analytics():
    rate = hits / max(total_requests, 1)
    saved = (11462*800 - processed_tokens) * 0.60 / 1e6
    return {
        'hitRate': round(rate, 2),
        'totalRequests': total_requests,
        'cacheHits': hits,
        'cacheMisses': total_requests - hits,
        'cacheSize': len(exact_cache) + len(semantic_cache),
        'costSavings': round(saved, 2),
        'savingsPercent': round(rate * 100),
        'strategies': ['exact match', 'semantic >0.95', 'LRU', 'TTL 24h']
    }

if __name__ == '__main__':
    app.run(debug=True)
