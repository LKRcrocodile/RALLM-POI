import os
import json
import numpy as np
from typing import List, Dict, Any
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import torch
import ast
import pickle
import argparse

try:
    from transformers import AutoTokenizer, AutoModel
except ImportError:
    AutoTokenizer = None
    AutoModel = None

class Embedder:
    def __init__(self, method='tfidf', device='cpu', model_name='bert-base-uncased'):
        self.method = method
        self.device = device
        self.model_name = model_name
        if method == 'tfidf':
            self.vectorizer = TfidfVectorizer()
        elif method == 'bert':
            if AutoTokenizer is None or AutoModel is None:
                raise ImportError("transformers library is required for BERT embeddings.")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name).to(device)
            self.model.eval()
        else:
            raise ValueError("Unknown embedding method")

    def fit(self, texts: List[str]):
        if self.method == 'tfidf':
            self.vectorizer.fit(texts)
        # No fitting needed for BERT

    def encode(self, texts: List[str]) -> np.ndarray:
        if self.method == 'tfidf':
            return self.vectorizer.transform(texts).toarray()
        elif self.method == 'bert':
            with torch.no_grad():
                embeddings = []
                for text in texts:
                    inputs = self.tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=128)
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                    outputs = self.model(**inputs)
                    # Use the [CLS] token representation
                    cls_emb = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                    embeddings.append(cls_emb.squeeze(0))
                return np.stack(embeddings)

class Retriever:
    def __init__(self, embedder: Embedder, top_k: int = 5, metric: str = 'cosine'):
        self.metric = metric
        self.embedder = embedder
        self.top_k = top_k

    def retrieve_embed(self, query_embeds: np.ndarray, corpus_embeds: np.ndarray) -> List[List[int]]:
        if self.metric == 'cosine':
            sims = cosine_similarity(query_embeds, corpus_embeds)
            topk_indices = np.argsort(-sims, axis=1)[:, :self.top_k]
            return topk_indices.tolist()
        else:
            raise ValueError("Unknown similarity metric")
    
    def retrieve_text(self, query_texts: List[str], corpus_texts: List[str]) -> List[List[int]]:
        # bm25
        if self.metric == 'bm25':
            from rank_bm25 import BM25Okapi
            bm25 = BM25Okapi(corpus_texts)
            top_k_texts = bm25.get_top_n(query_texts, corpus_texts, n=self.top_k)
            topk_indices = []
            for query_text in query_texts:
                topk_indices.append([corpus_texts.index(text) for text in top_k_texts if text in corpus_texts])
            return topk_indices
        # tfidf
        else:
            raise ValueError("Unknown similarity metric")



def load_list_of_lists(path: str) -> List[Any]:
    with open(path, 'rb') as f:
        return pickle.load(f)

def main(
    valid_hist_path: str,
    train_hist_path: str,
    valid_rec_path: str,
    train_rec_path: str,
    embed_method: str = 'tfidf',
    top_k: int = 5,
    device: str = 'cpu'
):
    # Load data
    valid_histories = load_list_of_lists(valid_hist_path)
    train_histories = load_list_of_lists(train_hist_path)
    train_recommendations = load_list_of_lists(train_rec_path)
    valid_recommendations = load_list_of_lists(valid_rec_path)


    valid_texts = [' '.join(map(str, h)) for h in valid_histories]
    train_texts = [' '.join(map(str, h)) for h in train_histories]

    # Build embedder
    embedder = Embedder(method=embed_method, device=device)
    if embed_method == 'tfidf':
        embedder.fit(train_texts + valid_texts)

    train_embeds = embedder.encode(train_texts)
    valid_embeds = embedder.encode(valid_texts)

    # Retrieve
    retriever = Retriever(embedder, top_k=top_k)
    topk_indices = retriever.retrieve(valid_embeds, train_embeds)
    print(f"Top {top_k} indices for each valid history: {topk_indices}")

    # Get corresponding recommendations
    results = []
    for idx, indices in enumerate(topk_indices):
        recs = [train_recommendations[i] for i in indices]
        results.append({
            'valid_index': idx,
            'valid_history': valid_histories[idx],
            'topk_train_indices': indices,
            'topk_train_histories': [train_histories[i] for i in indices],
            'topk_recommendations': recs,
            'gt_recommendation': valid_recommendations[idx],
        })

    # pretty print the results
    for result in results:
        print(f"Valid index: {result['valid_index']}")
        print(f"Valid history: {result['valid_history']}")
        print(f"Topk train indices: {result['topk_train_indices']}")
        print(f"Topk train histories: {result['topk_train_histories']}")
        print(f"Topk recommendations: {result['topk_recommendations']}")
        print(f"Ground truth recommendation: {result['gt_recommendation']}")
        print("-" * 50)

    # Calculate recall rate
    recall_count = 0
    for idx, result in enumerate(results):
        gt = result['gt_recommendation']
        topk_recs = result['topk_recommendations']
        # Flatten topk_recs if needed (in case each rec is a list)
        flat_topk_recs = [item for sublist in topk_recs for item in (sublist if isinstance(sublist, list) else [sublist])]
        if gt in flat_topk_recs:
            recall_count += 1
    recall_rate = recall_count / len(results)
    print(f"Recall@{top_k}: {recall_rate:.4f} | {recall_count}/{len(results)}")
    return recall_rate, results

def run_multiple_ks(
    valid_hist_path: str,
    train_hist_path: str,
    valid_rec_path: str,
    train_rec_path: str,
    embed_method: str = 'tfidf',
    similarity_metric= 'cosine',
    k_list = [1, 2, 3, 5, 10, 20],
    device: str = 'cpu'
):
    # Load data once
    valid_histories = load_list_of_lists(valid_hist_path)
    train_histories = load_list_of_lists(train_hist_path)
    train_recommendations = load_list_of_lists(train_rec_path)
    valid_recommendations = load_list_of_lists(valid_rec_path)

    valid_texts = [' '.join(map(str, h)) for h in valid_histories]
    train_texts = [' '.join(map(str, h)) for h in train_histories]

    # Build embedder
    embedder = Embedder(method=embed_method, device=device)
    if embed_method == 'tfidf':
        embedder.fit(train_texts + valid_texts)

    train_embeds = embedder.encode(train_texts)
    valid_embeds = embedder.encode(valid_texts)

    # # Compute similarity matrix once
    # # You can switch similarity metric here. For example, use dot product instead of cosine similarity:
    # # sims = np.dot(valid_embeds, train_embeds.T)

    # # Or use cosine similarity (default):
    # # sims = cosine_similarity(valid_embeds, train_embeds)

    # # If you want to use Euclidean distance (smaller is more similar, so we invert the sign for ranking):
    # sims = -euclidean_distances(valid_embeds, train_embeds)
    if similarity_metric == 'cosine':
        sims = cosine_similarity(valid_embeds, train_embeds)
    elif similarity_metric == 'euclidean':
        from sklearn.metrics.pairwise import euclidean_distances
        sims = -euclidean_distances(valid_embeds, train_embeds)
    elif similarity_metric == 'dot_product':
        sims = np.dot(valid_embeds, train_embeds.T)
    else:
        raise ValueError("Unknown similarity metric")
    recalls = []
    results_list = []

    for k in k_list:
        topk_indices = np.argsort(-sims, axis=1)[:, :k]
        results = []
        for idx, indices in enumerate(topk_indices):
            recs = [train_recommendations[i] for i in indices]
            results.append({
                'valid_index': idx,
                'valid_history': valid_histories[idx],
                'topk_train_indices': indices.tolist(),
                'topk_train_histories': [train_histories[i] for i in indices],
                'topk_recommendations': recs,
                'gt_recommendation': valid_recommendations[idx],
            })

        recall_count = 0
        for idx, result in enumerate(results):
            gt = result['gt_recommendation']
            topk_recs = result['topk_recommendations']
            flat_topk_recs = [item for sublist in topk_recs for item in (sublist if isinstance(sublist, list) else [sublist])]
            if gt in flat_topk_recs:
                recall_count += 1
        recall_rate = recall_count / len(results)
        print(f"Recall@{k}: {recall_rate:.4f} | {recall_count}/{len(results)}")
        recalls.append(recall_rate)
        results_list.append(results)

    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 5))
    plt.plot(k_list, recalls, marker='o')
    plt.title('Recall@k for different k values (embed: {}) (sim: {}) (total cases: {})'.format(embed_method, similarity_metric, len(results)))
    plt.xlabel('k')
    plt.ylabel('Recall')
    plt.grid(True)
    plt.savefig(f"recall_plot_{embed_method}_{similarity_metric}_sin.png")
    print(f"Recall rates for different k values: {recalls}")    
