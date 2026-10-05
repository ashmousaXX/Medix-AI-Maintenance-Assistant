import json
import math
import re
import chromadb

from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from config import (PROCESSED_DIR,VECTOR_DB_DIR,)
from groq_client import chat, HYDE_MODELS, LLMUnavailable

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
COLLECTION_NAME = "maintai_manuals"
MAX_DISTANCE = 0.5
BM25_TOP_K = 20
BM25_RESCUE_RANK = 3
BM25_RESCUE_MAX_DISTANCE = 0.65
PROBE_K = 12
DEVICE_MARGIN = 0.035
MAX_DEVICES_SEARCHED = 3

embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
_collection = None

def get_collection():
    global _collection
    if _collection is None:
        client = chromadb.PersistentClient(path=str(VECTOR_DB_DIR))
        _collection = client.get_collection(name=COLLECTION_NAME)
    return _collection

def _reset_caches():
    """Call after the vector DB is rebuilt."""
    global _collection
    _collection = None
    _bm25_cache.clear()
    _doc_embedding_cache.clear()

SYNONYM_EXPANSIONS = {
    "power supply": "power supply voltage supply internal supply voltage",
    "power problem": "voltage supply mains power",
    "blank screen": "blank display no display",
    "black screen": "blank display no display no picture dark screen",
    "screen is black": "blank display no display no picture dark screen",
    "display is black": "blank display no display no picture dark screen",
    "no picture": "blank display no display",
    "flow sensor": "flow transducer",
    "broken": "defective malfunction",
    "not working": "defective malfunction failure",
    "gas pressure": "gas supply pressure inlet pressure too low too high",
}

def expand_query(query, device_id=None):
    q = query.lower()
    extra = [v for k, v in SYNONYM_EXPANSIONS.items() if k in q]
    return query + (" " + " ".join(extra) if extra else "")

def get_available_devices():
    data = get_collection().get()
    devices = set()
    for metadata in data["metadatas"]:
        device_id = metadata.get("device_id", "")
        if device_id:
            devices.add(device_id)
    return sorted(list(devices))

def load_chunks():
    chunks_file = PROCESSED_DIR / "maintai_chunks.json"
    if not chunks_file.exists():
        raise FileNotFoundError(f"Chunks file not found: {chunks_file}")

    with open(chunks_file, "r", encoding="utf-8") as file:
        chunks = json.load(file)
    return chunks

def create_vector_database():
    print("=" * 70)
    print("CREATING VECTOR DATABASE")
    print("=" * 70)
    chunks = load_chunks()
    print(f"Loaded chunks: {len(chunks)}")
    VECTOR_DB_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(VECTOR_DB_DIR))
    try:
        client.delete_collection(COLLECTION_NAME)
        print("Old collection deleted")
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )
    texts = []
    ids = []
    metadatas = []
    for chunk in chunks:
        text = chunk.get("text", "").strip()
        if not text:
            continue
        chunk_id = chunk.get("chunk_id")
        if not chunk_id:
            continue

        metadata = {
            "device_id": str(chunk.get("device_id", "")),
            "device": str(chunk.get("device", "")),
            "manufacturer": str(chunk.get("manufacturer", "")),
            "page": int(chunk.get("page", 0)),
            "section": str(chunk.get("section", "")),
            "chunk_type": str(chunk.get("chunk_type", "text")),
            "error_code": (
                str(chunk["error_code"])
                if chunk.get("error_code") is not None
                else ""
            ),
        }
        texts.append(text)
        ids.append(chunk_id)
        metadatas.append(metadata)
    print(f"Documents prepared: {len(texts)}")

    embeddings = embedding_model.encode(
        texts,
        show_progress_bar=True,
        normalize_embeddings=True,
        batch_size=32,
    )

    CHROMA_BATCH_SIZE = 1000
    total = len(texts)
    for start in range(0, total, CHROMA_BATCH_SIZE):
        end = min(start + CHROMA_BATCH_SIZE, total)
        print(f"Adding batch {start} - {end} / {total}")
        collection.add(
            ids=ids[start:end],
            documents=texts[start:end],
            metadatas=metadatas[start:end],
            embeddings=embeddings[start:end].tolist(),
        )
    print()
    print(f"Stored vectors: {collection.count()}")
    print("=" * 70)
    _reset_caches()

def semantic_search(
    query,
    device_id=None,
    top_k=8,
    expand=False,
):
    collection = get_collection()
    search_query = expand_query(query, device_id=device_id) if expand else query
    query_embedding = embedding_model.encode(
        search_query,
        normalize_embeddings=True,
    )
    search_arguments = {
        "query_embeddings": [query_embedding.tolist()],
        "n_results": top_k,
    }

    if device_id:
        search_arguments["where"] = {"device_id": device_id}
    return collection.query(**search_arguments)

_hyde_cache = {}

HYDE_SYSTEM_PROMPT = (
    "Rewrite the user's symptom as a short technical phrase in the style "
    "of a medical-equipment service manual's troubleshooting table. One "
    "sentence. Keep every device and component word from the user's text "
    "exactly as written; do not correct, replace or invent words. Do not "
    "answer or solve anything, only rephrase."
)

def generate_hypothetical_passage(query):
    key = query.strip().lower()
    if key in _hyde_cache:
        return _hyde_cache[key]
    try:
        text, _model = chat(
            [
                {"role": "system", "content": HYDE_SYSTEM_PROMPT},
                {"role": "user", "content": query},
            ],
            HYDE_MODELS,
            max_tokens=200,
            temperature=0,
            label="hyde",
        )
    except LLMUnavailable as error:
        print(f"  HyDE rewrite skipped, using original query: {error}")
        return query

    text = text.strip().strip('"').strip()
    if not text or len(text) > 300:
        return query

    _hyde_cache[key] = text
    return text

_bm25_cache = {}

_STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "has", "have", "had",
    "my", "our", "your", "of", "to", "in", "on", "at", "it", "its", "and",
    "or", "does", "do", "not", "no", "with", "for", "this", "that", "i",
    "me", "we", "seems", "seem", "problem",
}

def _tokens(text):
    return [
        t for t in re.findall(r"[a-z0-9]+", text.lower())
        if t not in _STOPWORDS
    ]

def _get_bm25_index(device_id):
    if device_id in _bm25_cache:
        return _bm25_cache[device_id]

    data = get_collection().get(where={"device_id": device_id})
    texts = data["documents"]
    if not texts:
        _bm25_cache[device_id] = None
        return None
    entry = {
        "bm25": BM25Okapi([_tokens(text) or ["_"] for text in texts]),
        "ids": data["ids"],
        "texts": texts,
        "metadatas": data["metadatas"],
    }
    _bm25_cache[device_id] = entry
    return entry

def keyword_search(
    query,
    device_id,
    top_k=BM25_TOP_K,
):
    empty = {"ids": [], "documents": [], "metadatas": [], "scores": []}
    index = _get_bm25_index(device_id)
    if index is None:
        return empty

    tokenized_query = _tokens(expand_query(query))
    if not tokenized_query:
        return empty

    scores = index["bm25"].get_scores(tokenized_query)
    ranked_positions = sorted(
        range(len(scores)),
        key=lambda i: scores[i],
        reverse=True,
    )
    ranked_positions = [i for i in ranked_positions if scores[i] > 0][:top_k]
    return {
        "ids": [index["ids"][i] for i in ranked_positions],
        "documents": [index["texts"][i] for i in ranked_positions],
        "metadatas": [index["metadatas"][i] for i in ranked_positions],
        "scores": [scores[i] for i in ranked_positions],
    }
_doc_embedding_cache = {}

def _embed_docs(ids, documents):
    missing = [
        (doc_id, doc)
        for doc_id, doc in zip(ids, documents)
        if doc_id not in _doc_embedding_cache
    ]
    if missing:
        vectors = embedding_model.encode(
            [doc for _, doc in missing],
            normalize_embeddings=True,
            batch_size=32,
        )
        for (doc_id, _), vector in zip(missing, vectors):
            _doc_embedding_cache[doc_id] = vector
    return [_doc_embedding_cache[doc_id] for doc_id in ids]

def hybrid_search(
    query,
    device_id,
    top_k=8,
    hyde_query=None,
):
    if hyde_query is None:
        hyde_query = generate_hypothetical_passage(query)
    use_hyde = bool(hyde_query) and hyde_query.strip() != query.strip()
    semantic_top_k = max(top_k * 3, 24)
    empty_result = {
        "ids": [[]],
        "documents": [[]],
        "metadatas": [[]],
        "distances": [[]],
    }
    candidates = {}

    def _get(doc_id, document, metadata):
        if doc_id not in candidates:
            candidates[doc_id] = {
                "id": doc_id,
                "document": document,
                "metadata": metadata,
                "query_rank": None,
                "hyde_rank": None,
                "bm25_rank": None,
                "dist_query": None,
                "dist_hyde": None,
            }
        return candidates[doc_id]
    query_results = semantic_search(
        query=query,
        device_id=device_id,
        top_k=semantic_top_k,
        expand=True,
    )
    for rank, (doc_id, doc, meta) in enumerate(
        zip(
            query_results["ids"][0],
            query_results["documents"][0],
            query_results["metadatas"][0],
        ),
        start=1,
    ):
        _get(doc_id, doc, meta)["query_rank"] = rank

    if use_hyde:
        hyde_results = semantic_search(
            query=hyde_query,
            device_id=device_id,
            top_k=semantic_top_k,
            expand=False,
        )
        for rank, (doc_id, doc, meta) in enumerate(
            zip(
                hyde_results["ids"][0],
                hyde_results["documents"][0],
                hyde_results["metadatas"][0],
            ),
            start=1,
        ):
            _get(doc_id, doc, meta)["hyde_rank"] = rank

    keyword_results = keyword_search(
        query=query,
        device_id=device_id,
        top_k=BM25_TOP_K,
    )
    for rank, (doc_id, doc, meta) in enumerate(
        zip(
            keyword_results["ids"],
            keyword_results["documents"],
            keyword_results["metadatas"],
        ),
        start=1,
    ):
        _get(doc_id, doc, meta)["bm25_rank"] = rank

    if not candidates:
        return empty_result

    candidate_list = list(candidates.values())
    doc_embeddings = _embed_docs(
        [c["id"] for c in candidate_list],
        [c["document"] for c in candidate_list],
    )
    query_embedding = embedding_model.encode(query, normalize_embeddings=True)
    hyde_embedding = (
        embedding_model.encode(hyde_query, normalize_embeddings=True)
        if use_hyde
        else None
    )
    for candidate, doc_embedding in zip(candidate_list, doc_embeddings):
        candidate["dist_query"] = float(1.0 - (doc_embedding @ query_embedding))
        if hyde_embedding is not None:
            candidate["dist_hyde"] = float(1.0 - (doc_embedding @ hyde_embedding))
        distances = [
            d for d in (candidate["dist_query"], candidate["dist_hyde"])
            if d is not None
        ]
        candidate["semantic_distance"] = min(distances)

    filtered_candidates = [
        c for c in candidate_list
        if c["semantic_distance"] <= MAX_DISTANCE
        or (
            c["bm25_rank"] is not None
            and c["bm25_rank"] <= BM25_RESCUE_RANK
            and c["semantic_distance"] <= BM25_RESCUE_MAX_DISTANCE
        )
    ]
    if not filtered_candidates:
        return empty_result

    RRF_K = 60
    QUERY_WEIGHT = 1.5
    HYDE_WEIGHT = 1.0
    BM25_WEIGHT = 1.0

    for candidate in filtered_candidates:
        score = 0.0
        if candidate["query_rank"] is not None:
            score += QUERY_WEIGHT / (RRF_K + candidate["query_rank"])
        if candidate["hyde_rank"] is not None:
            score += HYDE_WEIGHT / (RRF_K + candidate["hyde_rank"])
        if candidate["bm25_rank"] is not None:
            score += BM25_WEIGHT / (RRF_K + candidate["bm25_rank"])
        candidate["rrf_score"] = score
    ranked_candidates = sorted(
        filtered_candidates,
        key=lambda item: (item["rrf_score"], -item["semantic_distance"]),
        reverse=True,
    )[:top_k]
    true_best_distance = min(
        c["semantic_distance"] for c in filtered_candidates
    )

    return {
        "ids": [[c["id"] for c in ranked_candidates]],
        "documents": [[c["document"] for c in ranked_candidates]],
        "metadatas": [[c["metadata"] for c in ranked_candidates]],
        "distances": [[c["semantic_distance"] for c in ranked_candidates]],
        "true_best_distance": true_best_distance,
    }

def detect_candidate_devices(query):
    probe = semantic_search(query=query, device_id=None, top_k=PROBE_K)
    if not probe["documents"][0]:
        return [], None, probe

    best_per_device = {}
    for metadata, distance in zip(probe["metadatas"][0], probe["distances"][0]):
        device = metadata.get("device_id", "")
        if device and (
            device not in best_per_device or distance < best_per_device[device]
        ):
            best_per_device[device] = distance

    if not best_per_device:
        return [], None, probe

    ranked = sorted(best_per_device.items(), key=lambda item: item[1])
    best_distance = ranked[0][1]

    if best_distance > MAX_DISTANCE:
        return [], best_distance, probe

    devices = [
        device for device, distance in ranked
        if distance <= best_distance + DEVICE_MARGIN
        and distance <= MAX_DISTANCE
    ][:MAX_DEVICES_SEARCHED]
    return devices, best_distance, probe

def search_devices(query, devices, top_k, hyde_query):
    per_device = []
    for device in devices:
        result = hybrid_search(
            query=query,
            device_id=device,
            top_k=top_k,
            hyde_query=hyde_query,
        )
        if result["documents"][0]:
            per_device.append(result)

    if not per_device:
        return {
            "ids": [[]],
            "documents": [[]],
            "metadatas": [[]],
            "distances": [[]],
        }

    if len(per_device) == 1:
        return per_device[0]

    per_device.sort(key=lambda r: r.get("true_best_distance", 1.0))
    take = max(3, math.ceil(top_k / len(per_device)))
    merged = {
        "ids": [[]],
        "documents": [[]],
        "metadatas": [[]],
        "distances": [[]],
        "true_best_distance": min(
            r.get("true_best_distance", 1.0) for r in per_device
        ),
    }
    for result in per_device:
        for key in ("ids", "documents", "metadatas", "distances"):
            merged[key][0].extend(result[key][0][:take])
    return merged

MAX_EXACT_MATCHES = 5

def exact_error_search(
    error_code,
    device_id=None,
):
    collection = get_collection()
    filters = [{"error_code": str(error_code)}]

    if device_id:
        filters.append({"device_id": device_id})
    if len(filters) == 1:
        where_clause = filters[0]
    else:
        where_clause = {"$and": filters}
    results = collection.get(where=where_clause)
    if len(results.get("ids", [])) > MAX_EXACT_MATCHES:
        for key in ("ids", "documents", "metadatas"):
            if key in results and results[key]:
                results[key] = results[key][:MAX_EXACT_MATCHES]
    return results

FALSE_POSITIVE_ERROR_CODES = {"382"}

def detect_error_code(query):
    patterns = [
        r"\berror\s+code\s+(\d{1,5})\b",
        r"\berror\s+(\d{1,5})\b",
        r"\bcode\s+(\d{1,5})\b",
        r"\bE[-\s]?(\d{1,5})\b",
        r"\bERR[-\s]?(\d{1,5})\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, query, re.IGNORECASE)
        if match:
            code = match.group(1)
            if code in FALSE_POSITIVE_ERROR_CODES:
                return None
            return code
    return None

def retrieve(
    query,
    device_id=None,
    error_code=None,
    top_k=8,
):
    if error_code is None:
        error_code = detect_error_code(query)
    if error_code:
        exact_results = exact_error_search(error_code, device_id)
        if exact_results["ids"]:
            return {
                "retrieval_type": "exact_error",
                "detected_error_code": error_code,
                "detected_device": exact_results["metadatas"][0].get(
                    "device_id", ""
                ),
                "results": exact_results,
            }
    if device_id:
        devices = [device_id]
    else:
        devices, _probe_distance, probe_results = detect_candidate_devices(query)
        if not devices:
            return {
                "retrieval_type": "not_found",
                "detected_error_code": error_code,
                "detected_device": None,
                "candidate_devices": [],
                "results": probe_results,
            }
    hyde_query = generate_hypothetical_passage(query)
    print(f"HyDE rewrite: {hyde_query}")
    print(f"Searching devices: {devices}")

    semantic_results = search_devices(
        query=query,
        devices=devices,
        top_k=top_k,
        hyde_query=hyde_query,
    )
    if not semantic_results["documents"][0]:
        return {
            "retrieval_type": "not_found",
            "detected_error_code": error_code,
            "detected_device": None,
            "candidate_devices": devices,
            "results": semantic_results,
        }
    best_distance = semantic_results.get(
        "true_best_distance",
        semantic_results["distances"][0][0],
    )
    print(f"Best semantic distance: {best_distance:.4f}")
    print(f"Maximum allowed distance: {MAX_DISTANCE:.4f}")

    if best_distance > MAX_DISTANCE:
        return {
            "retrieval_type": "not_found",
            "detected_error_code": error_code,
            "detected_device": None,
            "candidate_devices": devices,
            "results": semantic_results,
        }
    detected_device = semantic_results["metadatas"][0][0].get("device_id", "")
    return {
        "retrieval_type": "semantic",
        "detected_error_code": error_code,
        "detected_device": detected_device,
        "candidate_devices": devices,
        "results": semantic_results,
    }

def test_retrieve():
    query = "The ventilator has a power supply problem"
    result = retrieve(
        query=query,
        top_k=8,
    )
    print("=" * 70)
    print("TYPE:", result["retrieval_type"])
    print("DEVICE:", result["detected_device"])
    for doc in result["results"]["documents"][0]:
        print("-" * 70)
        print(doc[:300])

if __name__ == "__main__":
    test_retrieve()