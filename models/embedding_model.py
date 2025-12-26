from sentence_transformers import SentenceTransformer

_model = None  # global cache

def get_model():
    global _model
    if _model is None:
        print("🔹 Loading embedding model (first time)...")
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model

def get_embedding(text):
    model = get_model()
    return model.encode(text)
