from pathlib import Path
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


base_path = Path(__file__).parent.parent

data_path = base_path / "data" / "analysed_news.csv"
embeddings_path = base_path / "data" / "headline_embeddings.npy"

model = SentenceTransformer("all-MiniLM-L6-v2")

df = pd.read_csv(data_path)

headlines = df["headline"].fillna("").tolist()

print("Creating headline embeddings...")

embeddings = model.encode(
    headlines,
    convert_to_numpy=True,
    normalize_embeddings=True
)

np.save(embeddings_path, embeddings)

print(f"Saved embeddings to: {embeddings_path}")
print(f"Embedded {len(headlines)} headlines")