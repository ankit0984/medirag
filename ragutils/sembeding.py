import torch
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer
import os


class EmbeddedRag:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                 local_dir: str = "./hf_models/all-MiniLM-L6-v2"):
        self.MODEL_NAME = model_name
        self.LOCAL_DIR = local_dir

        # Ensure model is available
        self._ensure_model()

        # Create embeddings object
        self.embeddings = self._load_embeddings()

    def _ensure_model(self):
        """Download & save the model locally if not already present."""
        if not os.path.exists(self.LOCAL_DIR):
            print("Downloading and saving model...")
            model = SentenceTransformer(self.MODEL_NAME)
            model.save(self.LOCAL_DIR)
            print("Model saved successfully.")
        else:
            print("Model already downloaded.")

    def _load_embeddings(self):
        """Load embeddings from the local directory."""
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")

        embeddings = HuggingFaceEmbeddings(
            model_name=self.LOCAL_DIR,
            model_kwargs={"device": device}
        )
        return embeddings