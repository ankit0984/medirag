from langchain_astradb import AstraDBVectorStore
from tqdm import tqdm
import config
from ragutils import EmbeddedRag
from ragutils.pdfprocessor import filter_to_minimal_docs, PDFProcessor
from src.db import collection


def ragcollection():
    try:
        if config.db_collection_name == collection.name:
            # ✅ Check if collection already has documents
            existing_doc = collection.find_one({})
            if existing_doc is not None:
                print("⚠️ Collection already has documents.")
                return

            processor = PDFProcessor("data/medical_book_fixed.pdf")
            docs = processor.load_docs()
            print(f"📄 Loaded {len(docs)} documents")
            minimal_docs = filter_to_minimal_docs(docs)
            texts_chunk = processor.split_docs(minimal_docs)
            print(f"✂️ Split into {len(texts_chunk)} chunks")

            embedder = EmbeddedRag()
            vstore = AstraDBVectorStore.from_documents(
                embedding=embedder.embeddings,
                api_endpoint=config.db_endpoint,
                token=config.db_token,
                collection_name=config.db_collection_name,
                batch_size=100,
                documents=texts_chunk
            )

            batch_size = 100
            total_inserted = 0
            num_batches = (len(texts_chunk) + batch_size - 1) // batch_size

            for i in tqdm(range(0, len(texts_chunk), batch_size), total=num_batches, desc="📦 Inserting Batches"):
                batch = texts_chunk[i:i + batch_size]
                inserted_ids = vstore.add_documents(batch)
                total_inserted += len(inserted_ids)

            print(f"\n✅ Total inserted: {total_inserted}")

    except Exception as e:
        print(f"❌ ragcollection error: {e}")

