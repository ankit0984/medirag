from typing import List
import config
from langchain_community.document_loaders import PyPDFLoader
from langchain.schema import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def filter_to_minimal_docs(docs: List[Document]) -> List[Document]:
    """Keep only page_content and source metadata."""
    minimal_docs: List[Document] = []
    for doc in docs:
        src = doc.metadata.get("source")
        minimal_docs.append(
            Document(page_content=doc.page_content, metadata={"source": src})
        )
    return minimal_docs


class PDFProcessor:
    def __init__(self,  filepath: str, chunk_size: int = config.Chunk_Size, chunk_overlap: int = config.Chunk_Overlap):
        self.filepath = filepath
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    def load_docs(self):
        loader=PyPDFLoader(self.filepath)
        docs=loader.load()
        return docs


    def split_docs(self, docs: List[Document]) -> List[Document]:
        """Split documents into smaller chunks."""
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap
        )
        texts_chunk = splitter.split_documents(docs)
        return texts_chunk

