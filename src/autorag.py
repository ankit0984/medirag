from langchain_astradb import AstraDBVectorStore
from langchain_astradb.utils.astradb import SetupMode
from ragutils import EmbeddedRag
import config


def autorag():
    try:
        # ✅ Always attach, never recreate
        embedder = EmbeddedRag()
        vstore = AstraDBVectorStore(
            collection_name=config.db_collection_name,
            embedding=embedder.embeddings,
            api_endpoint=config.db_endpoint,
            token=config.db_token,
            namespace=config.astra_namespace,
            setup_mode=SetupMode.OFF   # ✅ key fix
        )

        retriever = vstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 3}
        )
        return retriever

    except Exception as e:
        print(f"❌ autorag exception: {e}")
        return None

auto_retriever = autorag()
