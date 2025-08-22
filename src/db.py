from astrapy.constants import VectorMetric
from astrapy.data.info.collection_descriptor import CollectionVectorOptions, CollectionDefinition
from astrapy.data.info.vectorize import VectorServiceOptions

import config
from astrapy import DataAPIClient


def database_client():
    client = DataAPIClient(config.db_token)
    database = client.get_database(config.db_endpoint)
    collection = database.get_collection(config.db_collection_name)

    if config.db_collection_name not in database.list_collection_names():
        collection_definition = CollectionDefinition(
            vector=CollectionVectorOptions(
                metric=VectorMetric.COSINE,
                dimension=384,
                service=VectorServiceOptions(
                    provider="huggingface",
                    model_name=config.Model_name,
                )
            )
        )

        collections = database.create_collection(
            config.db_collection_name,
            definition=collection_definition
        )

        print(f"* Collection: {collections.full_name}\n")
    else:
        print(f"Collection '{collection.name}' already exists")
    return database, collection


# Initialize DB + Collection
database, collection = database_client()
