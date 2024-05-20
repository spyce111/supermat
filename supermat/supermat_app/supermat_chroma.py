import json
import traceback
import uuid
from .constants import SupermatConstants
import chromadb
from chromadb.utils import embedding_functions
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction as OpenAIEF
from tqdm import tqdm



class ChromaDBConnection:
    def __init__(self, db_config) -> None:
        self.connection = chromadb.HttpClient(**db_config)
        self.sentence_embedd = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=SupermatConstants.EMBED_MDL_NAME)
        # self.openai_ef = OpenAIEF(**openai_config)

    def create_collection(self, collection_name):
        collection = self.connection.create_collection(collection_name, embedding_function=self.sentence_embedd)
        print(f"created collection {collection_name}")
        return collection

    def delete_collection(self, collection_name):
        self.connection.delete_collection(collection_name)
        print(f"deleted collection {collection_name}")
        return True

    def get_collection(self, collection_name):
        collection = self.connection.get_collection(collection_name, embedding_function=self.sentence_embedd)
        return collection

    def get_or_create_collection(self, collection_name):
        collection = self.connection.get_or_create_collection(collection_name, embedding_function=self.sentence_embedd)
        return collection

    def insert(self, collection_name, data_list):
        """
        collection_name: str - name of the collection
        data_list: dict - dictionary containing 'ids', 'documents', 'metadatas' lists

        data_list format:
        {
            'ids': [str], # list of data ids
            'documents': [str], # list of data values/chunks/paragraph
            'metadatas': [str] # Metadata for each data point
        }
        """
        import pdb;pdb.set_trace()
        collection = self.get_or_create_collection(collection_name)
        batch_size = 10
        num_data = len(data_list['ids'])  # Assuming 'ids' list as the reference for data length

        for i in tqdm(range(0, num_data, batch_size), desc="inserting"):
            batch_ids = data_list['ids'][i:i + batch_size]
            batch_docs = data_list['documents'][i:i + batch_size]
            batch_meta = data_list['metadatas'][i:i + batch_size]
            batch_data = {
                'ids': batch_ids,
                'documents': batch_docs,
                'metadatas': batch_meta
            }
            collection.add(**batch_data)
        print('Successfully inserted data for collection: {}'.format(collection_name))
        return collection_name, True

    def get_all_vectors(self, collection_name):
        collection = self.get_collection(collection_name)
        all_values = collection.get()
        return all_values

    def query(self, collection, query="", top_k=10, document_id=None):
        collection = self.get_or_create_collection(collection)
        if not document_id:
            values = collection.query(
                query_texts=[query],
                n_results=top_k
            )
        else:
            values = collection.query(
                query_texts = [query],
                n_results = top_k,
                where = {
                    "document_id": {
                        "$eq": document_id
                    }
                }
            )
        ids = values['ids'][0]
        results = collection.get(ids=ids)
        return results['documents']


# class ChromaConnector:
#     def __init__(self, hostname, port_no, embed_mdl_name):
#         self.cli = chromadb.HttpClient(hostname, port_no)
#         self.collection = None
#         self.sentence_embedd = embedding_functions.SentenceTransformerEmbeddingFunction(
#             model_name=embed_mdl_name)
#         # self.sentence_embedd =OpenAIEF(**OPENAI_CONFIG)
#
#     def push(self, indexes, documents):
#         self.collection.add(ids=indexes, documents=documents)
#
#     def create_collection(self, collection_name):
#         try:
#             self.collection = self.cli.create_collection(collection_name,
#                                                          embedding_function=self.sentence_embedd)
#         except:
#             print("Unable to create collection")
#
#     def list_collection(self):
#         try:
#             col_list = self.cli.list_collections()
#             return [i.name for i in col_list]
#         except:
#             return None
#
#     def delete_collection(self, collection_name):
#         try:
#             self.cli.delete_collection(collection_name)
#         except:
#             print(f'Unable to delete collection {collection_name}')
#
#     def query_data(self, collection_name, query, n_result=5):
#         self.collection = self.cli.get_collection(collection_name, embedding_function=self.sentence_embedd)
#         return self.collection.query(query_texts=[query], n_results=n_result)
#
#
#
#
# class ChromaService:
#     def __init__(self, collection_name=None):
#         self.conn = ChromaConnector(SupermatConstants.CHROMA_HOST, SupermatConstants.CHROMA_PORT, SupermatConstants.CHROMA_MODEL)
#         if not collection_name:
#             self.collection_name = "supermat_docs"
#         else:
#             self.collection_name = collection_name
#
#     def load(self, json_data):
#         print("Started creating collection")
#         coll_list = self.conn.list_collection()
#         if self.collection_name  not in coll_list:
#             self.conn.create_collection(self.collection_name)
#         print("Collection created")
#         print("Reading saved up data")
#         self.conn.push(json_ids, json_list)
#
#     def get_or_create_collection(self, collection_name):
#         collection = self.conn.get_or_create_collection(collection_name, embedding_function=self.openai_ef)
#         return collection
#     def query(self, collection, query="", top_k=10):
#         collection = self.get_or_create_collection(collection)
#         values = collection.query(
#             query_texts=[query],
#             n_results = top_k
#         )
#         return values
