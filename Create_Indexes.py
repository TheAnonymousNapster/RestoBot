import os
import logging
import sys

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

from llama_index import (
    VectorStoreIndex,
    SimpleDirectoryReader,
)

from llama_index.llms import Replicate
from llama_index import ServiceContext, set_global_service_context
from llama_index.llms.llama_utils import (
    messages_to_prompt,
    completion_to_prompt,
)

documents = SimpleDirectoryReader("Data").load_data()

from langchain.embeddings.huggingface import HuggingFaceBgeEmbeddings
from llama_index import ServiceContext

embed_model = HuggingFaceBgeEmbeddings(model_name="BAAI/bge-base-en")

service_context = ServiceContext.from_defaults(embed_model=embed_model)
index = VectorStoreIndex.from_documents(documents, service_context=service_context, show_progress=True)
index.storage_context.persist(persist_dir='Data_Indexes')