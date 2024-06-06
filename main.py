from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import BedrockEmbeddings
import os

import nest_asyncio
nest_asyncio.apply()

embeddings = BedrockEmbeddings(
    region_name="us-west-2"
)

def return_filelist(path : str) -> list:
    file_list = []
    for filename in os.listdir(path):
        f = os.path.join(path, filename)
        file_list.append(f)

    return file_list


def document_splitter(document):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 500,
        chunk_overlap = 100,
        length_function = len,
        is_separator_regex= False
    )

    chunks = text_splitter.split_documents([document])
    return chunks

def split_and_embed_document(document):
    for page in document:
        page_chunks = document_splitter(page)
        embeddings.aembed_documents(page_chunks) 


def load_file(file_path):
    loader = PyPDFLoader(file_path)
    document = loader.load()
    return document


file_path_list = return_filelist("documents")


for file_path in file_path_list:
    document = load_file(file_path)
    split_and_embed_document(document)




