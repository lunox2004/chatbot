from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import BedrockEmbeddings
import os

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

def embed_files(file_path_list):
    document_list = []
    
    for document in file_path_list:
        loader = PyPDFLoader(document)
        #document split into list of pages
        document = loader.load()
        pages = []
        
        for page in document:
            page_chunks = document_splitter(page)
            pages.append(page_chunks)
            embeddings.aembed_documents(page_chunks)

        document_list.append(pages)

    return document_list

file_path_list = return_filelist("documents")
document_list = embed_files(file_path_list)

