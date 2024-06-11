from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import BedrockEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv
import boto3
import botocore
import asyncio
import os

load_dotenv()

config = botocore.config.Config(
    read_timeout=900,
    connect_timeout=900,
    retries={"max_attempts": 0}
)

bedrock_runtime_client = boto3.client(
    'bedrock-runtime',
    region_name=os.getenv("AWS_REGION"),
    aws_access_key_id=os.getenv("ACCESS_KEY"),
    aws_secret_access_key=os.getenv("SECRET_ACCESS_KEY"),
    config=config
)

embeddings = BedrockEmbeddings(
    client= bedrock_runtime_client,
    region_name="us-west-2"
)

database = Chroma(embedding_function=embeddings)

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
        if(len(page_chunks) == 0):
            continue
        page_chunks_id = database.add_documents(page_chunks)



def load_file(file_path):
    loader = PyPDFLoader(file_path)
    document = loader.load()
    return document



async def main():
    file_path_list = return_filelist("documents")
    
    for file_path in file_path_list:
        documents = load_file(file_path)
        split_and_embed_document(documents)



    query = "What is the dress code policy?"
    answer = database.similarity_search(query)
    print(query)
    print("\n\n")
    print(answer)

asyncio.run(main())




