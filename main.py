from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import BedrockEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv
import chromadb
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


database_cliet = chromadb.Client()
databse_collection = database_cliet.get_or_create_collection("chatbot_database", embedding_function=embeddings)

def generate_consecutive_id(total_id, lenght_of_new_list):
    length = lenght_of_new_list 
    start_number = total_id + 1 
    consecutive_numbers = list(str(num) for num in range(start_number,start_number+length)) 
    return consecutive_numbers

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
        page_chunks_string = [page_chunks[i].page_content for i in range(len(page_chunks))]
        page_chunks_metadata =[page_chunks[i].metadata for i in range(len(page_chunks))]
        page_chunks_id = generate_consecutive_id(databse_collection.count(),len(page_chunks_string))
        databse_collection.add(ids = page_chunks_id,
                               documents= page_chunks_string,
                               metadatas= page_chunks_metadata)
        print(page_chunks_id)
        print("done")



def load_file(file_path):
    print(file_path)
    loader = PyPDFLoader(file_path)
    document = loader.load()
    return document



async def main():
    file_path_list = return_filelist("documents")
    
    for file_path in file_path_list:
        documents = load_file(file_path)
        split_and_embed_document(documents)

    langhchain_chroma_database = Chroma(
        client= database_cliet,
        collection_name= "chatbot_database",
        embedding_function=embeddings

    )


    query = "What is the dress code policy?"
    answer = langhchain_chroma_database.similarity_search(query)
    print(answer)
    langhchain_chroma_database.add_documents()

asyncio.run(main())




