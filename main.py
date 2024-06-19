from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import BedrockEmbeddings
from langchain_aws import ChatBedrock
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
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

params = {
    "temperature": 0.2,
    "top_k": 250,
    "top_p": 1,
    "stop_sequences": ["\n\nHuman"]
}

llm = ChatBedrock(
    client= bedrock_runtime_client,
    model_id="anthropic.claude-v2",
    model_kwargs=params
)


database = Chroma(embedding_function=embeddings, persist_directory="./database")

def return_filelist(path : str) -> list:
    file_list = []
    for filename in os.listdir(path):
        f = os.path.join(path, filename)
        file_list.append(f)

    return file_list


def document_splitter(document):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 2000,
        chunk_overlap = 400,
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
        print(page_chunks_id)


def load_file(file_path):
    loader = PyPDFLoader(file_path)
    document = loader.load()
    return document

def find_and_add_file_names_to_filestore(file_names_storage, filename):
    found = False
    
    try:
        with open(file_names_storage, 'r+') as file:
            lines = file.readlines()
            for line in lines:
                if filename in line:
                    found = True
                    break

            if not found:
                file.write(filename + '\n')
        
        return found

    except FileNotFoundError:
        with open(file_names_storage, 'w') as file:
            file.write(filename + '\n')
        return False

async def main():
    file_path_list = return_filelist("documents")
    
    for file_path in file_path_list:
        if(find_and_add_file_names_to_filestore("./database/file_name_store.txt",file_path)):
            print(f"\n{file_path} already in database")
            continue
        
        documents = load_file(file_path)
        split_and_embed_document(documents)
        print(f"\n{file_path} added to the database")  
    
    retriever = database.as_retriever(search_type="mmr")
    system_prompt = (
    "Use the given context to answer the question. "
    "If you don't know the answer, say you don't know. "
    "keep the answer concise. "
    "Context: {context}"
    )
    prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
    )
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    chain = create_retrieval_chain(retriever, question_answer_chain)
    
    while True:
        query = input("Enter the question(x to exit) : ")
        if(query.upper() == "X"):
            break
        
        print(chain.invoke({"input":query})["answer"])


        

asyncio.run(main())




