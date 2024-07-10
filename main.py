from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import BedrockEmbeddings
from langchain_aws import ChatBedrock
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import UnstructuredHTMLLoader
from langchain_chroma import Chroma
from dotenv import load_dotenv
import time
import streamlit as st
import boto3
import botocore
import asyncio
import fitz
from PIL import Image
import io
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
        chunk_size = 4000,
        chunk_overlap = 800,
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

def stream_generator(input):
    for word in input.split():
        yield word + " "
        time.sleep(0.02)


def load_file(file_path):
    loader = PyPDFLoader(file_path)
    document = loader.load()
    return document

def extract_pdf_page_as_image(pdf_path, page_number):
    pdf_document = fitz.open(pdf_path)
    page = pdf_document.load_page(page_number - 1)
    pix = page.get_pixmap()
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    
    return img

def extract_filename_without_extension(file_path):
    filename_with_ext = os.path.basename(file_path)
    filename, _ = os.path.splitext(filename_with_ext)
    return filename


def find_and_add_name_to_store(file_names_storage, filename):
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

def add_url(link):
    if (find_and_add_name_to_store("./database/url_name_store.txt",link)):
        print(f"\n{link} already in database")
    else:
        loader = UnstructuredHTMLLoader(link)
        data = loader.load()
        split_and_embed_document(data)
        print(f"\n{link} has been uploaded to database")

def make_retreiver_chain():
    retriever = database.as_retriever(search_type="mmr")
    system_prompt = (
    """Your role is that of am human resource manager answering questions employees have about various policies and any queries related to the company,
    keep the answer short and concise. 
    If there is no context given just say \"there is no context given \" and nothing else
    Context: {context}"""
    )
    prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
    )
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    chain = create_retrieval_chain(retriever, question_answer_chain)
    return chain

def clear_chat():
    st.session_state.messages = []


async def main():
    uploaded_files = []
    uploaded_url = ""
    response = ""
    file_path_list = return_filelist("documents")
    save_location = "documents"
    sidebar = st.sidebar
    upload_type = sidebar.radio("Select file type to upload", ["pdf" , "websites"])
    
    if upload_type == "pdf":
        uploaded_files = sidebar.file_uploader("Upload PDF files",key= 123, type=["pdf"], accept_multiple_files=True)
    else:
        uploaded_url = sidebar.text_input("enter url to upload")
    
    sidebar.button("clear chat", on_click=clear_chat)

    if uploaded_files:
        for uploaded_file in uploaded_files:
            with open(os.path.join(save_location, uploaded_file.name), "wb") as f:
                f.write(uploaded_file.getbuffer())
            st.success(f"Saved {uploaded_file.name}")

    for file_path in file_path_list:
        if(find_and_add_name_to_store("./database/file_name_store.txt",file_path)):
            print(f"\n{file_path} already in database")
            continue
        
        documents = load_file(file_path)
        split_and_embed_document(documents)
        print(f"\n{file_path} added to the database")  
    
    chain = make_retreiver_chain()
    
    if "messages" not in st.session_state:
        st.session_state.messages = []

    tab1 , tab2 = st.tabs(['chatbot','refererenced pdf pages'])
    
    with tab1:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                if message["role"] == "user":
                    st.markdown(message["content"])
                else:
                    if isinstance(message["content"], dict) and "answer" in message["content"]:
                        st.markdown(message["content"]["answer"])
                    else:
                        st.markdown(message["content"]) 

    with tab2:
        if not st.session_state.messages:
            st.write("nothing to display")
        else:
            last_message = st.session_state.messages[-1]
            if last_message["role"] == "assistant" and isinstance(last_message["content"], dict) and "context" in last_message["content"]:
                for page in last_message["content"]["context"]:
                    with st.expander(extract_filename_without_extension(page.metadata["source"])):
                        img = extract_pdf_page_as_image(page.metadata["source"], page.metadata["page"] + 1)
                        img_byte_arr = io.BytesIO()
                        img.save(img_byte_arr, format='PNG')
                        img_byte_arr = img_byte_arr.getvalue()
                        st.image(img_byte_arr)
            else:
                st.markdown("nothing to display")

    if prompt := st.chat_input("Ask a question"):
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        response = chain.invoke({"input": prompt})
        if response.get("context"):
            with st.chat_message("assistant"):
                st.markdown(response["answer"])
            st.session_state.messages.append({"role": "assistant", "content": response})
        else:
            st.markdown("No pdf's or urls given")
    
    print(response)


asyncio.run(main())




