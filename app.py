import os,time
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_tavily import TavilySearch
from langchain.agents import create_agent
from langchain.tools import tool

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")


os.environ["HF_HOME"] = os.path.expanduser("~/hf-cache")
os.environ["HF_HOME"] = os.path.expanduser("~/hf-cache")

#---------------------------------------------------------------------------
#Loading the PDF
file_path = "./who_hypertension_guidelines.pdf"
loader = PyPDFLoader(file_path)
docs = loader.load()

print("Loading PDF Done")
#---------------------------------------------------------------------------
#Splitting the PDF

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    add_start_index=True,
)
all_splits = text_splitter.split_documents(docs)

print("Splitting PDF Done")
#-----------------------------------------------------------------------------------
# Converting the Splitts into the embeddings and Storing In Vector Store

print("STEP 1: Initializing embedding model...", flush=True)

start = time.perf_counter()

embeddings = HuggingFaceEmbeddings(
   model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"batch_size": 4}
)

print(
    f"STEP 2: Embedding model loaded in "
    f"{time.perf_counter() - start:.2f} seconds",
    flush=True
)

print("STEP 3: Connecting to Chroma...", flush=True)

vector_store = Chroma(
    collection_name="clinical_guidelines",
    embedding_function=embeddings,
    persist_directory="./chroma_clinical_db",
)

print("STEP 4: Chroma connected", flush=True)
print(f"Number of chunks: {len(all_splits)}", flush=True)

if len(all_splits) > 0:
    print("STEP 5: Adding documents and generating embeddings...", flush=True)

    start = time.perf_counter()
    document_ids = vector_store.add_documents(documents=all_splits)

    print(
        f"STEP 6: Indexed {len(document_ids)} documents in "
        f"{time.perf_counter() - start:.2f} seconds",
        flush=True
    )
else:
    print("No document chunks found!", flush=True)

sample = vector_store.get(limit=1, include=["documents"])

print("Storing in Vectors Done", flush=True)

#-------------------------------------------------------------------------------------

#Creating the Model
model = init_chat_model(
    model_provider = "google_genai",
    model = "gemini-3.5-flash-lite",
    api_key=GOOGLE_API_KEY,
)

print("Model Is Created")
#------------------------------------------------------------------------------------

#System Prompt
system_prompt = """ You are a medical education assistant with access to two tools:
1. retrieve_from_guidelines: Use this to find information from the
   WHO Hypertension Guidelines PDF
2. TavilySearch: Use this to find current information
   not in the guidelines (recent approvals, updated trials, etc.)

Strategy:
- For drug dosages, thresholds, contraindications, diagnostic criteria → use retrieve_from_guidelines
- For recent drug approvals, updated evidence, guideline revisions → use TavilySearch
- For comparing guideline recommendations vs new research → use both tools
"""

#-------------------------------------------------------------------------------------
# Initializing the Tavily Search
tavily_search = TavilySearch(
    max_results = 5,
    search_depth = "advanced",
    tavily_api_key = TAVILY_API_KEY
)
print("Tavily API Initialized")
#------------------------------------------------------------------------------------


@tool
def retrieve_from_guidelines(query: str) -> str:
    """ Retrieve information from the WHO Hypertension Guidelines PDF. """
    retrived_docs = vector_store.similarity_search(query)
    docs_content = ""
    for doc in retrived_docs:
        docs_content += f"Source: {doc.metadata} \n"
        docs_content += f"Content: {doc.page_content} \n"
    print("Retrieving API Done")
    return docs_content, retrived_docs

#------------------------------------------------------------------------------------
#Creating the Agent
agent = create_agent(
    model = model,
    tools = [retrieve_from_guidelines, tavily_search],
    system_prompt = system_prompt,
    debug = True
)

#--------------------------------------------------------------------------------

#User Query

user_query = "What medicine does the WHO guideline recommend for high blood pressure, and what are the latest treatment options available?"

#--------------------------------------------------------------------------------

#Generating the response by the Agent
response = agent.invoke({
    "messages":{
        "role": "user",
        "content": user_query,
    }
})


print(response["messages"][-1].content[0])
print(f"The response is Generated according to User Query: \n {user_query}")