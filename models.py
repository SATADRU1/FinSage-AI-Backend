import os 
from langchain_ollama import OllamaEmbeddings, ChatOllama
#  from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
class Models:
    def __init__(self):
        #ollama pull  nomic-embed-text 
        self.embeddings_ollama = OllamaEmbeddings(
            model="nomic-embed-text"
        )
        #ollama pull llama3.2
        self.model_ollama = ChatOllama(
            model="llama3.2:latest",
            temperature=0,
        )

'''        # Azure OpenAI embeddings
        self.embeddings_openai = AzureOpenAIEmbeddings(
            model="text-embedding-3-large",
            dimensions=1536, # Can specify dimensions with new
            text-embedding-3 models
            azure_endpoint=os.environ.get
            ("AZURE_OPENAI_EMBEDDINGS_ENDPOINT"),
            api_key= os.environ.get("AZURE OPENAI_API_KEY"),
            api_version=os.environ.get
            ("AZURE_OPENAI_EMBEDDINGS_API_VERSION"),
        )
        #Azure Open Ai chat models 
        self.model_openai = AzureChatOpenAI(
            azure_deploymentwos.environ.get
            ("AZURE_OPENAI_API_DEPLOYMENT_NAME"),
            api_version=os.environ.get("AZURE_OPENAI_API_VERSION"),
            temperature=0,
            max_tokens=None,
            timeout=None, I
            max_retries=2,
        )
'''