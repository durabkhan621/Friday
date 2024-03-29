from langchain_community.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain.chains import RetrievalQAWithSourcesChain
import pickle
api_key = 'AIzaSyA-iD_a5JYihcpvMvOh3YwLYT_nNrLVb6c'


print("Reading data...")
loader = TextLoader('./data.txt')
documents = loader.load()

print('Splitting data into chunks....')
small_chunk_size = 500
small_chunk_overlap = 100
large_chunk_size = 4000  # Or any other desired size for larger chunks

# Create a CharacterTextSplitter for smaller chunks
small_text_splitter = CharacterTextSplitter(chunk_size=small_chunk_size, chunk_overlap=small_chunk_overlap)

# Split documents into smaller chunks
small_chunks = small_text_splitter.split_documents(documents)

large_text_splitter = CharacterTextSplitter(chunk_size=large_chunk_size, chunk_overlap=small_chunk_overlap)
# Aggregate smaller chunks into larger chunks for summarization
large_chunks = large_text_splitter.split_documents(documents)

all_chunks = small_chunks + large_chunks

print("creating embeddings....")
embeddings = GoogleGenerativeAIEmbeddings(google_api_key=api_key,
                                          model="models/embedding-001")


print('Createing retriever data base....')
vectorindex_openai = FAISS.from_documents(all_chunks, embeddings)
vectorindex_retriever = vectorindex_openai.as_retriever()



print('saving vectorindex_retriever as pkl')
with open('vectorindex_retriever.pkl', 'wb') as f:
    pickle.dump(vectorindex_retriever, f)
print('vectorindex_retriever saved at location: ./vectorindex_retriever.pkl')



print('Initiated LLMs....')
llm = GoogleGenerativeAI(model="models/text-bison-001", google_api_key=api_key, temperature=0.2)


print('initiating chain....')
chain = RetrievalQAWithSourcesChain.from_chain_type(llm=llm, chain_type="map_rerank", 
                                                    retriever=vectorindex_retriever, 
                                                    return_source_documents=False)


query = "What is the food allowance if i travel  domestically?"
input_data = {"question": query}
print(chain.invoke(query))

