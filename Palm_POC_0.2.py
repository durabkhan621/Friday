from langchain_community.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain.chains import RetrievalQAWithSourcesChain
import pickle
api_key = 'AIzaSyA-iD_a5JYihcpvMvOh3YwLYT_nNrLVb6c'


print("Reading data...")
with open('data_0.2.pkl', 'rb') as f:
        data = pickle.load(f)

print('Splitting data into chunks....')
small_chunk_size = 200
small_chunk_overlap = 50

medium_chunk_size = 1000
medium_chunk_overlap = 200

large_chunk_size = 4000  # Or any other desired size for larger chunks
large_chunk_over_lap = 400

very_large_chunk_size = 8000  # Or any other desired size for larger chunks
very_large_chunk_over_lap = 1000

separator="\n\n"
# Create a CharacterTextSplitter for smaller chunks
small_text_splitter = CharacterTextSplitter(separator=separator,
    chunk_size=small_chunk_size,
    chunk_overlap=small_chunk_overlap,
    length_function=len,
    is_separator_regex=False,)

small_chunks = small_text_splitter.split_documents(data)


medium_text_splitter = CharacterTextSplitter(separator=separator,
    chunk_size=medium_chunk_size,
    chunk_overlap=medium_chunk_overlap,
    length_function=len,
    is_separator_regex=False,)

# Split documents into mediumer chunks
medium_chunks = medium_text_splitter.split_documents(data)

large_text_splitter = CharacterTextSplitter(separator=separator,
    chunk_size=large_chunk_size,
    chunk_overlap=large_chunk_over_lap,
    length_function=len,
    is_separator_regex=False,)
# Aggregate smaller chunks into larger chunks for summarization
large_chunks = large_text_splitter.split_documents(data)



very_large_text_splitter = CharacterTextSplitter(separator=separator,
    chunk_size=very_large_chunk_size,
    chunk_overlap=very_large_chunk_over_lap,
    length_function=len,
    is_separator_regex=False,)
# Aggregate smaller chunks into very_larger chunks for summarization
very_large_chunks = very_large_text_splitter.split_documents(data)


all_chunks = small_chunks + medium_chunks + large_chunks + very_large_chunks

print("creating embeddings....")
embeddings = GoogleGenerativeAIEmbeddings(google_api_key=api_key,
                                          model="models/embedding-001")


print('Createing retriever data base....')
vectorindex_openai = FAISS.from_documents(all_chunks, embeddings)
vectorindex_retriever = vectorindex_openai.as_retriever()



print('saving vectorindex_retriever as pkl')
with open('vectorindex_retriever_0.2.pkl', 'wb') as f:
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

