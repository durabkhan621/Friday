import streamlit as st
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_google_genai import GoogleGenerativeAI
from langchain.chains import RetrievalQAWithSourcesChain
import pickle

api_key = 'AIzaSyA-iD_a5JYihcpvMvOh3YwLYT_nNrLVb6c'
synonyms_explain_in_detail = [
    "Elaborate",
    "Expound",
    "Clarify",
    "Illustrate",
    "Expand on",
    "Flesh out",
    "Detail",
    "Spell out",
    "Enlighten",
    "Illuminate",
    "Thoroughly explain",
    "Provide details on",
    "Give a comprehensive explanation of",
    "Offer a thorough explanation of",
    "Delineate",
    "Expatiate",
    "Exposit",
    "Exemplify",
    "Demonstrate",
    "Decipher"
]



# Define boolean variable to control initialization
initialized = False

# Function to initialize database and LLM
@st.cache_resource()
def initialize():
    print("Reading database from pkl format...")
    with open('vectorindex_retriever.pkl', 'rb') as f:
        db = pickle.load(f)


    llm = GoogleGenerativeAI(model="models/text-bison-001", google_api_key=api_key, temperature=0.2)
    print('Initiating chain....')
    chain_precise = RetrievalQAWithSourcesChain.from_chain_type(llm=llm, chain_type="map_rerank", 
                                                    retriever=db, 
                                                    return_source_documents=False)
    chain_detailed_refine = RetrievalQAWithSourcesChain.from_chain_type(llm=llm, chain_type="refine", 
                                                    retriever=db, 
                                                    return_source_documents=False)

    return chain_precise, chain_detailed_refine

# Define function to get answer
def get_answer_google_palm(question):
    global initialized
    if not initialized:
        global chain_precise
        global chain_detailed_refine
        chain_precise, chain_detailed_refine = initialize()
        initialized = True

    contains_synonym = any(word in question for word in synonyms_explain_in_detail)

    if contains_synonym:
        return chain_detailed_refine.invoke(question)['answer']
    return chain_precise.invoke(question)['answer']


