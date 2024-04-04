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
    "Explain"
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
    with open('vectorindex_retriever_0.2.pkl', 'rb') as f:
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
        # return chain_detailed_refine.invoke(question)['answer']
        return chain_precise.invoke(question)['answer']
    return chain_precise.invoke(question)['answer']

# Define the Streamlit app
def main():
    image_file = "./download.jpeg"
    st.sidebar.image(image_file, width=100)
    st.sidebar.title("Questions Asked")
    if 'questions' not in st.session_state:
        st.session_state.questions = []

    

    st.title("Hi I am Friday, Your HR Assistant")
    st.title("How May I help you?")

    # Add a text input box for the user to enter their question
    user_question = st.text_input("Question:")

    # Add a submit button
    # Add a submit button
    if st.button('Submit'):
    # Check if the user has entered a question
        if user_question:
            # Call the function to get the answer
            answer = get_answer_google_palm(user_question)

            # Append the question and answer to the conversation history
            st.session_state.questions.append((user_question, answer))

    # Update the sidebar with all questions in the history
    for idx, (question, _) in enumerate(st.session_state.questions[::-1]):
        st.sidebar.markdown(f"- [{question}](#question-{idx})")

    # Display the conversation history
    for idx, (question, answer) in enumerate(st.session_state.questions[::-1]):
        st.write(f'<span style="color: yellow;">You:</span> {question}', unsafe_allow_html=True)
        st.write(f'<span style="color: green;">Friday:</span> {answer}', unsafe_allow_html=True)
        st.write("---")

        # Create anchor links for each question
        st.markdown(f'<a id="question-{idx-1}"></a>', unsafe_allow_html=True)

# Run the app
if __name__ == "__main__":
    main()
