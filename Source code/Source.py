from streamlit_chat import message
from langchain.chains import ConversationalRetrievalChain
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.embeddings import HuggingFaceEmbeddings, OpenAIEmbeddings
from langchain.llms import CTransformers
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
import os
from dotenv import load_dotenv

def load_documents():
    # Function to load documents
    print('Started to load documents')
    loader = DirectoryLoader('data/', glob="*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    print('Documents loaded')
    return documents

# Function to split text into chunks
def split_text_into_chunks(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, chunk_overlap=50)
    text_chunks = text_splitter.split_documents(documents)
    print('Chunks created')
    return text_chunks

def make_vectorstore(text_chunks=None):
    load_dotenv()
    embeddings = OpenAIEmbeddings(deployment=os.getenv(
        "OPENAI_DEPLOYMENT_NAME"), chunk_size=1,)
    print('Embedding started')
    '''vectorstore = FAISS.from_documents(text_chunks, embeddings)
    print('Vector store received')
    vectorstore.save_local(
        '/Users/dinesh/college/proj/ResearchAssistant/knowledge/Vectorstore')
    # vs = np.array(vectorstore)
    # np.save('savefile.npy',vs)'''
    vectorstore = FAISS.load_local('Vectorstore', embeddings)
    print('Vector store received')
    print('Embedding completed')
    return vectorstore

# Function to create LLMS model
def create_llms_model():
    llm = CTransformers(model="mistral-7b-instruct-v0.1.Q4_K_M.gguf",
                        config={'max_new_tokens': 128, 'temperature': 0.01})
    return llm

# Initialize Streamlit app
st.title("AI research assistant")
st.markdown(
    '<style>h1{color: orange; text-align: center;}</style>', unsafe_allow_html=True)
st.subheader('Get help from the chatbot')
st.markdown(
    '<style>h3{color: pink; text-align: center;}</style>', unsafe_allow_html=True)

# loading of documents
# documents = load_documents()

# Split text into chunks
text_chunks = None
# text_chunks = split_text_into_chunks(documents)
vector_store = make_vectorstore(text_chunks)  # openAI

# Create LLMS model
llm = create_llms_model()

# Initialize conversation history
if 'history' not in st.session_state:
    st.session_state['history'] = []

if 'generated' not in st.session_state:
    st.session_state['generated'] = ["Hello! Ask me anything "]

if 'past' not in st.session_state:
    st.session_state['past'] = ["Hey! 👋"]

# Create memory
memory = ConversationBufferMemory(
    memory_key="chat_history", return_messages=True)

# Create chain
chain = ConversationalRetrievalChain.from_llm(llm=llm, chain_type='stuff',
                                              retriever=vector_store.as_retriever(
                                                  search_kwargs={"k": 2}),
                                              memory=memory)

# Define chat function
def conversation_chat(query):
    result = chain(
        {"question": query, "chat_history": st.session_state['history']})
    st.session_state['history'].append((query, result["answer"]))
    return result["answer"]

# Display chat history
reply_container = st.container()
container = st.container()

with container:
    with st.form(key='my_form', clear_on_submit=True):
        user_input = st.text_input(
            "Question:", placeholder="Ask about your research", key='input')
        submit_button = st.form_submit_button(label='Send')

    if submit_button and user_input:
        output = conversation_chat(user_input)
        st.session_state['past'].append(user_input)
        st.session_state['generated'].append(output)

if st.session_state['generated']:
    with reply_container:
        for i in range(len(st.session_state['generated'])):
            message(st.session_state["past"][i], is_user=True, key=str(
                i) + '_user', avatar_style="thumbs")
            message(st.session_state['generated'][i],
                    key=str(i), avatar_style="fun-emoji")
