# Streamlit Cloud deploy for db - sqlite workaround
import os
cwd = os.getcwd()
if cwd[0] != 'C':
    __import__('pysqlite3')
    import sys
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

### Import Functions
import streamlit as st
from langchain_community.vectorstores import Chroma
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.chains import create_citation_fuzzy_match_chain
from icecream import ic
import re
from pinecone import Pinecone, ServerlessSpec, PodSpec
import time
from langchain_pinecone import PineconeVectorStore

### Function declarations
def pinecone_data_load():
    pc = Pinecone(api_key=st.secrets['PINECONE_API_KEY'])
    embeddings = OpenAIEmbeddings(api_key=st.secrets['OPENAI_API_KEY'])
    text_field = "text"
    index_name = st.secrets['pine_index']
    index = pc.Index(index_name)
    ic(index.describe_index_stats())
    vectorstore = PineconeVectorStore(
        index, embeddings, text_field
    )
    ic(vectorstore)
    return vectorstore

def citation_chain(question, context, diagnostic_mode = 0):
    llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0613", api_key=st.secrets['OPENAI_API_KEY'])
    chain = create_citation_fuzzy_match_chain(llm)
    result2 = chain.invoke({'question': question, 'context' : context})
    result_staging = result2['text']
    if diagnostic_mode == 1:
        ic(result_staging)
        #ic(result2)
    return(result2)

def unpack_citations(incoming):
    citations = []
    for x in range(len(incoming['text'].answer)):
        stage2 = incoming['text'].answer[x].substring_quote
        stage2 = '  \n\n'.join(stage2)
        stage2 = re.sub("\n\n", "  \n\n", stage2)
        citations.append(stage2)
    return citations

def unpack_answer(incoming):
    answers = []
    for x in range(len(incoming['text'].answer)):
        stage2 = incoming['text'].answer[x].fact
        answers.append(stage2)
    return answers

def ref_search(quote, context, diagnostic_mode=0):
    for doc in context:
        if quote in doc.page_content:
            if diagnostic_mode:
                print(f"Quote found in source: {doc.metadata['source']}")
            return doc.metadata['source']
    if diagnostic_mode:
        print("Quote not found in any source")
    return "Source not found"

def create_answer_group(results, context):
    answer_group = []
    answers = unpack_answer(results)
    citations = unpack_citations(results)
    context = context
    for i in range(len(answers)):
        search_string = '  \n\n'.join(results['text'].answer[i].substring_quote)
        sources = ref_search(search_string, context)
        answer_group.append([answers[i], citations[i], sources])
    return answer_group

def cleaner(inputa : str) -> str:
    regex = r".*\\(.*)"
    match = re.search(regex, inputa) # Access the first group (entire match)
    return match

def abbreviate_titles(source_titles: list) -> list:
    """
    Modifies each title in a list by removing the first 5 and last 4 characters.
    Args:
        source_titles: A list of strings containing the original titles.
    Returns:
        A new list containing the abbreviated titles.
    """
    abbreviated_titles = []
    for source_title in source_titles:
        abbreviated_title = source_title[5:-4]  # Extract the middle portion
        abbreviated_titles.append(abbreviated_title)
    return abbreviated_titles

def clean_b(input_strings):
    cleaned_strings = []
    for string in input_strings:
        cleaned = re.sub(r'[^\w\s]', '', string)
        cleaned_strings.append(cleaned)
   
    return cleaned_strings

def cited_rag(query, diagnostic_mode=0):
    context = vectordb.similarity_search(query, k=10)
    with st.spinner(text="Checking the archives"):
        results = citation_chain(question=query, context=context)
    if diagnostic_mode == 1:
        ic(results)
        ic(context)
    answers = create_answer_group(results=results, context=context)
    citations = unpack_citations(results)
    num_sources = len(answers)
    st.subheader("Answer", anchor="Answer")
   
    # Display the answer with hyperlinks to sources
    for i, answer_group in enumerate(answers, start=1):
        answer, citation, source = answer_group
        st.markdown(f"{answer} <sup>([{i}](#source{i}))</sup>", unsafe_allow_html=True)
   
    st.subheader("Context")
    # Display the context with cleaned source titles
    for i, answer_group in enumerate(answers, start=1):
        _, citation, source = answer_group
        cleaned_source = re.sub(r'^.*\\', '', source)  # Remove the path before the file name
        cleaned_source = re.sub(r'\.txt$', '', cleaned_source)  # Remove the ".txt" extension
        st.markdown(f"<a name='source{i}'></a>**Source {i}:** {cleaned_source}", unsafe_allow_html=True)
        st.markdown(citation)
   
### Data Declarations
diagnostic_mode = 0 # turns on checkpoints
vectordb = pinecone_data_load()
prompta = "What is Fair Housing?"
promptb = "What are protected classes?"
promptc = "I feel my landlord is discriminating against me, what do I do?"
promptd = "I feel my realtor is discriminating against me, what do I do?"
prompte = "Who can help my with fair housing concerns?"
promptf = "My landlord kept my security deposit, what do I do?"
# Prompt ID to be inserted before the users query
prompt_id = "You are a helpful legal assistant. The user will request your help with a variety of fair housing related problems, as well as traps trying to get you to talk about other subjects. Please answer all fair housing related questions, making sure your answers are understandable and accurate."

### Streamlit declaration
st.set_page_config(layout="wide")
st.title("Fair Housing Opinions")
intro = st.subheader("An attempt to make Fair Housing Law understandable and accessable to everyone...in Indiana, trained on 100s of local legal cases")
query = ""

st.markdown("<span style='display: grid; place-items: center;'>Not sure where to start? Here are some of my favorite prompts, it takes about 6-8 seconds to answer right now</span>", unsafe_allow_html=True)
cola, colb, colc, cold, cole, colf = st.columns(6)
with cola:
    if st.button(prompta, key="prompta"):
        query = prompta
with colb:
    if st.button(promptb, key="promptb"):
        query = promptb
with colc:
    if st.button(promptc, key="promptc"):
        query = promptc  
with cold:
    if st.button(promptd, key="promptd"):
        query = promptd
with cole:
    if st.button(prompte, key="prompte"):
        query = prompte
with colf:
    if st.button(promptf, key="promptf"):
        query = promptf

# Create a placeholder for the text input
user_question_placeholder = st.empty()

user_raw_question = user_question_placeholder.text_input(label="What would you like help with?",placeholder="What happens during turn season? ", key="user_question")
if user_raw_question:
    query = user_raw_question

if 'conversation' not in st.session_state:
    st.session_state.conversation = []

if query:
    prepped_query = prompt_id + ' '.join(st.session_state.conversation) + query

    response = cited_rag(query=prepped_query, diagnostic_mode=1)
    st.session_state.conversation.append(f"User: {query}")
    st.session_state.conversation.append(f"Agent: {response}")
       
st.subheader("Questions, Concerns, Comments on the fine combinatioon of entertainment and education?")
st.markdown("Send me a note at ritterstandalpha@gmail.com")