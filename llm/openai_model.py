import openai
import json
import os
from config.settings import OPENAI_API_KEY, OPENAI_MODEL, OPENAI_TEMPERATURE
from utils.tools import ItemPrice, GenerateCards, Speaker
from utils.kb_reader import KnowledgeBaseReader, KnowledgeBaseLangchainReader
from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_chroma import Chroma
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

openai.api_key = OPENAI_API_KEY

# Init KnowledgeBaseLangchainReader
KnowledgeBaseLangchainReader.read_file()


def call_openai_with_kb_langchain(messages):
    ''' messages including system_prompt + history(user & assistant messages) + new message 
        Instead of relying on LLM tool to do intent detection (with semantic search) & entity extraction, we'll manually do it by ourself
        Langchain handle all chat history, we just need to pass the last message as input when calling conversation_chain.invoke()
    '''

    ##### Create Vector DB
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # # Chroma (SQL Lite)    
    # db_name = "product_vector_db"
    # if os.path.exists(db_name):
    #     Chroma(persist_directory=db_name, embedding_function=embeddings).delete_collection()
    # vectorstore = Chroma.from_documents(documents=KnowledgeBaseLangchainReader.chunks, embedding=embeddings, persist_directory=db_name)
    
    # FAISS
    vectorstore = FAISS.from_documents(KnowledgeBaseLangchainReader.chunks, embedding=embeddings)

    ##### Create Conversation Chain
    # create a new Chat with OpenAI
    llm = ChatOpenAI(temperature=0.7, model_name=OPENAI_MODEL)

    # set up the conversation memory for the chat
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)

    # the retriever is an abstraction over the VectorStore that will be used during RAG
    retriever = vectorstore.as_retriever() # search_kwargs={"k": 25} : can tune how many chunks to use in RAG

    # putting it together: set up the conversation chain with the GPT 4o-mini LLM, the vector store and memory
    conversation_chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=retriever, memory=memory) # , callbacks=[StdOutCallbackHandler()] : use to investigate what happen behind the scene

    # Wrapping in a function - note that history isn't used, as the memory is in the conversation_chain (in 'chat_history')
    response_text = conversation_chain.invoke({"question": messages[-1]['content']})
    print("response_text: ", response_text)
    response_text = response_text['answer']

    image = None

    # call speaker
    # Speaker.speak(response_text)
    
    return response_text, image



def call_openai_with_kb(messages):
    ''' messages including system_prompt + history(user & assistant messages) + new message 
        Instead of relying on LLM tool to do intent detection (with semantic search) & entity extraction, we'll manually do it by ourself
    '''

    # Retrieve context from the last user message to get Product Description & Price
    context_message_prompt = "\n\nThe following additional context might be relevant in answering this question:\n\n"
    context_message = KnowledgeBaseReader.retrieve_context(messages[-1]["content"]) 
    print("context_message: ", context_message)

    if context_message != "":
        # Give more info to LLM
        context_message_prompt += context_message
        messages.append({"role": "user", "content": context_message_prompt})

    image = None
    response = openai.chat.completions.create(
        model=OPENAI_MODEL,
        messages=messages,
        tools= GenerateCards.tools,
        temperature=OPENAI_TEMPERATURE
    )

    if response.choices[0].finish_reason=="tool_calls": # default finish_reason = 'stop'
        # assistant call tool
        message = response.choices[0].message 
        response, image = handle_tool_call(message)
        messages.append(message)
        messages.append(response)

        # assistant wrap retrieved item price and write answer text back to user
        response = openai.chat.completions.create(
            model=OPENAI_MODEL, 
            messages=messages
        ) 

    response_text = response.choices[0].message.content

    # call speaker
    # Speaker.speak(response_text)
    
    return response_text, image


############################################################


def call_openai_with_tool(messages):
    ''' messages including system_prompt + history(user & assistant messages) + new message '''
    image = None

    response = openai.chat.completions.create(
        model=OPENAI_MODEL,
        messages=messages,
        tools= ItemPrice.tools + GenerateCards.tools,
        temperature=OPENAI_TEMPERATURE
    )

    if response.choices[0].finish_reason=="tool_calls": # default finish_reason = 'stop'
        # assistant call tool
        message = response.choices[0].message 
        response, image = handle_tool_call(message)
        messages.append(message)
        messages.append(response)

        # assistant wrap retrieved item price and write answer text back to user
        response = openai.chat.completions.create(
            model=OPENAI_MODEL, 
            messages=messages
        ) 

    response_text = response.choices[0].message.content

    # call speaker
    # Speaker.speak(response_text)
    
    return response_text, image


def handle_tool_call(message):
    tool_call = message.tool_calls[0]
    arguments = json.loads(tool_call.function.arguments)
    tool_name = tool_call.function.name
    image = None

    # tool reponse with price (or unknown)
    if tool_name == "get_item_price":
        item = arguments.get('item')
        price = ItemPrice.get_item_price(item)
        response = {
            "role": "tool",
            "content": json.dumps({"item": item, "price": price}),
            "tool_call_id": tool_call.id
        } 
    elif tool_name == "generate_gift_card":
        name = arguments.get('name')
        response = {
            "role": "tool",
            "content": name,
            "tool_call_id": tool_call.id
        } 
        image = GenerateCards.genImage(name)
    else:
        response = {
            "role": "tool",
            "content": "unknown",
            "tool_call_id": tool_call.id
        } 

    return response, image
