import openai
import json
import os
from config.settings import OPENAI_API_KEY, OPENAI_MODEL, OPENAI_TEMPERATURE
from utils.tools import ItemPrice, GenerateCards, Speaker
from utils.kb_reader import KnowledgeBaseReader, KnowledgeBaseLangchainReader
from utils.dedup_retriever import DedupRetriever
from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_chroma import Chroma
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain_community.chat_message_histories import FileChatMessageHistory
from langchain.chains import ConversationalRetrievalChain, LLMChain, RetrievalQA
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, \
    HumanMessagePromptTemplate, MessagesPlaceholder
from langchain.schema import SystemMessage
import langchain

openai.api_key = OPENAI_API_KEY

# langchain.debug = True

class OpenAIChatbot():

    def __init__(self):
        # self.init_openai_with_kb_langchain()
        self.init_openai_with_kb()
        # self.init_openai_with_tool()


    def init_openai_with_kb_langchain(self):
        ##### Prompt
        system_prompt = '''
                You are a helpful and professional virtual assistant for Parcel Perform, a global parcel tracking and delivery performance platform.

                Your main tasks are to:
                - Assist users in tracking their parcels by providing status updates based on tracking numbers.
                - Answer common questions about shipping, delivery times, delays, and returns.
                - Provide clear instructions on how to use Parcel Perform services.
                - Escalate complex issues politely by suggesting users contact customer support.

                Always respond politely, clearly, and concisely. Use simple language that anyone can understand. 

                If the user provides a tracking number, help them check the latest status and estimated delivery date.

                Example interactions:

                User: "Can you track my package with tracking number 123456789?"
                Assistant: "Sure! Let me check the status of tracking number 123456789... Your parcel is currently in transit and expected to arrive on May 24th."

                User: "What should I do if my package is delayed?"
                Assistant: "I’m sorry for the delay. You can contact the sender or your local courier for more details. Would you like me to help find their contact info?"

                User: "How do I return a parcel?"
                Assistant: "To return a parcel, please follow the return instructions provided by the seller or courier. If you need specific help, I can guide you through the steps."

                Be friendly, supportive, and helpful in every response.

                Remember, you are an expert in answering accurate questions for Parcel Perform. Give brief, accurate answers. If you don't know the answer, say so. Do not make anything up if you haven't been provided with relevant context.
            '''

        chat_prompt = ChatPromptTemplate(
            input_variables=["chat_history", "question"],
            messages = [
                SystemMessagePromptTemplate.from_template(system_prompt),
                MessagesPlaceholder(variable_name="chat_history"),
                HumanMessagePromptTemplate.from_template("{question}")
            ]
        )

        ##### Create Vector DB
        print("Load embeddings...")
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

        # # Chroma (SQL Lite)    
        # db_name = "resources/product_vector_db"
        # if not os.path.exists(db_name):
        #     knowledgebaes_langchain_read = KnowledgeBaseLangchainReader()
        #     vectorstore = Chroma.from_documents(documents=knowledgebaes_langchain_read.chunks, embedding=embeddings, persist_directory=db_name)
        # else:
        #     vectorstore = Chroma(persist_directory=db_name, embedding_function=embeddings)

        faiss_index_path = "resources/faiss_index"
        if (not os.path.exists(faiss_index_path)):
            print("Initializing Retriever, this may take a while....")

            # Init KnowledgeBaseLangchainReader
            knowledgebaes_langchain_read = KnowledgeBaseLangchainReader()

            # FAISS
            vectorstore = FAISS.from_documents(knowledgebaes_langchain_read.chunks, embedding=embeddings)
            vectorstore.save_local(faiss_index_path)
        else:
            print("Load faiss index from file...")
            vectorstore = FAISS.load_local(faiss_index_path, embeddings, allow_dangerous_deserialization=True)
        
        # the retriever is an abstraction over the VectorStore that will be used during RAG
        # retriever = vectorstore.as_retriever() # search_kwargs={"k": 25} : can tune how many chunks to use in RAG
        retriever = DedupRetriever(
            embeddings=embeddings,
            db=vectorstore
        )

        ##### Create Conversation Chain
        # create a new Chat with OpenAI
        llm = ChatOpenAI(temperature=0.7, model_name=OPENAI_MODEL) # verbose=True

        # set up the conversation memory for the chat
        memory = ConversationBufferMemory(memory_key='chat_history', 
                                            # chat_memory=FileChatMessageHistory("resources/chat_history.json"), # when history is long, input is very lengthy => better to use ConversationSummaryMemory than ConversationBufferMemory + FileChatMessageHistory (but with the cost of slower result because of 2 LLM passes)
                                            return_messages=True)
        
        # # RetrievalQA: only embedding retrieval for QA (1 turn, not for chat)
        # self.conversation_chain = RetrievalQA(llm=llm,
        #                                       retriever=retriever,
        #                                       chain_type="stuff")

        # # LLMChain: only chat memory context
        # self.conversation_chain = LLMChain(llm=llm,
        #                                 memory=memory,  
        #                                 prompt=chat_prompt) # verbose=True

        # ConversationalRetrievalChain: chat memory context + embedding retrieval
        self.conversation_chain = ConversationalRetrievalChain.from_llm(llm=llm, 
                                                                    retriever=retriever, 
                                                                    memory=memory)
                                                                    # combine_docs_chain_kwargs={"prompt": chat_prompt}) # , callbacks=[StdOutCallbackHandler()] : use to investigate what happen behind the scene


    def call_openai_with_kb_langchain(self, messages):
        ''' messages including system_prompt + history(user & assistant messages) + new message 
                => Langchain already handle chat history, we just need to use the new message
            Instead of relying on LLM tool to do intent detection (with semantic search) & entity extraction, we'll manually do it by ourself
            Langchain handle all chat history, we just need to pass the last message as input when calling conversation_chain.invoke()
        '''

        # Wrapping in a function - note that history isn't used, as the memory is in the conversation_chain (in 'chat_history')
        response_text = self.conversation_chain.invoke({"question": messages[-1]['content']})
        print("response_text: ", response_text)

        response_text = response_text['answer'] # for ConversationalRetrievalChain
        # response_text = response_text['text'] # for LLMChain

        image = None

        # call speaker
        # Speaker.speak(response_text)
        
        return response_text, image


    def init_openai_with_kb(self):
        self.knowledgebase_reader = KnowledgeBaseReader()
        self.item_price = ItemPrice()
        self.generate_cards = GenerateCards()


    def call_openai_with_kb(self, messages):
        ''' messages including system_prompt + history(user & assistant messages) + new message 
            Instead of relying on LLM tool to do intent detection (with semantic search) & entity extraction, we'll manually do it by ourself
        '''

        # Retrieve context from the last user message to get Product Description & Price
        context_message_prompt = "\n\nThe following additional context might be relevant in answering this question:\n\n"
        context_message = self.knowledgebase_reader.retrieve_context(messages[-1]["content"]) 
        print("context_message: ", context_message)

        if context_message != "":
            # Give more info to LLM
            context_message_prompt += context_message
            messages.append({"role": "user", "content": context_message_prompt})

        image = None
        response = openai.chat.completions.create(
            model=OPENAI_MODEL,
            messages=messages,
            tools=self.generate_cards.tools,
            temperature=OPENAI_TEMPERATURE
        )

        if response.choices[0].finish_reason=="tool_calls": # default finish_reason = 'stop'
            # assistant call tool
            message = response.choices[0].message 
            response, image = self.handle_tool_call(message)
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

    def init_openai_with_tool(self):
        self.item_price = ItemPrice()
        self.generate_cards = GenerateCards()


    def call_openai_with_tool(self, messages):
        ''' messages including system_prompt + history(user & assistant messages) + new message '''
        image = None

        response = openai.chat.completions.create(
            model=OPENAI_MODEL,
            messages=messages,
            tools=self.item_price.tools + self.generate_cards.tools,
            temperature=OPENAI_TEMPERATURE
        )

        if response.choices[0].finish_reason=="tool_calls": # default finish_reason = 'stop'
            # assistant call tool
            message = response.choices[0].message 
            response, image = self.handle_tool_call(message)
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


    def handle_tool_call(self, message):
        tool_call = message.tool_calls[0]
        arguments = json.loads(tool_call.function.arguments)
        tool_name = tool_call.function.name
        image = None

        # tool reponse with price (or unknown)
        if tool_name == "get_item_price":
            item = arguments.get('item')
            price = self.item_price.get_item_price(item)
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
            image = self.generate_cards.genImage(name)
        else:
            response = {
                "role": "tool",
                "content": "unknown",
                "tool_call_id": tool_call.id
            } 

        return response, image
