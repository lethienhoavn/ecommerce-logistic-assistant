import pandas as pd
from langchain_core.documents import Document
from langchain.text_splitter import CharacterTextSplitter



class KnowledgeBaseLangchainReader:

    chunks = []

    def read_file():
        # Load CSV
        df = pd.read_csv("knowledge-base/item_price.csv", sep=';')

        # Turn each line into 1 Langchain document
        documents = []
        for _, row in df.iterrows():
            content = f"Item Name: {row['item_name']}\nDescription: {row['description']}\nPrice: {row['price']} {row['currency']}"
            metadata = {
                "item_id": row['item_id'],
                "currency": row['currency'],
                "last_updated": row['last_updated'],
                "doc_type": "product_description"
            }
            documents.append(Document(page_content=content, metadata=metadata))
        
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        KnowledgeBaseLangchainReader.chunks = text_splitter.split_documents(documents)

        print("len chunks: ", len(KnowledgeBaseLangchainReader.chunks))

        

class KnowledgeBaseReader:

    item2desc = dict()

    def read_file():
        df = pd.read_csv('knowledge-base/item_price.csv', sep=";")
        KnowledgeBaseReader.item2desc = {
            row['item_name']: f"Description: {row['description']} | Price: {row['price']} {row['currency']}"
            for _, row in df.iterrows()
        }

    def retrieve_context(ask_item):
        if len(KnowledgeBaseReader.item2desc) == 0:
            KnowledgeBaseReader.read_file()

        print("ask_item: ", ask_item)

        description = set()
        for item, desc in KnowledgeBaseReader.item2desc.items():
            for word in ask_item.split(): # Drawback: word split & look up is not really right here => better to use phrase vector with semantic search !
                if word.lower() in item.lower():
                    description.add(desc)
        
        return ", ".join(list(description))
