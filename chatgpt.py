import os
import sys
from flask import Flask, render_template, request 
import time


# the imports used
import openai
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain, RetrievalQA
from langchain_community.chat_models import ChatOpenAI
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain.indexes import VectorstoreIndexCreator
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain_community.llms import openai
from langchain_community.vectorstores import chroma

import constants

app = Flask(__name__, static_url_path='/static')
os.environ["OPENAI_API_KEY"] = constants.APIKEY  # MY API KEY


def read_data_file():
    data_file_path = os.path.join('data', 'data.txt')
    with open(data_file_path, 'r') as file:
        data = file.readlines()
    return [line.strip() for line in data]

data_content = read_data_file()


def create_chain():
    PERSIST = False

    if not hasattr(app, 'chain'):
        loader = DirectoryLoader("data/")
        index = VectorstoreIndexCreator().from_loaders([loader])

        chain = ConversationalRetrievalChain.from_llm(
            llm=ChatOpenAI(model="gpt-3.5-turbo"),
            retriever=index.vectorstore.as_retriever(search_kwargs={"k": 1}),
        )
        app.chain = chain  

# defining flask routes for integration 
@app.route('/')
def index():
    create_chain()  
    return render_template('index.html')

@app.route('/get_prompt', methods=['POST'])
def get_prompt():
    create_chain()  

    query = request.form['query']
    chat_history = []  

    start_time = time.time() #timing response


    #checking data.txt
    matched_response = None
    for line in data_content:
        if query.lower() in line.lower():
            matched_response = line
            break

    if matched_response:
        response = matched_response
    else:
        # if no match
        result = app.chain({"question": query, "chat_history": chat_history})
        response = result['answer']

    elapsed_time = time.time() - start_time

    chat_history.append((query, response))

    return render_template('index.html', response=response, elapsed_time=elapsed_time)


if __name__ == '__main__':
    app.run(debug=True)