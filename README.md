# Chat Documents

## Preface
Chat document is a chatbot application which takes a file as an input and answers user's query. The goal of this application is to accurately provide answers based on the uploaded file. This application could be used as an assistant to quickly answer questions or summarize facts from files containing large amounts of text data, making our lives easier.

## Project structure

In this project you find 2 directories

1. `backend` containing the server side **python** code
2. `frontend` containing the client side **typescript** code.

### **Backend**
The RAG architecture is implemented in a seperate module `rag.py` with a class named RAG. 
The three main components of this class are:
   
1. `isCSV`: it works like a toggle to switch betweem the pandas dataframe agent to query csv file and the chain to query documents like text, pdf, docx.
2. `agent`: it is an agent to talk with pandas dataframe created using *'create_pandas_dataframe_agent'*.
3. `chain`: It is conversational_retrieval_chain that implements the RAG architecture using BGE-Embeddings and Gemini-pro model. It uses two chains for retrieval with history awareness and text generation.

### Discussing conversational_retrieval_chain

It is a sequence of following two chains 
   - **query_transforming_retriever_chain**: This is LCEL chain takes the previous `chat_history` and the current question to generate a serch query to search for relevant information form the vectorstore. It contains `SrtOutputParser` to make sure the output is a string from the chain.

   - **document_chain**: It is a simple stuff_document_chain that uses the orignal question and the context received from the retriver to generate the final answer.

   Both the above chain use a seprate prompt to get the best results from the gemini model.

`RunnablePassthrough` is used to pass the output from the retriver as the context and get the final answer from the document chain.
   
### Some other functions of the RAG class 
   - **prepare_chain**: This function is used to prepare the conversation_retrieval_chain. It first takes the raw text as input and used the prepare_vectorstore function to create the vectorstore.
   - **prepare_agent**: it is used to create the pandas dataframe agent in case user uploads a csv file.
   - **predict**: it uses the toggle `isCSV` to decide to use agent or chain. It is also responsible to maintain `chat_history` by adding the user question and chain out. We are using it as a chain only because gemini-pro is not compatible with takin a list of `AIMessage` and `HumanMessage` as input.

### Server side code 
The predict endpoint handles the post request from the user. It uses an object of class RAG to implement the function of this application. 

The predict function takes care of all the edge case like filesize>100mb or unsupportd file type.
It uses the filename member variable to avoid generating the vectorstore for each call for the same file. 

#### Running the backend server

To launch the server, navigate to the `backend` directory and run:

##### `uvicorn main:app --reload`

This will start the server at [http://127.0.0.1:8000/](http://127.0.0.1:8000/)

## **Frontend**

The frontend is implemented using react and tailwind for styling.

#### How to launch the react app

1. Navigate to the `frontend` directory and run `npm install`
2. Then you can run:

   ##### `npm start`

   This will launch the app in development mode.\
   Open [http://localhost:3000](http://localhost:3000) to view it in the browser.



