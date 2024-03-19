import pandas as pd
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableBranch
from langchain_core.runnables import RunnablePassthrough
from langchain_core.messages import HumanMessage
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores.faiss import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain.chains.combine_documents import create_stuff_documents_chain

class RAG:
    def __init__(self) -> None:
        self.vectorstore=None
        self.chain=None
        self.isCSV=False
        self.agent=None
        self.filename=""

        self.query_transform_prompt = ChatPromptTemplate.from_messages(
            [
                MessagesPlaceholder(variable_name="messages"),
                (
                    "user",
                    "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation. Only respond with the query, nothing else.",
                ),
            ]
        )
        SYSTEM_TEMPLATE = """
        Elaborate Answer the question based only on given context

        <context>
        {context}
        </context>
        Elaborate the answer related to the question from the context in english, unless stated otherwise.
        Dont state about the context in the answer.
        Answer:
        """

        self.question_answering_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    SYSTEM_TEMPLATE,
                ),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )

    def prepare_vectorstore(self, text:str)->None:
        text_splitter=RecursiveCharacterTextSplitter(separators=["\n"], chunk_size=1000, chunk_overlap=200, length_function=len)
        chunks=text_splitter.split_text(text)
        embeddings = HuggingFaceBgeEmbeddings(model_name="BAAI/bge-small-en", model_kwargs={"device": "cpu"}, encode_kwargs={"normalize_embeddings": True})
        self.vectorstore=FAISS.from_texts(texts=chunks,embedding=embeddings)

    def prepare_chain(self, text:str)->None:
        self.prepare_vectorstore(text)
        retriever=self.vectorstore.as_retriever()
        llm = ChatGoogleGenerativeAI(model='gemini-pro', temperature=0.3, convert_system_message_to_human=True, verbose=True)
        query_transforming_retriever_chain = RunnableBranch(
                (
                    lambda x: len(x.get("messages", [])) == 1,
                    (lambda x: x["messages"][-1].content) | retriever,
                ),
                self.query_transform_prompt | llm | StrOutputParser() | retriever,
            ).with_config(run_name="chat_retriever_chain")
        document_chain =create_stuff_documents_chain(llm, self.question_answering_prompt)
        conversational_retrieval_chain = RunnablePassthrough.assign(
            context=query_transforming_retriever_chain,
            ).assign(
                answer=document_chain,
            )
        self.chain=conversational_retrieval_chain

    def prepare_agent(self, df:pd.DataFrame)->None:
        self.agent=create_pandas_dataframe_agent(llm=ChatGoogleGenerativeAI(model='gemini-pro', temperature=0.3, convert_system_message_to_human=True), df=df, verbose=True)
    
    def predict(self, question: str) -> dict:
        if self.isCSV and self.agent:
            self.agent.run(question)
        elif not self.isCSV and self.chain:
            res=self.chain.invoke({
                            "messages": [
                                HumanMessage(content=question)
                            ]})['answer']
        else:
           res='Error: No file uploaded'
        return res
    