from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_openai.chat_models import ChatOpenAI
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from  sentence_transformers.cross_encoder import CrossEncoder

def get_multiqueries(query,n):
    '''
    Returns n multiqueries similar to original query
    '''
    llm = ChatOpenAI()
    prompt_multiquery = ChatPromptTemplate.from_template("""Create  multiple queries similar to the input query given number of queries to create (n) :
    query : {query}
    n: {n}
    Do not put any number or formatting in front of queries
    """)
    output_parser =  StrOutputParser()
    chain = prompt_multiquery | llm | output_parser
    multiquery = chain.invoke({"query":query,"n":n}).split('\n')
    return multiquery


def get_docs(filepath):
    loader = PyPDFLoader(filepath)
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter()
    documents = text_splitter.split_documents(docs)
    return documents

def create_faiss_vector_index(documents):
    embeddings = OpenAIEmbeddings()
    vector = FAISS.from_documents(documents, embeddings)
    return vector

def multiquery_similarity_search(multiquery,vector):
    '''
    Returns similarity search results for each of the multiquery from the vector db
    '''
    
    multiquery_ss_results = []
    for query in multiquery : 
        query_results =  vector.similarity_search(query)
        multiquery_ss_results = multiquery_ss_results + [result.page_content for result in query_results ]
    multiquery_ss_results = list(set(multiquery_ss_results))
    
    return multiquery_ss_results


def get_cross_encoder_score(multiquery_ss_results,query):
    '''
    Returns bert cross encoder scores for original query to the multiquery from vector db 
    '''
    model = CrossEncoder("cross-encoder/stsb-distilroberta-base")
    score_query_pair = [[q,model.predict([[q,query]][0])] for q in multiquery_ss_results]
    score_query_pair.sort(key=lambda x : x[1],reverse=True)
    return score_query_pair

def limp_rerank(sorted_list):
    '''
    Re-ranks the vector result to solve lost in the middle problem (limp):
    [1,2,3,4,5,6] -> [1,3,5,6,4,2]
    '''
    beginning_list = [sorted_list[i] for i in range(0,len(sorted_list),2)]
    end_list = [sorted_list[i] for i in range(1,len(sorted_list),2)][::-1]
    return  beginning_list + end_list

def get_context(query,vector,num_multiqueries=3):
    
    multi_query = get_multiqueries(query,num_multiqueries)
    multiquery_ss_result = multiquery_similarity_search(multi_query,vector)
    score_context_pair = get_cross_encoder_score(multiquery_ss_result,query)
    score_context_pair_reranked = limp_rerank(score_context_pair)
    context_reranked = [q[0] for q in score_context_pair_reranked]
    
    context = ('\n').join(context_reranked)
    return context

def run():
    document_file = input("Enter path to the Document")
    documents = get_docs(document_file)
    vector = create_faiss_vector_index(documents) 
    llm = ChatOpenAI()
    prompt = ChatPromptTemplate.from_template("""Answer the following question based only on the provided context:

    <context>
    {context}
    </context>

    Question: {input}""")
    output_parser = StrOutputParser()
    chain = prompt | llm | output_parser
    query = input("Enter your query (type END to terminate the program):")
    while(query!="END"):
        context = get_context(query,vector)
        for chunk in   chain.stream({"input":query,"context":context}) :
            print(chunk,end="",flush = True)
        query = input("Enter your query (type END to terminate the program):")
        
if __name__ == "__main__" : 
    
    run()
        
    