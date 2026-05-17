import os
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from operator import itemgetter

load_dotenv()

print('Initializing components...')

embeddings = OpenAIEmbeddings()
llm = ChatOpenAI()

vector_store = PineconeVectorStore(
    embedding=embeddings,
    index_name=os.getenv("INDEX_NAME")
    )

retriever = vector_store.as_retriever(search_kwargs={"k": 3})

prompt_template = ChatPromptTemplate.from_template(
    """Answer the question based only on the following context:

        {context}

        Question: {question}

        Provide a detailed answer:"""
)

def format_docs(docs):
    """Format retrieved documents into a single string."""
    return "\n\n".join(doc.page_content for doc in docs)

# ===============================================================================================
# IMPLEMENTATION 1: Without LCEL (LangChain Expression Language) (Simple Function-Based Approach)
# ===============================================================================================
def retrieval_chain_without_lcel(query: str):
    """
    Simple retrieval chain without LCEL.
    Manually retrieves documents, formats them, and generates a response.

    Limitations:
    - Manual step-by-step execution
    - No built-in streaming support
    - No async support without additional code
    - Harder to compose with other chains
    - More verbose and error-prone
    """
    # Step 1: Retrieve relevant documents
    docs = retriever.invoke(query)

    # Step 2: Format documents into context string
    context = format_docs(docs)

    # Step 3: Format the prompt with context and question
    messages = prompt_template.format_messages(context=context, question=query)

    # Step 4: Invoke LLM with the formatted messages
    response = llm.invoke(messages)

    # Step 5: Return the content
    return response.content


# ============================================================================
# IMPLEMENTATION 2: With LCEL (LangChain Expression Language) - BETTER APPROACH
# ============================================================================
def create_retrieval_chain_with_lcel():
    """
    Create a retrieval chain using LCEL (LangChain Expression Language).
    Returns a chain that can be invoked with {"question": "..."}

    Advantages over non-LCEL approach:
    - Declarative and composable: Easy to chain operations with pipe operator (|)
    - Built-in streaming: chain.stream() works out of the box
    - Built-in async: chain.ainvoke() and chain.astream() available
    - Batch processing: chain.batch() for multiple inputs
    - Type safety: Better integration with LangChain's type system
    - Less code: More concise and readable
    - Reusable: Chain can be saved, shared, and composed with other chains
    - Better debugging: LangChain provides better observability tools
    """

    # retrieval_chain = (
    #     RunnablePassthrough.assign(
    #         context=itemgetter("question") | retriever | format_docs
    #     )
    #     | prompt_template
    #     | llm
    #     | StrOutputParser()
    # )

    # Create a LangChain LCEL pipeline called retrieval_chain
    # This is a RAG pipeline: question -> retrieve docs -> build prompt -> LLM -> string answer
    retrieval_chain = (
            # RunnablePassthrough.assign keeps the original input dict and adds new keys to it
            RunnablePassthrough.assign(
                # Add a new key "context" to the input dict
                context=(
                    # itemgetter("question") pulls the "question" value from the input dict
                    # Example input: {"question": "What is RAG?"} -> outputs "What is RAG?"
                    itemgetter("question")

                    # | pipes that question string into the retriever
                    # retriever searches your vector DB and returns List[Document]
                    | retriever

                    # | pipes the List[Document] into format_docs
                    # format_docs should take docs and join them into 1 string
                    # Example: "\n\n".join([doc.page_content for doc in docs])
                    | format_docs
                )
            )
            # | pipes the dict {"question":..., "context":...} into the prompt
            # prompt_template uses both {question} and {context} placeholders
            | prompt_template

            # | sends the formatted ChatPromptValue to the LLM
            # llm can be ChatOpenAI, Ollama, etc. Returns an AIMessage
            | llm

            # | parses the LLM output and extracts just the text content
            # Converts AIMessage(content="...") -> "..."
            | StrOutputParser()
        )
    return retrieval_chain

if __name__ == "__main__":
    print("Retrieving...")

    # Query
    query = "what is Pinecone in machine learning?"

    # ========================================================================
    # Option 0: Raw invocation without RAG
    # ========================================================================
    print("\n" + "=" * 70)
    print("IMPLEMENTATION 0: Raw LLM Invocation (No RAG)")
    print("=" * 70)
    result_raw = llm.invoke([HumanMessage(content=query)])
    print("\nAnswer:")
    print(result_raw.content)

    # =========================================================================
    # Option 1: Use implementation WITHOUT LCEL (LangChain Expression Language)
    # =========================================================================
    print("\n" + "=" * 70)
    print("IMPLEMENTATION 1: Without LCEL")
    print("=" * 70)
    result_without_lcel = retrieval_chain_without_lcel(query)
    print("\nAnswer:")
    print(result_without_lcel)

    # ========================================================================
    # Option 2: Use implementation WITH LCEL (Better Approach)
    # ========================================================================
    print("\n" + "=" * 70)
    print("IMPLEMENTATION 2: With LCEL - Better Approach")
    print("=" * 70)
    print("Why LCEL is better:")
    print("- More concise and declarative")
    print("- Built-in streaming: chain.stream()")
    print("- Built-in async: chain.ainvoke()")
    print("- Easy to compose with other chains")
    print("- Better for production use")
    print("=" * 70)

    chain_with_lcel = create_retrieval_chain_with_lcel()
    result_with_lcel = chain_with_lcel.invoke({"question": query})
    print("\nAnswer:")
    print(result_with_lcel)