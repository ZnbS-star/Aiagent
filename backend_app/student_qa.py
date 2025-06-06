import os # For environment variables & path checks
from langchain_chroma import Chroma
from langchain_community.chat_models import ChatZhipuAI # For LLM interaction
from langchain_core.prompts import ChatPromptTemplate   # For LLM interaction
from langchain_core.output_parsers import StrOutputParser # For LLM interaction
# import sys # Only if sys.stdin.read() was planned, not for basic input()


def search_knowledge_base_for_answer(student_question, embeddings_model_instance, vector_store_dir, top_k=10):
    """
    Performs a RAG search using the student's question against a Chroma vector store.

    Args:
        student_question (str): The student's question.
        embeddings_model_instance: An initialized ZhipuAIEmbeddings instance.
        vector_store_dir (str): The directory of the Chroma vector store.
        top_k (int): The number of top documents to retrieve.

    Returns:
        list: A list of strings, where each string is the page_content of a retrieved document.
              Returns an empty list if no documents are found or an error occurs.
    """
    if not student_question:
        print("No question provided for RAG search.")
        return []

    print(f"\nSearching knowledge base for answers related to: '{student_question}' (top_k={top_k})...")

    try:
        # The embeddings_model_instance is passed in, already initialized.
        # ZHIPUAI_API_KEY should be set in the environment for it to work.
        if not os.path.exists(vector_store_dir):
            print(f"Error: Chroma DB directory '{vector_store_dir}' not found. Cannot perform RAG search.")
            return []

        vector_store = Chroma(
            persist_directory=vector_store_dir,
            embedding_function=embeddings_model_instance
        )
        
        retrieved_docs_with_scores = vector_store.similarity_search_with_score(student_question, k=top_k)
        
        retrieved_contents = []
        if retrieved_docs_with_scores:
            print(f"Retrieved {len(retrieved_docs_with_scores)} relevant snippets from the knowledge base.")
            for doc, score in retrieved_docs_with_scores:
                retrieved_contents.append(doc.page_content)
                # print(f"  Score: {score:.4f} - Snippet: {doc.page_content[:100]}...") # Optional for debugging
        else:
            print("No relevant snippets found in the knowledge base for this question.")
        
        return retrieved_contents
    except Exception as e:
        print(f"An error occurred during RAG search: {e}")
        print("Ensure the vector store is correctly set up and ZHIPUAI_API_KEY is valid for embeddings.")
        return []

def construct_student_qa_prompt(student_question, rag_snippets):
    """
    Constructs the system and human messages for the student Q&A prompt.

    Args:
        student_question (str): The student's original question.
        rag_snippets (list): A list of strings, where each string is a relevant snippet 
                             from the knowledge base.

    Returns:
        dict: A dictionary containing "system_message" and "human_message".
    """

    system_message = (
        "You are a helpful and friendly teaching assistant. Your primary role is to answer the student's "
        "question based *only* on the provided excerpts from their learning materials. "
        "If the excerpts do not contain enough information to answer the question thoroughly, "
        "clearly state that the information is not fully available in the provided context. "
        "Do not use any external knowledge or make assumptions beyond the provided text. "
        "Answer concisely and directly to the student's query."
    )

    human_message_parts = []
    human_message_parts.append("Here are some relevant excerpts from your learning materials that might help answer your question:")
    human_message_parts.append("\n--- Start of Excerpts ---") # Corrected newline
    if rag_snippets:
        for i, snippet in enumerate(rag_snippets):
            human_message_parts.append(f"\n[Excerpt {i+1}]:\n{snippet}") # Corrected newlines
    else:
        human_message_parts.append("\n(No specific excerpts were retrieved that seem relevant to your question.)") # Corrected newline
    human_message_parts.append("\n--- End of Excerpts ---\n") # Corrected newline
    human_message_parts.append("Based *only* on the excerpts provided above (if any), please answer the following question:")
    human_message_parts.append(f"\nStudent's Question: {student_question}") # Corrected newline

    human_message_content = "\n".join(human_message_parts)

    return {
        "system_message": system_message,
        "human_message": human_message_content
    }

def get_llm_response_to_student(system_prompt, human_prompt):
    """
    Sends the prompt to the LLM and returns the response.

    Args:
        system_prompt (str): The system message for the LLM.
        human_prompt (str): The human message for the LLM (containing question and context).
        llm_api_key (str): The API key for the LLM.

    Returns:
        str: The LLM's response, or None if an error occurs.
    """
    try:
        llm = ChatZhipuAI(temperature=0.3) # Lower temp for more factual Q&A
    except Exception as e:
        print(f"Error initializing LLM (ChatZhipuAI). Details: {e}")
        return None

    prompt_template = ChatPromptTemplate.from_messages([
        ("system", "{system_message_var}"),
        ("human", "{human_message_var}")
    ])
    output_parser = StrOutputParser()
    chain = prompt_template | llm | output_parser

    print("\nAsking LLM for an answer based on provided context...")
    try:
        response = chain.invoke({
            "system_message_var": system_prompt,
            "human_message_var": human_prompt
        })
        return response
    except Exception as e:
        print(f"An error occurred during LLM Q&A interaction: {e}")
        return None

