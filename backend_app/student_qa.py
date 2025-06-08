import os
from typing import List 
from langchain_chroma import Chroma
from langchain_community.chat_models import ChatZhipuAI # For LLM interaction
from langchain_core.output_parsers import StrOutputParser # For LLM interaction
from langchain.prompts import ChatPromptTemplate

from backend_app.models import ChatMessage


def search_knowledge_base_for_answer(student_question, embeddings_model_instance, vector_store_dir, top_k=10):
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
    human_message_parts.append(f"\nStudent's Question: {student_question}")

    human_message_content = "\n".join(human_message_parts)
    # This function is likely for initial, non-history based QA.
    # The new refine_student_question_service will use a different approach for history.
    # However, if this function is STILL used for initial QA, its output needs to be messages.
    # For now, let's assume it's for initial and keep its structure,
    # but the refine service will call a new version or directly construct messages.

    # To align with the goal of returning messages for LLM:
    from langchain_core.messages import SystemMessage, HumanMessage
    return [
        SystemMessage(content=system_message),
        HumanMessage(content=human_message_content)
    ]

# Adapted for refine flow primarily
def construct_student_qa_prompt_with_history(
    history_messages: list, # List of Langchain HumanMessage, AIMessage
    new_query_content: str, 
    rag_snippets: list[str]
) -> list:
    from langchain_core.messages import SystemMessage, HumanMessage

    # System message remains consistent for QA tasks
    system_message_content = (
        "You are a helpful and friendly teaching assistant. Your primary role is to answer the student's "
        "question based *only* on the provided excerpts from their learning materials (if any) and the conversation history. "
        "If the excerpts and history do not contain enough information to answer the question thoroughly, "
        "clearly state that the information is not fully available in the provided context. "
        "Do not use any external knowledge or make assumptions beyond the provided text. "
        "Answer concisely and directly to the student's query."
    )
    
    # Start with the system message
    final_messages = [SystemMessage(content=system_message_content)]
    
    # Add history messages
    final_messages.extend(history_messages)
    
    # Construct the content for the latest human query, including RAG snippets
    human_message_parts = []
    if rag_snippets:
        human_message_parts.append("Here are some relevant excerpts from your learning materials that might help answer your question:")
        human_message_parts.append("--- Start of Excerpts ---")
        for i, snippet in enumerate(rag_snippets):
            human_message_parts.append(f"[Excerpt {i+1}]:\n{snippet}")
        human_message_parts.append("--- End of Excerpts ---")
        human_message_parts.append("\nBased on the conversation history and *only* on the excerpts provided above (if any), please answer the following question:")
    else:
        human_message_parts.append("\n(No specific excerpts were retrieved for your latest question.)")
        human_message_parts.append("\nBased on the conversation history, please answer the following question:")
    
    human_message_parts.append(f"Student's Question: {new_query_content}")
    
    final_human_content = "\n".join(human_message_parts)
    final_messages.append(HumanMessage(content=final_human_content))
    
    return final_messages

# Accepts a list of Langchain message objects
def get_llm_response_to_student(messages: list, rag_snippets: list[str] = None): # rag_snippets no longer needed here if construct_student_qa_prompt_with_history is used
    try:
        llm = ChatZhipuAI(temperature=0.3) # Lower temp for more factual Q&A
    except Exception as e:
        print(f"Error initializing LLM (ChatZhipuAI). Details: {e}")
        return None

    output_parser = StrOutputParser()
    chain = llm | output_parser # Simpler chain

    print("\nAsking LLM for an answer based on provided messages and context...")
    try:

        response = chain.invoke(messages) # Pass the list of messages directly
        return response
    except Exception as e:
        print(f"An error occurred during LLM Q&A interaction: {e}")
        return None
    
async def _rewrite_query_with_history(chat_history: List[ChatMessage], new_query: str) -> str:
    """
    使用LLM根据对话历史重写用户的新查询，使其成为一个独立的、可用于搜索的问题。
    """
    print("SERVICE (Refine): Rewriting query with history...")
    
    history_str = "\n".join([f"{msg.role}: {msg.content}" for msg in chat_history])
    
    rewrite_prompt_template = ChatPromptTemplate.from_messages([
        ("system", "你是一个查询优化助手。你的任务是根据对话历史，将用户最新的、可能不完整的后续问题，改写成一个独立的、包含完整上下文的、可以用于搜索引擎或向量数据库检索的问题。只输出改写后的问题，不要添加任何额外解释。"),
        ("human", f"以下是对话历史:\n---\n{history_str}\n---\n\n用户的最新问题是: \"{new_query}\"\n\n请将这个最新问题改写成一个独立的、完整的检索查询:")
    ])
    
    try:
        llm = ChatZhipuAI(model="glm-4", temperature=0.0) 
        chain = rewrite_prompt_template | llm | StrOutputParser()
        rewritten_query = await chain.ainvoke({})
        print(f"SERVICE (Refine): Original query: '{new_query}', Rewritten query: '{rewritten_query}'")
        return rewritten_query.strip()
    except Exception as e:
        print(f"SERVICE ERROR (Refine): Failed to rewrite query: {e}. Falling back to original query.")
        return new_query 

