from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_core.output_parsers import StrOutputParser
from langchain_chroma import Chroma
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage # Added
from typing import List # To use List type hint


def perform_rag_search(keywords_list, embeddings_model_instance, vector_store_dir, top_k=10):
    if not keywords_list:
        print("No keywords provided for RAG search.")
        return []
    query = " ".join(keywords_list) # Combine keywords into a single query string
    try:
        vector_store = Chroma(
            persist_directory=vector_store_dir,
            embedding_function=embeddings_model_instance
        )
        retrieved_docs_with_scores = vector_store.similarity_search_with_score(query, k=top_k)
        retrieved_contents = []
        if retrieved_docs_with_scores:
            for doc, score in retrieved_docs_with_scores:
                retrieved_contents.append(doc.page_content)
        else:
            print("No relevant documents found in the knowledge base for the keywords.")
        return retrieved_contents
    except Exception as e:
        print(f"An error occurred during RAG search: {e}")
        print("Ensure the vector store exists at the specified directory and ZHIPUAI_API_KEY is valid for embeddings.")
        return []
    

def construct_assessment_prompt(teaching_plan_content, retrieved_rag_snippets, question_preferences):
    # System Message
    system_message = (
        "你是一位严格且专业的出题专家。你的任务是根据提供的核心教学内容和题目要求，生成一份高质量的考核试卷。你必须严格、精确地遵守题目类型和数量的要求，不能多也不能少。最终输出必须是完整的考核内容，语言为中文。"
    )
    

    # Human Message Construction
    human_message_parts = []
    human_message_parts.append(f"""
    请严格按照以下要求创建一份考核试卷。

    1.  **核心教学内容**:
        {teaching_plan_content}

    2.  **必须遵守的题目要求 (这是最高优先级的指令)**:
        - **这份试卷必须且只能包含以下题型和数量：{question_preferences}。**
        - **请确保最终生成的题目总数和每种题型的数量与此要求完全一致。**
        - **在生成完毕后，请在心里默数一下，检查数量是否正确。**
        - **这是一个严格的限制，不能多，也不能少。**

    3.  **输出格式要求**:
        - 对于每种题型，请先写出题型标题（例如：“一、选择题”）。
        - **请为每个题目进行编号，例如“题目1”、“题目2”...**
        - 每个题目都需要有清晰的题干。
        - 选择题需要提供 A, B, C, D 等选项。
        - 在试卷的最后，请提供一个清晰的“参考答案”部分，列出所有题目的正确答案和必要的解析。""")

    # Add RAG snippets if available
    if retrieved_rag_snippets: # Check if the list is not empty
        human_message_parts.append("\n### Relevant Excerpts from Source Textbooks (for additional context) ###")
        for i, snippet in enumerate(retrieved_rag_snippets):
            human_message_parts.append(f"--- Snippet {i+1} ---\n{snippet}\n--- End Snippet {i+1} ---")
        human_message_parts.append("\nNote: These excerpts are for context. Generate questions PRIMARILY based on the Teaching Plan Content provided above, using these excerpts for clarification or detail where appropriate.")
    else:
        human_message_parts.append("\n(No additional textbook excerpts were retrieved or provided for context.)")

    human_message_content = "\n".join(human_message_parts)

    return {
        "system_message": system_message,
        "human_message": human_message_content
    }

# Accepts a list of Langchain message objects
def generate_assessment_with_llm(messages: list):
    try:
        # Assuming DASHSCOPE_API_KEY is in the environment
        llm = ChatTongyi(temperature=0.7)
    except Exception as e:
        print(f"Error initializing LLM (ChatTongyi). Ensure DASHSCOPE_API_KEY is set correctly. Details: {e}")
        return None

    # The LLM can directly take a list of message objects
    output_parser = StrOutputParser()
    chain = llm | output_parser # Simpler chain

    print("\nSending request to LLM for assessment generation...")
    try:
        # The 'messages' parameter should be a list of Langchain HumanMessage, AIMessage, SystemMessage objects
        response = chain.invoke(messages) # Pass the list of messages directly
        return response
    except Exception as e:
        print(f"An error occurred during LLM interaction: {e}")
        return None

def construct_assessment_prompt_with_history(
    history: List, # List of Langchain HumanMessage, AIMessage, SystemMessage
    new_query:str,
    rag_snippets: List[str],
    user_intent:str
) -> List:
    if user_intent == "REWRITE":
        system_prompt = "根据用户的最新指令，生成一份【全新】的考核试卷。请忽略所有历史和原始试卷内容。"
    elif user_intent == "REVISION" or user_intent == "STYLE_CHANGE":
        system_prompt = "根据原始试卷和用户的最新指令，生成一份【修改后】的【完整】考核试卷。你的输出应该是替换掉整个旧试卷的新版本。"
    else: # INCREMENTAL_ADD 或 UNKNOWN
        system_prompt = "根据原始试卷和用户的最新指令，【只生成需要新增或修改】的那部分内容。不要重复原始试卷中未被修改的部分。"
    final_messages_for_llm = [SystemMessage(content=system_prompt)]
    final_messages_for_llm.extend(history)
    final_messages_for_llm.extend(new_query)

    if rag_snippets:
        rag_context_str = "\n--- Relevant Context from Knowledge Base ---\n"
        for i, snippet in enumerate(rag_snippets):
            rag_context_str += f"[Snippet {i+1}]: {snippet}\n"
        rag_context_str += "--- End of Context ---"
        
        # Add RAG context as a new HumanMessage
        final_messages_for_llm.append(HumanMessage(content=rag_context_str))

    return final_messages_for_llm
