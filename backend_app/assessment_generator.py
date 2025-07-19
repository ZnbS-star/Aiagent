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
    

def construct_assessment_prompt(teaching_plan_content, retrieved_rag_snippets, question_preferences, subject=None):
    # System Message 保持不变
    system_message = (
        "你是一位精通多学科的资深教育出题专家。你的任务是根据指定的学科背景、核心教学内容和题目要求，生成一份高质量、专业且格式规范的考核试卷。"
    )

    # Human Message Construction
    human_message_parts = []
    
    subject_context = subject if subject and subject.strip() else "通用"
    
    human_message_parts.append(f"""
        请严格按照以下要求，为 **“{subject_context}”** 学科创建一份考核试卷。

        **1. 核心出题依据 (Primary Source Material):**
        {teaching_plan_content}
        """)

    if retrieved_rag_snippets:
        # This part remains unchanged
        rag_context_info = "\n".join([f"- {snippet}" for snippet in retrieved_rag_snippets])
        human_message_parts.append(f"""
                    **2. 补充参考材料 (Retrieved Context):**
                        {rag_context_info}
        """)

    human_message_parts.append(f"""
**3. 必须遵守的题目要求 (Strict Requirements):**
这份试卷必须且只能包含以下题型和数量：**{question_preferences}**。
请确保最终生成的题目总数和每种题型的数量与此要求完全一致。

**4. 格式与内容规范 (Formatting and Content Rules) - 这是最高优先级的指令！**
- 对于每种题型，请先写出大标题（例如：“一、选择题”）。
- 请为每个题目进行全局连续编号（例如“题目1”、“题目2”...）。
- **在试卷的最后，你必须另起一行，并提供一个清晰的、标题为“---参考答案与解析---”的部分。**
- **所有题目的答案和解析都必须且只能在这个部分统一列出。**
- 试卷的题目区域（在“---参考答案与解析---”之前）必须保持纯净，只包含题干和选项。
""")
    
    if "编程题" in question_preferences:
        human_message_parts.append("""
- **如果要求包含“编程题”，请务必遵循以下结构：**
    - **问题描述:** 清晰、无歧义地描述需要解决的问题。
    - **输入格式:** 明确说明输入的格式和数据类型。
    - **输出格式:** 明确说明期望的输出格式。
    - **示例:** 提供至少一个清晰的“输入/输出”示例。
    - **编程题的答案部分必须提供：**
        - (a) 一份可直接运行的、注释良好的示例代码。
        - (b) 对关键代码逻辑或算法思想的简要解析。""")

    human_message_parts.append("\n请现在开始生成试卷。")
    
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
    history: List, 
    new_query:str,
    rag_snippets: List[str],
    user_intent:str
) -> List:
    if user_intent == "REWRITE":
        system_prompt = "根据用户的最新指令，生成一份【全新】的考核试卷。请忽略所有历史和原始试卷内容。"
    elif user_intent == "REVISION" or user_intent == "Deletion":
        system_prompt = "根据原始试卷和用户的最新指令，生成一份【修改后】的【完整】考核试卷。你的输出应该是替换掉整个旧试卷的新版本。"
    else: 
        system_prompt = (
            "你是一个出题专家。你的任务是根据用户的【新增要求】，为一份【已有的试卷】生成补充题目。"
            "你的输出【必须只包含新增的题目及其答案和解析】，并且必须遵循与原始试卷一致的格式（例如，先写题目，最后在`---参考答案与解析---`部分统一给出答案）。"
            "你的回答中【绝对不能】重复任何已有的、未被修改的题目。"
        )
    final_messages_for_llm = [SystemMessage(content=system_prompt)]
    final_messages_for_llm.append(f"""**. 格式与内容规范 (Formatting and Content Rules):**
- 对于每种题型，请先写出大标题（例如：“一、选择题”）。
- **请为每个题目进行全局连续编号（例如“题目1”、“题目2”...）。**
- **如果要求包含“编程题”(Programming Question)，请务必遵循以下结构：**
    - **问题描述:** 清晰、无歧义地描述需要解决的问题。
    - **输入格式:** 明确说明输入的格式和数据类型。
    - **输出格式:** 明确说明期望的输出格式。
    - **示例:** 提供至少一个清晰的“输入/输出”示例，帮助学生理解。
    - **(可选) 限制/提示:** 如有必要，提供时间/空间复杂度限制或解题提示。

- 在试卷的**最后**，请提供一个清晰的“**参考答案与解析**”部分，其中：
    - 列出所有题目的正确答案。
    - **对于编程题，答案部分必须提供:**
        - **(a) 一份可直接运行的、注释良好的示例代码 (推荐使用Python或与主题最相关的语言)。**
        - **(b) 对关键代码逻辑或算法思想的简要解析。**

请现在开始生成试卷。""")
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
