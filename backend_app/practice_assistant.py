from langchain_chroma import Chroma         
from langchain_community.chat_models.tongyi import ChatTongyi 
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import re 


def search_knowledge_for_practice_topic(topic_keywords, embeddings_model_instance, vector_store_dir, top_k=10):

    if not topic_keywords:
        print("No topic keywords provided for RAG search.")
        return []

    print(f"\nSearching knowledge base for practice materials related to: '{topic_keywords}' (top_k={top_k})...")

    try:
        vector_store = Chroma(
            persist_directory=vector_store_dir,
            embedding_function=embeddings_model_instance
        )
        
        retrieved_docs_with_scores = vector_store.similarity_search_with_score(topic_keywords, k=top_k)
        
        retrieved_contents = []
        if retrieved_docs_with_scores:
            print(f"Retrieved {len(retrieved_docs_with_scores)} relevant snippets from the knowledge base for practice context.")
            for doc, score in retrieved_docs_with_scores:
                retrieved_contents.append(doc.page_content)

        else:
            print("No relevant snippets found in the knowledge base for this practice topic.")
        
        return retrieved_contents
    except Exception as e:
        print(f"An error occurred during RAG search for practice materials: {e}")
        print("Ensure the vector store is correctly set up and ZHIPUAI_API_KEY is valid for embeddings.")
        return []

def construct_practice_question_prompt(practice_topic, question_preferences, student_history_summary=None,retrieved_context_snippets=None):

    system_message = (
        "你是一位教学经验丰富的AI助教，擅长根据特定知识点为学生生成练习题以巩固学习。请严格、精确地遵守题目类型和数量，以及输出格式的要求。"
    )

    # 将 {'选择题': 2} 转换成更自然的 "2道选择题"
    prefs_str = ", ".join([f"{count}道{q_type}" for q_type, count in question_preferences.items()])

    human_message_parts = []
    
    # --- 关键修改：在Prompt中加入对特殊标签的强制要求 ---
    human_message_parts.append(f"""
    请为学生创建一组关于以下主题的练习题。

    1.  **练习主题**:
        {practice_topic}

    2.  **必须遵守的题目要求 (最高优先级)**:
        - 练习题必须且只能包含以下题型和数量：{prefs_str}。
        - 请确保最终生成的题目数量与此要求完全一致。
        {student_history_summary}

    3.  3.  **必须严格遵守的输出格式要求 (这是最重要的指令！)**:
    - 你的输出必须被一个明确的分隔符分为两部分：题目部分和答案部分。
    - **首先，请完整地输出所有题目的题干和选项。**
    - **在所有题目内容完全结束后，你必须另起一个新行，并且这一行只能包含以下文本作为分隔符，不能有任何其他字符：**
      `---参考答案与解析---`
    - 在这个分隔符之后，再按顺序、完整地列出所有题目的答案和解析。
    - **请务必输出这个分隔符。**

        这是一个正确的输出结构示例：
        (所有题目...)

        ---参考答案与解析---

        (所有答案...)

        请现在严格按此格式生成。
    """)

    if retrieved_context_snippets:
        human_message_parts.append("\n### Relevant Context Snippets (for reference) ###")
        for i, snippet in enumerate(retrieved_context_snippets):
            human_message_parts.append(f"--- Snippet {i+1} ---\n{snippet}\n--- End Snippet {i+1} ---")
        human_message_parts.append("\nNote: Use these snippets to inform the questions, ensuring they are grounded in this context where appropriate.")
    else:
        human_message_parts.append("\n(No specific context snippets provided; generate general questions for the topic.)")
    

    human_message_content = "\n\n".join(human_message_parts)

    return {
        "system_message": system_message,
        "human_message": human_message_content
    }

def construct_feedback_prompt(question_text, model_answer_text, student_answer_text):

    system_message = (
        "你是一位友善、耐心且富有鼓励性的AI辅导老师。你的任务是为学生的练习题答案提供清晰、有建设性的即时反馈。请帮助学生理解知识点，而不是仅仅给出对错。"
    )

    human_message_content = f"""
        请为以下学生的练习回答提供反馈。

        2.  **参考答案**:
            {model_answer_text}

        3.  **学生的回答**:
            {student_answer_text if student_answer_text.strip() else "(学生没有提供答案)"}

        4.  **你的评判任务 (必须严格遵守以下格式)**:
            你的整个输出**必须**以一个单独的评估行开始，格式**必须**完全如下：
            `Overall Assessment: [assessment]`

            其中 `[assessment]` 必须是以下三个字符串之一：`Correct`, `Partially Correct`, `Incorrect`。
            **在这一行之前或之后，不要有任何其他文本或换行。你的回答的第一行就是它。**

            在输出了这第一行评估之后，你才可以开始提供详细的、有建设性的反馈，可以包括对错误原因的分析、值得肯定的地方以及改进建议。请保持鼓励的语气。

        这是一个正确的输出格式示例：
        Overall Assessment: Partially Correct
        很棒的尝试！你已经抓住了核心思想，但忽略了参考答案中提到的另一个关键因素。下次可以试着从这个角度再思考一下。

        请现在严格按此格式开始你的评判："""

    return {
        "system_message": system_message,
        "human_message": human_message_content
    }

def parse_feedback_and_correctness(llm_feedback_str):
    correctness = "Undetermined" 
    detailed_feedback = llm_feedback_str 

    if not llm_feedback_str or not llm_feedback_str.strip():
        return {"correctness": correctness, "detailed_feedback": "No feedback content received."}


    first_line_end_index = llm_feedback_str.find('\n')
    first_line = llm_feedback_str[:first_line_end_index if first_line_end_index != -1 else len(llm_feedback_str)].strip()

    assessment_prefix = "Overall Assessment: "
    if first_line.startswith(assessment_prefix):
        status_str = first_line[len(assessment_prefix):].strip()

        valid_statuses = ["Correct", "Partially Correct", "Incorrect"]
        if status_str in valid_statuses:
            correctness = status_str

            if first_line_end_index != -1:
                detailed_feedback = llm_feedback_str[first_line_end_index + 1:].strip()
            else: 
                detailed_feedback = "" 
            print(f"Parsed correctness: {correctness}") 
        else:
            print(f"Warning: Parsed status '{status_str}' is not one of {valid_statuses}. Using raw feedback.")

    else:
        print("Warning: LLM feedback did not start with 'Overall Assessment:'. Using raw feedback.")


    return {"correctness": correctness, "detailed_feedback": detailed_feedback}

def get_llm_feedback_on_answer(system_prompt, human_prompt):

    try:

        llm = ChatTongyi(temperature=0.7) 
    except Exception as e:
        print(f"Error initializing LLM (ChatTongyi) for feedback. Ensure DASHSCOPE_API_KEY is set. Details: {e}")
        return None

    prompt_template = ChatPromptTemplate.from_messages([
        ("system", "{system_message_var}"),
        ("human", "{human_message_var}")
    ])
    output_parser = StrOutputParser()
    chain = prompt_template | llm | output_parser

    print("\nRequesting feedback on student's answer from LLM...")
    try:
        response = chain.invoke({
            "system_message_var": system_prompt,
            "human_message_var": human_prompt
        })
        return response
    except Exception as e:
        print(f"An error occurred during LLM interaction for feedback: {e}")
        return None

def get_llm_practice_questions(system_prompt, human_prompt):

    try:

        llm = ChatTongyi(temperature=0.6) 
    except Exception as e:
        print(f"Error initializing LLM (ChatTongyi). Ensure DASHSCOPE_API_KEY is set. Details: {e}")
        return None

    prompt_template = ChatPromptTemplate.from_messages([
        ("system", "{system_message_var}"),
        ("human", "{human_message_var}")
    ])
    output_parser = StrOutputParser()
    chain = prompt_template | llm | output_parser

    print("\nRequesting practice questions from LLM...")
    try:
        response = chain.invoke({
            "system_message_var": system_prompt,
            "human_message_var": human_prompt
        })
        return response
    except Exception as e:
        print(f"An error occurred during LLM interaction for practice question generation: {e}")
        return None

def parse_questions_and_answers(llm_response_str):
    parsed_qa_pairs = []
    if not llm_response_str or not llm_response_str.strip():
        return parsed_qa_pairs


    qa_blocks = re.findall(r"\[START_QUESTION\](.*?)\[END_QUESTION\]\s*\[START_MODEL_ANSWER\](.*?)\[END_MODEL_ANSWER\]", llm_response_str, re.DOTALL)

    for q_content, a_content in qa_blocks:
        question_full = q_content.strip() 
        model_answer = a_content.strip()
        

        q_type = "Unknown"
        question_text_only = question_full 
        
        type_match = re.search(r"Type:\s*(.*)", question_full)
        if type_match:
            q_type = type_match.group(1).strip()

            question_text_only = re.sub(r"Type:\s*.*\n?", "", question_full, count=1).strip()


        parsed_qa_pairs.append({
            "question_type": q_type, 
            "question": question_text_only,
            "model_answer": model_answer
        })

    if not parsed_qa_pairs and llm_response_str.strip(): 
        print("Warning: Could not parse any Q&A pairs from LLM response using regex. LLM output might not conform to expected format. Displaying raw output as a fallback can be implemented if desired.")

        pass

    return parsed_qa_pairs

