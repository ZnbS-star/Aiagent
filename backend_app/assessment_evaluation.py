import sys # For potential sys.exit()
import os # For environment variables
import mysql.connector # For mysql.connector.Error
from langchain_community.chat_models import ChatZhipuAI # For LLM evaluation
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import sys # For potential sys.exit()
import os # For environment variables
from backend_app.database_utils import get_mysql_connection, get_or_create_student, save_student_assessment_answer # For database interaction
import mysql.connector # For mysql.connector.Error
from langchain_community.chat_models import ChatZhipuAI # For LLM evaluation
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.embeddings import ZhipuAIEmbeddings # For RAG
from langchain_chroma import Chroma         # For RAG


def get_assessment_content_by_id(db_conn, assessment_id):

    if not db_conn:
        print("No database connection provided to get_assessment_content_by_id.")
        return None
    
    cursor = db_conn.cursor(dictionary=True)
    try:
        # --- 关键修改：查询新的列 ---
        sql_query = """
            SELECT 
                id, title, teacher_id, created_at,
                questions_text, 
                answers_text ,
                subject
            FROM assessments 
            WHERE id = %s
        """
        cursor.execute(sql_query, (assessment_id,))
        assessment_data = cursor.fetchone()
        
        if assessment_data:
            questions = assessment_data.get('questions_text', '')
            answers = assessment_data.get('answers_text', '')
            
            assessment_data['content'] = f"{questions}\n\n---参考答案与解析---\n\n{answers}"
            
            # （可选）可以删除原始字段，避免混淆
            # del assessment_data['questions_text']
            # del assessment_data['answers_text']

            return assessment_data
        else:
            print(f"No assessment found with ID: {assessment_id}")
            return None
    except mysql.connector.Error as err:

        print(f"Error retrieving assessment with ID {assessment_id}: {err}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred in get_assessment_content_by_id: {e}")
        return None
    finally:
        cursor.close()

def get_rag_context_for_grading(question_text, embeddings_model_instance, vector_store_dir, top_k=2): # top_k=2 for concise context
    if not question_text:
        print("No question text provided for RAG search.")
        return []

    print(f"\nPerforming RAG search for grading context related to: '{question_text[:100]}...' (top_k={top_k})")

    try:
        if not os.path.exists(vector_store_dir):
            print(f"Warning: Chroma DB directory '{vector_store_dir}' not found. Cannot perform RAG search for grading.")
            return []

        vector_store = Chroma(
            persist_directory=vector_store_dir,
            embedding_function=embeddings_model_instance
        )
        
        retrieved_docs = vector_store.similarity_search(question_text, k=top_k) # Using similarity_search
        
        retrieved_contents = []
        if retrieved_docs:
            print(f"Retrieved {len(retrieved_docs)} RAG snippets for grading context.")
            for doc in retrieved_docs:
                retrieved_contents.append(doc.page_content)
        else:
            print("No relevant RAG snippets found for grading context.")
        return retrieved_contents
    except Exception as e:
        print(f"An error occurred during RAG search for grading: {e}")
        return []

def construct_evaluation_prompt(assessment_full_content, question_identifier, student_answer_text, rag_context_snippets=None):
    system_message = (
        "你是一位严格、公正、专业的AI评分助教。你的任务是根据提供的'正确答案与解析'，来评判'学生答案'是否正确。请给出明确的评判结果和简要理由。"
    )

    human_message_parts = []
    human_message_parts.append("### Full Assessment Content ###")
    human_message_parts.append(assessment_full_content)
    
    human_message_parts.append(f"\n### Question to Evaluate (Identifier) ###")
    human_message_parts.append(question_identifier)
    
    human_message_parts.append("\n### Student's Submitted Answer for the above question ###")
    human_message_parts.append(student_answer_text if student_answer_text.strip() else "(No answer provided by student for this question)")

    if rag_context_snippets:
        human_message_parts.append("\n### Relevant Context from Knowledge Base (for your reference during evaluation) ###")
        for i, snippet in enumerate(rag_context_snippets):
            human_message_parts.append(f"--- Context Snippet {i+1} ---\n{snippet}\n--- End Snippet {i+1} ---")
        human_message_parts.append("\nNote: Use these context snippets to verify factual accuracy, understand expected depth, or identify nuances in the student's answer where applicable.")
    else:
        human_message_parts.append("\n(No additional context from knowledge base was retrieved for this question.)")

    human_message_parts.append("\n### 你的评判任务 (必须严格遵守以下格式) ###")
    human_message_parts.append(
        """
            请根据以上所有信息，对学生的答案进行评判。
            你的输出必须严格遵循以下两行格式，不要添加任何额外的前导或后置文字：

            第一行：必须以 "Correctness: " 开头，后面跟上你的评判结果。评判结果只能是以下四个选项之一：`Correct`, `Partially Correct`, `Incorrect`, `Cannot Determine`。如果没有答案的话，直接判断是`Incorrect`
            第二行及以后：是你的详细评语和解释 (Detailed Feedback)。

            这是一个正确的输出格式示例：
            Correctness: Partially Correct
            这名学生答对了一半的要点，但忽略了另一个关键因素。

            请现在开始你的评判：
            """
    )

    human_message_content = "\n\n".join(human_message_parts)

    return {
        "system_message": system_message,
        "human_message": human_message_content
    }

def get_llm_evaluation_for_answer(system_prompt, human_prompt, llm_api_key):
    try:
        # Using ChatZhipuAI for evaluation. Temperature might be lower for more deterministic eval.
        # Assuming ChatZhipuAI defaults to reading ZHIPUAI_API_KEY from environment if api_key is not provided.
        llm = ChatZhipuAI(temperature=0.4) 
    except Exception as e:
        print(f"Error initializing LLM (ChatZhipuAI) for evaluation. Ensure ZHIPUAI_API_KEY is set in environment. Details: {e}")
        return None

    prompt_template = ChatPromptTemplate.from_messages([
        ("system", "{system_message_var}"),
        ("human", "{human_message_var}")
    ])
    output_parser = StrOutputParser()
    chain = prompt_template | llm | output_parser

    print("\nRequesting LLM evaluation of the student's answer...")
    try:
        response = chain.invoke({
            "system_message_var": system_prompt,
            "human_message_var": human_prompt
        })
        return response
    except Exception as e:
        print(f"An error occurred during LLM evaluation: {e}")
        return None

def parse_llm_evaluation(llm_evaluation_str):
    """
    Parses the LLM's evaluation string to extract structured correctness and detailed feedback.

    Args:
        llm_evaluation_str (str): The raw string output from the LLM.

    Returns:
        dict: A dictionary with keys 'llm_assessed_correctness' and 'llm_evaluation_feedback'.
              Defaults to "Undetermined" and the original string if parsing fails.
    """
    assessed_correctness = "Undetermined" # Default status
    evaluation_feedback = llm_evaluation_str # Default to the full string

    if not llm_evaluation_str or not llm_evaluation_str.strip():
        return {
            "llm_assessed_correctness": assessed_correctness,
            "llm_evaluation_feedback": "No evaluation content received from LLM."
        }

    # Attempt to parse the "Correctness: [status]" line
    first_line_end_index = llm_evaluation_str.find('\n')
    first_line = llm_evaluation_str[:first_line_end_index if first_line_end_index != -1 else len(llm_evaluation_str)].strip()

    prefix = "Correctness: "
    if first_line.startswith(prefix):
        status = first_line[len(prefix):].strip()
        # Basic validation against common expected statuses. Can be expanded.
        # The prompt asks for: Correct, Partially Correct, Incorrect, or Cannot Determine
        valid_statuses = ["Correct", "Partially Correct", "Incorrect", "Cannot Determine"]
        if status in valid_statuses:
            assessed_correctness = status
            if first_line_end_index != -1:
                evaluation_feedback = llm_evaluation_str[first_line_end_index + 1:].strip()
            else: # Only one line was returned, and it was the status line
                evaluation_feedback = "" 
            print(f"Parsed LLM Correctness Assessment: {assessed_correctness}") # For debugging
        else:
            print(f"Warning: LLM returned an unexpected status '{status}'. Using raw feedback for details.")
            # Keep defaults: assessed_correctness="Undetermined", evaluation_feedback=llm_evaluation_str
    else:
        print("Warning: LLM evaluation output did not start with 'Correctness:'. Using raw feedback for details.")
        # Keep defaults

    return {
        "llm_assessed_correctness": assessed_correctness,
        "llm_evaluation_feedback": evaluation_feedback
    }



