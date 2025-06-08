from datetime import datetime
import os
import sys
from typing import List
from langchain_community.chat_models import ChatZhipuAI # For LLM interaction
from langchain_core.output_parsers import StrOutputParser # For LLM interaction
from langchain.prompts import ChatPromptTemplate
from typing import Literal

IntentType = Literal["INCREMENTAL_ADD", "REVISION", "DELETION", "REWRITE", "UNKNOWN"]
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from backend_app.models import (
    StudentQuestionInput, StudentQuestionOutput,
    RefineStudentQAInput, RefineTeachingPlanInput, RefineAssessmentInput, ChatMessage  # Added new models
)
from backend_app.database_utils import get_mysql_connection, get_or_create_student,get_practice_question_details_by_id, get_teaching_plan_by_id 
from backend_app.student_qa import (
    _rewrite_query_with_history,
    search_knowledge_base_for_answer, 
    construct_student_qa_prompt, # Keep for initial
    construct_student_qa_prompt_with_history, # Add for refine
    get_llm_response_to_student
)
from backend_app.models import (
    AssessmentInput, 
    StudentAssessmentInput, StudentAssessmentEvaluationOutput,
    PracticeQuestionsInput, PracticeQuestionItem, PracticeQuestionsOutput,
    PracticeFeedbackInput, PracticeFeedbackOutput, StudentPerformanceDetail 
)

from fastapi import HTTPException 

from backend_app.assessment_generator import (
        perform_rag_search,
        construct_assessment_prompt,
        generate_assessment_with_llm,
        construct_assessment_prompt_with_history # Added import
    )
from backend_app.assessment_evaluation import (
        get_assessment_content_by_id, 
        construct_evaluation_prompt,
        get_llm_evaluation_for_answer, 
        parse_llm_evaluation
    )
from backend_app.practice_assistant import ( 
        search_knowledge_for_practice_topic,
        construct_practice_question_prompt,
        get_llm_practice_questions,
        construct_feedback_prompt as pa_construct_feedback_prompt, 
        get_llm_feedback_on_answer as pa_get_llm_feedback_on_answer,
        parse_feedback_and_correctness as pa_parse_feedback_and_correctness
    )

from backend_app.database_utils import (
        save_student_assessment_answer, 
        get_student_history_summary, 
        save_practice_question_to_catalog,
        save_practice_attempt, 
        get_student_performance_for_assessment 
    )


from langchain_community.embeddings import ZhipuAIEmbeddings
from langchain_chroma import Chroma

from langchain_community.chat_models import ChatZhipuAI

from langchain_core.messages import SystemMessage, HumanMessage,AIMessage


CHROMA_PERSIST_DIR = 'chroma_db_zhipu' 

async def _identify_user_intent(chat_history: List[ChatMessage], new_query: str) -> IntentType:
    print("SERVICE (Intent ID): Identifying user intent...")
    history_str = "\n".join([f"{msg.role}: {msg.content}" for msg in chat_history])
    
    intent_prompt = f"""
    你是一个执行严格指令的文本分类器。你的唯一任务是分析对话，并将用户的最新指令分类为以下五种意图之一：INCREMENTAL_ADD, REVISION, DELETION, REWRITE, UNKNOWN。

    **规则：**
    1.  仔细阅读提供的对话历史和用户最新指令。
    2.  根据定义的意图类型进行分类。
    3.  你的输出【必须只包含一个单词】，即你选择的意图类型。
    4.  【绝对不能】包含任何解释、分析、标点符号、或除了意图类型单词之外的任何文本。

    **意图定义:**
    - INCREMENTAL_ADD: 在原文基础上增加新内容。
    - REVISION: 修改原文的某一部分。
    - Deletion: 删除什么什么模块，功能，板块。
    - REWRITE: 抛弃原文，根据新要求重新生成。
    - UNKNOWN: 无法判断。

    ---
    对话历史:
    {history_str}
    ---
    用户最新指令: "{new_query}"
    ---

    输出分类结果（一个单词）:
    """
    
    try:
        # 使用一个快速、低成本的模型
        llm = ChatZhipuAI(model="glm-4", temperature=0.0)
        # 这里可以加一个OutputParser来确保输出格式正确，但简单起见，我们先直接解析字符串
        response = await llm.ainvoke(intent_prompt)
        intent = response.content.strip()
        
        valid_intents = ["INCREMENTAL_ADD", "REVISION", "DELETION", "REWRITE", "UNKNOWN"]
        if intent in valid_intents:
            print(f"SERVICE (Intent ID): Identified intent as: {intent}")
            return intent
        else:
            print(f"SERVICE WARNING (Intent ID): LLM returned an invalid intent '{intent}'. Defaulting to UNKNOWN.")
            return "UNKNOWN"
            
    except Exception as e:
        print(f"SERVICE ERROR (Intent ID): Failed to identify intent: {e}")
        return "UNKNOWN"


async def _identify_assessment_intent(chat_history: List[ChatMessage], new_query: str) -> IntentType:
    print("SERVICE (Assessment Intent ID): Identifying user intent...")
    history_str = "\n".join([f"{msg.role}: {msg.content}" for msg in chat_history])
    
    # Prompt 针对“试卷”场景进行微调
    intent_prompt = f"""
    你是一个用户意图分析专家，专注于分析【考核试卷】生成相关的对话。请分析对话历史和用户最新指令，将其核心意图分类为以下五种之一：

 - INCREMENTAL_ADD: 用户要求在原试卷基础上【增加】新的题目、题型或说明，而【不改变】已有内容。 (例如: "再加两道选择题", "在末尾加上评分标准")
 - REVISION: 用户要求【修改、替换或重写】原试卷中【已经存在】的某道题目或某个部分。 (例如: "把第一题改得更难一些", "选择题部分全部换成新的")
 - DELETION:用户要求杀掉什么题
 -REWRITE:全部重新生成
 -UNKNOWN:无法判断
    ---
    对话历史:
    {history_str}
    ---
    用户最新指令: "{new_query}"
    ---

    你的回答【必须只包含一个单词】，即你选择的意图类型。不要有任何解释。
    """
    
    # ... (try/except 逻辑和之前完全一样，这里省略)
    try:
        # ... (LLM调用和解析逻辑)
        # ...
        return "COMPLETE_REWRITE" # 伪代码
    except:
        return "UNKNOWN"

def _get_zhipuai_api_key():
    api_key = os.environ.get("ZHIPUAI_API_KEY")
    if api_key is None:
        print("Warning: ZHIPUAI_API_KEY not found in environment. Using default key for service layer.")
        
    return api_key



async def process_student_question_service(input_data: StudentQuestionInput) -> StudentQuestionOutput:

    print(f"SERVICE: Processing student question: '{input_data.question}'")
    api_key = _get_zhipuai_api_key()
    rag_snippets = []
    llm_answer = "Could not determine an answer."
    error_message = None
    try:

        try:
            embeddings = ZhipuAIEmbeddings() 
        except Exception as e:
            print(f"SERVICE ERROR: Failed to initialize embeddings model: {e}")
            return StudentQuestionOutput(
                student_question=input_data.question,
                rag_context=None,
                llm_answer="Error: Could not initialize embeddings model.",
                error_message=f"Failed to initialize embeddings model: {e}"
            )

        rag_snippets = search_knowledge_base_for_answer(
            student_question=input_data.question,
            embeddings_model_instance=embeddings,
            vector_store_dir=CHROMA_PERSIST_DIR,
            top_k=10 
        )
        print(f"SERVICE: RAG search retrieved {len(rag_snippets)} snippets.")

        
        messages_for_llm = construct_student_qa_prompt(
            student_question=input_data.question,
            rag_snippets=rag_snippets
        )
        print("SERVICE: Prompt constructed.")

     
        llm_response = get_llm_response_to_student(
            messages=messages_for_llm
        )

        if llm_response:
            llm_answer = llm_response
            print("SERVICE: LLM response received.")
        else:
            llm_answer = "Failed to get a response from the LLM."
            error_message = "LLM did not provide an answer."
            print("SERVICE ERROR: LLM did not provide an answer.")
            
    except Exception as e:
        print(f"SERVICE ERROR: An unexpected error occurred: {e}")
        error_message = f"An unexpected error occurred: {str(e)}"
        # Ensure llm_answer reflects error if it happened before LLM call
        if llm_answer == "Could not determine an answer.": # Check if it's still the initial default
            llm_answer = "An error occurred while processing the question."

    return StudentQuestionOutput(
        student_question=input_data.question,
        rag_context=rag_snippets if rag_snippets else None,
        llm_answer=llm_answer,
        error_message=error_message
    )

async def generate_initial_teaching_plan_service(

    initial_outline: str, # Used for RAG query and as initial_human_task
    style_tone:str,
    output_structure:str
    # LLM and Embeddings will be initialized inside, using env vars for keys
) -> tuple[str , List[str] ]: # Returns (plan_content, rag_snippets_used) or (None, None)
    zhipuai_api_key = _get_zhipuai_api_key() # Uses the existing helper
    if not style_tone or not style_tone.strip():
        final_style_tone = "清晰、专业且易于理解" 
        print(f"SERVICE INFO: 'style_tone' was empty, using default: '{final_style_tone}'")
    else:
        final_style_tone = style_tone
    # 2. 设置 output_structure 的默认值
    if not output_structure or not output_structure.strip():
        final_output_structure = "请为以下教学大纲生成一个完整的教案。内容应包括：1. 教学目标；2. 知识点详解；3. 课堂活动与互动环节建议；4. 简单的实训练习及其指导；5. 预估的时间分布。" # 这是一个非常全面和实用的默认结构
        print(f"SERVICE INFO: 'output_structure' was empty, using default: '{final_output_structure}'")
    else:
        final_output_structure = output_structure
    retrieved_rag_snippets = []
    try:
        print(f"SERVICE: Performing RAG search for query: '{initial_outline[:100]}...'")
        if not os.path.exists(CHROMA_PERSIST_DIR):
            print(f"SERVICE WARNING: Chroma DB directory '{CHROMA_PERSIST_DIR}' not found. Proceeding without RAG context.")
        else:
            embeddings_for_rag = ZhipuAIEmbeddings() 
            vector_store = Chroma(
                persist_directory=CHROMA_PERSIST_DIR,
                embedding_function=embeddings_for_rag
            )
            rag_docs = vector_store.similarity_search(initial_outline, k=10)
            if rag_docs:
                retrieved_rag_snippets = [doc.page_content for doc in rag_docs]
                print(f"SERVICE: Retrieved {len(retrieved_rag_snippets)} snippets from RAG.")
            else:
                print("SERVICE: No relevant snippets found from RAG.")
    except Exception as e:
        print(f"SERVICE ERROR during RAG search: {e}. Proceeding without RAG context.")

    system_message = (
        f"你是一位经验丰富的教师，你的任务是根据提供的大纲和补充材料，撰写一份详细的教案初稿。需注重内容的清晰性、准确性，并全面覆盖要点。最后输出语言是中文"
    )
    
    human_message_parts = [
        f"教师提供的初始大纲是:\n{initial_outline}\n"
    ]
    if retrieved_rag_snippets:
        human_message_parts.append("考虑以下来自源材料的相关摘录，以获取更多背景或细节:")
        for i, snippet in enumerate(retrieved_rag_snippets):
            human_message_parts.append(f"--- Snippet {i+1} ---\n{snippet}\n--- End Snippet {i+1} ---")
    else:
        human_message_parts.append("(No supplementary materials from RAG were available or retrieved.)")
    human_message_parts.append(f"输出语言风格要求：{style_tone}")
    human_message_parts.append(f"输出结构要求:{output_structure}")
    human_message_content = "\n".join(human_message_parts)

    # C. LLM Call for Plan Generation
    try:
        print("SERVICE: Initializing LLM for teaching plan generation (ChatZhipuAI)...")
        # Using ZhipuAI (glm-4) as it's generally good for generation tasks.
        # API key is passed directly or picked from env by ChatZhipuAI
        llm_for_plan = ChatZhipuAI(model="glm-4", temperature=0.7, api_key=zhipuai_api_key) 

        messages = [
            SystemMessage(content=system_message),
            HumanMessage(content=human_message_content)
        ]
        
        print("SERVICE: Generating teaching plan via LLM...")
        ai_message = await llm_for_plan.ainvoke(messages) # Use await for async
        generated_plan_content = ai_message.content

        if generated_plan_content and generated_plan_content.strip():
            print("SERVICE: Teaching plan generated successfully.")
            return generated_plan_content, retrieved_rag_snippets
        else:
            print("SERVICE ERROR: LLM returned empty content for teaching plan.")
            return None, retrieved_rag_snippets 
    except Exception as e:
        print(f"SERVICE ERROR during LLM call for teaching plan generation: {e}")
        return None, retrieved_rag_snippets

async def generate_assessment_service(
    input_data: AssessmentInput,
) -> tuple[str , List[str] ]: 

    print(f"SERVICE: Initiating assessment generation for teacher_id: {input_data.teacher_id or input_data.teacher_name}")

    if not input_data.question_preferences:
       
        final_question_prefs = {"选择题": 3, "简答题": 2, "判断题": 2}
        print(f"SERVICE INFO: 'question_preferences' was empty, using default: {final_question_prefs}")
    else:
        final_question_prefs = input_data.question_preferences

 
    retrieved_rag_snippets = []
    generated_assessment_content = None

    try:

        print("SERVICE: Initializing LLM for keyword extraction (ChatTongyi)...")

        embeddings_for_rag = ZhipuAIEmbeddings() # API key from env
        if os.path.exists(CHROMA_PERSIST_DIR):
            retrieved_rag_snippets = perform_rag_search(
            input_data.teaching_plan_content, 
            embeddings_for_rag, 
            CHROMA_PERSIST_DIR
            )
            print(f"SERVICE: RAG search retrieved {len(retrieved_rag_snippets)} snippets.")
        else:
            print(f"SERVICE WARNING: Chroma DB directory '{CHROMA_PERSIST_DIR}' not found. Proceeding without RAG.")
        

        assessment_prompt_components = construct_assessment_prompt(
            input_data.teaching_plan_content,
            retrieved_rag_snippets,
            input_data.question_preferences
        )
        print("SERVICE: Assessment prompt constructed.")
        system_msg_content = assessment_prompt_components["system_message"]
        human_msg_content = assessment_prompt_components["human_message"]
        messages_for_llm = [
            SystemMessage(content=system_msg_content),
            HumanMessage(content=human_msg_content)
        ]
        generated_assessment_content = generate_assessment_with_llm(
            messages_for_llm
        )

        if generated_assessment_content and generated_assessment_content.strip():
            print("SERVICE: Assessment content generated successfully.")
        else:
            print("SERVICE ERROR: LLM returned empty content for assessment.")
            generated_assessment_content = None # Ensure it's None if empty

    except Exception as e:
        print(f"SERVICE ERROR during assessment generation pipeline: {e}")

        return None 

    return generated_assessment_content

async def evaluate_student_assessment_answers_service(
    input_data: StudentAssessmentInput
) -> List[StudentAssessmentEvaluationOutput]:

    print(f"SERVICE: Initiating evaluation for student_id: {input_data.student_id or input_data.student_name} on assessment_id: {input_data.assessment_id}")
    zhipuai_api_key = _get_zhipuai_api_key() # For ChatZhipuAI used in assessment_evaluation.py
    results: List[StudentAssessmentEvaluationOutput] = []
    db_conn = None
    MYSQL_DB_NAME = os.environ.get("MYSQL_DB")
    try:
        db_conn = get_mysql_connection(db_name=MYSQL_DB_NAME)
        if not db_conn:
            raise Exception("Failed to connect to the database.")

        # 1. Get/Create Student ID
        actual_student_id = input_data.student_id
        if not actual_student_id:
            if not input_data.student_name: # Should be caught by Pydantic model if student_name is mandatory when id is not
                raise ValueError("Student name or ID is required.")
            actual_student_id = get_or_create_student(db_conn, input_data.student_name)
            if not actual_student_id:
                raise Exception(f"Failed to get or create student: {input_data.student_name}")
        
        assessment_data = get_assessment_content_by_id(db_conn, input_data.assessment_id)
        if not assessment_data or not assessment_data.get("content"):
            raise ValueError(f"Could not retrieve content for assessment ID {input_data.assessment_id}.")
        assessment_content = assessment_data["content"]

        for answer_item in input_data.answers:
            question_id_str = answer_item.question_identifier
            student_ans_text = answer_item.student_answer_text
            
            print(f"SERVICE: Evaluating answer for Q: {question_id_str}, Student: {actual_student_id}")

            # Construct prompt
            eval_prompt_components = construct_evaluation_prompt(
                assessment_content, question_id_str, student_ans_text
            )
            

            raw_llm_evaluation = get_llm_evaluation_for_answer(
                eval_prompt_components["system_message"],
                eval_prompt_components["human_message"],
                zhipuai_api_key 
            )

            parsed_eval = parse_llm_evaluation(raw_llm_evaluation)
            
            # Save the evaluated answer
            answer_db_id = save_student_assessment_answer(
                db_conn,
                input_data.assessment_id,
                question_id_str,
                actual_student_id,
                student_ans_text,
                parsed_eval["llm_evaluation_feedback"],
                parsed_eval["llm_assessed_correctness"]
            )
            
            results.append(StudentAssessmentEvaluationOutput(
                answer_id=answer_db_id if answer_db_id else None,
                assessment_id=input_data.assessment_id,
                question_identifier=question_id_str,
                student_id=actual_student_id,
                student_answer_text=student_ans_text,
                llm_assessed_correctness=parsed_eval["llm_assessed_correctness"],
                llm_evaluation_feedback=parsed_eval["llm_evaluation_feedback"],
                error_message=None if answer_db_id else "Failed to save this answer."
            ))
            
    except Exception as e:
        print(f"SERVICE ERROR in evaluate_student_assessment_answers_service: {e}")

        if not results or results[-1].error_message != "Database not configured.":
             results.append(StudentAssessmentEvaluationOutput(
                assessment_id=input_data.assessment_id, 
                question_identifier="Overall Error", 
                student_id=input_data.student_id or 0,
                student_answer_text="N/A",
                llm_assessed_correctness="Error",
                llm_evaluation_feedback=str(e),
                error_message=str(e)
            ))
    finally:
        if db_conn and db_conn.is_connected():
            db_conn.close()
            print("SERVICE: DB connection closed for evaluate_student_assessment_answers_service.")
            
    return results

async def generate_practice_questions_service(
    input_data: PracticeQuestionsInput
) -> PracticeQuestionsOutput:
    print(f"SERVICE: Generating practice questions for topic: {input_data.practice_topic}")
    db_conn = None
    student_id = input_data.student_id
    history_summary = "No specific student performance history provided."
    retrieved_rag_snippets = []
    generated_q_items: List[PracticeQuestionItem] = []
    MYSQL_DB_NAME = os.environ.get("MYSQL_DB")
    if not MYSQL_DB_NAME:
        return PracticeQuestionsOutput(generated_questions=[], error_message="Database not configured.")
    try:
        db_conn = get_mysql_connection(db_name=MYSQL_DB_NAME)
        if not db_conn:
            raise Exception("Failed to connect to the database.")

        # 3. Get Student History Summary (if student_id is available)
        if student_id:
            summary = get_student_history_summary(db_conn, student_id)
            if summary: history_summary = summary # Use default if summary is empty/None
            print(f"SERVICE: Student history summary: {history_summary}")
        
        # 4. RAG Search
        embeddings_for_rag = ZhipuAIEmbeddings()
        if os.path.exists(CHROMA_PERSIST_DIR):
            retrieved_rag_snippets = search_knowledge_for_practice_topic(
                input_data.practice_topic, embeddings_for_rag, CHROMA_PERSIST_DIR
            )
        else:
            print(f"SERVICE WARNING: Chroma DB directory '{CHROMA_PERSIST_DIR}' not found. No RAG context.")

        # 5. Construct Prompt for LLM
        prompt_components = construct_practice_question_prompt(
            input_data.practice_topic,
            input_data.question_preferences,
            student_history_summary=history_summary,
            retrieved_context_snippets=retrieved_rag_snippets
        )

        # 6. Generate Practice Questions (uses ChatTongyi via assessment_generator import)
        raw_generated_questions = get_llm_practice_questions(
            prompt_components["system_message"],
            prompt_components["human_message"]

        )

        # 7. Parse and Save Questions to Catalog
        if raw_generated_questions and raw_generated_questions.strip():

            separator = "---参考答案与解析---"
            if separator in raw_generated_questions:
                parts = raw_generated_questions.split(separator, 1)
                all_questions_text = parts[0].strip()
                all_answers_text = parts[1].strip()
            else:
                # 如果找不到分隔符，做一个降级处理
                print("SERVICE WARNING: Could not find the answer separator. Saving the entire content as question_text.")
                all_questions_text = raw_generated_questions.strip()
                all_answers_text = "（答案解析未找到，可能包含在题目文本中）"

            # 将整套题目作为一个单元保存到数据库
            concepts = [input_data.practice_topic]
            catalog_id = save_practice_question_to_catalog( # 使用一个新的保存函数
                db_conn,
                all_questions_text,
                all_answers_text,
                concepts_list=concepts
            )

            if catalog_id:
                # 成功保存，返回包含ID和完整内容的单个PracticeQuestionItem
                generated_item = PracticeQuestionItem(
                    catalog_id=catalog_id,
                    question_text=all_questions_text,
                    model_answer=all_answers_text,
                    # question_type 字段不再需要
                )
                return PracticeQuestionsOutput(generated_questions=[generated_item], error_message=None)
            else:
                # 保存失败
                return PracticeQuestionsOutput(generated_questions=[], error_message="生成了练习题但保存至题库失败。")

        else: # LLM返回空内容
            return PracticeQuestionsOutput(generated_questions=[], error_message="AI未能生成练习题内容。")
            
    except Exception as e:
        print(f"SERVICE ERROR in generate_practice_questions_service: {e}")
        return PracticeQuestionsOutput(generated_questions=[], error_message=f"An unexpected error occurred: {str(e)}")
    finally:
        if db_conn and db_conn.is_connected():
            db_conn.close()
            print("SERVICE: DB connection closed for generate_practice_questions_service.")
            
    return PracticeQuestionsOutput(generated_questions=generated_q_items)

async def get_practice_feedback_service(
    input_data: PracticeFeedbackInput
) -> PracticeFeedbackOutput:
    print(f"SERVICE: Getting feedback for student {input_data.student_id} on catalog_id {input_data.catalog_id}")
    db_conn = None
    MYSQL_DB_NAME = os.environ.get("MYSQL_DB")
    try:
        db_conn = get_mysql_connection(db_name=MYSQL_DB_NAME)
        if not db_conn:
            raise Exception("无法连接到数据库。")


        question_details = get_practice_question_details_by_id(db_conn, input_data.catalog_id)
        
        if not question_details:
            raise ValueError(f"在题库中未找到ID为 {input_data.catalog_id} 的练习题。")

        # --- 2. 构建 Prompt (使用从数据库获取的数据) ---
        feedback_prompt_components =pa_construct_feedback_prompt( 
            question_text=question_details["question_text"],
            model_answer_text=question_details["model_answer"],
            student_answer_text=input_data.student_answer,
        )

        raw_llm_feedback = pa_get_llm_feedback_on_answer(
            feedback_prompt_components["system_message"],
            feedback_prompt_components["human_message"]
        )

        if not raw_llm_feedback:
            return PracticeFeedbackOutput(correctness_assessment="Error", detailed_feedback="LLM failed to provide feedback.", error_message="LLM feedback generation failed.")

        # 3. Parse feedback
        parsed_feedback = pa_parse_feedback_and_correctness(raw_llm_feedback)
        
        # 4. Save practice attempt
        db_conn = get_mysql_connection(db_name=MYSQL_DB_NAME)
        if not db_conn:
            raise Exception("Failed to connect to database to save practice attempt.")

        attempt_id = save_practice_attempt(
            db_conn,
            input_data.student_id,
            input_data.catalog_id,
            input_data.student_answer,
            parsed_feedback["correctness"],
            parsed_feedback["detailed_feedback"]
        )

        if not attempt_id:
            return PracticeFeedbackOutput(
                correctness_assessment=parsed_feedback["correctness"],
                detailed_feedback=parsed_feedback["detailed_feedback"],
                error_message="Feedback generated but failed to save practice attempt."
            )

        return PracticeFeedbackOutput(
            attempt_id=attempt_id,
            correctness_assessment=parsed_feedback["correctness"],
            detailed_feedback=parsed_feedback["detailed_feedback"]
        )

    except Exception as e:
        print(f"SERVICE ERROR in get_practice_feedback_service: {e}")
        return PracticeFeedbackOutput(correctness_assessment="Error", detailed_feedback=str(e), error_message=f"An unexpected error occurred: {str(e)}")
    finally:
        if db_conn and db_conn.is_connected():
            db_conn.close()
            print("SERVICE: DB connection closed for get_practice_feedback_service.")

# Helper function to convert Pydantic ChatMessage to Langchain messages
def convert_chat_messages_to_langchain_format(chat_history: List[ChatMessage], new_query: str) -> List:
    langchain_messages = []
    for msg in chat_history:
        if msg.role == "user":
            langchain_messages.append(HumanMessage(content=msg.content))
        elif msg.role == "assistant":
            langchain_messages.append(AIMessage(content=msg.content))
        elif msg.role == "system": # If you expect system messages in history
            langchain_messages.append(SystemMessage(content=msg.content))
    langchain_messages.append(HumanMessage(content=new_query))
    return langchain_messages

async def refine_student_question_service(input_data: RefineStudentQAInput) -> StudentQuestionOutput:
    print(f"SERVICE: Refining student question for student_id: {input_data.student_id}")
    # ... (其他初始化)
    rag_snippets = []
    llm_answer = "Could not determine a refined answer."
    error_message = None

    try:
        # --- 1. 【新增】查询重写 ---
        standalone_query_for_rag = await _rewrite_query_with_history(input_data.history, input_data.new_query)
        
        # --- 2. 【修改】使用重写后的查询进行RAG搜索 ---
        try:
            embeddings = ZhipuAIEmbeddings() 
        except Exception as e:
            # ... (错误处理)
            pass

        rag_snippets = search_knowledge_base_for_answer(
            student_question=standalone_query_for_rag, # <--- 使用重写后的查询
            embeddings_model_instance=embeddings,
            vector_store_dir=CHROMA_PERSIST_DIR,
            top_k=10 
        )
        print(f"SERVICE (Refine): RAG search retrieved {len(rag_snippets)} snippets for rewritten query.")

        # --- 3. 【保持不变】生成最终答案 ---
        # 准备LangChain消息列表 (这部分你的代码应该已经有了)
        history_langchain_messages = []
        for msg in input_data.history:
            if msg.role == "user" :
                history_langchain_messages.append(HumanMessage(content=msg.content))
            elif msg.role == "assistant":
                history_langchain_messages.append(AIMessage(content=msg.content))

        final_messages_for_llm = construct_student_qa_prompt_with_history(
            history_messages=history_langchain_messages,
            new_query_content=input_data.new_query, 
            rag_snippets=rag_snippets
        )
        
        llm_response = get_llm_response_to_student(
             messages=final_messages_for_llm
        )

        if llm_response:
            llm_answer = llm_response
        else:
            # ... (错误处理)
            pass

    except Exception as e:
        # ... (错误处理)
        pass

    return StudentQuestionOutput(
        student_question=input_data.new_query,
        rag_context=rag_snippets if rag_snippets else None,
        llm_answer=llm_answer,
        error_message=error_message
    )

async def refine_teaching_plan_service(input_data: RefineTeachingPlanInput) -> tuple[str , List[str]]:
    user_intent = await _identify_user_intent(input_data.history, input_data.new_query)
    print(f"SERVICE: Refining teaching plan for teacher_id: {input_data.teacher_id}")
    zhipuai_api_key = _get_zhipuai_api_key()
    retrieved_rag_snippets = []
    generated_plan_content = None
    db_conn = None
    original_plan_content = ""
    MYSQL_DB_NAME = os.environ.get("MYSQL_DB")
    try:
            if input_data.base_teaching_plan_id:
                db_conn = get_mysql_connection(db_name=MYSQL_DB_NAME)
                if db_conn:
                    original_plan = get_teaching_plan_by_id(db_conn, input_data.base_teaching_plan_id)
                    if original_plan and original_plan.get('content'):
                        original_plan_content = original_plan['content']
                        print(f"SERVICE (Refine): Loaded original content from plan ID {input_data.base_teaching_plan_id}.")
                    if original_plan_content:
                        original_content_message = ChatMessage(
                            role="system", 
                            content=f"你正在修改以下这份原始教案：\n\n--- 原始教案开始 ---\n{original_plan_content}\n--- 原始教案结束 ---"
                        )
                        input_data.history.insert(0, original_content_message)
                else:
                    print("SERVICE (Refine): Could not connect to DB to fetch original plan.")
            standalone_query_for_rag = await _rewrite_teaching_plan_query(input_data.history, input_data.new_query)
            try:
                embeddings_for_rag = ZhipuAIEmbeddings()
                if os.path.exists(CHROMA_PERSIST_DIR):
                    vector_store = Chroma(persist_directory=CHROMA_PERSIST_DIR, embedding_function=embeddings_for_rag)
                    rag_docs = vector_store.similarity_search(standalone_query_for_rag, k=5) 
                    if rag_docs:
                        retrieved_rag_snippets = [doc.page_content for doc in rag_docs]
                        print(f"SERVICE (Refine): Retrieved {len(retrieved_rag_snippets)} RAG snippets for rewritten query.")
                else:
                    print("SERVICE WARNING (Refine): Chroma DB directory not found.")
            except Exception as e:
                print(f"SERVICE ERROR (Refine) during RAG for teaching plan: {e}.")
            langchain_messages = convert_chat_messages_to_langchain_format(input_data.history, input_data.new_query)
            if user_intent == "REWRITE":
                system_prompt = "根据用户的最新指令，生成一份【全新】的考核试卷。请忽略所有历史和原始试卷内容。"
            elif user_intent == "REVISION" or user_intent == "DELETION":
                system_prompt = "根据原始试卷和用户的最新指令，生成一份【修改后】的【完整】考核试卷。你的输出应该是替换掉整个旧试卷的新版本。"
            else: # INCREMENTAL_ADD 或 UNKNOWN
                system_prompt = "根据原始试卷和用户的最新指令，【只生成需要新增或修改】的那部分内容。不要重复原始试卷中未被修改的部分。"
            final_messages_for_llm = [SystemMessage(content=system_prompt)] + langchain_messages 
            if retrieved_rag_snippets:
                rag_context_str = "\n\n--- 相关参考资料 ---\n" + "\n---\n".join(retrieved_rag_snippets)
                if isinstance(final_messages_for_llm[-1], HumanMessage):
                    final_messages_for_llm[-1].content += rag_context_str
            llm_for_plan = ChatZhipuAI(model="glm-4", temperature=0.7, api_key=zhipuai_api_key)
            print("SERVICE (Refine): Generating refined teaching plan via LLM...")
            ai_message = await llm_for_plan.ainvoke(final_messages_for_llm)
            generated_content = ai_message.content
            if user_intent == "INCREMENTAL_ADD":
                print("SERVICE: Handling INCREMENTAL_ADD. Appending new content.")
                final_full_content = (
            f"{original_plan_content}\n\n"
            f"--- 更新于 {datetime.now().strftime('%Y-%m-%d %H:%M')} ---\n\n"
            f"{generated_content}"
        )
    
            elif user_intent == "REVISION" or user_intent == "DELETION":
                print(f"SERVICE: Handling {user_intent}. Replacing original content with new version.")
                final_full_content = generated_content

            elif user_intent == "REWRITE":
                print("SERVICE: Handling REWRITE. Replacing original content with new version.")
                final_full_content = generated_content

            else: # UNKNOWN 或其他未处理的意图
                print("SERVICE: Intent is UNKNOWN. Defaulting to incremental add behavior.")
                # 默认行为：为了安全起见，我们选择最常见的“增量添加”作为默认操作。
                final_full_content = (
            f"{original_plan_content}\n\n"
            f"--- 更新于 {datetime.now().strftime('%Y-%m-%d %H:%M')} ---\n\n"
            f"[意图未知，内容已附加]：\n{generated_content}"
        )

 
            if final_full_content is None:
                print("SERVICE ERROR: No final content was constructed due to logic error.")
                return None, [] # 返回空，让上层处理

            return final_full_content, rag_context_str

    except Exception as e:
            print(f"SERVICE ERROR (Refine) in refine_teaching_plan_service: {e}")
            return None, []
    finally:
            if db_conn and db_conn.is_connected():
                db_conn.close()
    

        
async def refine_assessment_service(input_data: RefineAssessmentInput) -> str :
    print(f"SERVICE: Starting refine process for assessment ID: {input_data.base_assessment_id}")
    
    # --- 1. 意图识别 ---
    user_intent = await _identify_assessment_intent(input_data.history, input_data.new_query)
    db_conn = None
    original_assessment_content = ""
    MYSQL_DB_NAME = os.environ.get("MYSQL_DB")
    
    try:
        db_conn = get_mysql_connection(db_name=MYSQL_DB_NAME)
        if db_conn:
                original_assessment = get_assessment_content_by_id(db_conn, input_data.base_assessment_id)
                if original_assessment and original_assessment.get('content'):
                    original_assessment_content = original_assessment['content']
                    # 将原始试卷内容注入history
                    input_data.history.insert(0, ChatMessage(
                        role="system",
                        content=f"你正在修改以下这份原始试卷：\n\n--- 原始试卷开始 ---\n{original_assessment_content}\n--- 原始试卷结束 ---"
                    ))
        standalone_query_for_rag = await _rewrite_teaching_assessment_query(input_data.history, input_data.new_query)
        embeddings = ZhipuAIEmbeddings()
        rag_snippets = perform_rag_search(standalone_query_for_rag, embeddings, CHROMA_PERSIST_DIR)
        langchain_messages = convert_chat_messages_to_langchain_format(input_data.history, input_data.new_query)
        final_messages_for_llm = construct_assessment_prompt_with_history(
            history=langchain_messages[:-1], # 传入不包含new_query的历史
            new_query=langchain_messages[-1].content, # 单独传入new_query
            rag_snippets=rag_snippets,
            user_intent=user_intent
        )
        generated_content = generate_assessment_with_llm(messages=final_messages_for_llm)
        if not generated_content:
            return None, []

        final_full_content = None

        if user_intent == "INCREMENTAL_ADD":
            print("SERVICE: Handling INCREMENTAL_ADD. Appending new content.")
            # 增量添加：拼接原始内容和新内容
            final_full_content = (
                f"{original_assessment_content}\n\n"
                f"--- 更新于 {datetime.now().strftime('%Y-%m-%d %H:%M')} ---\n\n"
                f"{generated_content}"
            )
        
        elif user_intent == "REVISION" or user_intent == "DELETION":
            print(f"SERVICE: Handling {user_intent}. Replacing original content with new version.")
            # 修改或风格变更：此时LLM应该被引导生成【完整】的新版本。
            # 因此，最终内容就是LLM新生成的内容。
            # 注意：你需要确保你的Prompt在这种意图下能引导LLM生成全文。
            final_full_content = generated_content

        elif user_intent == "REWRITE":
            print("SERVICE: Handling REWRITE. Replacing original content with new version.")
            # 完全重写：最终内容就是新生成的内容。
            final_full_content = generated_content

        else: # UNKNOWN 或其他未处理的意图
            print("SERVICE: Intent is UNKNOWN. Defaulting to incremental add behavior.")
            # 默认行为：为了安全起见，我们选择最常见的“增量添加”作为默认操作。
            final_full_content = (
                f"{original_assessment_content}\n\n"
                f"--- 更新于 {datetime.now().strftime('%Y-%m-%d %H:%M')} ---\n\n"
                f"[意图未知，内容已附加]：\n{generated_content}"
            )

        # 确保 final_full_content 有值
        if final_full_content is None:
            print("SERVICE ERROR: No final content was constructed due to logic error.")
            return None, [] # 返回空，让上层处理

        # 返回元组 (完整内容, RAG片段)
        return final_full_content, rag_snippets
    
    except Exception as e:
        print(f"SERVICE ERROR in refine_assessment_service: {e}")
        return None, []
    finally:
        if db_conn and db_conn.is_connected():
            db_conn.close()

    

async def get_student_assessment_performance_service(assessment_id: int) -> List[StudentPerformanceDetail]:

    print(f"SERVICE: Call received for get_student_assessment_performance_service with assessment_id: {assessment_id}")
    
    db_conn = None
    performance_details: List[StudentPerformanceDetail] = []
    
    MYSQL_DB_NAME = os.environ.get("MYSQL_DB")
    if not MYSQL_DB_NAME:
        print("SERVICE ERROR: MYSQL_DB environment variable not set. Cannot fetch performance details.")

        raise HTTPException(status_code=500, detail="Database configuration error.")

    try:
        db_conn = get_mysql_connection(db_name=MYSQL_DB_NAME)
        if not db_conn:
            print("SERVICE ERROR: Failed to connect to the database.")
            raise HTTPException(status_code=500, detail="Failed to connect to the database.")

        raw_performance_data = get_student_performance_for_assessment(db_conn, assessment_id)

        if not raw_performance_data:
            # This isn't necessarily an error; it could be a valid assessment with no submissions yet.
            print(f"SERVICE: No performance data found for assessment_id: {assessment_id}. Returning empty list.")
            return []

        # Map dictionary results to StudentPerformanceDetail Pydantic models
        for row in raw_performance_data:

            try:
                performance_details.append(StudentPerformanceDetail(**row))
            except Exception as pydantic_err: # Catch potential Pydantic validation errors
                print(f"SERVICE ERROR: Pydantic validation error for row {row}: {pydantic_err}")
                # Decide how to handle: skip this row, or raise an error for the whole request.
                # For now, let's skip problematic rows and log.
                continue 
        
        print(f"SERVICE: Successfully retrieved and mapped {len(performance_details)} performance records for assessment_id: {assessment_id}")

    except HTTPException as he: # Re-raise HTTPExceptions from connection attempts
        raise he
    except Exception as e:
        print(f"SERVICE ERROR: An unexpected error occurred in get_student_assessment_performance_service: {e}")
        # Log the full error e for server-side debugging
        # For a teacher-facing endpoint, returning a 500 is appropriate for unexpected issues.
        raise HTTPException(status_code=500, detail=f"An internal server error occurred while fetching performance data: {str(e)}")
    finally:
        if db_conn and db_conn.is_connected():
            db_conn.close()
            print(f"SERVICE: DB connection closed for get_student_assessment_performance_service (assessment_id: {assessment_id}).")
            
    return performance_details

async def _rewrite_teaching_plan_query(chat_history: List[ChatMessage], new_query: str) -> str:
    print("SERVICE (Teaching Plan Refine): Rewriting query with history...")
    history_str = "\n".join([f"{msg.role}: {msg.content}" for msg in chat_history])
    rewrite_prompt_template = ChatPromptTemplate.from_messages([
        ("system", "你是一个教学大纲优化助手。你的任务是分析一段关于“教案生成”的对话历史，并将用户最新的、可能不完整的修改指令，改写成一个独立的、包含核心主题的、可以用于信息检索的查询。只输出改写后的查询，不要包含任何额外解释。"),
        ("human", f"""
            对话历史:
            ---
            {history_str}
            ---

            用户的最新修改指令是: "{new_query}"

            请根据以上对话历史，将这个指令改写成一个独立的、包含教案核心主题的检索查询。

            例如，如果历史是关于“一战历史”的教案，用户的指令是“增加一些关于萨拉热窝事件的细节”，你应该输出“一战历史中的萨拉热窝事件细节”。
            如果历史是关于“Python列表推导式”的教案，用户的指令是“再加几个练习题”，你应该输出“关于Python列表推导式的练习题示例”。

            现在，请开始改写：
            """)
    ])
    try:
        llm = ChatZhipuAI(model="glm-4", temperature=0.0) 
        chain = rewrite_prompt_template | llm | StrOutputParser()
        rewritten_query = await chain.ainvoke({})
        print(f"SERVICE (Teaching Plan Refine): Original query: '{new_query}', Rewritten query: '{rewritten_query}'")
        return rewritten_query.strip()
    except Exception as e:
        print(f"SERVICE ERROR (Teaching Plan Refine): Failed to rewrite query: {e}. Falling back to original query.")
        return new_query

async def _rewrite_teaching_assessment_query(chat_history: List[ChatMessage], new_query: str) -> str:
    history_str = "\n".join([f"{msg.role}: {msg.content}" for msg in chat_history])
    rewrite_prompt_template = ChatPromptTemplate.from_messages([
        ("system", "你是一个教学测试题优化助手。你的任务是分析一段关于“测试题生成”的对话历史，并将用户最新的、可能不完整的修改指令，改写成一个独立的、包含核心主题的、可以用于信息检索的查询。只输出改写后的查询，不要包含任何额外解释。"),
        ("human", f"""
            对话历史:
            ---
            {history_str}
            ---

            用户的最新修改指令是: "{new_query}"

            请根据以上对话历史，将这个指令改写成一个独立的的检索查询。

            例如，如果历史是关于“一战历史”的习题，用户的指令是“增加一些关于萨拉热窝事件的习题”，你应该输出“有关一战历史中的萨拉热窝事件习题”。

            现在，请开始改写：
            """)
    ])
    try:
        llm = ChatZhipuAI(model="glm-4", temperature=0.0) 
        chain = rewrite_prompt_template | llm | StrOutputParser()
        rewritten_query = await chain.ainvoke({})
        return rewritten_query.strip()
    except Exception as e:
        return new_query