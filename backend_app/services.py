from datetime import datetime
import json
import os
import re
import sys
from typing import List, Optional
from langchain_community.chat_models import ChatZhipuAI # For LLM interaction
from langchain_core.output_parsers import StrOutputParser # For LLM interaction
from langchain.prompts import ChatPromptTemplate
from typing import Literal

from backend_app.nlp_utils import parse_query_with_llm

IntentType = Literal["INCREMENTAL_ADD", "REVISION", "DELETION", "REWRITE", "UNKNOWN"]
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from backend_app.models import (
    PracticeChatInput, PracticeChatOutput, 
    FeedbackItem, PracticeQuestionDetailOutput, PracticeQuestionListItem, PracticeQuestionListOutput, PracticeQuestionNLInput, StudentAssessmentSummary, StudentQuestionInput, StudentQuestionOutput,
    RefineStudentQAInput, RefineTeachingPlanInput, RefineAssessmentInput, ChatMessage, TeacherAssessmentListItem, TeacherAssessmentListOutput  # Added new models
)
from backend_app.database_utils import get_aggregated_student_performance, get_all_assessment_for_student_view,  get_assessment_details_by_id, get_assessments_by_teacher_id, get_mysql_connection, get_or_create_student,get_practice_question_details_by_id, get_teaching_plan_by_id, publish_assessment 
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

PracticeIntent = Literal["ADD_QUESTIONS", "REWRITE_QUESTIONS", "SUBMIT_ANSWER", "GENERATE_NEW", "UNKNOWN"]
CHROMA_PERSIST_DIR = 'chroma_db_zhipu' 

async def _identify_user_intent(chat_history: List[ChatMessage], new_query: str) -> IntentType:
    print("SERVICE : Identifying user intent...")
    history_str = "\n".join([f"{msg.role}: {msg.content}" for msg in chat_history])
    
    intent_prompt = f"""
    你是一个执行严格指令的文本分类器。你的唯一任务是分析对话，并将用户的最新指令分类为以下五种意图之一：INCREMENTAL_ADD, REVISION, DELETION, REWRITE, UNKNOWN。

    **规则：**
    1.  仔细阅读提供的对话历史和用户最新指令。
    2.  根据定义的意图类型进行分类。
    3.  你的输出【必须只包含一个单词】，即你选择的意图类型。
    4.  【绝对不能】包含任何解释、分析、标点符号、或除了意图类型单词之外的任何文本。

    **意图定义:**
    - INCREMENTAL_ADD: 在原文基础上增加新内容,再加上，另外等字段代表这个意图。
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
            print(f"SERVICE : Identified intent as: {intent}")
            return intent
        else:
            print(f"SERVICE WARNING : LLM returned an invalid intent '{intent}'. Defaulting to UNKNOWN.")
            return "UNKNOWN"
            
    except Exception as e:
        print(f"SERVICE ERROR : Failed to identify intent: {e}")
        return "UNKNOWN"

async def _identify_practice_intent(history: List[ChatMessage], new_query: str) -> PracticeIntent:
    """使用LLM分析对话，判断用户更精确的意图。"""
    print("SERVICE: Identifying nuanced practice assistant intent...")
    history_str = "\n".join([f"{msg.role}: {msg.content}" for msg in history])

    intent_prompt = f"""
    你是一个文本分类器，任务是分析学生与AI练习助手的对话，判断学生最新输入的意图。
    意图只能是以下五种之一: ADD_QUESTIONS, REWRITE_QUESTIONS, SUBMIT_ANSWER, GENERATE_NEW, UNKNOWN。

    **意图定义:**
    - ADD_QUESTIONS: 用户明确要求在上一轮题目基础上【增加】新的题目。
      (关键词: "再来几道", "加上", "多出点", "还想要2道")
    - REWRITE_QUESTIONS: 用户对上一轮的题目不满意，要求【替换】或【重写】。这包括改变难度、换内容、或完全换题型。
      (关键词: "太难了", "简单点", "换一批", "不要这个", "换成选择题")
    - SUBMIT_ANSWER: 用户正在提供问题的答案。
      (明显特征: "答案是", "第一题选A", "我的代码如下", 或者直接是一段看起来像答案的文本)
    - GENERATE_NEW: 用户想开始一个【全新的练习主题】，与上一轮无关。
      (关键词: "我们来练习...", "换个主题", "我想学...", "出点关于...的题")
    - UNKNOWN: 无法判断或闲聊。

    **对话历史:**
    ---
    {history_str}
    ---
    **学生最新输入:** "{new_query}"
    ---
    你的输出【必须只包含一个单词】，即你选择的意图类型。
    """
    try:
        llm = ChatZhipuAI(model="glm-4", temperature=0.0)
        response = await llm.ainvoke(intent_prompt)
        intent = response.content.strip()
        
        valid_intents: List[PracticeIntent] = ["ADD_QUESTIONS", "REWRITE_QUESTIONS", "SUBMIT_ANSWER", "GENERATE_NEW", "UNKNOWN"]
        if intent in valid_intents:
            print(f"SERVICE: Identified intent as: {intent}")
            return intent
        print(f"SERVICE WARNING: LLM returned invalid intent '{intent}'. Defaulting to UNKNOWN.")
        return "UNKNOWN"
    except Exception as e:
        print(f"SERVICE ERROR: Failed to identify practice intent: {e}")
        return "UNKNOWN"

async def process_practice_chat_service(input_data: PracticeChatInput) -> PracticeChatOutput:
    """
    Handles the ongoing conversation for the practice assistant, implementing advanced
    logic for rewriting, adding, and generating questions based on nuanced user intent.
    """
    
    intent = await _identify_practice_intent(input_data.history, input_data.new_query)

    # --- Branch 1: User is submitting an answer ---
    if intent == "SUBMIT_ANSWER":
        if not input_data.active_catalog_id:
            return PracticeChatOutput(assistant_response_text="我好像不知道你在回答哪一套题，请先让我出题。", intent_detected=intent)
        try:
            feedback_input = PracticeFeedbackInput(student_id=input_data.student_id, catalog_id=input_data.active_catalog_id, student_answer=input_data.new_query)
            feedback_result = await get_practice_feedback_service(feedback_input)
            return PracticeChatOutput(assistant_response_text="这是你本次作答的反馈：", intent_detected=intent, feedback=feedback_result)
        except Exception as e:
            print(f"Error during feedback service call: {e}")
            return PracticeChatOutput(assistant_response_text="抱歉，在评价你的答案时出错了。", intent_detected=intent, error_message=str(e))

    # --- Branch 2: User wants to generate a completely new set of questions ---
    elif intent == "GENERATE_NEW":
        nlp_prompt_new = """
        你是一个分析专家。请从以下文本中提取【核心练习主题】和【明确的题目偏好】。
        ---
        "{new_query}"
        ---
        你需要提取:
        1. "practice_topic" (字符串, 必填): 核心练习主题。
        2. "question_preferences" (对象, 必填): 一个只包含【题型:数量】的JSON对象。键必须是字符串，值必须是整数。如果用户未提及任何题型，则返回一个空的JSON对象{{}}。
        你的最终响应必须是且只能是一个JSON对象。
        """
        FULL_PROMPT = nlp_prompt_new.format(new_query=input_data.new_query)
        parsed_entities = await parse_query_with_llm(FULL_PROMPT)

        if "error" in parsed_entities or not parsed_entities.get("practice_topic"):
            return PracticeChatOutput(assistant_response_text="我不太明白你想要练习什么，可以再说清楚一点吗？", intent_detected=intent)

        practice_topic = parsed_entities.get("practice_topic")
        
        raw_prefs = parsed_entities.get("question_preferences")
        cleaned_prefs = {}
        if isinstance(raw_prefs, dict):
            for key, value in raw_prefs.items():
                if isinstance(key, str) and isinstance(value, int):
                    cleaned_prefs[key] = value
        
        final_prefs = cleaned_prefs or {"选择题": 2, "简答题": 1}

        service_input = PracticeQuestionsInput(student_id=input_data.student_id, practice_topic=practice_topic, question_preferences=final_prefs)
        new_questions_result = await generate_practice_questions_service(service_input)

        if new_questions_result.error_message:
             return PracticeChatOutput(assistant_response_text=f"抱歉，生成题目时出错了: {new_questions_result.error_message}", intent_detected=intent)
        
        return PracticeChatOutput(
            assistant_response_text="好的，这是为你准备的新题目：",
            intent_detected=intent,
            new_questions=new_questions_result
        )

    # --- Branch 3: User wants to add more questions (Merge Logic) ---
    elif intent == "ADD_QUESTIONS":
        if not input_data.active_catalog_id:
            return PracticeChatOutput(assistant_response_text="我需要先为你出一套题，才能在它的基础上修改哦。", intent_detected=intent)

        # Get old questions and reliable topic from DB
        db_conn = None
        original_questions, reliable_topic = "", "相关主题"
        try:
            db_conn = get_mysql_connection(db_name=os.environ.get("MYSQL_DB"))
            if db_conn:
                cursor = db_conn.cursor(dictionary=True)
                # We need to get original_answers as well now
                cursor.execute("SELECT question_text, model_answer, concepts_covered FROM practice_questions_catalog WHERE catalog_id = %s", (input_data.active_catalog_id,))
                old_set = cursor.fetchone()
                if old_set:
                    original_answers = old_set.get("model_answer", "")
                    original_questions = old_set.get("question_text", "")
                    concepts_str = old_set.get("concepts_covered", "")
                    try: reliable_topic = json.loads(concepts_str)[0] if concepts_str else "相关主题"
                    except (json.JSONDecodeError, IndexError): pass
        finally:
            if db_conn and db_conn.is_connected(): db_conn.close()

        if not original_questions:
            return PracticeChatOutput(assistant_response_text="抱歉，我找不到你上一轮的题目了，我们重新开始吧？", intent_detected=intent)
        
        # Use NLP to get ONLY the new preferences to add
        nlp_prompt_add = """
        你是一个分析专家。你的任务是从以下用户的【新增指令】中，提取出【明确的题目偏好】。
        ---
        "{new_query}"
        ---
        你需要提取一个名为 "question_preferences" 的JSON对象，它只包含【题型:数量】的键值对。
        **规则:**
        1. 键必须是字符串(题型)，值必须是整数(数量)。
        2. 如果用户提到了题型但数量模糊（例如 "几道选择题"），你【必须】使用默认数量 `3`。
        3. 如果用户没有提到任何题型，返回一个空对象 `{{}}`。
        
        **示例:**
        - 输入: "再来2道编程题" -> 输出: `{{"question_preferences": {{"编程题": 2}}}}`
        - 输入: "再出几道选择题吧" -> 输出: `{{"question_preferences": {{"选择题": 3}}}}`
        """
        FULL_PROMPT_ADD = nlp_prompt_add.format(new_query=input_data.new_query)
        parsed_entities = await parse_query_with_llm(FULL_PROMPT_ADD)
        
        raw_prefs = parsed_entities.get("question_preferences")
        final_prefs = {}
        if isinstance(raw_prefs, dict):
            for key, value in raw_prefs.items():
                if isinstance(key, str) and isinstance(value, int):
                    final_prefs[key] = value
        
        if not final_prefs:
             return PracticeChatOutput(assistant_response_text="你想增加什么类型的题目呢？可以说得更具体一点吗？", intent_detected=intent)

        merge_prompt = f"""
        你是一个AI出题助手，任务是基于一个【已有的练习集】和用户的【新增题目要求】，生成一个【全新的、完整的】练习集。

        --- 已有的练习集 ---
        【已有题目部分】:
        {original_questions}

        【已有答案部分】:
        {original_answers}
        ---

        --- 用户的新增题目要求 ---
        请在【已有题目部分】的基础上，增加以下新题目：
        主题: {reliable_topic}
        题型和数量: {final_prefs}
        ---

        **你的任务**:
        返回一个完整的、合并后的新版本。

        **输出格式规则 (必须严格遵守！)**:
        1.  **题目区**: 你的输出必须先包含【所有旧题目】，然后紧接着是【所有新题目】。
        2.  **分隔符**: 在所有题目结束后，另起一行，只包含 `---参考答案与解析---`。
        3.  **答案区**: 在分隔符后，你的输出必须包含【所有旧答案】，然后紧接着是【所有新题目的答案】。

        请现在开始生成**合并后的完整新版**：
        """
        system_message = "你是一个教学经验丰富的AI助教，擅长根据用户的指令修改和整合练习题。"
        raw_generated_questions = get_llm_practice_questions(system_message, merge_prompt)
        
        service_input = PracticeQuestionsInput(student_id=input_data.student_id, practice_topic=reliable_topic, question_preferences=final_prefs)
        new_questions_result = await generate_practice_questions_service(service_input, raw_generated_questions=raw_generated_questions)

        if new_questions_result.error_message:
            return PracticeChatOutput(assistant_response_text=f"抱歉，在处理题目时出错了: {new_questions_result.error_message}", intent_detected=intent)

        return PracticeChatOutput(
            assistant_response_text="好的，已为你添加新题目。这是修改后的完整练习题：",
            intent_detected=intent,
            new_questions=new_questions_result
        )

    # --- Branch 4: User wants to rewrite the questions (Your intelligent rewrite logic) ---
    elif intent == "REWRITE_QUESTIONS":
        if not input_data.active_catalog_id:
            return PracticeChatOutput(assistant_response_text="我需要先为你出一套题，才能在它的基础上修改哦。", intent_detected=intent)

        # Step 1: Get old question text and reliable topic from DB
        db_conn = None
        original_questions = ""
        reliable_topic = "相关主题"
        try:
            db_conn = get_mysql_connection(db_name=os.environ.get("MYSQL_DB"))
            if db_conn:
                cursor = db_conn.cursor(dictionary=True)
                cursor.execute("SELECT question_text, concepts_covered FROM practice_questions_catalog WHERE catalog_id = %s", (input_data.active_catalog_id,))
                old_set = cursor.fetchone()
                if old_set:
                    original_questions = old_set.get("question_text", "")
                    concepts_str = old_set.get("concepts_covered", "")
                    try:
                        reliable_topic = json.loads(concepts_str)[0] if concepts_str else "相关主题"
                    except (json.JSONDecodeError, IndexError):
                        pass
        finally:
            if db_conn and db_conn.is_connected(): db_conn.close()

        if not original_questions:
            return PracticeChatOutput(assistant_response_text="抱歉，我找不到你上一轮的题目了，我们重新开始吧？", intent_detected=intent)

        # --- 【核心修正：采用两步生成，并在第一步中注入智能分析指令】 ---

        # === Step 2.1: Generate ONLY the new, rewritten questions ===
        prompt_step1_questions = f"""
        你是一个非常智能且严格遵守指令的AI出题专家。你的任务是根据用户的反馈，重写一套练习题的【题目部分】。

        --- 这是上一轮的题目，你需要先分析它的【题型和数量】 ---
        {original_questions}
        ---

        --- 这是用户的【修改要求】 ---
        "{input_data.new_query}"
        ---

        **你的核心任务 (必须严格遵守):**

        1.  **分析**: 在你的“脑中”分析【上一轮的题目】，搞清楚它的【题型和数量】。例如，你可能会分析出它是 "3道编程题"。
        
        2.  **重写**:
            *   **完全抛弃**旧的题目内容。
            *   **严格保持**你刚才分析出的【题型和数量】不变。
            *   根据用户的【修改要求】（例如：更简单），并围绕【主题】"{reliable_topic}"，从零开始生成一套全新的题目。
            *   **=> 换句话说：如果上一轮是3道编程题，用户说“太难了”，你就必须生成3道更简单的【编程题】，而不是换成选择题。**

        3.  **输出要求**: 你的输出【只能包含新题目的题干】，绝对不能包含任何答案、解析或"---参考答案与解析---"这样的分隔符。

        请现在只输出【新的题目部分】：
        """
        system_message_step1 = "你是一个只负责出题的AI专家，不提供答案。"
        generated_questions_only = get_llm_practice_questions(system_message_step1, prompt_step1_questions)

        if not generated_questions_only or not generated_questions_only.strip():
            return PracticeChatOutput(assistant_response_text="抱歉，我在构思新题目时遇到了问题。", intent_detected=intent)

        # === Step 2.2: Generate ONLY the answers for the questions from Step 1 ===
        prompt_step2_answers = f"""
        你是一个AI解题专家。你的任务是为以下提供的题目，生成详细的答案和解析。

        --- 需要解答的题目 ---
        {generated_questions_only}
        ---

        **你的任务**:
        为上面提供的【每一道题】，都生成对应的【答案和解析】。对于编程题，答案必须包含可运行的代码。
        你的输出【只能包含答案和解析部分】，不要重复题目，也不要包含"---参考答案与解析---"这样的分隔符。

        请现在只输出【答案和解析部分】：
        """
        system_message_step2 = "你是一个只负责解题和写解析的AI专家。"
        generated_answers_only = get_llm_practice_questions(system_message_step2, prompt_step2_answers)

        if not generated_answers_only or not generated_answers_only.strip():
            return PracticeChatOutput(assistant_response_text="抱歉，我想出了题目但没想出答案。", intent_detected=intent)

        # === Step 2.3: Combine them in reliable Python code ===
        raw_generated_questions = f"{generated_questions_only.strip()}\n\n---参考答案与解析---\n\n{generated_answers_only.strip()}"
        
        # Step 2.4: Call the unified downstream service to save and format the result.
        service_input = PracticeQuestionsInput(
            student_id=input_data.student_id, 
            practice_topic=reliable_topic, 
            question_preferences={} # This is okay, as the LLM handled type/quantity internally
        )
        new_questions_result = await generate_practice_questions_service(service_input, raw_generated_questions=raw_generated_questions)

        if new_questions_result.error_message:
            return PracticeChatOutput(assistant_response_text=f"抱歉，在处理题目时出错了: {new_questions_result.error_message}", intent_detected=intent)

        return PracticeChatOutput(
            assistant_response_text="好的，这是根据你的要求修改后的完整练习题：",
            intent_detected=intent,
            new_questions=new_questions_result
        )
    
    # --- Branch 5: Fallback for any other unhandled case ---
    else: # UNKNOWN
        return PracticeChatOutput(
            assistant_response_text="我不太确定该怎么做，你可以尝试让我“出题”、“修改题目”或者直接“提交你的答案”。",
            intent_detected="UNKNOWN"
        )
    
async def _identify_assessment_intent(chat_history: List[ChatMessage], new_query: str) -> IntentType:
    print("SERVICE : Identifying user intent...")
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
    

    try:
        llm = ChatZhipuAI(model="glm-4", temperature=0.0)
        # 这里可以加一个OutputParser来确保输出格式正确，但简单起见，我们先直接解析字符串
        response = await llm.ainvoke(intent_prompt)
        intent = response.content.strip()
        
        valid_intents = ["INCREMENTAL_ADD", "REVISION", "DELETION", "REWRITE", "UNKNOWN"]
        if intent in valid_intents:
            print(f"SERVICE : Identified intent as: {intent}")
            return intent
        else:
            print(f"SERVICE WARNING : LLM returned an invalid intent '{intent}'. Defaulting to UNKNOWN.")
            return "UNKNOWN"
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
) -> tuple[str , str ]: 
    print(f"SERVICE: Initiating assessment generation for subject: {input_data.subject or 'General'}")

    final_question_prefs = input_data.question_preferences
    # --- 使用 subject 进行智能推荐 ---
    if not final_question_prefs:
        print("SERVICE INFO: No question preferences. Applying smart defaults based on subject.")
        subject_lower = (input_data.subject or "").lower()
        if "computer" in subject_lower or "编程" in subject_lower or "programming" in subject_lower:
            final_question_prefs = {"选择题": 2, "编程题": 2}
        else:
            final_question_prefs = {"选择题": 3, "简答题": 2}
        print(f"SERVICE INFO: Using defaults for '{input_data.subject}': {final_question_prefs}")
    
    retrieved_rag_snippets = []
    generated_assessment_content = None

    try:
        assessment_prompt_components = construct_assessment_prompt(
            teaching_plan_content=input_data.teaching_plan_content,
            retrieved_rag_snippets=retrieved_rag_snippets,
            question_preferences=final_question_prefs,
            subject=input_data.subject 
        )
        

        messages_for_llm = [
            SystemMessage(content=assessment_prompt_components["system_message"]),
            HumanMessage(content=assessment_prompt_components["human_message"])
        ]
        generated_assessment_content = generate_assessment_with_llm(messages_for_llm)

        if generated_assessment_content and generated_assessment_content.strip():
            print("SERVICE: Splitting generated content into questions and answers.")
            

            answer_separator = "参考答案与解析"
            parts = generated_assessment_content.split(answer_separator, 1)
            questions_part = parts[0].strip()
            answers_part = parts[1].strip() if len(parts) > 1 else "（无答案信息）"
            
            return questions_part, answers_part
        else:
            print("SERVICE ERROR: LLM returned empty content for assessment.")
            return None, None

    except Exception as e:
        print(f"SERVICE ERROR during assessment generation pipeline: {e}")
        return None, None

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
    input_data: PracticeQuestionsInput,
    raw_generated_questions: Optional[str] = None
) -> PracticeQuestionsOutput:
    """
    Generates or processes practice questions. If raw_generated_questions is provided,
    it processes and saves that. Otherwise, it generates new questions from scratch.
    This version uses a robust regex for splitting questions and answers to handle
    minor LLM formatting errors.
    """
    print(f"SERVICE: Generating/Saving practice questions for topic: {input_data.practice_topic}")
    
    # If raw content isn't provided, generate it now.
    if raw_generated_questions is None:
        print("SERVICE: No raw content provided, generating from scratch...")
        history_summary = "No specific student performance history provided."
        db_conn_hist = None
        try:
            if input_data.student_id:
                db_conn_hist = get_mysql_connection(db_name=os.environ.get("MYSQL_DB"))
                if db_conn_hist:
                    summary = get_student_history_summary(db_conn_hist, input_data.student_id)
                    if summary: history_summary = summary
        finally:
            if db_conn_hist and db_conn_hist.is_connected():
                db_conn_hist.close()

        from langchain_community.embeddings import ZhipuAIEmbeddings
        embeddings = ZhipuAIEmbeddings()
        rag_snippets = search_knowledge_for_practice_topic(input_data.practice_topic, embeddings, CHROMA_PERSIST_DIR)
        
        prompt_components = construct_practice_question_prompt(
            input_data.practice_topic,
            input_data.question_preferences,
            student_history_summary=history_summary,
            retrieved_context_snippets=rag_snippets
        )
        raw_generated_questions = get_llm_practice_questions(
            prompt_components["system_message"],
            prompt_components["human_message"]
        )

    # Unified processing and saving logic
    db_conn_save = None
    try:
        if not (raw_generated_questions and raw_generated_questions.strip()):
            return PracticeQuestionsOutput(generated_questions=[], error_message="AI未能生成练习题内容。")

        # --- 【核心修正点：使用正则表达式进行宽容分割】 ---
        # Regex to find "---参考答案与解析---" with potential variations
        # It allows for:
        # - Optional leading/trailing whitespace/newlines (\s*)
        # - Optional hyphens at the start (---)?
        # - The core text "参考答案与解析"
        # - Optional hyphens at the end (---)?
        separator_pattern = re.compile(r'\s*---?\s*参考答案与解析\s*---?\s*', re.IGNORECASE)
        
        match = separator_pattern.search(raw_generated_questions)

        if match:
            # If a match is found, split the string at the match position
            all_questions_text = raw_generated_questions[:match.start()].strip()
            all_answers_text = raw_generated_questions[match.end():].strip()
            print("SERVICE INFO: Successfully split questions and answers using regex.")
        else:
            # Fallback if the separator is still not found
            all_questions_text = raw_generated_questions.strip()
            all_answers_text = "（答案解析未找到，AI可能未按指定格式输出）"
            print("SERVICE WARNING: Separator pattern not found. Could not split questions and answers.")

        # --- (分割逻辑修正结束) ---

        db_conn_save = get_mysql_connection(db_name=os.environ.get("MYSQL_DB"))
        if not db_conn_save:
            raise Exception("Failed to connect to the database for saving.")

        concepts_list = [input_data.practice_topic] if input_data.practice_topic else []
        catalog_id = save_practice_question_to_catalog(
            db_conn_save, all_questions_text, all_answers_text, concepts_list=concepts_list
        )

        if catalog_id:
            generated_item = PracticeQuestionItem(question_text=all_questions_text, model_answer=all_answers_text)
            return PracticeQuestionsOutput(generated_questions=[generated_item], catalog_id=catalog_id)
        else:
            return PracticeQuestionsOutput(generated_questions=[], error_message="生成了练习题但保存至题库失败。")
            
    except Exception as e:
        # It's helpful to log the full exception for debugging
        import traceback
        print(f"SERVICE ERROR in generate_practice_questions_service: {e}")
        traceback.print_exc()
        return PracticeQuestionsOutput(generated_questions=[], error_message=f"An unexpected error occurred: {str(e)}")
    finally:
        if db_conn_save and db_conn_save.is_connected():
            db_conn_save.close()


async def get_practice_feedback_service(
    input_data: PracticeFeedbackInput
) -> PracticeFeedbackOutput:
    print(f"SERVICE: Getting holistic feedback for student {input_data.student_id} on catalog_id {input_data.catalog_id}")

    db_conn = None
    questions_text = ""
    model_answers_text = ""
    try:
        MYSQL_DB_NAME = os.environ.get("MYSQL_DB")
        db_conn = get_mysql_connection(db_name=MYSQL_DB_NAME)
        # ... (获取 questions_text 和 model_answers_text 的逻辑)
        practice_set_details = get_practice_question_details_by_id(db_conn, input_data.catalog_id)
        if not practice_set_details:
             raise HTTPException(status_code=404, detail=f"Practice set with catalog_id {input_data.catalog_id} not found.")
        questions_text = practice_set_details.get("question_text", "")
        model_answers_text = practice_set_details.get("model_answer", "")
    finally:
        if db_conn and db_conn.is_connected():
            db_conn.close()

    holistic_feedback_prompt = f"""
    你是一位顶级的AI辅导老师，善于对学生的一次性整体作答给出全面、细致、有条理的反馈。

    **【上下文信息】**

    1.  **试卷的全部题目:**
        ---
        {questions_text}
        ---

    2.  **所有题目的参考答案与解析:**
        ---
        {model_answers_text}
        ---

    3.  **学生提交的全部回答:**
        ---
        {input_data.student_answer}
        ---

    **【你的任务】**
    请一步到位，完成以下所有任务，并严格按照指定的JSON格式输出。

    1.  **总体评价 (Overall Comment)**: 首先，请对学生的整体作答情况给出一个简短、鼓励性的总体评价。
    2.  **逐题反馈 (Detailed Feedback)**: 接着，对学生回答的**每一个问题**，进行独立的分析和反馈。你需要：
        a.  将学生的回答匹配到具体的题目。
        b.  判断该回答的正确性 (`Correct`, `Partially Correct`, `Incorrect`, `Not Answered`)。
        c.  给出有建设性的、详细的反馈。
    
    **【输出格式 - 必须严格遵守】**
    你的输出必须是且只能是一个单一的JSON对象，结构如下：
    ```json
    {{
      "overall_comment": "（这里是你的总体评价，例如：这次练习完成得很不错！对大部分知识点掌握得很好。）",
      "feedback_details": [
        {{
          "question_identifier": "（题目的原始标识符，如 '题目1'）",
          "student_answer": "（你从学生回答中匹配到的、针对这个问题的具体内容）",
          "correctness": "（'Correct', 'Partially Correct', 'Incorrect', 或 'Not Answered'）",
          "feedback": "（针对这个问题的详细反馈）"
        }},
        // ... 为其他所有题目生成类似的反馈对象 ...
      ]
    }}
    ```
    - **重要**: `feedback_details` 数组必须包含对**试卷中所有题目**的反馈，即使学生没有回答某个问题（此时 `correctness` 应为 `Not Answered`）。

    现在，请开始生成反馈JSON：
    """

    from backend_app.nlp_utils import parse_query_with_llm
    parsed_feedback_json = await parse_query_with_llm(holistic_feedback_prompt)

    if "error" in parsed_feedback_json or not parsed_feedback_json.get("feedback_details"):
        raise HTTPException(status_code=500, detail="Failed to get a valid holistic feedback from AI.")
        

    overall_comment = parsed_feedback_json.get("overall_comment", "No overall comment provided.")
    feedback_details_raw = parsed_feedback_json.get("feedback_details", [])
    total_questions = len(feedback_details_raw)
    score = 0.0
    if total_questions > 0:
        for item in feedback_details_raw:
            correctness = item.get("correctness")
            if correctness == "Correct":
                score += 1.0
            elif correctness == "Partially Correct":
                score += 0.5
    
    # 计算准确率
        accuracy_percentage = (score / total_questions) * 100
    # 将结果格式化为字符串，例如 "85.7%" 或 "A+"
    # 这里我们直接存百分比字符串，因为它信息量大且可读
        overall_correctness_for_db = f"{accuracy_percentage:.1f}%"
    else:
        overall_correctness_for_db = "N/A" # 没有题目，无法评估

# --- 2. 准备保存到数据库 ---
    feedback_for_db = json.dumps(feedback_details_raw, ensure_ascii=False)

    db_conn = None
    attempt_id = None
    try:
        MYSQL_DB_NAME = os.environ.get("MYSQL_DB")
        db_conn = get_mysql_connection(db_name=MYSQL_DB_NAME)
        if not db_conn:
            raise Exception("Database connection failed for saving.")
        
    # 调用 save_practice_attempt，传入新计算出的评估结果
        attempt_id = save_practice_attempt(
            db_conn,
            input_data.student_id,
            input_data.catalog_id,
            input_data.student_answer,
            overall_correctness_for_db,       # <--- 修正：保存计算出的准确率
            feedback_for_db
    )
        if not attempt_id:
            raise Exception("Failed to save the practice attempt to the database.")
        
    finally:
        if db_conn and db_conn.is_connected():
            db_conn.close()

    # --- 5. 构造并返回API响应 ---
    # 使用 Pydantic 模型进行验证和序列化
    feedback_items = [FeedbackItem(**item) for item in feedback_details_raw]
    
    return PracticeFeedbackOutput(
        attempt_id=attempt_id,
        overall_comment=overall_comment,
        feedback_details=feedback_items
    )

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
            else: 
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
    

        
async def refine_assessment_service(input_data: RefineAssessmentInput) -> tuple[str, list, str ]:
    print(f"SERVICE: Starting refine process for assessment ID: {input_data.base_assessment_id}")
    
    user_intent = await _identify_assessment_intent(input_data.history, input_data.new_query)
    db_conn = None
    original_assessment_content = ""
    original_answers_text = ""
    original_subject = None  
    MYSQL_DB_NAME = os.environ.get("MYSQL_DB")
    answer_separator = "参考答案与解析"
    
    try:
        db_conn = get_mysql_connection(db_name=MYSQL_DB_NAME)
        if db_conn:
                original_assessment = get_assessment_details_by_id(db_conn, input_data.base_assessment_id)
                if original_assessment:
                    original_assessment_content = original_assessment.get('content', '')
                    original_answers_text = original_assessment.get('answers_text', '') 
                    original_subject = original_assessment.get('subject') 
                    if original_assessment_content:
                        input_data.history.insert(0, ChatMessage(
                            role="system",
                            content=f"你正在修改以下这份原始试卷：\n\n--- 原始试卷开始 ---\n{original_assessment_content}\n--- 原始试卷结束 ---"
                        ))

        standalone_query_for_rag = await _rewrite_teaching_assessment_query(input_data.history, input_data.new_query)
        embeddings = ZhipuAIEmbeddings()
        rag_snippets = perform_rag_search(standalone_query_for_rag, embeddings, CHROMA_PERSIST_DIR)
        langchain_messages = convert_chat_messages_to_langchain_format(input_data.history, input_data.new_query)
        
        final_messages_for_llm = construct_assessment_prompt_with_history(
            history=langchain_messages[:-1],
            new_query=langchain_messages[-1].content,
            rag_snippets=rag_snippets,
            user_intent=user_intent
        )
        
        generated_content = generate_assessment_with_llm(messages=final_messages_for_llm)
        if not generated_content:
            return None, [], None

        final_full_content = None

        # ... (意图处理逻辑保持不变) ...
        if user_intent == "INCREMENTAL_ADD":
            print("SERVICE: Handling INCREMENTAL_ADD. Appending new content and answers.")
            
            # 分割LLM生成的新内容
            new_parts = generated_content.split(answer_separator, 1)
            new_questions_part = new_parts[0].strip()
            new_answers_part = new_parts[1].strip() if len(new_parts) > 1 else ""

            # 拼接完整的问题部分
            full_questions = f"{original_assessment_content}\n\n{new_questions_part}"
            
            # 拼接完整的答案部分
            full_answers = f"{original_answers_text}\n\n{new_answers_part}".strip()
            
            # 重新组合成最终的完整内容
            final_full_content = f"{full_questions}\n\n{answer_separator}\n{full_answers}"
        
        elif user_intent in ["REVISION", "DELETION", "REWRITE"]:
            print(f"SERVICE: Handling {user_intent}. Replacing original content with new version.")
            # 对于这些意图, LLM应被引导生成完整的新版本，所以直接使用其输出
            final_full_content = generated_content

        else: # UNKNOWN 或其他未处理的意图
            print("SERVICE: Intent is UNKNOWN. Defaulting to incremental add behavior (with answer concatenation).")
            # 同样应用增量添加逻辑作为安全默认值
            new_parts = generated_content.split(answer_separator, 1)
            new_questions_part = f"[意图未知，内容已附加]：\n{new_parts[0].strip()}"
            new_answers_part = new_parts[1].strip() if len(new_parts) > 1 else ""

            full_questions = f"{original_assessment_content}\n\n{new_questions_part}"
            full_answers = f"{original_answers_text}\n\n{new_answers_part}".strip()
            final_full_content = f"{full_questions}\n\n{answer_separator}\n{full_answers}"

        if final_full_content is None:
            print("SERVICE ERROR: No final content was constructed due to logic error.")
            return None, [], None

        # ## 修改点3: 返回元组 (完整内容, RAG片段, 原始学科)
        return final_full_content, rag_snippets, original_subject
    
    except Exception as e:
        print(f"SERVICE ERROR in refine_assessment_service: {e}")
        # ## 修改点4: 调整返回以匹配新签名
        return None, [], None
    finally:
        if db_conn and db_conn.is_connected():
            db_conn.close()

    

async def get_student_assessment_performance_service(assessment_id: int) -> List[StudentAssessmentSummary]:
    """
    Retrieves and aggregates student performance for a specific assessment, calculating accuracy for each student.
    """
    print(f"SERVICE: Call received for aggregated performance on assessment_id: {assessment_id}")
    
    db_conn = None
    summary_list: List[StudentAssessmentSummary] = []
    
    MYSQL_DB_NAME = os.environ.get("MYSQL_DB")
    if not MYSQL_DB_NAME:
        raise HTTPException(status_code=500, detail="Database configuration error.")

    try:
        db_conn = get_mysql_connection(db_name=MYSQL_DB_NAME)
        if not db_conn:
            raise HTTPException(status_code=500, detail="Failed to connect to the database.")

        # Call the new aggregation function from database_utils
        aggregated_data = get_aggregated_student_performance(db_conn, assessment_id)

        if not aggregated_data:
            print(f"SERVICE: No performance data found for assessment_id: {assessment_id}. Returning empty list.")
            return []

        # Process each student's aggregated data
        for row in aggregated_data:
            total = row.get('total_answered', 0)
            if total > 0:
                correct = row.get('correct_count', 0)
                partial = row.get('partially_correct_count', 0)
                # Calculate accuracy: (Correct * 1 + Partial * 0.5) / Total * 100
                accuracy = ((correct + 0.5 * partial) / total) * 100
            else:
                accuracy = 0.0

            # Create the Pydantic model for the response
            summary_item = StudentAssessmentSummary(
                student_id=row['student_id'],
                student_name=row.get('student_name'),
                total_answered=total,
                correct_count=row.get('correct_count', 0),
                partially_correct_count=row.get('partially_correct_count', 0),
                incorrect_count=row.get('incorrect_count', 0),
                accuracy=round(accuracy, 2) # Round to 2 decimal places
            )
            summary_list.append(summary_item)
        
        print(f"SERVICE: Successfully aggregated performance for {len(summary_list)} students on assessment_id: {assessment_id}")
        return summary_list

    except Exception as e:
        print(f"SERVICE ERROR: An unexpected error occurred in get_student_assessment_performance_service: {e}")
        raise HTTPException(status_code=500, detail="An internal server error occurred while processing performance data.")
    finally:
        if db_conn and db_conn.is_connected():
            db_conn.close()
            print(f"SERVICE: DB connection closed for get_student_assessment_performance_service (assessment_id: {assessment_id}).")



async def _rewrite_teaching_plan_query(chat_history: List[ChatMessage], new_query: str) -> str:
    print("SERVICE (Teaching Plan Refine): Rewriting query with history...")
    history_str = "\n".join([f"{msg.role}: {msg.content}" for msg in chat_history])
    # (修改1) 使用模板变量而不是f-string
    rewrite_prompt_template = ChatPromptTemplate.from_template("""
        你是一个教学大纲优化助手。你的任务是分析一段关于“教案生成”的对话历史，并将用户最新的、可能不完整的修改指令，改写成一个独立的、包含核心主题的、可以用于信息检索的查询。只输出改写后的查询，不要包含任何额外解释。
        对话历史:
        ---
        {history}
        ---

        用户的最新修改指令是: "{query}"

        请根据以上对话历史，将这个指令改写成一个独立的、包含教案核心主题的检索查询。

        例如，如果历史是关于“一战历史”的教案，用户的指令是“增加一些关于萨拉热窝事件的细节”，你应该输出“一战历史中的萨拉热窝事件细节”。
        如果历史是关于“Python列表推导式”的教案，用户的指令是“再加几个练习题”，你应该输出“关于Python列表推导式的练习题示例”。

        现在，请开始改写：
        """)
    try:
        llm = ChatZhipuAI(model="glm-4", temperature=0.0) 
        chain = rewrite_prompt_template | llm | StrOutputParser()
        # (修改2) 将变量通过字典传递给ainvoke
        rewritten_query = await chain.ainvoke({"history": history_str, "query": new_query})
        print(f"SERVICE (Teaching Plan Refine): Original query: '{new_query}', Rewritten query: '{rewritten_query}'")
        return rewritten_query.strip()
    except Exception as e:
        print(f"SERVICE ERROR (Teaching Plan Refine): Failed to rewrite query: {e}. Falling back to original query.")
        return new_query


async def _rewrite_teaching_assessment_query(chat_history: List[ChatMessage], new_query: str) -> str:
    history_str = "\n".join([f"{msg.role}: {msg.content}" for msg in chat_history])
    
    # ## 修改点：优化Prompt，增加更复杂的示例和规则 ##
    rewrite_prompt_template = ChatPromptTemplate.from_template("""
    你是一个教学测试题优化助手。你的任务是分析一段关于“测试题生成”的对话历史，并将用户最新的、可能不完整的修改指令，改写成一个独立的、包含核心主题和【最新约束】的、可以用于信息检索的查询。

    **核心规则：**
    1.  **继承主题**：查询必须包含对话的核心主题（例如“一战历史”、“TensorFlow.js”）。
    2.  **覆盖约束**：如果用户的最新指令中包含了新的约束条件（如题型、难度、数量），这些新约束必须【覆盖】历史记录中的旧约束。
    3.  **保持简洁**：只输出改写后的查询，不要包含任何额外解释。

    ---
    **对话历史:**
    {history}
    ---
    **用户的最新修改指令是:** "{query}"
    ---

    **示例学习:**

    *   **示例1 (主题继承):**
        *   历史: 用户要求生成关于“一战历史”的习题。
        *   最新指令: “增加一些关于萨拉热窝事件的”
        *   改写后输出: `有关一战历史中的萨拉热窝事件的习题`

    *   **示例2 (题型覆盖 - 这非常重要!):**
        *   历史: 用户要求生成“5道关于Python基础知识的选择题”。
        *   最新指令: “再来2道填空题”
        *   改写后输出: `关于Python基础知识的填空题` (注意：题型从“选择题”变成了“填空题”)

    *   **示例3 (您的场景):**
        *   历史: 用户要求生成“关于TensorFlow.js编程的选择题”。
        *   最新指令: “再生成几道编程题”
        *   改写后输出: `关于TensorFlow.js的编程题` (注意：题型从“选择题”变成了“编程题”)

    现在，请根据以上规则和示例，对以下内容进行改写。

    **对话历史:**
    ---
    {history}
    ---
    **用户的最新修改指令是:** "{query}"

    **改写后的查询:**
    """)
    try:
        llm = ChatZhipuAI(model="glm-4", temperature=0.0) 
        chain = rewrite_prompt_template | llm | StrOutputParser()
        rewritten_query = await chain.ainvoke({"history": history_str, "query": new_query})
        print(f"DEBUG: Original Query: '{new_query}' | Rewritten Query: '{rewritten_query.strip()}'") # 添加这行来调试
        return rewritten_query.strip()
    except Exception as e:
        print(f"SERVICE ERROR (_rewrite_teaching_assessment_query): Failed to rewrite query. Error: {e}. Falling back to original query.")
        return new_query
    
async def get_assessment_list_service() -> PracticeQuestionListOutput:
    print("SERVICE: Call received for get_practice_question_list_service")
    db_conn = None
    MYSQL_DB_NAME = os.environ.get("MYSQL_DB")
    if not MYSQL_DB_NAME:
        raise HTTPException(status_code=500, detail="Database not configured.")
        
    try:
        db_conn = get_mysql_connection(db_name=MYSQL_DB_NAME)
        if not db_conn:
            raise HTTPException(status_code=500, detail="Failed to connect to the database.")


        raw_question_list = get_all_assessment_for_student_view(db_conn)
        

        question_list = [PracticeQuestionListItem(**item) for item in raw_question_list]

        return PracticeQuestionListOutput(questions=question_list)

    except HTTPException as he:
        raise he
    except Exception as e:
        print(f"SERVICE ERROR: An unexpected error occurred in get_practice_question_list_service: {e}")
        raise HTTPException(status_code=500, detail="An internal server error occurred while fetching the practice list.")
    finally:
        if db_conn and db_conn.is_connected():
            db_conn.close()


async def get_assessment_detail_service(assessment_id: int) -> PracticeQuestionDetailOutput:
    print(f"SERVICE: Call received for get_assessment_detail_service with id: {assessment_id}")
    db_conn = None
    MYSQL_DB_NAME = os.environ.get("MYSQL_DB")
    if not MYSQL_DB_NAME:
        raise HTTPException(status_code=500, detail="Database not configured.")

    try:
        db_conn = get_mysql_connection(db_name=MYSQL_DB_NAME)
        if not db_conn:
            raise HTTPException(status_code=500, detail="Failed to connect to the database.")
            
        assessment_details_dict = get_assessment_details_by_id(db_conn, assessment_id)
        
        if not assessment_details_dict:
            raise HTTPException(status_code=404, detail=f"Assessment with ID {assessment_id} not found.")
            
        return PracticeQuestionDetailOutput(**assessment_details_dict)

    except HTTPException as he:
        raise he
    except Exception as e:
        print(f"SERVICE ERROR: An unexpected error occurred in get_assessment_detail_service: {e}")
        raise HTTPException(status_code=500, detail=f"An internal server error occurred while fetching assessment details.")
    finally:
        if db_conn and db_conn.is_connected():
            db_conn.close()

async def get_teacher_assessments_service(teacher_id: int) -> TeacherAssessmentListOutput:
    db_conn = None
    try:
        MYSQL_DB_NAME = os.environ.get("MYSQL_DB")
        db_conn = get_mysql_connection(db_name=MYSQL_DB_NAME)
        if not db_conn:
            raise HTTPException(status_code=500, detail="Database connection failed.")
        
        assessments_raw = get_assessments_by_teacher_id(db_conn, teacher_id)
        
        assessments_list = [TeacherAssessmentListItem(**item) for item in assessments_raw]
        return TeacherAssessmentListOutput(assessments=assessments_list)
        
    finally:
        if db_conn and db_conn.is_connected():
            db_conn.close()

async def publish_assessment_service(assessment_id: int, teacher_id: int):
    db_conn = None
    try:
        MYSQL_DB_NAME = os.environ.get("MYSQL_DB")
        db_conn = get_mysql_connection(db_name=MYSQL_DB_NAME)
        if not db_conn:
            raise HTTPException(status_code=500, detail="Database connection failed.")
        success = publish_assessment(db_conn, assessment_id, teacher_id)
        
        if not success:
            # 这可能是因为它已经被发布了
            raise HTTPException(status_code=409, detail="Assessment is already published or does not exist.")
            
        return {"message": f"Assessment {assessment_id} published successfully."}
        
    finally:
        if db_conn and db_conn.is_connected():
            db_conn.close()