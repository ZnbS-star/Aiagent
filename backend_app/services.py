from collections import defaultdict
import csv
from datetime import datetime
import io
import json
import os
import re
import sys
from typing import Any, Dict, List, Optional
from langchain_community.chat_models import ChatZhipuAI # For LLM interaction
from langchain_core.output_parsers import StrOutputParser # For LLM interaction
from langchain.prompts import ChatPromptTemplate
from typing import Literal

from backend_app.nlp_utils import parse_query_with_llm
from backend_app.security import get_password_hash

IntentType = Literal["INCREMENTAL_ADD", "REVISION", "DELETION", "REWRITE", "UNKNOWN"]
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from backend_app.models import (
    ActivityStat, AdminCreateUserInput, AdminResetPasswordInput, AdminResourceDetailView, AssessmentAnalysis, AssessmentAnalysisOutput, ConceptStat, DailyAccuracy, DashboardUsageResponse, PaginatedAdminResourcesResponse, PaginatedUsersResponse, PracticeChatInput, PracticeChatOutput, 
    FeedbackItem, PracticeQuestionDetailOutput, PracticeQuestionListItem, PracticeQuestionListOutput, PracticeQuestionNLInput, PublishedAssessmentInfo, StudentAssessmentSummary, StudentEffectivenessResponse, StudentQuestionInput, StudentQuestionOutput,
    RefineStudentQAInput, RefineTeachingPlanInput, RefineAssessmentInput, ChatMessage, SubjectPerformance, TeacherAssessmentListItem, TeacherAssessmentListOutput, TeacherEfficiencyStat, UsageStats  # Added new models
)
from backend_app.database_utils import admin_create_user, admin_delete_user, admin_update_user_password, get_activity_stats, get_aggregated_student_performance, get_all_assessment_for_student_view, get_all_practice_attempts_with_concepts, get_all_subjects,  get_assessment_details_by_id, get_assessment_question_stats, get_assessments_by_teacher_id, get_daily_accuracy_trend, get_low_performing_subjects, get_mysql_connection,get_practice_question_details_by_id, get_published_assessments_by_teacher, get_student_for_auth, get_teacher_content_creation_stats, get_teacher_for_auth, get_teacher_resource_detail, get_teaching_plan_by_id, get_unified_resources_by_subject, get_users_by_role, publish_assessment 
from backend_app.student_qa import (
    _rewrite_query_with_history,
    search_knowledge_base_for_answer, 
    construct_student_qa_prompt, 
    construct_student_qa_prompt_with_history, 
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
        log_activity
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
        llm = ChatZhipuAI(model="glm-4", temperature=0.0)
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

async def _identify_practice_intent(history: List[ChatMessage], new_query: str, active_question_text: Optional[str] = None) -> PracticeIntent:
    """使用LLM分析对话，判断用户更精确的意图。"""
    print("SERVICE: Identifying nuanced practice assistant intent...")
    history_str = "\n".join([f"{msg.role}: {msg.content}" for msg in history])

    active_question_context = ""
    if active_question_text and active_question_text.strip():
        active_question_context = f"""
    **当前激活的题目 (如果学生正在作答):**
    ---
    {active_question_text}
    ---
    """
    else:
        active_question_context = "**当前没有激活的题目。**"



    intent_prompt = f"""
    你是一个文本分类器，任务是分析学生与AI练习助手的对话，判断学生最新输入的意图。
    意图只能是以下五种之一: ADD_QUESTIONS, REWRITE_QUESTIONS, SUBMIT_ANSWER, GENERATE_NEW, UNKNOWN。

    {active_question_context}

    **意图定义:**
    - ADD_QUESTIONS: 用户明确要求在上一轮题目基础上【增加】新的题目。
      (关键词: "再来几道", "加上", "多出点", "还想要2道")
    - REWRITE_QUESTIONS: 用户对上一轮的题目不满意，要求【替换】或【重写】。这包括改变难度、换内容、或完全换题型。
      (关键词: "太难了", "简单点", "换一批", "不要这个", "换成选择题")
    - SUBMIT_ANSWER: 用户正在提供问题的答案。**这是一个高优先级的判断**。如果【当前激活的题目】存在，并且用户的输入可以被合理解释为该题目的答案（**即使是很短的词或数字**），则意图应为 SUBMIT_ANSWER。例如，题目是填空题，用户的输入只是一个词。
    - GENERATE_NEW: 用户想开始一个【全新的练习主题】，与上一轮无关。
      (关键词: "我们来练习...", "换个主题", "我想学...", "出点关于...的题")
    - UNKNOWN: 无法判断或闲聊。

    **对话历史:**
    ---
    {history_str}
    ---
    **学生最新输入:** "{new_query}"
    ---

    **示例 (填空题作答):**
    - 当前激活的题目: "法国的首都是____。"
    - 学生最新输入: "巴黎"
    - 你的输出: SUBMIT_ANSWER

    你的输出【必须只包含一个单词】，即你选择的意图类型。
    """
    try:
        llm = ChatZhipuAI(model="glm-4", temperature=0.0)
        response = await llm.ainvoke(intent_prompt)
        intent = response.content.strip()
        
        valid_intents: List[PracticeIntent] = ["ADD_QUESTIONS", "REWRITE_QUESTIONS", "SUBMIT_ANSWER", "GENERATE_NEW", "UNKNOWN"]
        if intent in valid_intents:
            print(f"SERVICE: Identified intent as: {intent} (with active question context)")
            return intent
        print(f"SERVICE WARNING: LLM returned invalid intent '{intent}'. Defaulting to UNKNOWN.")
        return "UNKNOWN"
    except Exception as e:
        print(f"SERVICE ERROR: Failed to identify practice intent: {e}")
        return "UNKNOWN"

async def process_practice_chat_service(input_data: PracticeChatInput) -> PracticeChatOutput:
    db_conn = get_mysql_connection(db_name=os.environ.get("MYSQL_DB"))
    active_question_text = None
    practice_set_details = None

    try:
        if input_data.active_catalog_id and db_conn:
            print(f"SERVICE: Active catalog ID {input_data.active_catalog_id} found. Fetching question text for context.")
            practice_set_details = get_practice_question_details_by_id(db_conn, input_data.active_catalog_id)
            if practice_set_details:
                active_question_text = practice_set_details.get("question_text")

        # 将获取到的题目文本传递给意图识别函数
        intent = await _identify_practice_intent(input_data.history, input_data.new_query, active_question_text)

        # --- Branch 1: User is submitting an answer ---
        if intent == "SUBMIT_ANSWER":
            if not input_data.active_catalog_id:
                return PracticeChatOutput(assistant_response_text="我好像不知道你在回答哪一套题，请先让我出题。", intent_detected=intent)
            
            if not practice_set_details:
                 return PracticeChatOutput(assistant_response_text=f"抱歉，我找不到ID为 {input_data.active_catalog_id} 的题目了，我们重新开始吧？", intent_detected=intent, error_message="Active practice set not found in DB.")

            try:
                feedback_input = PracticeFeedbackInput(student_id=input_data.student_id, catalog_id=input_data.active_catalog_id, student_answer=input_data.new_query)
                feedback_result = await get_practice_feedback_service(feedback_input)
                log_activity(
                        db_conn,
                        user_id=input_data.student_id,
                        user_role="student",
                        activity_type="PRACTICE_SUBMIT_ANSWER",
                        details={"catalog_id": input_data.active_catalog_id, "attempt_id": feedback_result.attempt_id}
                    )
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
            log_activity(
                        db_conn,
                        user_id=input_data.student_id,
                        user_role="student",
                        activity_type="GENERATE_NEW_PRACTICE",
                        details={"new_catalog_id": new_questions_result.catalog_id, "based_on_id": input_data.active_catalog_id}
                    )
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
            original_questions, reliable_topic = "", "相关主题"
            
            # Reuse details fetched earlier
            if practice_set_details:
                original_answers = practice_set_details.get("model_answer", "")
                original_questions = practice_set_details.get("question_text", "")
                concepts_str = practice_set_details.get("concepts_covered", "")
                try: reliable_topic = json.loads(concepts_str)[0] if concepts_str else "相关主题"
                except (json.JSONDecodeError, IndexError): pass
            
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
            log_activity(
                        db_conn,
                        user_id=input_data.student_id,
                        user_role="student",
                        activity_type="PRACTICE_" + intent, # 动态记录是 ADD 还是 REWRITE
                        details={"new_catalog_id": new_questions_result.catalog_id, "based_on_id": input_data.active_catalog_id}
                    )
            
            return PracticeChatOutput(
                assistant_response_text="好的，已为你添加新题目。这是修改后的完整练习题：",
                intent_detected=intent,
                new_questions=new_questions_result
            )
            

        # --- Branch 4: User wants to rewrite the questions (Your intelligent rewrite logic) ---
        elif intent == "REWRITE_QUESTIONS":
            if not input_data.active_catalog_id:
                return PracticeChatOutput(assistant_response_text="我需要先为你出一套题，才能在它的基础上修改哦。", intent_detected=intent)

            # Get old question text and reliable topic from DB
            original_questions = ""
            reliable_topic = "相关主题"
            if practice_set_details:
                original_questions = practice_set_details.get("question_text", "")
                concepts_str = practice_set_details.get("concepts_covered", "")
                try:
                    reliable_topic = json.loads(concepts_str)[0] if concepts_str else "相关主题"
                except (json.JSONDecodeError, IndexError):
                    pass

            if not original_questions:
                return PracticeChatOutput(assistant_response_text="抱歉，我找不到你上一轮的题目了，我们重新开始吧？", intent_detected=intent)

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
            log_activity(
                        db_conn,
                        user_id=input_data.student_id,
                        user_role="student",
                        activity_type="PRACTICE_" + intent, 
                        details={"new_catalog_id": new_questions_result.catalog_id, "based_on_id": input_data.active_catalog_id}
                    )
            
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
    finally:
        if db_conn and db_conn.is_connected():
            db_conn.close()
    
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
            db_conn = get_mysql_connection(db_name=os.environ.get("MYSQL_DB"))
            log_activity(
                        db_conn,
                        user_id=input_data.student_id,
                        user_role="student",
                        activity_type="STUDENT_QA",
                        details={"question_length": len(input_data.question)}
                    )
            db_conn.close()
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

    initial_outline: str,
    style_tone:str,
    output_structure:str
) -> tuple[str , List[str] ]:
    zhipuai_api_key = _get_zhipuai_api_key()
    if not style_tone or not style_tone.strip():
        final_style_tone = "清晰、专业且易于理解"
        print(f"SERVICE INFO: 'style_tone' was empty, using default: '{final_style_tone}'")
    else:
        final_style_tone = style_tone
    if not output_structure or not output_structure.strip():
        final_output_structure = "请为以下教学大纲生成一个完整的教案。内容应包括：1. 教学目标；2. 知识点详解；3. 课堂活动与互动环节建议；4. 简单的实训练习及其指导；5. 预估的时间分布。" # 这是一个非常全面和实用的默认结构
        print(f"SERVICE INFO: 'output_structure' was empty, using default: '{final_output_structure}'")
    else:
        final_output_structure = output_structure
    retrieved_rag_snippets = []
    try:
        print(f"SERVICE: Performing RAG search for query: '{initial_outline[:100]}...'")
        embeddings = ZhipuAIEmbeddings()
        retrieved_rag_snippets = perform_rag_search(
            keywords_list=[initial_outline],
            embeddings_model_instance=embeddings,
            vector_store_dir=CHROMA_PERSIST_DIR,
            top_k=10
        )
    except Exception as e:
        print(f"SERVICE ERROR during RAG search: {e}. Proceeding without RAG context.")

    system_message = (
        f"你是一位经验丰富的教师，你的任务是根据提供的大纲和补充材料，撰写一份详细的教案初稿。需注重内容的清晰性、准确性，并全面覆盖要点。最后输出语言是中文"
    )
    
    human_message_parts = [
        f"请根据以下信息，为我生成一份教案初稿。\n",
        f"**1. 核心教学大纲:**\n{initial_outline}\n"
    ]

    if retrieved_rag_snippets:
        human_message_parts.append("**2. 补充参考材料 (用于丰富内容):**")
        for i, snippet in enumerate(retrieved_rag_snippets):
            human_message_parts.append(f"--- Snippet {i+1} ---\n{snippet}\n--- End Snippet {i+1} ---")
    else:
        human_message_parts.append("(无补充参考材料)")

    human_message_parts.append(f"\n**3. 输出语言风格要求:**\n{style_tone}\n")


    human_message_parts.append(
        f"""
**4. 必须严格遵守的输出结构 (这是最高优先级的指令！):**
你的输出【必须且只能】包含以下几个部分，并严格使用我提供的一级标题。不要增加、减少或修改任何一级标题。

---
{output_structure}
---

请现在开始，严格按照上述第4点的结构要求生成教案。
        """
    )
    human_message_content = "\n".join(human_message_parts)


    try:
        print("SERVICE: Initializing LLM for teaching plan generation (ChatZhipuAI)...")
        llm_for_plan = ChatZhipuAI(model="glm-4", temperature=0.7, api_key=zhipuai_api_key) 

        messages = [
            SystemMessage(content=system_message),
            HumanMessage(content=human_message_content)
        ]
        
        print("SERVICE: Generating teaching plan via LLM...")
        ai_message = await llm_for_plan.ainvoke(messages) 
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
        # 在调用construct_assessment_prompt之前，先执行RAG搜索
        print(f"SERVICE: Performing RAG search for assessment content: '{input_data.teaching_plan_content[:100]}...'")
        embeddings = ZhipuAIEmbeddings()
        retrieved_rag_snippets = perform_rag_search(
            keywords_list=[input_data.teaching_plan_content],
            embeddings_model_instance=embeddings,
            vector_store_dir=CHROMA_PERSIST_DIR,
            top_k=5  # 为试卷生成获取更精确的5个片段
        )
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
    zhipuai_api_key = _get_zhipuai_api_key() 
    results: List[StudentAssessmentEvaluationOutput] = []
    db_conn = None
    MYSQL_DB_NAME = os.environ.get("MYSQL_DB")
    try:
        db_conn = get_mysql_connection(db_name=MYSQL_DB_NAME)
        if not db_conn:
            raise Exception("Failed to connect to the database.")

        actual_student_id = input_data.student_id
        
        assessment_data = get_assessment_content_by_id(db_conn, input_data.assessment_id)
        if not assessment_data or not assessment_data.get("content"):
            raise ValueError(f"Could not retrieve content for assessment ID {input_data.assessment_id}.")
        assessment_content = assessment_data["content"]
        log_activity(
            db_conn,
            user_id=actual_student_id,
            user_role="student",
            activity_type="SUBMIT_ASSESSMENT",
            details={"assessment_id": input_data.assessment_id, "answer_count": len(input_data.answers)}
        )
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
    print(f"SERVICE: Generating/Saving practice questions for topic: {input_data.practice_topic}")
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

    db_conn_save = None
    try:
        if not (raw_generated_questions and raw_generated_questions.strip()):
            return PracticeQuestionsOutput(generated_questions=[], error_message="AI未能生成练习题内容。")
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
            return PracticeQuestionsOutput(generated_questions=generated_item, catalog_id=catalog_id)
        else:
            return PracticeQuestionsOutput(generated_questions=None, error_message="生成了练习题但保存至题库失败。")
            
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

    3.  **学生提交的全部回答 (作为一个整体):**
        ---
        {input_data.student_answer}
        ---

    **【你的任务】**
    请一步到位，完成以下所有任务，并严格按照指定的JSON格式输出。

    1.  **总体评价 (Overall Comment)**: 首先，请对学生的整体作答情况给出一个简短、鼓励性的总体评价。
    2.  **逐题反馈 (Detailed Feedback)**: 接着，对学生回答的**每一个问题**，进行独立的分析和反馈。你需要：
        a.  从【学生提交的全部回答】中，**精确地抽取出**针对当前题目的那部分回答。
        b.  判断该回答的正确性 (`Correct`, `Partially Correct`, `Incorrect`, `Not Answered`)。
        c.  给出有建设性的、详细的反馈。
    
    **【输出格式 - 必须严格遵守！】**
    你的输出必须是且只能是一个单一的JSON对象，结构如下：
    ```json
    {{
      "overall_comment": "（这里是你的总体评价...）",
      "feedback_details": [
        {{
          "question_identifier": "（题目的原始标识符，如 '题目1'）",
          "student_answer": "（【必须】从学生回答中【原文复制】针对这个问题的具体内容。此字段【绝对不能】包含'见学生答案'、'如上'等任何占位符或缩写，必须是学生答案的原文。）",
          "correctness": "（'Correct', 'Partially Correct', 'Incorrect', 或 'Not Answered'）",
          "feedback": "（针对这个问题的详细反馈）"
        }},
        // ... 为其他所有题目生成类似的反馈对象 ...
      ]
    }}
    ```
    - **重要**: `feedback_details` 数组必须包含对**试卷中所有题目**的反馈，即使学生没有回答某个问题（此时 `student_answer` 应为空字符串 `""`，`correctness` 应为 `Not Answered`）。

    现在，请开始生成反馈JSON：
    """
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
    

        accuracy_percentage = (score / total_questions) * 100

        overall_correctness_for_db = f"{accuracy_percentage:.1f}%"
    else:
        overall_correctness_for_db = "N/A" # 没有题目，无法评估


    feedback_for_db = json.dumps(feedback_details_raw, ensure_ascii=False)

    db_conn = None
    attempt_id = None
    try:
        MYSQL_DB_NAME = os.environ.get("MYSQL_DB")
        db_conn = get_mysql_connection(db_name=MYSQL_DB_NAME)
        if not db_conn:
            raise Exception("Database connection failed for saving.")
        
        attempt_id = save_practice_attempt(
            db_conn,
            input_data.student_id,
            input_data.catalog_id,
            input_data.student_answer,
            overall_correctness_for_db,       
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
        feedback_details=feedback_items,
        error_message=None
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
    rag_snippets = []
    llm_answer = "Could not determine a refined answer."
    error_message = None

    try:
        standalone_query_for_rag = await _rewrite_query_with_history(input_data.history, input_data.new_query)
        
        try:
            embeddings = ZhipuAIEmbeddings() 
        except Exception as e:
            pass

        rag_snippets = search_knowledge_base_for_answer(
            student_question=standalone_query_for_rag, 
            embeddings_model_instance=embeddings,
            vector_store_dir=CHROMA_PERSIST_DIR,
            top_k=10 
        )
        print(f"SERVICE (Refine): RAG search retrieved {len(rag_snippets)} snippets for rewritten query.")

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

            pass

    except Exception as e:

        pass

    return StudentQuestionOutput(
        student_question=input_data.new_query,
        rag_context=rag_snippets if rag_snippets else None,
        llm_answer=llm_answer,
        error_message=error_message
    )


async def refine_teaching_plan_service(input_data: RefineTeachingPlanInput) -> tuple[str, str, List[str]]:
    print(f"SERVICE: Starting refine process for teaching plan ID: {input_data.base_teaching_plan_id}")

    user_intent = await _identify_user_intent(input_data.history, input_data.new_query)
    db_conn = None
    original_plan_content = ""
    original_title = "未命名教案"
    MYSQL_DB_NAME = os.environ.get("MYSQL_DB")
    zhipuai_api_key = _get_zhipuai_api_key()

    try:
        db_conn = get_mysql_connection(db_name=MYSQL_DB_NAME)
        if db_conn and input_data.base_teaching_plan_id:
            original_plan = get_teaching_plan_by_id(db_conn, input_data.base_teaching_plan_id)
            if original_plan:
                original_title = original_plan.get('title', '未命名教案')
                original_plan_content = original_plan.get('content', '').strip()

        final_full_content = None
        # Perform RAG search based on the refinement query
        print(f"SERVICE: Performing RAG search for teaching plan refinement: '{input_data.new_query[:100]}...'")
        embeddings = ZhipuAIEmbeddings()
        rag_snippets = perform_rag_search(
            keywords_list=[input_data.new_query],
            embeddings_model_instance=embeddings,
            vector_store_dir=CHROMA_PERSIST_DIR,
            top_k=5  # Get 5 focused snippets for refinement
        )
        if user_intent == "INCREMENTAL_ADD":
            print(f"SERVICE: Handling {user_intent} for teaching plan with dedicated creation-then-append flow.")

            # 1. LLM 创作新内容
            creation_prompt = f"""
            你是一位教案设计专家。你的任务是根据一个【核心主题】和用户的【补充要求】，创作出【新增的教案章节】。
            核心主题: **{original_title}**
            用户的补充要求: **{input_data.new_query}**
            你的输出【只应包含你新创作的章节内容】，不要重复任何已有内容。
            """

            llm_for_plan = ChatZhipuAI(model="glm-4", temperature=0.7, api_key=zhipuai_api_key)
            generated_new_content = await llm_for_plan.ainvoke(creation_prompt)
            generated_new_content = generated_new_content.content # 提取内容

            if not generated_new_content or not generated_new_content.strip():
                raise Exception("LLM failed to generate new content for the ADD request.")

            # 2. Python 代码负责拼接
            final_full_content = (
                f"{original_plan_content}\n\n"
                f"--- 更新于 {datetime.now().strftime('%Y-%m-%d %H:%M')} ---\n\n"
                f"{generated_new_content}"
            )

        else: # REVISION, DELETION, REWRITE 走一个统一的、更强大的直接生成流程
            print(f"SERVICE: Handling {user_intent} for teaching plan with direct full-regeneration flow.")

            system_prompt = "你是一位经验丰富的教师和教案设计专家。你的任务是根据提供的原始教案和用户的修改指令，生成一份修改后的、全新的、完整的教案。"
            human_prompt_parts = [
                f"--- 原始教案 (供你参考和修改) ---\n{original_plan_content}\n--- 原始教案结束 ---",
            ]

            if user_intent in ["REVISION", "DELETION"]:
                human_prompt_parts.append(
                    "**【核心任务：修改教案】**\n请在继承原始教案的基础上，根据用户的最新指令进行修改。你的输出必须是修改后的【完整新版本】。"
                )
            else: # REWRITE
                human_prompt_parts.append(
                    "**【核心任务：重写教案】**\n请【完全忽略】上述原始教案，根据用户最新指令，从零开始创作一份【全新的、完整的】教案。"
                )

            human_prompt_parts.append(f"用户的最新指令是：'{input_data.new_query}'")
            if rag_snippets:
                rag_context_str = "\n".join([f"- {snippet}" for snippet in rag_snippets])
                human_prompt_parts.append(f"\n--- 补充参考材料 ---\n{rag_context_str}")
            human_prompt_content = "\n\n".join(human_prompt_parts)

            llm_for_plan = ChatZhipuAI(model="glm-4", temperature=0.7, api_key=zhipuai_api_key)
            ai_message = await llm_for_plan.ainvoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=human_prompt_content)
            ])
            final_full_content = ai_message.content

        # 生成标题并返回
        new_title = await _generate_semantic_title_for_refinement(original_title, input_data.new_query)
        if final_full_content is None:
            raise Exception("Failed to construct final content.")

        return new_title, final_full_content, rag_snippets

    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"SERVICE ERROR in refine_teaching_plan_service: {e}")
        return None, None, []
    finally:
        if db_conn and db_conn.is_connected():
            db_conn.close()
    

def _apply_structured_edits(original_q_text, original_a_text, instructions, answer_separator):
    q_to_delete_identifiers = set(instructions.get("questions_to_delete", []))
    q_to_add = instructions.get("questions_to_add", [])

    # --- 1. 精确解析原始题目和答案 ---
    def parse_content(text_content):
        # 按 "题目X" 分割，同时保留题号
        # 使用 findall 来捕获所有题目块
        pattern = r'(题目\s*\d+\s*[:：.][\s\S]*?)(?=\n题目\s*\d+\s*[:：.]|\Z)'
        items = re.findall(pattern, text_content)
        
        # 将其转换为 { "题目1": "题目1的完整内容...", "题目2": "..." } 的字典
        item_dict = {}
        for item in items:
            match = re.match(r'(题目\s*\d+)', item.strip())
            if match:
                item_id = match.group(1).replace(" ", "")
                item_dict[item_id] = item.strip()
        return item_dict

    # 找到原始试卷中所有题型的大标题，并保留它们的顺序
    original_type_titles = re.findall(r'^[一二三四五六七八九十、\w\s]+题$', original_q_text, re.MULTILINE)

    original_questions_dict = parse_content(original_q_text)
    original_answers_dict = parse_content(original_a_text)

    # --- 2. 执行删除操作 ---
    for del_id in q_to_delete_identifiers:
        if del_id in original_questions_dict:
            del original_questions_dict[del_id]
        if del_id in original_answers_dict:
            del original_answers_dict[del_id]
            
    # --- 3. 执行添加操作 ---
    for new_q_obj in q_to_add:
        q_text = new_q_obj.get("question_text", "")
        a_text = new_q_obj.get("model_answer", "")
        
        # 从新题目文本中提取题号
        match = re.match(r'(题目\s*\d+)', q_text.strip())
        if match:
            new_q_id = match.group(1).replace(" ", "")
            original_questions_dict[new_q_id] = q_text
            original_answers_dict[new_q_id] = a_text
    
    # --- 4. 重新组装试卷 ---
    # 首先，按题号对字典进行排序，以保持正确的题目顺序
    sorted_q_ids = sorted(original_questions_dict.keys(), key=lambda x: int(re.search(r'\d+', x).group()))

    new_questions_section = "\n\n".join([original_questions_dict[qid] for qid in sorted_q_ids])

    # 构建答案区
    new_answers_section = "\n\n".join([original_answers_dict.get(qid, "") for qid in sorted_q_ids])

    return f"{new_questions_section}\n\n{answer_separator}\n\n{new_answers_section}"


async def refine_assessment_service(input_data: RefineAssessmentInput) -> tuple[str, str, list, str]:
    print(f"SERVICE: Starting refine process for assessment ID: {input_data.base_assessment_id}")

    user_intent = await _identify_assessment_intent(input_data.history, input_data.new_query)
    db_conn = None
    original_assessment_content = ""
    original_answers_text = ""
    original_subject = None
    original_title = "未命名考核"
    MYSQL_DB_NAME = os.environ.get("MYSQL_DB")
    answer_separator = "---参考答案与解析---"

    try:
        db_conn = get_mysql_connection(db_name=MYSQL_DB_NAME)
        if db_conn and input_data.base_assessment_id:
            original_assessment = get_assessment_details_by_id(db_conn, input_data.base_assessment_id)
            if original_assessment:
                original_title = original_assessment.get('title', '未命名考核')
                original_assessment_content = original_assessment.get('content', '').strip()
                original_answers_text = original_assessment.get('answers_text', '').strip()
                original_subject = original_assessment.get('subject')

        final_full_content = None
        # RAG Search based on the new query
        print(f"SERVICE: Performing RAG search for assessment refinement: '{input_data.new_query[:100]}...'")
        embeddings = ZhipuAIEmbeddings()
        rag_snippets = perform_rag_search(
            keywords_list=[input_data.new_query],
            embeddings_model_instance=embeddings,
            vector_store_dir=CHROMA_PERSIST_DIR,
            top_k=5
        )
        if user_intent == "INCREMENTAL_ADD":
            print(f"SERVICE: Handling {user_intent} with a dedicated creation-then-merge flow.")
            creation_prompt = f"""
            你是一位出题专家。你的任务是根据一个【核心主题】和用户的【具体要求】，创作出符合要求的【新增题目和答案】。
            核心主题: **{original_title}**
            用户的具体要求: **{input_data.new_query}**
            补充知识: **{rag_snippets}**
            你的输出【只应包含你新创作的题目和答案】，并使用 "{answer_separator}" 分隔。新题目要包含题型大标题，如“一、选择题”。
            """
            generated_new_content = generate_assessment_with_llm(messages=[
                SystemMessage(content="你是一个只负责内容创作的AI。"),
                HumanMessage(content=creation_prompt)
            ])
            if not generated_new_content or not generated_new_content.strip():
                raise Exception("LLM failed to generate new content for the ADD request.")
            final_full_content = f"{original_assessment_content}\n\n{original_answers_text}\n\n{generated_new_content}"
            # 我们将拼接逻辑简化，让前端或后续步骤处理合并，以确保所有内容都存在
            new_parts = generated_new_content.split(answer_separator, 1)
            new_questions_part = new_parts[0].strip()
            new_answers_part = new_parts[1].strip() if len(new_parts) > 1 else ""
            full_questions = f"{original_assessment_content}\n\n{new_questions_part}"
            full_answers = f"{original_answers_text}\n\n{new_answers_part}".strip()
            final_full_content = f"{full_questions}\n\n{answer_separator}\n\n{full_answers}"

        elif user_intent in ["REVISION", "DELETION"]:
            print(f"SERVICE: Handling {user_intent} with structured output approach.")
            structured_edit_prompt = f"""
            你是一个精准的文本编辑指令生成器。你的任务是分析一份【原始试卷】和用户的【修改要求】，然后生成一个描述如何修改的JSON指令。
            --- 原始试卷 ---
            {original_assessment_content}
            ---
            --- 用户的修改要求 ---
            {input_data.new_query}
            ---
            --- 补充知识 ---
            {rag_snippets}
            ---
            【你的任务】
            请生成一个JSON对象，该对象包含两个键：
            1. `questions_to_delete`: 一个包含【需要被删除的题目编号】的数组。编号必须是 "题目X" 的格式。例如 ["题目6"]。如果不需要删除任何题目，则为空数组 `[]`。
            2. `questions_to_add`: 一个包含【需要新增的题目】的数组，每个题目是一个包含 "question_text" 和 "model_answer" 的对象。如果不需要新增题目，则为空数组 `[]`。
            你的输出必须是且只能是一个JSON对象。
            """
            edit_instructions = await parse_query_with_llm(structured_edit_prompt)
            if "error" in edit_instructions:
                raise Exception(f"LLM failed to generate valid edit instructions: {edit_instructions['error']}")
            final_full_content = _apply_structured_edits(original_assessment_content, original_answers_text, edit_instructions, answer_separator)

        else: # REWRITE
            print(f"SERVICE: Handling {user_intent} with direct generation approach.")
            system_prompt = "你是一位顶级的出题专家，请根据用户指令重写试卷。"
            human_prompt_parts = [
                "**【核心任务：重写试卷】**\n请【完全忽略】所有上下文，根据用户最新指令，从零开始生成一份【全新的、完整的】试卷。",
                f"用户的最新指令是：'{input_data.new_query}'",
                f"补充知识: {rag_snippets}",
                f"\n【输出格式规范】\n你的输出必须包含 {answer_separator} 分隔的题目和答案部分。"
            ]
            human_prompt_content = "\n".join(human_prompt_parts)
            messages = [SystemMessage(content=system_prompt), HumanMessage(content=human_prompt_content)]
            final_full_content = generate_assessment_with_llm(messages=messages)

        new_title = await _generate_semantic_title_for_refinement(original_title, input_data.new_query)
        if final_full_content is None:
            raise Exception("Failed to construct final content.")
        return new_title, final_full_content, [], original_subject

    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"SERVICE ERROR in refine_assessment_service: {e}")
        return None, None, [], None
    finally:
        if db_conn and db_conn.is_connected():
            db_conn.close()

async def get_student_assessment_performance_service(assessment_id: int) -> List[StudentAssessmentSummary]:

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
        log_activity(
                    db_conn,
                    user_id=teacher_id,
                    user_role="teacher",
                    activity_type="PUBLISH_ASSESSMENT",
                    details={"assessment_id": assessment_id}
                )
        
        if not success:
            # 这可能是因为它已经被发布了
            raise HTTPException(status_code=409, detail="Assessment is already published or does not exist.")
            
        return {"message": f"Assessment {assessment_id} published successfully."}
        
    finally:
        if db_conn and db_conn.is_connected():
            db_conn.close()

async def list_users_service(role: str, page: int, page_size: int, search: Optional[str]) -> PaginatedUsersResponse:
    db_conn = get_mysql_connection(db_name="Aiagent")
    if not db_conn:
        raise HTTPException(status_code=500, detail="Database connection failed.")
    try:
        result = get_users_by_role(db_conn, role, page, page_size, search)
        return PaginatedUsersResponse(**result)
    finally:
        if db_conn.is_connected():
            db_conn.close()

async def create_user_by_admin_service(user_data: AdminCreateUserInput) -> Dict[str, Any]:

    if user_data.role == 'student':
        existing_user = await get_student_for_auth(user_data.username)
    else:
        existing_user = await get_teacher_for_auth(user_data.username)

    if existing_user:
        raise HTTPException(status_code=409, detail=f"Username '{user_data.username}' already exists.")

    hashed_password = get_password_hash(user_data.password)
    
    db_conn = get_mysql_connection(db_name=os.environ.get("MYSQL_DB"))
    if not db_conn:
        raise HTTPException(status_code=500, detail="Database connection failed.")
    try:
        new_user_id = admin_create_user(db_conn, user_data.username, hashed_password, user_data.role)
        if not new_user_id:
            raise HTTPException(status_code=500, detail="Failed to create user in database.")
        return {"id": new_user_id, "username": user_data.username}
    finally:
        if db_conn.is_connected():
            db_conn.close()

async def reset_user_password_service(user_id: int, role: str, password: str):
    new_hashed_password = get_password_hash(password)
    db_conn = get_mysql_connection(db_name=os.environ.get("MYSQL_DB"))
    if not db_conn:
        raise HTTPException(status_code=500, detail="Database connection failed.")
    try:
        success = admin_update_user_password(db_conn, user_id, new_hashed_password, role)
        if not success:
            raise HTTPException(status_code=404, detail=f"User with role '{role}' and id {user_id} not found.")
        return {"message": "Password updated successfully."}
    finally:
        if db_conn.is_connected():
            db_conn.close()

async def delete_user_service(user_id: int, role: str):
    db_conn = get_mysql_connection(db_name=os.environ.get("MYSQL_DB"))
    if not db_conn:
        raise HTTPException(status_code=500, detail="Database connection failed.")
    try:
        success = admin_delete_user(db_conn, user_id, role)
        if not success:
            raise HTTPException(status_code=404, detail=f"User with role '{role}' and id {user_id} not found.")
        return {"message": "User deleted successfully."}
    finally:
        if db_conn.is_connected():
            db_conn.close()


async def get_teacher_resource_detail_service(resource_type: str, resource_id: int) -> AdminResourceDetailView:
    db_conn = get_mysql_connection(db_name=os.environ.get("MYSQL_DB"))
    if not db_conn:
        raise HTTPException(status_code=500, detail="Database connection failed.")
    try:
        details = get_teacher_resource_detail(db_conn, resource_type, resource_id)
        if not details:
            raise HTTPException(status_code=404, detail="Resource not found.")
        return AdminResourceDetailView(**details)
    finally:
        if db_conn and db_conn.is_connected():
            db_conn.close()



async def list_all_subjects_service() -> List[str]:

    db_conn = get_mysql_connection(db_name=os.environ.get("MYSQL_DB"))
    if not db_conn:
        raise HTTPException(status_code=500, detail="Database connection failed.")
    try:
        subjects = get_all_subjects(db_conn)
        return subjects
    finally:
        if db_conn.is_connected():
            db_conn.close()


async def list_resources_by_subject_service(
    subject: str, page: int, page_size: int, search: Optional[str]
) -> PaginatedAdminResourcesResponse:
    db_conn = get_mysql_connection(db_name=os.environ.get("MYSQL_DB"))
    if not db_conn:
        raise HTTPException(status_code=500, detail="Database connection failed.")
    try:
        data = get_unified_resources_by_subject(db_conn, subject, page, page_size, search)
        return PaginatedAdminResourcesResponse(**data)
    finally:
        if db_conn.is_connected():
            db_conn.close()

async def export_single_resource_service(resource_type: str, resource_id: int) -> Dict[str, str]:
    """
    获取单个资源的内容，并准备用于导出的数据。
    返回一个包含安全文件名和文件内容的字典。
    """
    db_conn = get_mysql_connection(db_name=os.environ.get("MYSQL_DB"))
    if not db_conn:
        raise HTTPException(status_code=500, detail="Database connection failed.")
    
    try:
        # 复用获取详情的函数
        details = get_teacher_resource_detail(db_conn, resource_type,resource_id)
        if not details:
            raise HTTPException(status_code=404, detail="Resource not found.")
        
        # --- 准备文件名和文件内容 ---
        
        # 清理标题，移除不适合做文件名的字符
        title = details.get('title', f'resource_{resource_id}')
        # 移除非法字符，并将空格替换为下划线
        safe_title = re.sub(r'[\\/*?:"<>|]', "", title).replace(' ', '_')
        filename = f"{resource_type}_{safe_title}.txt"
        
        # 准备文件内容，格式化输出
        file_content = f"标题: {details.get('title', 'N/A')}\n"
        file_content += f"学科: {details.get('subject', 'N/A')}\n"
        file_content += f"资源ID: {details.get('id')}\n"
        file_content += f"========================================\n\n"
        file_content += details.get('full_content', '')
        
        return {"filename": filename, "content": file_content}
        
    finally:
        if db_conn and db_conn.is_connected():
            db_conn.close()

async def get_dashboard_usage_service() -> DashboardUsageResponse:
    db_conn = get_mysql_connection(db_name=os.environ.get("MYSQL_DB"))
    if not db_conn:
        raise HTTPException(status_code=500, detail="Database connection failed.")
    
    try:
        daily_raw = get_activity_stats(db_conn, 'daily')
        weekly_raw = get_activity_stats(db_conn, 'weekly')
        
        daily_stats = UsageStats(teacher=[], student=[])
        for row in daily_raw:
            stat = ActivityStat(activity_type=row['activity_type'], count=row['count'])
            if row['user_role'] == 'teacher':
                daily_stats.teacher.append(stat)
            elif row['user_role'] == 'student':
                daily_stats.student.append(stat)
        
        weekly_stats = UsageStats(teacher=[], student=[])
        for row in weekly_raw:
            stat = ActivityStat(activity_type=row['activity_type'], count=row['count'])
            if row['user_role'] == 'teacher':
                weekly_stats.teacher.append(stat)
            elif row['user_role'] == 'student':
                weekly_stats.student.append(stat)

        return DashboardUsageResponse(daily=daily_stats, weekly=weekly_stats)
        
    finally:
        if db_conn.is_connected():
            db_conn.close()

async def get_teaching_efficiency_service() -> List[TeacherEfficiencyStat]:
    """
    分析教师的备课和修正活动，计算教学效率指数。
    """
    db_conn = get_mysql_connection(db_name=os.environ.get("MYSQL_DB"))
    if not db_conn:
        raise HTTPException(status_code=500, detail="Database connection failed.")
    
    try:
        # 1. 从数据库获取原始统计数据
        raw_stats = get_teacher_content_creation_stats(db_conn)
        
        # 2. 在Python中处理和聚合数据
        # 使用一个字典来按 teacher_id 聚合数据
        teacher_data = {}

        for row in raw_stats:
            teacher_id = row['teacher_id']
            teacher_name = row['teacher_name']
            activity = row['activity_type']
            count = row['count']
            
            # 如果是第一次见到这位老师，为他初始化一个数据结构
            if teacher_id not in teacher_data:
                teacher_data[teacher_id] = {
                    "teacher_id": teacher_id,
                    "teacher_name": teacher_name,
                    "plans_created": 0,
                    "assessments_created": 0,
                    "plans_refined": 0,
                    "assessments_refined": 0
                }
            
            # 根据活动类型累加次数
            if activity == 'GENERATE_TEACHING_PLAN':
                teacher_data[teacher_id]['plans_created'] += count
            elif activity == 'REFINE_TEACHING_PLAN':
                teacher_data[teacher_id]['plans_refined'] += count
            elif activity == 'GENERATE_ASSESSMENT':
                teacher_data[teacher_id]['assessments_created'] += count
            elif activity == 'REFINE_ASSESSMENT':
                teacher_data[teacher_id]['assessments_refined'] += count
        
        # 3. 计算每个老师的效率指数并格式化为最终结果
        final_results = []
        for teacher_id, data in teacher_data.items():
            # 计算教案效率指数
            total_plan_actions = data['plans_created'] + data['plans_refined']
            if total_plan_actions > 0:
                plan_efficiency = (data['plans_created'] / total_plan_actions) * 100
            else:
                plan_efficiency = 0.0 # 或者 100.0，取决于如何定义无操作的效率

            # 计算考核效率指数
            total_assessment_actions = data['assessments_created'] + data['assessments_refined']
            if total_assessment_actions > 0:
                assessment_efficiency = (data['assessments_created'] / total_assessment_actions) * 100
            else:
                assessment_efficiency = 0.0

            # 创建 Pydantic 模型实例
            stat_entry = TeacherEfficiencyStat(
                teacher_id=data['teacher_id'],
                teacher_name=data['teacher_name'],
                plans_created=data['plans_created'],
                assessments_created=data['assessments_created'],
                plans_refined=data['plans_refined'],
                assessments_refined=data['assessments_refined'],
                plan_efficiency_index=round(plan_efficiency, 2),
                assessment_efficiency_index=round(assessment_efficiency, 2)
            )
            final_results.append(stat_entry)
            
        return final_results
        
    finally:
        if db_conn.is_connected():
            db_conn.close()

async def get_low_performing_subjects_service() -> List[SubjectPerformance]:
    db_conn = get_mysql_connection(db_name=os.environ.get("MYSQL_DB"))
    if not db_conn:
        raise HTTPException(status_code=500, detail="Database connection failed.")
    try:
        raw_data = get_low_performing_subjects(db_conn, limit=5)
        return [SubjectPerformance(**item) for item in raw_data]
    finally:
        if db_conn.is_connected():
            db_conn.close()

async def get_student_effectiveness_service() -> StudentEffectivenessResponse:
    db_conn = None
    db_conn = get_mysql_connection(db_name=os.environ.get("MYSQL_DB"))
    if not db_conn:
        raise HTTPException(status_code=500, detail="Database connection failed.")
        
    try:
        # --- 添加的调试日志 ---
        print("\n--- [SERVICE] START: get_student_effectiveness_service ---")
        print(f"[SERVICE] 1. Before get_daily_accuracy_trend: Connection is connected? -> {db_conn.is_connected()}")

        # 1. 获取正确率趋势
        trend_data = get_daily_accuracy_trend(db_conn)

        print("[SERVICE] 1.1. SKIPPED get_daily_accuracy_trend call.")
        # --- 添加的调试日志 ---
        print(f"[SERVICE] 2. After get_daily_accuracy_trend: Connection is connected? -> {db_conn.is_connected()}")

        # 2. 获取知识点数据并分析
        attempts_with_concepts = get_all_practice_attempts_with_concepts(db_conn)
        
        print("[SERVICE] 3. After get_all_practice_attempts_with_concepts: All DB calls finished.")
        # --- 调试日志结束 ---
        
        concept_stats = defaultdict(lambda: {'total': 0, 'score': 0.0, 'incorrect': 0})
        
        for attempt in attempts_with_concepts:
            concepts_str = attempt.get('concepts_covered')
            correctness = attempt.get('correctness_assessment', '')
            
            if not concepts_str: continue

            try:
                concepts = json.loads(concepts_str)
                for concept in concepts:
                    concept_stats[concept]['total'] += 1
                    # '85.7%'
                    if '%' in correctness:
                        score_val = float(correctness.replace('%', '')) / 100.0
                        concept_stats[concept]['score'] += score_val
                        if score_val < 0.5: # 假设低于50%算错误
                            concept_stats[concept]['incorrect'] += 1
                    # 'Correct', 'Partially Correct'
                    elif correctness == 'Correct':
                        concept_stats[concept]['score'] += 1.0
                    elif correctness == 'Partially Correct':
                        concept_stats[concept]['score'] += 0.5
                    elif correctness == 'Incorrect':
                        concept_stats[concept]['incorrect'] += 1
            except (json.JSONDecodeError, TypeError, ValueError):
                continue
        
        # 3. 计算结果
        concept_results = []
        for concept, stats in concept_stats.items():
            if stats['total'] > 0:
                mastery_rate = (stats['score'] / stats['total']) * 100
                concept_results.append(ConceptStat(
                    concept=concept,
                    mastery_rate=round(mastery_rate, 2),
                    total_attempts=stats['total'],
                    incorrect_attempts=stats['incorrect']
                ))
        
        # 按掌握率从低到高排序
        weakest_concepts = sorted(concept_results, key=lambda x: x.mastery_rate)[:10] # 只取最弱的10个

        return StudentEffectivenessResponse(
            accuracy_trend=[DailyAccuracy(**item) for item in trend_data],
            weakest_concepts=weakest_concepts
        )

    finally:
        if db_conn and db_conn.is_connected():
            print("[SERVICE] FINALLY: Closing connection.")
            db_conn.close()
        print("--- [SERVICE] END: get_student_effectiveness_service ---\n")

async def get_teacher_published_assessments_service(teacher_id: int) -> List[PublishedAssessmentInfo]:
    db_conn = None
    try:
        MYSQL_DB_NAME = os.environ.get("MYSQL_DB")
        db_conn = get_mysql_connection(db_name=MYSQL_DB_NAME)
        if not db_conn:
            raise HTTPException(status_code=500, detail="Database connection failed.")
        
        assessments_raw = get_published_assessments_by_teacher(db_conn, teacher_id)
        
        return [PublishedAssessmentInfo(**item) for item in assessments_raw]
        
    finally:
        if db_conn and db_conn.is_connected():
            db_conn.close()

async def analyze_assessment_performance_service(assessment_id: int) -> AssessmentAnalysisOutput:
    db_conn = None
    try:
        MYSQL_DB_NAME = os.environ.get("MYSQL_DB")
        db_conn = get_mysql_connection(db_name=MYSQL_DB_NAME)
        if not db_conn:
            raise HTTPException(status_code=500, detail="Database connection failed.")

        stats_raw = get_assessment_question_stats(db_conn, assessment_id)
        if not stats_raw:
            raise HTTPException(status_code=404, detail="No student answer data found for this assessment.")

        assessment_data = get_assessment_content_by_id(db_conn, assessment_id)
        if not assessment_data:
            raise HTTPException(status_code=404, detail=f"Assessment with ID {assessment_id} not found.")
        
        assessment_title = assessment_data.get('title', 'Untitled Assessment')
        questions_text = assessment_data.get('content', '').split('---参考答案与解析---')[0].strip()

        # 3. 构造给LLM的提示 (Prompt Engineering)
        analysis_prompt = f"""
        你是一位顶级的教育数据分析专家。你的任务是分析一份考核的学情数据，并提供一份简洁、深刻、有洞察力的分析报告。

        **1. 考核基本信息:**
        - 考核标题: "{assessment_title}"

        **2. 考核题目内容:**
        ---
        {questions_text}
        ---

        **3. 各题作答情况统计:**
        以下是每个题目的作答情况统计：
        ```json
        {json.dumps(stats_raw, indent=2, ensure_ascii=False)}
        ```

        **你的任务 (必须严格遵守):**
        请根据以上所有信息，生成一份学情分析报告。你的输出必须是且只能是一个单一的、结构化的JSON对象，格式如下：

        ```json
        {{
          "assessment_title": "{assessment_title}",
          "overall_summary": "（这里是对班级整体表现的概括性总结，比如：整体掌握情况良好，但在...方面存在普遍困难。）",
          "question_analysis": [
            {{
              "question_identifier": "（错误率最高的题号，如 '题目3'）",
              "question_text": "（该题目的完整题干）",
              "correct_rate": "（该题的正确率，计算方式为 Correct / Total * 100）",
              "main_knowledge_point": "（根据题干，提炼出这道题考察的核心知识点或技能）",
              "common_errors": "（推测学生可能的常见错误或思维误区）"
            }},
            // ... (为其他错误率较高的2-3个题目生成类似对象) ...
          ],
          "teaching_suggestions": [
            "（基于以上分析，提出第一条具体的、可操作的教学建议）",
            "（提出第二条教学建议，例如：可以针对...知识点设计专项练习）",
            "（提出第三条教学建议，例如：下次授课时可以多举一些...的例子）"
          ]
        }}
        ```

        **分析要点:**
        - 在 `question_analysis` 中，请重点分析错误率最高或最值得关注的2-4个问题。
        - `teaching_suggestions` 必须具体、有针对性，能够直接帮助老师改进教学。

        现在，请开始生成你的JSON分析报告。
        """

        # 4. 调用LLM并解析 (LLM Call & Parsing)
        parsed_analysis = await parse_query_with_llm(analysis_prompt)
        
        if "error" in parsed_analysis or not parsed_analysis.get("question_analysis"):
            print(f"LLM parsing failed. Raw response: {parsed_analysis}")
            raise HTTPException(status_code=500, detail="Failed to get a valid analysis from the AI model.")

        # 使用Pydantic模型进行验证和返回
        return AssessmentAnalysisOutput(**parsed_analysis)

    except HTTPException as he:
        raise he # 直接重新抛出HTTP异常
    except Exception as e:
        print(f"SERVICE ERROR in analyze_assessment_performance_service: {e}")
        raise HTTPException(status_code=500, detail=f"An internal server error occurred during analysis: {str(e)}")
    finally:
        if db_conn and db_conn.is_connected():
            db_conn.close()


async def analyze_assessment_performance(assessment_id: int) -> AssessmentAnalysisOutput:
    db_conn = None
    try:
        db_conn = get_mysql_connection(db_name=os.environ.get("MYSQL_DB"))
        if not db_conn:
            raise HTTPException(status_code=500, detail="Database connection failed.")

        # --- 1. 数据聚合 (Data Aggregation) ---
        stats_raw = get_assessment_question_stats(db_conn, assessment_id)
        if not stats_raw:
            raise HTTPException(status_code=404, detail="该考核尚无学生作答数据，无法进行分析。")

        # --- 2. 获取上下文 (Get Context) ---
        assessment_data = get_assessment_content_by_id(db_conn, assessment_id)
        if not assessment_data:
            raise HTTPException(status_code=404, detail=f"ID为 {assessment_id} 的考核未找到。")
        
        assessment_title = assessment_data.get('title', '未命名考核')
        # 从 'content' 中分离出纯题目部分
        questions_text = assessment_data.get('content', '').split('---参考答案与解析---')[0].strip()

        # --- 3. 构造 Prompt (Prompt Engineering) ---
        analysis_prompt = f"""
        你是一位顶级的教育数据分析专家和教学顾问。你的任务是深入分析一份在线考核的学情统计数据，并为任课教师生成一份全面、深刻且极具操作性的学情分析报告。

        **【输入信息】**

        1.  **考核基本信息:**
            - 考核标题: "{assessment_title}"

        2.  **考核的全部题目:**
            ---
            {questions_text}
            ---

        3.  **各题作答情况的统计数据 (JSON格式):**
            ---
            {json.dumps(stats_raw, indent=2, ensure_ascii=False)}
            ---

        **【你的核心任务】**
        请基于以上所有信息，生成一份结构化的学情分析报告。你的输出必须是且只能是一个符合下面描述的、格式严谨的JSON对象。

        **【必需的输出JSON结构】**
        ```json
        {{
          "assessment_title": "{assessment_title}",
          "overall_summary": "（这里是对班级整体表现的高度概括性总结，必须简明扼要，点出核心问题。例如：本次考核显示，学生对基础概念掌握较为扎实，但在综合应用和解决复杂问题方面能力不足。）",
          "strength_points": [
            "（根据数据和题目，提炼出学生普遍掌握得最好的1-3个知识点或技能）",
            "（例如：学生对'Python基础语法'的记忆性知识掌握牢固）"
          ],
          "weakness_points": [
            "（根据数据和题目，提炼出学生普遍存在的、最关键的1-3个薄弱知识点或技能）",
            "（例如：学生在'递归思想的理解与应用'上存在普遍困难）"
          ],
          "problematic_questions": [
            {{
              "question_identifier": "（选择错误率最高或最能反映问题的题号，如 '题目3'）",
              "question_text": "（从上面提供的题目中，复制该题的完整题干）",
              "correct_rate": "（根据统计数据，计算并填入该题的正确率【纯数字，不要带百分号%】，例如 55.5）",
              "main_knowledge_point": "（精炼概括这道题考察的核心知识点或能力，例如：'链表的逆序操作'）",
              "common_error_analysis": "（深入分析学生为什么会在这道题上出错，是概念混淆、计算失误，还是审题不清？给出具体推测。）"
            }}
            // ... (为其他1-2个最值得关注的问题，生成同样结构的对象) ...
          ],
          "teaching_suggestions": [
            "（第一条教学建议：必须非常具体、可操作。例如：'建议在下节课用15分钟时间，通过画图和实例，重新讲解递归的执行过程，特别是回溯阶段。'）",
            "（第二条教学建议：例如：'可以设计一个关于链表操作的专项练习，包含头插法、尾插法和逆序，帮助学生巩固。'）",
            "（第三条教学建议：例如：'鼓励学生在编程题中多写注释，解释自己的思路，有助于暴露其思维误区。'）"
          ]
        }}
        ```
        **分析要求:**
        - **深刻洞察**: 不要只做表面描述，要深入分析数据背后的原因。
        - **聚焦重点**: `problematic_questions` 只需选择最关键的2-3个进行分析。
        - ** actionable**: `teaching_suggestions` 必须是老师看完就能直接用的具体方法，而不是空泛的口号。

        现在，请开始生成你的专业JSON分析报告。
        """

        # --- 4. 调用 LLM 并返回 ---
        parsed_analysis = await parse_query_with_llm(analysis_prompt)
        
        if "error" in parsed_analysis or not parsed_analysis.get("problematic_questions"):
            print(f"LLM parsing failed. Raw response: {parsed_analysis}")
            raise HTTPException(status_code=500, detail="AI模型未能生成有效的分析报告。")
        if 'problematic_questions' in parsed_analysis and isinstance(parsed_analysis['problematic_questions'], list):
            for question_data in parsed_analysis['problematic_questions']:
               if 'correct_rate' in question_data:
                    rate_val = question_data['correct_rate']
                    try:
                        # 如果是字符串, 移除 '%' 并转换为 float
                        if isinstance(rate_val, str):
                           cleaned_rate = float(rate_val.strip().replace('%', ''))
                        # 如果已经是数字, 确保是 float
                        else:
                            cleaned_rate = float(rate_val)
                        question_data['correct_rate'] = cleaned_rate
                    except (ValueError, TypeError):
                        # 如果转换失败, 打印警告并设置为默认值 0.0
                        print(f"Warning: Could not parse correct_rate '{rate_val}'. Defaulting to 0.0.")
                        question_data['correct_rate'] = 0.0
        return AssessmentAnalysis(**parsed_analysis)

    except HTTPException as he:
        raise he
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"分析过程中发生内部错误: {str(e)}")
    finally:
        if db_conn and db_conn.is_connected():
            db_conn.close()


async def _generate_semantic_title_for_refinement(original_title: str, user_query: str) -> str:
    """
    使用LLM根据原始标题和用户追问，生成一个描述性的新标题。
    """
    print(f"SERVICE: Generating semantic title. Original: '{original_title}', Query: '{user_query}'")
    
    title_generation_prompt = ChatPromptTemplate.from_template(
        """
        你是一位精通文件命名的专家。你的任务是根据一个【原始标题】和用户的【修改指令】，生成一个简洁、清晰、描述性的【新标题】。

        规则：
        1. 新标题必须保留【原始标题】的核心内容。
        2. 新标题应该简要地概括用户的【修改指令】。
        3. 新标题应该看起来专业，避免使用 "Refined", "ID" 等技术词汇。
        4. 新标题末尾可以加上一个版本标识，如 "(修订版)" 或 "(补充版)"。
        5. 你的回答【只能包含最终的标题字符串】，不要有任何额外的解释或引号。

        ---
        【示例 1】
        原始标题: "Python入门教案"
        修改指令: "再加两道编程练习题"
        你的输出: Python入门教案 - 补充编程练习

        【示例 2】
        原始标题: "一战历史考核"
        修改指令: "把选择题的难度提高一些"
        你的输出: 一战历史考核 (难度提升版)
        ---

        现在，请为以下输入生成新标题：
        原始标题: "{original_title}"
        修改指令: "{user_query}"
        """
    )
    
    try:
        # 使用一个快速、低成本的模型
        llm = ChatZhipuAI(model="glm-4", temperature=0.1)
        chain = title_generation_prompt | llm | StrOutputParser()
        new_title = await chain.ainvoke({"original_title": original_title, "user_query": user_query})
        return new_title.strip().replace('"', '') # 清理可能的引号
    except Exception as e:
        print(f"SERVICE WARNING: Failed to generate semantic title: {e}. Falling back to default.")
        # 如果LLM失败，提供一个仍然比之前好的后备标题
        return f"{original_title} (修订版 - {datetime.now().strftime('%H%M')})"