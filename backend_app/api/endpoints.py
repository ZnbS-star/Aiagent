from datetime import datetime
from fastapi import APIRouter,  HTTPException, Body, Query
from fastapi.responses import StreamingResponse
import urllib
from backend_app.auth_service import unified_register_service, unified_login_service
from backend_app.models import (
    AdminCreateUserInput, AdminResetPasswordInput, AdminResourceDetailView, AdminResourceType, AssessmentAnalysisOutput, DashboardUsageResponse, PaginatedAdminResourcesResponse, PaginatedUsersResponse, PracticeQuestionDetailOutput, PracticeQuestionListOutput, PublishAssessment, PublishedAssessmentInfo, StudentAssessmentSummary, StudentEffectivenessResponse, StudentQuestionInput, StudentQuestionOutput, SubjectPerformance, TeacherAssessmentListOutput, TeacherEfficiencyStat, 
    TeachingPlanNLInput, TeachingPlanOutput, 
    AssessmentInput, AssessmentNLInput, AssessmentOutput, 
    StudentAssessmentInput, StudentAssessmentNLInput,  StudentAssessmentAnswerItem, 
    PracticeQuestionsInput, PracticeQuestionNLInput, PracticeQuestionsOutput, 
    Message, Token, UserCreate, UserLogin,
    RefineStudentQAInput, RefineTeachingPlanInput, RefineAssessmentInput,
    PracticeChatInput, PracticeChatOutput, 
)
from backend_app.services import (
    analyze_assessment_performance_service,
    analyze_assessment_service,
    create_user_by_admin_service,
    delete_user_service,
    export_resources_by_subject_service,
    get_dashboard_usage_service,
    get_low_performing_subjects_service,
    get_student_effectiveness_service,
    get_teacher_published_assessments_service,
    get_teacher_resource_detail_service,
    get_teaching_efficiency_service,
    list_all_subjects_service,
    list_resources_by_subject_service,
    list_users_service,
    process_practice_chat_service, 
    get_assessment_detail_service,
    get_assessment_list_service,
    get_teacher_assessments_service,
    process_student_question_service, 
    generate_initial_teaching_plan_service,
    generate_assessment_service,
    evaluate_student_assessment_answers_service,
    get_student_assessment_performance_service, 
    generate_practice_questions_service,
    publish_assessment_service,
    refine_assessment_service,
    refine_student_question_service,
    refine_teaching_plan_service,
    reset_user_password_service
)
from backend_app.nlp_utils import parse_query_with_llm 
from typing import List, Literal, Optional
import os
from backend_app.database_utils import get_mysql_connection, get_question_identifiers_from_assessment, log_activity, save_teaching_plan,save_assessment

router = APIRouter()

@router.post("/register", summary="注册")
async def unified_register_endpoint(user_data: UserCreate):

    new_user = await unified_register_service(user_data)
    return new_user

@router.post("/login", response_model=Token, summary="登录 ")
async def login_teacher_endpoint(form_data: UserLogin): 

    token = await unified_login_service(form_data)
    return token
    


@router.post(
    "/student-qa", 
    response_model=StudentQuestionOutput, 
    summary="Process a student's question using RAG and LLM",
    description="Receives a student's question, optionally a student ID. "
                "It performs a RAG search on the knowledge base, "
                "then uses an LLM to generate an answer based on the question and retrieved context.",
    responses={
        200: {"description": "Successful response with the LLM's answer and context."},
        400: {"model": Message, "description": "Bad Request (e.g., invalid input)"},
        500: {"model": Message, "description": "Internal Server Error"}
    }
)

async def student_question_answer(input_data: StudentQuestionInput = Body(..., examples={
    "simple_question": {
        "summary": "A basic question",
        "description": "A student asks a question without providing a student ID.",
        "value": {"question": "What is the main function of a CPU?"}
    },
    "question_with_id": {
        "summary": "Question with student ID",
        "description": "A student asks a question and provides their ID for potential history tracking.",
        "value": {"question": "How does photosynthesis work?", "student_id": 101}
    }
})):
    if not input_data.question or not input_data.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    try:
        result = await process_student_question_service(input_data)
        if result.error_message:
            pass 
        return result
    except HTTPException as he: 
        raise he
    except Exception as e:
        print(f"API ERROR: An unexpected error occurred in /student-qa/ endpoint: {e}")
        raise HTTPException(status_code=500, detail=f"An internal server error occurred: {str(e)}")


@router.post(
    "/student-qa/refine",
    response_model=StudentQuestionOutput,
    summary="Refine a student's question based on conversation history",
    description="Receives conversation history and a new query to refine and regenerate an answer.",
    responses={
        200: {"description": "Successful response with the refined LLM's answer."},
        400: {"model": Message, "description": "Bad Request (e.g., empty history or new_query)"},
        500: {"model": Message, "description": "Internal Server Error"}
    }
)
async def refine_student_question(input_data: RefineStudentQAInput = Body(...)):
    if not input_data.history:
        raise HTTPException(status_code=400, detail="History cannot be empty.")
    if not input_data.new_query or not input_data.new_query.strip():
        raise HTTPException(status_code=400, detail="New query cannot be empty.")
    try:
        result = await refine_student_question_service(input_data) 
        return result 
    except HTTPException as he:
        raise he
    except Exception as e:
        print(f"API ERROR: An unexpected error occurred in /student-qa/refine endpoint: {e}")
        raise HTTPException(status_code=500, detail=f"An internal server error occurred: {str(e)}")

@router.post(
    "/teaching-plans",

)
async def create_initial_teaching_plan(input_data: TeachingPlanNLInput):
    if not input_data.query or not input_data.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")


    FULL_PROMPT_FOR_LLM = f"""
    请分析以下用户查询，并从中提取特定信息用于创建教案。

    --- 用户查询开始 ---
    {input_data.query}
    --- 用户查询结束 ---

    你的任务是根据上述查询，提取以下实体：
    1.  "teaching_outline" (字符串, 必填): 用户明确要求的核心教学主题。例如 "TensorFlow.js编程"。
    2.  "subject" (字符串, 可选): 查询中明确或暗示的学科领域。例如 "计算机科学", "生物", "历史"。如果未提及，请基于 `teaching_outline` 生成一个科目。
    3.  "style_tone" (字符串, 可选): 教案的特定风格或语气。如果用户未提及，则此字段应为 null。
    4.  "output_structure" (字符串, 可选): 所需的输出结构。如果用户未提及，则此字段应为 null。
    5.  "title_for_db" (字符串, 可选): 用于保存教案的特定标题。如果用户未指定，请基于 `teaching_outline` 生成一个简洁的标题。

    你的最终响应必须是且只能是一个符合以下描述的 JSON 对象。不要包含任何解释性文字或前导/后置文本，直接输出 JSON。
    """
    parsed_entities_dict = await parse_query_with_llm(FULL_PROMPT_FOR_LLM)

    if "error" in parsed_entities_dict:
        raise HTTPException(status_code=400, detail=f"NLP processing error: {parsed_entities_dict['error']}")
    

    teaching_outline = parsed_entities_dict.get("teaching_outline")
    subject = parsed_entities_dict.get("subject") 
    style_tone = parsed_entities_dict.get("style_tone")
    output_structure = parsed_entities_dict.get("output_structure")
    title_for_db = parsed_entities_dict.get("title_for_db")
    final_teacher_id = input_data.teacher_id

    db_conn = None
    try:
        generated_content, _ = await generate_initial_teaching_plan_service(
            initial_outline=teaching_outline,
            style_tone=style_tone,
            output_structure=output_structure
        )
        if not generated_content:
            raise HTTPException(status_code=500, detail="Failed to generate teaching plan content from LLM service. Check service logs.")
        title_to_save = title_for_db 
        MYSQL_DB_NAME = os.environ.get("MYSQL_DB")
        if not MYSQL_DB_NAME:
            print("API WARNING: MYSQL_DB environment variable not set. Cannot save teaching plan.")
            return TeachingPlanOutput(
                title=title_to_save, 
                generated_plan_content=generated_content,
                error_message="Plan generated but not saved; MYSQL_DB not configured."
            )
        db_conn = get_mysql_connection(db_name=MYSQL_DB_NAME)
        if not db_conn:
            print("API ERROR: Failed to connect to the database for saving teaching plan.")
            return TeachingPlanOutput(
                title=title_to_save,
                generated_plan_content=generated_content,
                error_message="Plan generated but failed to connect to DB for saving."
            )
        plan_id = save_teaching_plan(
            db_conn,
            title_to_save,
            generated_content,
            final_teacher_id,
            subject=subject  
        )
        if plan_id:
            if final_teacher_id: # 只有在知道用户ID时才记录
                log_activity(
                    db_conn,
                    user_id=final_teacher_id,
                    user_role="teacher",
                    activity_type="GENERATE_TEACHING_PLAN",
                    details={"plan_id": plan_id, "title": title_to_save}
                )
            return TeachingPlanOutput(
                teaching_plan_id=plan_id,
                title=title_to_save,
                generated_plan_content=generated_content,
                teacher_id=final_teacher_id 
            )
        else:
            return TeachingPlanOutput(
                title=title_to_save,
                generated_plan_content=generated_content,
                error_message="Plan generated but failed to save to database. Check server logs."
            )
    except HTTPException as he: 
        raise he
    except Exception as e:
        print(f"API ERROR: An unexpected error occurred in /teaching-plans/generate-initial/ endpoint: {e}")
        raise HTTPException(status_code=500, detail=f"An internal server error occurred: {str(e)}")
    finally:
        if db_conn and db_conn.is_connected():
            print("Closing DB connection for /teaching-plans/generate-initial/ endpoint.")
            db_conn.close()



@router.post(
    "/teaching-plans/refine",
    response_model=TeachingPlanOutput,
    summary="Refine a teaching plan based on conversation history",
    description="Receives conversation history and a new query to refine an existing teaching plan.",
    responses={
        200: {"description": "Teaching plan refined successfully."},
        400: {"model": Message, "description": "Bad Request (e.g., empty history or new_query)"},
        500: {"model": Message, "description": "Internal Server Error"}
    }
)
async def refine_teaching_plan(input_data: RefineTeachingPlanInput = Body(...)):
    db_conn = None
    try:
        full_generated_content, _ = await refine_teaching_plan_service(input_data)

        if not full_generated_content:
            raise HTTPException(status_code=500, detail="Failed to generate refined teaching plan content.")
        new_title = f"Refined Plan (based on ID {input_data.base_teaching_plan_id}) - {datetime.now().strftime('%H%M%S')}"

        MYSQL_DB_NAME = os.environ.get("MYSQL_DB")
        db_conn = get_mysql_connection(db_name=MYSQL_DB_NAME)
        if not db_conn:

            return TeachingPlanOutput(
                title=new_title,
                generated_plan_content=full_generated_content,
                teacher_id=input_data.teacher_id,
                error_message="Content generated but failed to connect to DB for saving."
            )


        new_plan_id = save_teaching_plan(
            db_conn,
            new_title,
            full_generated_content,
            input_data.teacher_id
        )


        if new_plan_id:
            log_activity(
                    db_conn,
                    user_id=input_data.teacher_id,
                    user_role="teacher",
                    activity_type="REFINE_TEACHING_PLAN", 
                    details={"plan_id": new_plan_id, "title": new_title}
            )
            return TeachingPlanOutput(
                teaching_plan_id=new_plan_id, 
                title=new_title,
                generated_plan_content=full_generated_content,
                teacher_id=input_data.teacher_id
            )
        else:
            raise HTTPException(status_code=500, detail="Content generated but failed to save the new version to the database.")

    except HTTPException as he:
        raise he
    except Exception as e:
        print(f"API ERROR: An unexpected error occurred in /teaching-plans/refine endpoint: {e}")
        raise HTTPException(status_code=500, detail=f"An internal server error occurred: {str(e)}")
    finally:
        if db_conn and db_conn.is_connected():
            db_conn.close()


@router.post(
    "/assessments/generate",
    
)
async def create_assessment_endpoint(input_data: AssessmentNLInput): 
    if not input_data.query or not input_data.query.strip():
        raise HTTPException(status_code=400, detail="查询内容不能为空。")

    FULL_PROMPT_FOR_ASSESSMENT_NLP = f"""
    请分析以下用户查询，该查询旨在生成一份考核。你的任务是从中提取特定信息。

    --- 用户查询开始 ---
    {input_data.query}
    --- 用户查询结束 ---

    你需要根据上述查询，提取以下实体：
    1.  "teaching_plan_content" (字符串, 必填): 考核应涵盖的核心内容、主题或材料摘要。
    2.  "subject" (字符串, 可选): 查询中明确或暗示的学科领域。例如 "计算机科学", "物理", "文学"。如果未提及，则根据问题生成一个subject。
    3.  "question_preferences" (对象, 可选): 一个指定所需问题类型和数量的字典。
        *   **如果用户明确指定了问题类型 (例如，“选择题”，“填空题”等)，则 `question_preferences` 字典中必须只包含用户明确要求的类型。不要添加任何用户未提及的类型。**
        *   如果用户指定了类型但未指定数量，为每种指定的类型设定一个合理的默认数量 (例如，3-5 道)。
        *   **只有当用户完全没有提及任何问题类型时**，你才可以根据 `teaching_plan_content` 自动推荐并生成合适的类型和数量 (例如: {{"选择题": 3, "简答题": 2}})。
        *   如果用户只要求了数量而未指定类型 (例如，“出5道题”），则你可以根据内容推荐类型，并将总数分配给这些类型。
        *   示例：如果用户说“帮我出几道选择题”，你应该提取出 {{"选择题": 默认数量}} (例如 {{"选择题": 5}})。如果用户说“关于Python基础的选择题和判断题”，你应该提取出 {{"选择题": 默认数量, "判断题": 默认数量}}。
    4.  "title_for_db" (字符串, 可选): 用于保存考核的特定标题。如果用户未指定，请基于 `teaching_plan_content` 生成一个简洁的标题。

    你的最终响应必须是且只能是一个符合以下描述的 JSON 对象。不要包含任何解释性文字或前导/后置文本，直接输出 JSON。
    """
    parsed_entities_dict = await parse_query_with_llm(FULL_PROMPT_FOR_ASSESSMENT_NLP)

    if "error" in parsed_entities_dict:
        raise HTTPException(status_code=400, detail=f"NLP 处理错误: {parsed_entities_dict['error']}")

    teaching_plan_content = parsed_entities_dict.get("teaching_plan_content")
    if not teaching_plan_content:
        raise HTTPException(status_code=400, detail="NLP 未能从查询中提取必需的 'teaching_plan_content'。")

    subject = parsed_entities_dict.get("subject") 
    nlp_title_for_db = parsed_entities_dict.get("title_for_db")
    question_preferences = parsed_entities_dict.get("question_preferences")
    print(question_preferences)

    assessment_service_input = AssessmentInput(
        teaching_plan_content=teaching_plan_content,
        teacher_id=input_data.teacher_id,
        subject=subject, 
        question_preferences=question_preferences if question_preferences else {}, 
        title_for_db=nlp_title_for_db 
    )
    
    db_conn = None
    try:

        questions_part, answers_part = await generate_assessment_service(assessment_service_input)

        if not questions_part:
            raise HTTPException(status_code=500, detail="服务未能生成考核内容。")
       
        title_to_save = nlp_title_for_db if nlp_title_for_db else f"关于“{teaching_plan_content[:20]}...”的考核"
        

        
        MYSQL_DB_NAME = os.environ.get("MYSQL_DB")
        db_conn = get_mysql_connection(db_name=MYSQL_DB_NAME)
        assessment_id = save_assessment(
            db_conn,
            title_to_save,
            questions_part,    
            answers_part,      
            input_data.teacher_id,
            subject=subject    
        )
        if assessment_id:
            log_activity(
                    db_conn,
                    user_id=input_data.teacher_id,
                    user_role="teacher",
                    activity_type="GENERATE_ASSESSMENT",
                    details={"assessment_id": assessment_id, "title": title_to_save}
                )
            return AssessmentOutput(
                assessment_id=assessment_id,
                title=title_to_save,
                generated_assessment_content=questions_part, 
                teacher_id=input_data.teacher_id,
                subject=subject 
            )
        else:
            # 即使保存失败，也只返回问题部分
            return AssessmentOutput(
                title=title_to_save,
                generated_assessment_content=questions_part,
                error_message="考核已生成但保存至数据库失败。",
                subject=subject
            )
    except HTTPException as he:
        raise he
    except Exception as e:
        print(f"API ERROR in /assessments/generate/ endpoint: {e}")
        raise HTTPException(status_code=500, detail=f"内部服务器错误: {str(e)}")
    finally:
        if db_conn and db_conn.is_connected():
            db_conn.close()

@router.post(
    "/assessments/refine",
    response_model=AssessmentOutput,
    summary="Refine an assessment based on conversation history",
    description="Receives conversation history and a new query to refine an existing assessment.",
    responses={
        200: {"description": "Assessment refined successfully."},
        400: {"model": Message, "description": "Bad Request (e.g., empty history or new_query)"},
        500: {"model": Message, "description": "Internal Server Error"}
    }
)
async def refine_assessment(input_data: RefineAssessmentInput = Body(...)):
    if not input_data.history or not input_data.new_query:
        raise HTTPException(status_code=400, detail="History and new_query are required.")

    db_conn = None
    try:
        # 1. 调用服务，并解构返回的元组
        full_generated_content,_,subject= await refine_assessment_service(input_data)
        answer_separator = "参考答案与解析"
        parts = full_generated_content.split(answer_separator, 1)
        questions_part = parts[0].strip()
        answers_part = parts[1].strip() if len(parts) > 1 else "（无答案信息）"
        # 2. 检查服务是否成功生成内容
        if not full_generated_content:
            raise HTTPException(status_code=500, detail="Failed to generate refined assessment content.")

        # 3. 保存新副本到数据库
        new_title = f"Refined Assessment (based on ID {input_data.base_assessment_id}) - {datetime.now().strftime('%H%M%S')}"
        
        MYSQL_DB_NAME = os.environ.get("MYSQL_DB")
        db_conn = get_mysql_connection(db_name=MYSQL_DB_NAME)
        if not db_conn:
            # 返回内容，但提示保存失败
            return AssessmentOutput(
                title=new_title,
                generated_assessment_content=full_generated_content,
                teacher_id=input_data.teacher_id,
                error_message="Content generated but failed to connect to DB for saving."
            )

        new_assessment_id = save_assessment(
            db_conn,
            new_title,
            questions_part,      
            answers_part,        
            input_data.teacher_id,
            subject=subject 
            
        )

        if new_assessment_id:
            log_activity(
                    db_conn,
                    user_id=input_data.teacher_id,
                    user_role="teacher",
                    activity_type="REFINE_ASSESSMENT",
                    details={"assessment_id": new_assessment_id, "title": new_title}
                )
            return AssessmentOutput(
                assessment_id=new_assessment_id,
                title=new_title,
                generated_assessment_content=questions_part,
                teacher_id=input_data.teacher_id
            )
        else:
            raise HTTPException(status_code=500, detail="Content generated but failed to save the new version.")

    except HTTPException as he:
        raise he
    except Exception as e:
        print(f"API ERROR: An unexpected error occurred in /assessments/refine endpoint: {e}")
        raise HTTPException(status_code=500, detail=f"An internal server error occurred: {str(e)}")
    finally:
        if db_conn and db_conn.is_connected():
            db_conn.close()


@router.post(
    "/assessments/evaluate-answers",
    response_model=Message, 
    summary="Evaluate Student's Assessment Answers from Natural Language Query",
    description="Accepts a natural language query detailing a student's answers to an assessment. "
                "Uses an LLM to extract the answers, then evaluates them using the service layer, "
                "and saves the evaluation. Key details like `assessment_id`, `student_id` (optional), "
                "and `student_name` (optional, but one of student_id or student_name must be resolvable) "
                "are provided directly in the input alongside the query.",
    responses={
        200: {"model": Message, "description": "Student answers processed and saved successfully."}, # Changed description and model if it was different
        400: {"model": Message, "description": "Bad Request (e.g., invalid query, missing essential info after NLP, NLP processing error, or missing required direct fields like assessment_id)"},
        422: {"model": Message, "description": "Validation Error (e.g., assessment_id is not an int)"},
        500: {"model": Message, "description": "Internal Server Error / LLM or DB failure"}
    }
)
async def evaluate_student_answers(input_data: StudentAssessmentNLInput ):
    if not input_data.query or not input_data.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
    if not input_data.assessment_id: 
        raise HTTPException(status_code=422, detail="assessment_id is required.")
    FULL_PROMPT_FOR_EVAL_NLP = f"""
    你的任务是分析并提取学生针对一份考核所提交的答案。请仔细阅读以下文本。

    --- 用户提供的文本开始 ---
    {input_data.query}
    --- 用户提供的文本结束 ---

    请根据上述文本，严格按照以下规则提取实体并输出一个JSON对象：

    1.  **答案提取规则**:
        - 你的目标是提取一个名为 "answers" 的对象数组。
        - 数组中的每个对象都必须包含 "question_identifier" (字符串, 必填) 和 "student_answer_text" (字符串, 必填)。
        - **示例**: `{{ "question_identifier": "题目1", "student_answer_text": "A" }}`

    2.  **特殊情况处理规则 (最重要!)**:
        - **如果文本明确表示学生不会回答或放弃作答** (例如 "我不会"、"不知道"、"交白卷"、"放弃")，你的JSON输出中，"answers" 字段必须是一个 **空数组 `[]`**。
        - **如果文本既不包含任何可识别的答案，也不包含明确的放弃信息** (例如 "这是什么题目？"、"老师好")，你的JSON输出中，"answers" 字段也必须是一个 **空数组 `[]`**。
        - **在以上特殊情况下，你可以选择性地在JSON中增加一个 "status" 字段**，值为 "NO_ANSWERS_PROVIDED"。

    3.  **最终输出格式**:
        - 你的最终响应必须是且只能是一个JSON对象。
        - 不要包含任何解释性文字或前导/后置文本。

    **正确输出示例 1 (成功提取):**
    ```json
    {{
      "answers": [
        {{"question_identifier": "题目1", "student_answer_text": "A"}},
        {{"question_identifier": "简答题1", "student_answer_text": "地心引力。"}}
      ]
    }}
    ```

    **正确输出示例 2 (用户表示不会):**
    ```json
    {{
      "answers": [],
      "status": "NO_ANSWERS_PROVIDED"
    }}
    ```
    
    现在，请开始分析并生成JSON：
    """
    parsed_entities_dict = await parse_query_with_llm(FULL_PROMPT_FOR_EVAL_NLP)

    if "error" in parsed_entities_dict:
        raise HTTPException(status_code=400, detail=f"NLP processing error: {parsed_entities_dict['error']} - Details: {parsed_entities_dict.get('details', 'N/A')}")

    answers_raw = parsed_entities_dict.get("answers")
    parsed_answers_for_service: List[StudentAssessmentAnswerItem] = []
    if not isinstance(answers_raw, list) or not answers_raw:
        # 如果LLM没有返回有效的答案列表 (包括返回空列表)
        print("API INFO: No answers extracted by LLM. Assuming student submitted a blank paper.")
        
        db_conn_temp = None
        try:
            # 需要连接数据库来获取这份试卷的所有题目
            MYSQL_DB_NAME = os.environ.get("MYSQL_DB")
            db_conn_temp = get_mysql_connection(db_name=MYSQL_DB_NAME)
            if not db_conn_temp:
                raise HTTPException(status_code=500, detail="Database connection failed, cannot process blank submission.")
            
            # 获取该试卷的所有问题标识符
            all_question_ids = get_question_identifiers_from_assessment(db_conn_temp, input_data.assessment_id)
            
            if not all_question_ids:
                # 如果试卷本身没有可识别的题目，那也无法处理
                print(f"API WARNING: Could not find any question identifiers for assessment_id {input_data.assessment_id}. Cannot create blank entries.")
                return Message(message="Submission acknowledged, but no questions found in the assessment to mark as unanswered.")

            # 为每个题目创建一个“空答案”的条目
            for q_id in all_question_ids:

                parsed_answers_for_service.append(
                    StudentAssessmentAnswerItem(question_identifier=q_id, student_answer_text="") # 答案设为空字符串
                )

        finally:
            if db_conn_temp and db_conn_temp.is_connected():
                db_conn_temp.close()
    else:
        # 如果LLM成功提取了答案，就走原来的逻辑
        for i, ans_item in enumerate(answers_raw):
            q_id = ans_item.get("question_identifier")
            s_ans = ans_item.get("student_answer_text")
            if not q_id or not isinstance(q_id, str) or not q_id.strip():
                raise HTTPException(...)
            if s_ans is None or not isinstance(s_ans, str): 
                raise HTTPException(...)
            parsed_answers_for_service.append(StudentAssessmentAnswerItem(question_identifier=q_id.strip(), student_answer_text=s_ans))
    
    
    if not parsed_answers_for_service:
        return Message(message="No answers to evaluate.")
    student_assessment_service_input = StudentAssessmentInput(
        student_id=input_data.student_id,
        assessment_id=input_data.assessment_id,
        answers=parsed_answers_for_service
    )

    try:
        evaluation_results = await evaluate_student_assessment_answers_service(student_assessment_service_input)

        if evaluation_results and any(result.error_message for result in evaluation_results if result.question_identifier == "Overall Error"):
            overall_error = next((res.error_message for res in evaluation_results if res.question_identifier == "Overall Error"), "Service error")
            raise HTTPException(status_code=500, detail=overall_error)
        
  
        if evaluation_results and any(result.error_message and not result.answer_id for result in evaluation_results):
             print("API WARNING: Some answers may not have been saved successfully during the evaluation process. Check service logs for details.")

        return Message(message="Student assessment answers processed and saved successfully.")
            
    except HTTPException as he:
        raise he
    except ValueError as ve: 
        raise HTTPException(status_code=404, detail=str(ve))
    except Exception as e:
        print(f"API ERROR: An unexpected error occurred in /student-assessments/evaluate-answers/ endpoint: {e}")
        
        raise HTTPException(status_code=500, detail=f"An internal server error occurred: {str(e)}")



@router.post(
    "/practice-questions/generate",
    response_model=PracticeQuestionsOutput,
    summary="Generate Practice Questions from Natural Language Query",
    description="Accepts a natural language query to generate practice questions. Uses an LLM to extract "
                "the practice topic and any question preferences. Student ID and name can also be provided "
                "for personalized question generation.",
    responses={
        200: {"description": "Practice questions generated successfully."},
        400: {"model": Message, "description": "Bad Request (e.g., invalid query, missing essential info after NLP, NLP processing error)"},
        500: {"model": Message, "description": "Internal Server Error / LLM or RAG failure during question generation"}
    }
)
async def generate_practice_questions_endpoint(input_data: PracticeQuestionNLInput = Body):
    if not input_data.query or not input_data.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    FULL_PROMPT_FOR_PRACTICE_NLP = f"""
    请分析以下用户的请求，该请求旨在生成一些练习题。你的任务是从中提取特定信息。

    --- 用户请求开始 ---
    {input_data.query}
    --- 用户请求结束 ---

    你需要根据上述请求，提取以下实体：
    1.  "practice_topic" (字符串, 必填): 用户想要练习的具体主题或知识点。例如："Python列表推导式"、"光合作用的基本原理"。
    2.  "question_preferences" (对象, 可选): 一个指定所需问题类型和数量的字典。例如：{{"选择题": 2, "填空题": 3}}。如果用户没有明确指定，则随便返回一个数值。

    你的最终响应必须是且只能是一个符合以下描述的 JSON 对象。不要包含任何解释性文字或前导/后置文本，直接输出 JSON。
"""
    parsed_entities_dict = await parse_query_with_llm(FULL_PROMPT_FOR_PRACTICE_NLP)

    if "error" in parsed_entities_dict:
        raise HTTPException(status_code=400, detail=f"NLP processing error: {parsed_entities_dict['error']} - Details: {parsed_entities_dict.get('details', 'N/A')}")

    practice_topic = parsed_entities_dict.get("practice_topic")
    if not practice_topic:
        raise HTTPException(status_code=400, detail="NLP could not extract required entity 'practice_topic' from query.")

    nlp_question_preferences = parsed_entities_dict.get("question_preferences")

    if nlp_question_preferences and isinstance(nlp_question_preferences, dict) and nlp_question_preferences:
       
        final_preferences = nlp_question_preferences
        print(f"API INFO: Using user-specified question preferences: {final_preferences}")
    else:
        final_preferences = {"选择题": 2, "判断题": 1} 
        print(f"API INFO: User did not specify preferences, using system default: {final_preferences}")
    
    try:

        valid_preferences = {}
        for key, value in final_preferences.items():
            if value is not None:
                valid_preferences[key] = int(value)
        
        final_preferences = valid_preferences
        
        
        if not final_preferences:
            print("API WARNING: Preferences became empty after validation, reapplying default.")
            final_preferences = {"选择题": 2, "判断题": 1}

    except (ValueError, TypeError):
         raise HTTPException(status_code=400, detail="NLP返回的题目数量不是有效的整数。")

      
    practice_questions_service_input = PracticeQuestionsInput(
        practice_topic=practice_topic,
        question_preferences=final_preferences, 
        student_id=input_data.student_id
    )

    try:
        result = await generate_practice_questions_service(practice_questions_service_input)
        
        return result
    except HTTPException as he: 
        raise he
    except Exception as e:
       
        print(f"API ERROR: An unexpected error occurred in /practice-questions/generate/ endpoint: {e}")
        raise HTTPException(status_code=500, detail=f"An internal server error occurred: {str(e)}")


@router.post(
    "/practice-assistant/chat",
    response_model=PracticeChatOutput,
    summary="[Student] Interact with the Practice Assistant",
    description="Handles the ongoing conversation for practice questions. "
                "Send the entire chat history and the user's new message. "
                "The API will detect the user's intent (refine questions, submit answer, etc.) "
                "and respond accordingly.",
    responses={
        200: {"description": "Successful interaction."},
        400: {"model": Message, "description": "Bad Request (e.g., empty history or query)"},
        500: {"model": Message, "description": "Internal Server Error"}
    }
)
async def practice_assistant_chat(input_data: PracticeChatInput):
    if not input_data.history or not input_data.new_query:
        raise HTTPException(status_code=400, detail="History and new_query are required for a chat interaction.")
    
    try:
        result = await process_practice_chat_service(input_data)
        return result
    except Exception as e:
        print(f"API ERROR in /practice-assistant/chat endpoint: {e}")
        raise HTTPException(status_code=500, detail=f"An internal server error occurred: {str(e)}")


@router.get(
    "/assessments/{assessment_id}/student-performance/",
    response_model=List[StudentAssessmentSummary],
    summary="Get Aggregated Student Performance for a Specific Assessment",
    description="Retrieves an aggregated list of performance summaries for all students who took a specific assessment. "
                "Each summary includes counts of correct/incorrect answers and a calculated accuracy rate.",
    responses={
        200: {"description": "Successfully retrieved student performance summary."},
        404: {"model": Message, "description": "Assessment not found."},
        500: {"model": Message, "description": "Internal Server Error."}
    }
)
async def get_assessment_performance_for_teacher(assessment_id: int):
    try:
        performance_summary = await get_student_assessment_performance_service(assessment_id=assessment_id)
        return performance_summary
    except HTTPException as he:
        raise he 
    except Exception as e:
        print(f"API ERROR: An unexpected error occurred in /assessments/{assessment_id}/student-performance/ endpoint: {e}")
        raise HTTPException(status_code=500, detail=f"An internal server error occurred: {str(e)}")
    
@router.get(
    "/assessments/list",
    response_model=PracticeQuestionListOutput,
    summary="Get List of Available Practice Questions",
    description="Retrieves a list of all available practice question sets, including their ID and a generated title, for a student to choose from.",
    responses={
        200: {"description": "Successfully retrieved the list of practice questions."},
        500: {"model": Message, "description": "Internal Server Error."}
    }
)
async def get_practice_list_endpoint():
    try:
        return await get_assessment_list_service()
    except HTTPException as he:
        raise he
    except Exception as e:
        print(f"API ERROR: An unexpected error occurred in /practice-questions/list endpoint: {e}")
        raise HTTPException(status_code=500, detail="An internal server error occurred.")


@router.get(
    "/assessments/{assessment_id}/details",  # <-- 建议使用更具描述性的路由
    response_model=PracticeQuestionDetailOutput,
    summary="Get Details of a Specific Assessment",
    description="Retrieves the full content of a specific assessment by its ID.",
    responses={
        200: {"description": "Successfully retrieved assessment details."},
        404: {"model": Message, "description": "Assessment not found."},
        500: {"model": Message, "description": "Internal Server Error."}
    }
)
async def get_assessment_detail_endpoint(assessment_id: int): # <-- 参数名也更清晰
    try:
        # 调用修正后的服务函数
        return await get_assessment_detail_service(assessment_id)
    except HTTPException as he:
        raise he
    except Exception as e:
        print(f"API ERROR: An unexpected error occurred in /assessments/{assessment_id}/details endpoint: {e}")
        raise HTTPException(status_code=500, detail="An internal server error occurred.")
    
@router.get(
    "/teacher/assessments/{teacher_id}",
    response_model=TeacherAssessmentListOutput,
    summary="[Teacher] Get all assessments created by the teacher",
)

async def get_teacher_assessments_endpoint(teacher_id: int): 

    return await get_teacher_assessments_service(teacher_id)


@router.post(
    "/teacher/assessments/publish",
    response_model=Message,
    summary="[Teacher] Publish an assessment",

)
async def publish_assessment_endpoint(inputdata:PublishAssessment =Body): 
    return await publish_assessment_service(inputdata.assessment_id,inputdata.teacher_id)

@router.get(
    "/admin/users", 
    response_model=PaginatedUsersResponse, 
    summary="[Admin] Get a paginated list of users"
)
async def admin_get_users(
    role: Literal['student', 'teacher'],
    page: int = 1,
    page_size: int = Query(10, ge=1, le=100),
    search: Optional[str] = None
):

    return await list_users_service(role, page, page_size, search)

@router.post(
    "/admin/users",
    status_code=201,
    summary="[Admin] Create a new user"
)
async def admin_create_user_endpoint(user_data: AdminCreateUserInput):

    return await create_user_by_admin_service(user_data)

@router.put(
    "/admin/users/{role}/{user_id}/reset-password",
    summary="[Admin] Reset a user's password"
)
async def admin_reset_password_endpoint(
    role: Literal['student', 'teacher'], 
    user_id: int, 
    password_data: AdminResetPasswordInput
):

    return await reset_user_password_service(user_id, role, password_data)

@router.delete(
    "/admin/users/{role}/{user_id}",
    status_code=204,
    summary="[Admin] Delete a user"
)
async def admin_delete_user_endpoint(
    role: Literal['student', 'teacher'], 
    user_id: int
):

    return await delete_user_service(user_id, role)
    

@router.get(
    "/admin/subjects", 
    response_model=List[str],
    summary="[Admin] Get a list of all subjects"
)
async def admin_get_all_subjects():
    return await list_all_subjects_service()

@router.get(
    "/admin/resources/by-subject/{subject}", 
    response_model=PaginatedAdminResourcesResponse,
    summary="[Admin] Get a paginated list of resources for a specific subject"
)
async def admin_get_resources_by_subject(
    subject: str,
    page: int = 1,
    page_size: int = Query(10, ge=1, le=100),
    search: Optional[str] = None
):

    return await list_resources_by_subject_service(subject, page, page_size, search)

@router.get(
    "/admin/resources/{resource_type}/{resource_id}",
    response_model=AdminResourceDetailView,
    summary="[Admin] Get details of a specific teacher resource"
)
async def admin_get_teacher_resource_detail(resource_type: AdminResourceType, resource_id: int):
    return await get_teacher_resource_detail_service(resource_type, resource_id)

@router.get(
    "/admin/resources/by-subject/{subject}/export",
    summary="[Admin] Export resources for a subject to CSV"
)
async def admin_export_resources_by_subject_endpoint(
    subject: str, 
    search: Optional[str] = None
):
    csv_data = await export_resources_by_subject_service(subject, search)
    response = StreamingResponse(iter([csv_data]), media_type="text/csv")
    original_filename = f"resources_{subject}_{datetime.now().strftime('%Y%m%d')}.csv"
    

    encoded_filename = urllib.parse.quote(original_filename)
    
    response.headers["Content-Disposition"] = f"attachment; filename*=UTF-8''{encoded_filename}"
    return response

@router.get(
    "/admin/dashboard/usage-stats",
    response_model=DashboardUsageResponse,
    summary="[Admin Dashboard] Get daily and weekly usage statistics"
)
async def admin_get_usage_stats():
    return await get_dashboard_usage_service()

@router.get(
    "/admin/dashboard/teaching-efficiency",
    response_model=List[TeacherEfficiencyStat],
    summary="[Admin Dashboard] Get teaching efficiency statistics"
)
async def admin_get_teaching_efficiency():

    return await get_teaching_efficiency_service()

@router.get(
    "/admin/dashboard/low-performing-subjects",
    response_model=List[SubjectPerformance],
    summary="[Admin Dashboard] Get subjects with the lowest average scores"
)
async def admin_get_low_performing_subjects():

    return await get_low_performing_subjects_service()

@router.get(
    "/admin/dashboard/student-effectiveness",
    response_model=StudentEffectivenessResponse,
    summary="[Admin Dashboard] Get student learning effectiveness metrics"
)
async def admin_get_student_effectiveness():

    return await get_student_effectiveness_service()

@router.get(
    "/teacher/{teacher_id}/published-assessments",
    response_model=List[PublishedAssessmentInfo],
    summary="[Teacher] Get all published assessments for learning analysis",
    tags=["Teacher Toolkit Features"]
)
async def get_teacher_published_assessments_endpoint(teacher_id: int):

    return await get_teacher_published_assessments_service(teacher_id)

@router.get(
    "/assessments/{assessment_id}/analysis",
    response_model=AssessmentAnalysisOutput,
    summary="[Teacher] Get a detailed learning analysis for an assessment",
    tags=["Teacher Toolkit Features"]
)
async def get_assessment_analysis_endpoint(assessment_id: int):

    return await analyze_assessment_performance_service(assessment_id)
