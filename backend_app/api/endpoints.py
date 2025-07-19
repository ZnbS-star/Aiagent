from datetime import datetime

from urllib.parse import quote
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
    analyze_assessment_performance,
    analyze_assessment_performance_service,

    create_user_by_admin_service,
    delete_user_service,
    export_single_resource_service,
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
)

async def student_question_answer(input_data: StudentQuestionInput ):
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
    "/student-qa/refine"
)
async def refine_student_question(input_data: RefineStudentQAInput ):
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
    "/teaching-plans/refine"
)
async def refine_teaching_plan(input_data: RefineTeachingPlanInput ):
    db_conn = None
    try:
        # 接收服务层返回的新标题和内容
        new_title, full_generated_content, _ = await refine_teaching_plan_service(input_data)

        if not full_generated_content or not new_title:
            raise HTTPException(status_code=500, detail="Failed to generate refined teaching plan content or title.")

        # 使用服务层生成的新标题进行保存
        title_to_save = new_title

        MYSQL_DB_NAME = os.environ.get("MYSQL_DB")
        db_conn = get_mysql_connection(db_name=MYSQL_DB_NAME)
        if not db_conn:
            return TeachingPlanOutput(
                title=title_to_save,
                generated_plan_content=full_generated_content,
                teacher_id=input_data.teacher_id,
                error_message="Content generated but failed to connect to DB for saving."
            )

        new_plan_id = save_teaching_plan(
            db_conn,
            title_to_save,
            full_generated_content,
            input_data.teacher_id
            # 注意：这里可能还需要传递 subject，如果您的 refine_teaching_plan_service 也返回了它
        )

        if new_plan_id:
            log_activity(
                db_conn,
                user_id=input_data.teacher_id,
                user_role="teacher",
                activity_type="REFINE_TEACHING_PLAN",
                details={"plan_id": new_plan_id, "title": title_to_save}
            )
            return TeachingPlanOutput(
                teaching_plan_id=new_plan_id,
                title=title_to_save,
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
)
async def refine_assessment(input_data: RefineAssessmentInput ):
    if not input_data.history or not input_data.new_query:
        raise HTTPException(status_code=400, detail="History and new_query are required.")
    db_conn = None
    try:
        # 接收包含新标题的元组
        new_title, full_generated_content, _, subject = await refine_assessment_service(input_data)

        if not full_generated_content or not new_title:
            raise HTTPException(status_code=500, detail="Failed to generate refined assessment content or title.")

        answer_separator = "参考答案与解析"
        parts = full_generated_content.split(answer_separator, 1)
        questions_part = parts[0].strip()
        answers_part = parts[1].strip() if len(parts) > 1 else "（无答案信息）"

        # 使用服务层生成的新标题
        title_to_save = new_title

        MYSQL_DB_NAME = os.environ.get("MYSQL_DB")
        db_conn = get_mysql_connection(db_name=MYSQL_DB_NAME)
        if not db_conn:
            # 返回内容，但提示保存失败
            return AssessmentOutput(
                title=title_to_save,
                generated_assessment_content=questions_part, # 返回问题部分
                teacher_id=input_data.teacher_id,
                error_message="Content generated but failed to connect to DB for saving."
            )

        new_assessment_id = save_assessment(
            db_conn,
            title_to_save,
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
                details={"assessment_id": new_assessment_id, "title": title_to_save}
            )
            return AssessmentOutput(
                assessment_id=new_assessment_id,
                title=title_to_save,
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
    "/practice-questions/generate"
)
async def generate_practice_questions_endpoint(input_data: PracticeQuestionNLInput = Body):
    if not input_data.query or not input_data.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    FULL_PROMPT_FOR_PRACTICE_NLP = f"""
    你是一个执行严格指令的文本分析器。请分析以下用户的练习题生成请求。

    --- 用户请求开始 ---
    {input_data.query}
    --- 用户请求结束 ---

    你的任务是提取以下两个实体，并以JSON格式输出：

    1.  **"practice_topic" (字符串, 必填)**: 
        用户想要练习的【首要、核心】主题。即使请求中包含子主题（如“关于万历皇帝”），你也应该提取更宏观的主题（如“明朝历史”或“万历十五年”）。
        
    2.  **"question_preferences" (JSON对象, 可选)**: 
        一个【键值对】对象，用于指定题型和数量。
        - **【规则1 - 结构必须简单】**: Key 必须是题型 (字符串)，Value 必须是该题型的数量 (【整数】)。绝对不能是嵌套的JSON对象。
        - **【规则2 - 忽略子主题】**: 在提取数量时，【忽略】掉所有关于具体人物、章节等子主题的限定。你只关心题型和总数。
        - **【规则3 - 合理默认】**: 如果用户没有明确指定题型和数量，可以返回一个合理的默认值，例如 `{{"选择题": 3, "判断题": 2}}`。

    ---
    **【重要示例】**

    *   **输入**: "我正在看黄仁宇的《万历十五年》，想做几道题。请出2道关于“万历皇帝”的选择题和2道关于“申时行”的简答题。"
    *   **你的正确输出**:
        ```json
        {{
          "practice_topic": "《万历十五年》相关历史",
          "question_preferences": {{
            "选择题": 2,
            "简答题": 2
          }}
        }}
        ```

    *   **输入**: "请给我出5道关于TensorFlow.js的选择题"
    *   **你的正确输出**:
        ```json
        {{
          "practice_topic": "TensorFlow.js",
          "question_preferences": {{
            "选择题": 5
          }}
        }}
        ```
    ---

    现在，请严格按照以上规则和示例，分析用户请求并生成JSON输出。你的响应必须是且只能是一个JSON对象。
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
    "/assessments/list",
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
    "/assessments/{assessment_id}/details", 

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

)
async def admin_get_users(
    role: Literal['student', 'teacher'],
    page: int = 1,
    page_size: int = Query(10, ge=1, le=100),
    search: Optional[str] = None
):

    return await list_users_service(role, page, 10, search)

@router.post(
    "/admin/users",

)
async def admin_create_user_endpoint(user_data: AdminCreateUserInput):

    return await create_user_by_admin_service(user_data)

@router.put(
    "/admin/users/{role}/{user_id}/reset-password",
)
async def admin_reset_password_endpoint(
    role: Literal['student', 'teacher'], 
    user_id: int, 
    password: str
):

    return await reset_user_password_service(user_id, role, password)

@router.delete(
    "/admin/users/{role}/{user_id}",

)
async def admin_delete_user_endpoint(
    role: Literal['student', 'teacher'], 
    user_id: int
):

    return await delete_user_service(user_id, role)
    

@router.get(
    "/admin/subjects", 

)
async def admin_get_all_subjects():
    return await list_all_subjects_service()

@router.get(
    "/admin/resources/by-subject/{subject}", 

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

)
async def admin_get_teacher_resource_detail(resource_type: AdminResourceType, resource_id: int):
    return await get_teacher_resource_detail_service(resource_type, resource_id)

@router.get(
    "/admin/resources/{resource_type}/{resource_id}/export",
    summary="[Admin] Export a SINGLE resource to a text file"
)
async def admin_export_single_resource_endpoint(
    resource_type: AdminResourceType, 
    resource_id: int
):
    export_data = await export_single_resource_service(resource_type, resource_id)
    file_content = export_data["content"]
    filename = export_data["filename"]
    response = StreamingResponse(iter([file_content]), media_type="text/plain; charset=utf-8")
    encoded_filename = quote(filename)
    response.headers["Content-Disposition"] = f"attachment; filename*=UTF-8''{encoded_filename}"
    return response

@router.get(
    "/admin/dashboard/usage-stats",
)
async def admin_get_usage_stats():
    return await get_dashboard_usage_service()

@router.get(
    "/admin/dashboard/teaching-efficiency",
)
async def admin_get_teaching_efficiency():

    return await get_teaching_efficiency_service()

@router.get(
    "/admin/dashboard/low-performing-subjects",
)
async def admin_get_low_performing_subjects():

    return await get_low_performing_subjects_service()

@router.get(
    "/admin/dashboard/student-effectiveness",
)
async def admin_get_student_effectiveness():

    return await get_student_effectiveness_service()

@router.get(
    "/teacher/{teacher_id}/published-assessments",
)
async def get_teacher_published_assessments_endpoint(teacher_id: int):

    return await get_teacher_published_assessments_service(teacher_id)

@router.get(
    "/assessments/{assessment_id}/analysis",
)
async def get_assessment_analysis_endpoint(assessment_id: int):

    return await analyze_assessment_performance(assessment_id)
