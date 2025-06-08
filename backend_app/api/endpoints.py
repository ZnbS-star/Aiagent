from fastapi import APIRouter,  HTTPException, Body
from backend_app.auth_service import unified_register_service, unified_login_service
from backend_app.models import (
    StudentQuestionInput, StudentQuestionOutput, 
    TeachingPlanNLInput, TeachingPlanOutput, 
    AssessmentInput, AssessmentNLInput, AssessmentOutput, 
    StudentAssessmentInput, StudentAssessmentNLInput,  StudentAssessmentAnswerItem, 
    PracticeQuestionsInput, PracticeQuestionNLInput, PracticeQuestionsOutput, 
    PracticeFeedbackInput,  PracticeFeedbackOutput, 
    Message, StudentPerformanceDetail, Token, UserCreate, UserLogin 
)
from backend_app.services import (
    process_student_question_service, 
    generate_initial_teaching_plan_service,
    generate_assessment_service,
    evaluate_student_assessment_answers_service,
    get_student_assessment_performance_service, 
    generate_practice_questions_service,
    get_practice_feedback_service
)
from backend_app.nlp_utils import parse_query_with_llm 
from typing import List
import os
from backend_app.database_utils import get_mysql_connection, save_teaching_plan,save_assessment

router = APIRouter()

@router.post("register", summary="注册")
async def unified_register_endpoint(user_data: UserCreate):

    new_user = await unified_register_service(user_data)
    return new_user

@router.post("/login", response_model=Token, summary="登录 ")
async def login_teacher_endpoint(form_data: UserLogin): 

    token = await unified_login_service(form_data)
    return token



@router.post(
    "/student-qa/", 
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
    "/teaching-plans/generate-initial/",
    response_model=TeachingPlanOutput,
    summary="Generate and Save Initial Teaching Plan from Natural Language Query",
    description="Accepts a natural language query to generate a first draft of a teaching plan using an LLM "
                "to extract parameters, then RAG and another LLM call for content generation. "
                "The generated plan is then saved to the database.",
    responses={
        200: {"description": "Teaching plan generated and saved successfully."},
        400: {"model": Message, "description": "Bad Request (e.g., invalid query, missing essential info after NLP, NLP processing error)"},
        500: {"model": Message, "description": "Internal Server Error / LLM or RAG failure during plan generation"}
    }
)
async def create_initial_teaching_plan(input_data: TeachingPlanNLInput ):
    if not input_data.query or not input_data.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")


    FULL_PROMPT_FOR_LLM = f"""
请分析以下用户查询，并从中提取特定信息。

--- 用户查询开始 ---
{input_data.query}
    --- 用户查询结束 ---

    你的任务是根据上述查询，提取以下实体：
    1.  "teaching_outline"（字符串，必填）：用户明确要求的核心教学主题。例如 "TensorFlow.js编程"。
    2.  "style_tone"（字符串，可选）：教案的特定风格或语气。如果用户未提及，则此字段应为 null。
    3.  "output_structure"（字符串，可选）：所需的输出结构。如果用户未提及，则此字段应为 null。
    4.  "title_for_db"（字符串，可选）：用于保存教案的特定标题。如果用户未指定，请基于 `teaching_outline` 生成一个简洁的标题。

    你的最终响应必须是且只能是一个符合以下描述的 JSON 对象。不要包含任何解释性文字或前导/后置文本，直接输出 JSON。
"""
    parsed_entities_dict = await parse_query_with_llm(FULL_PROMPT_FOR_LLM)

    if "error" in parsed_entities_dict:
        raise HTTPException(status_code=400, detail=f"NLP processing error: {parsed_entities_dict['error']} - Details: {parsed_entities_dict.get('details', 'N/A')}")
    teaching_outline = parsed_entities_dict.get("teaching_outline")
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
            final_teacher_id 
        )

        if plan_id:
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
    "/assessments/generate/",
    response_model=AssessmentOutput,
    summary="Generate and Save an Assessment from Natural Language Query",
    description="Accepts a natural language query to generate an assessment. Uses an LLM to extract "
                "teaching plan content, question preferences, and other details. The assessment is then "
                "generated by the service layer (which might involve RAG and another LLM call) "
                "and saved to the database.",
    responses={
        200: {"description": "Assessment generated and saved successfully."},
        400: {"model": Message, "description": "Bad Request (e.g., invalid query, missing essential info after NLP, NLP processing error)"},
        500: {"model": Message, "description": "Internal Server Error / LLM or RAG failure during assessment generation"}
    }
)
async def create_assessment_endpoint(input_data: AssessmentNLInput): 

    if not input_data.query or not input_data.query.strip():
        raise HTTPException(status_code=400, detail="查询内容不能为空。")

    # 1. 构建一个包含了所有指令和用户查询的、完整的中文用户提示 (User Prompt)
    FULL_PROMPT_FOR_ASSESSMENT_NLP = f"""
    请分析以下用户查询，该查询旨在生成一份考核。你的任务是从中提取特定信息。

    --- 用户查询开始 ---
    {input_data.query}
    --- 用户查询结束 ---

    你需要根据上述查询，提取以下实体：
    1.  "teaching_plan_content" (字符串, 必填): 考核应涵盖的核心内容、主题或材料摘要。例如："一战的起因"、"宝可梦关都地区的图鉴、道馆馆主和四天王"。
    2.  "question_preferences" (对象, 可选): 一个指定所需问题类型和数量的字典。例如：{{"选择题": 3, "简答题": 2}}。如果用户没有明确指定数量和类型，请返回 null。
    3.  "title_for_db" (字符串, 可选): 用于保存考核的特定标题。如果用户未指定，请返回 null。

    你的最终响应必须是且只能是一个符合以下描述的 JSON 对象。不要包含任何解释性文字或前导/后置文本，直接输出 JSON。
"""

    parsed_entities_dict = await parse_query_with_llm(FULL_PROMPT_FOR_ASSESSMENT_NLP)


    if "error" in parsed_entities_dict:
        raise HTTPException(status_code=400, detail=f"NLP 处理错误: {parsed_entities_dict['error']} - 详情: {parsed_entities_dict.get('details', 'N/A')}")

    teaching_plan_content = parsed_entities_dict.get("teaching_plan_content")
    if not teaching_plan_content:
        raise HTTPException(status_code=400, detail="NLP 未能从查询中提取必需的实体 'teaching_plan_content'。")

    nlp_title_for_db = parsed_entities_dict.get("title_for_db")
    question_preferences = parsed_entities_dict.get("question_preferences") 

    if question_preferences and not isinstance(question_preferences, dict):

        print(f"WARNING: 'question_preferences' from LLM was not a dict: {question_preferences}")
        question_preferences = {} 

    assessment_service_input = AssessmentInput(
        teaching_plan_content=teaching_plan_content,
        teacher_id=input_data.teacher_id,
        question_preferences=question_preferences if question_preferences else {}, 
        title_for_db=nlp_title_for_db 
    )
    


    db_conn = None
    try:
        
        generated_content = await generate_assessment_service(assessment_service_input)

        if not generated_content:
            raise HTTPException(status_code=500, detail="服务未能从LLM生成考核内容。")

       
        title_to_save = nlp_title_for_db if nlp_title_for_db else f"关于“{teaching_plan_content[:20]}...”的考核"
        

        final_teacher_id = input_data.teacher_id
        
        MYSQL_DB_NAME = os.environ.get("MYSQL_DB")
        db_conn = get_mysql_connection(db_name=MYSQL_DB_NAME)
        assessment_id = save_assessment(
            db_conn,
            title_to_save,
            generated_content,
            final_teacher_id
        )
        if assessment_id:
            return AssessmentOutput(
                assessment_id=assessment_id,
                title=title_to_save,
                generated_assessment_content=generated_content,
                teacher_id=final_teacher_id
            )
        else:
            return AssessmentOutput(
                title=title_to_save,
                generated_assessment_content=generated_content,
                error_message="考核已生成但保存至数据库失败。"
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
    "/student-assessments/evaluate-answers/",
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
    请从以下文本中，提取出学生针对一份考核所提交的全部答案。

    --- 用户提供的文本开始 ---
    {input_data.query}
    --- 用户提供的文本结束 ---

    你的任务是根据上述文本，提取以下实体：
    - "answers" (对象数组, 必填): 一个包含学生所有答案的列表。数组中的每一个对象都必须包含以下两个键：
        - "question_identifier" (字符串, 必填): 问题的标识符（例如: "题目1", "一、选择题-1", "T1"）。
        - "student_answer_text" (字符串, 必填): 学生对该问题的回答内容。

    这是一个 "answers" 字段的正确格式示例:
    `[
        {{"question_identifier": "题目1", "student_answer_text": "A"}},
        {{"question_identifier": "题目2", "student_answer_text": "B"}},
        {{"question_identifier": "简答题1", "student_answer_text": "主要原因是地心引力。"}}
    ]`

    如果无法从文本中提取出符合上述格式的答案列表，请在JSON的"error"字段中说明原因。
    你的最终响应必须是且只能是一个符合以下描述的 JSON 对象。不要包含任何解释性文字或前导/后置文本，直接输出 JSON。
"""
    parsed_entities_dict = await parse_query_with_llm(FULL_PROMPT_FOR_EVAL_NLP)

    if "error" in parsed_entities_dict:
        raise HTTPException(status_code=400, detail=f"NLP processing error: {parsed_entities_dict['error']} - Details: {parsed_entities_dict.get('details', 'N/A')}")

    answers_raw = parsed_entities_dict.get("answers")
    if not answers_raw or not isinstance(answers_raw, list) or not answers_raw:
        raise HTTPException(status_code=400, detail="NLP could not extract a valid list of answers from the query, or the list was empty.")

    parsed_answers_for_service = []
    for i, ans_item in enumerate(answers_raw):
        if not isinstance(ans_item, dict):
            raise HTTPException(status_code=400, detail=f"Invalid item in extracted 'answers' list: item at index {i} is not a dictionary.")
        q_id = ans_item.get("question_identifier")
        s_ans = ans_item.get("student_answer_text")
        if not q_id or not isinstance(q_id, str) or not q_id.strip():
            raise HTTPException(status_code=400, detail=f"Missing or invalid 'question_identifier' in extracted answer item at index {i}.")
        if s_ans is None or not isinstance(s_ans, str): 
            raise HTTPException(status_code=400, detail=f"Missing or invalid 'student_answer_text' in extracted answer item at index {i}.")
        parsed_answers_for_service.append(StudentAssessmentAnswerItem(question_identifier=q_id.strip(), student_answer_text=s_ans))
    

    

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
    "/practice-questions/generate/",
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
    "/practice-questions/feedback/",
    response_model=PracticeFeedbackOutput,
    summary="Get Feedback on a Student's Natural Language Answer to a Practice Question",
    description="Submits a student's natural language answer (`student_query_answer`) to a specific practice question, "
                "along with context like student ID, question ID, question text, model answer, and question type. "
                "The backend service then uses an LLM to generate and return feedback on the student's answer.",
    responses={
        200: {"description": "Feedback provided successfully."},
        422: {"model": Message, "description": "Validation Error (e.g., missing required fields in input)"}, # More specific than generic 400
        500: {"model": Message, "description": "Internal Server Error / LLM failure during feedback generation"}
    }
)
async def get_practice_feedback_endpoint(input_data: PracticeFeedbackInput = Body):

    try:
        result = await get_practice_feedback_service(input_data)
        return result
    except HTTPException as he: 
        raise he
    except Exception as e:
        
        print(f"API ERROR: An unexpected error occurred in /practice-questions/feedback/ endpoint: {e}")
        raise HTTPException(status_code=500, detail=f"An internal server error occurred: {str(e)}")


@router.get(
    "/assessments/{assessment_id}/student-performance/",
    response_model=List[StudentPerformanceDetail],
    summary="Get All Student Performance Details for a Specific Assessment",
    description="Retrieves a list of detailed performance records for all students who took a specific assessment. "
                "This is typically for a teacher to view class performance on a given assessment.",
    responses={
        200: {"description": "Successfully retrieved student performance details."},
        404: {"model": Message, "description": "Assessment not found."},
        500: {"model": Message, "description": "Internal Server Error."}
    }
)
async def get_assessment_performance_for_teacher(assessment_id: int):
    try:
        performance_details = await get_student_assessment_performance_service(assessment_id=assessment_id)
        return performance_details
    except HTTPException as he:
        raise he 
    except Exception as e:
        print(f"API ERROR: An unexpected error occurred in /assessments/{assessment_id}/student-performance/ endpoint: {e}")
        # Log the full error e for server-side debugging
        raise HTTPException(status_code=500, detail=f"An internal server error occurred: {str(e)}")