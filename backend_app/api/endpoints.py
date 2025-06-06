from fastapi import APIRouter, HTTPException, Body, Depends
from backend_app.models import (
    StudentQuestionInput, StudentQuestionOutput, 
    TeachingPlanNLInput, TeachingPlanOutput, 
    AssessmentInput, AssessmentNLInput, AssessmentOutput, 
    StudentAssessmentInput, StudentAssessmentNLInput, StudentAssessmentEvaluationOutput, StudentAssessmentAnswerItem, # Added StudentAssessmentNLInput and StudentAssessmentAnswerItem
    PracticeQuestionsInput, PracticeQuestionNLInput, PracticeQuestionsOutput, 
    PracticeFeedbackInput, PracticeFeedbackNLInput, PracticeFeedbackOutput, 
    Message, StudentPerformanceDetail # Added StudentPerformanceDetail
)
from backend_app.services import (
    process_student_question_service, 
    generate_initial_teaching_plan_service,
    generate_assessment_service,
    evaluate_student_assessment_answers_service,
    get_student_assessment_performance_service, # Added new service
    generate_practice_questions_service,
    get_practice_feedback_service
)
from backend_app.nlp_utils import parse_query_with_llm 
import json # Added for parsing question_preferences
from typing import Union, List, Optional
import os
import sys 
from backend_app.database_utils import get_mysql_connection, save_teaching_plan, get_or_create_teacher,save_assessment

# Initialize APIRouter
# All routes defined with this router will be prefixed with /api (as configured in main.py later)
router = APIRouter()
@router.post(
    "/student-qa/", 
    response_model=StudentQuestionOutput, # Primary response model
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
    """
    Endpoint to answer a student's question.
    - It takes a **question** string.
    - Optionally, it can take a `student_id` for future context personalization (not fully used yet in service).
    - It returns the original question, any RAG context found, the LLM's answer, and an optional error message.
    """
    if not input_data.question or not input_data.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    try:
        result = await process_student_question_service(input_data)
        if result.error_message:
            # Depending on the severity of error from service, you might choose different status codes
            # For now, if service indicates an error, let's consider it a 500 unless it's clearly client-side
            # (which should ideally be caught before calling the service or by the service raising HTTPException)
            # However, StudentQuestionOutput is designed to carry the error message.
            # So, we can return 200 OK with the error message inside the response body.
            # If the error implies a total failure that shouldn't be a 200, then:
            # raise HTTPException(status_code=500, detail=result.error_message)
            pass # The error is already in the result, will be returned as part of StudentQuestionOutput
        
        return result

    except HTTPException as he: # Re-raise HTTPExceptions to let FastAPI handle them
        raise he
    except Exception as e:
        # Log the exception e for server-side debugging
        print(f"API ERROR: An unexpected error occurred in /student-qa/ endpoint: {e}")
        # Return a generic 500 error to the client
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
async def create_initial_teaching_plan(input_data: TeachingPlanNLInput = Body(..., examples={
    "simple_bio_plan": {
        "summary": "Simple Biology Plan Query",
        "value": {"query": "I need a biology plan for high school covering cells and genetics."}
    },
    "physics_with_teacher_id_and_style": {
        "summary": "Physics Plan with Teacher ID and Style",
        "description": "Query includes style, and teacher_id is provided separately.",
        "value": {"query": "Generate an AP Physics 1 plan for Prof. Curie. Focus on kinematics, dynamics, and energy. Make it inquiry-based.", "teacher_id": 5}
    },
    "detailed_history_plan": {
        "summary": "Detailed History Plan Query",
        "value": {"query": "Create a teaching plan for Mr. Harrison on the American Revolution, suitable for 11th graders. It should include key battles, major figures, and the Declaration of Independence. Please provide a week-by-week breakdown and suggest some project ideas. Title it 'American Revolution Comprehensive'."}
    }
})):
    """
    Endpoint to generate an initial teaching plan from a natural language query.
    - Processes the `query` using NLP to extract subject, outline, teacher name, style, etc.
    - `teacher_id` can be optionally provided in the input if known (e.g., from user session).
    - The extracted information is then used to generate the teaching plan content.
    - The plan is saved to the database, associated with the teacher (either by `teacher_id` or by creating/finding teacher by `teacher_name` extracted from query).
    """
    if not input_data.query or not input_data.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    TEACHING_PLAN_SYSTEM_PROMPT = """
You are an AI assistant. Your task is to extract specific information from a user's query to help generate a teaching plan.
Output these entities as a JSON object.

**Important Instruction on Query Format:** The user's query might sometimes start with a phrase like "Teaching objectives:", "Note:", "User asks:", or similar instructional prefixes. You should IGNORE such prefixes and extract the information from the core request that follows. For example, if the query is "Teaching objectives: Create a math plan on algebra", you should process "Create a math plan on algebra".

Extract the following entities from the core request:
- "teaching_outline" (string, required): The specific topics, units, or key areas to be covered within that subject. If the query only states the subject for the plan without detailing specific topics, this outline can be the same as the subject. (e.g., "Core concepts, installation, basic examples", "TensorFlow Programming", "Cell structure, genetics, evolution").
- "style_tone" (string, optional): Specific style or tone for the plan (e.g., "inquiry-based", "formal", "project-based").
- "output_structure" (string, optional): Desired structure for the output (e.g., "week-by-week breakdown", "include project ideas").
- "title_for_db" (string, optional): A specific title for saving the plan.
如果没有style_tone,output_structure,title_for_db,可以根据teaching_outline生成
Output MUST be a JSON object.
"""
    parsed_entities_dict = await parse_query_with_llm(input_data.query, TEACHING_PLAN_SYSTEM_PROMPT)

    if "error" in parsed_entities_dict:
        raise HTTPException(status_code=400, detail=f"NLP processing error: {parsed_entities_dict['error']} - Details: {parsed_entities_dict.get('details', 'N/A')}")
    # Extract entities from NLP response
    teaching_outline = parsed_entities_dict.get("teaching_outline")
    style_tone = parsed_entities_dict.get("style_tone")
    output_structure = parsed_entities_dict.get("output_structure")
    title_for_db = parsed_entities_dict.get("title_for_db")
    print(teaching_outline)
    print(style_tone)
    print(output_structure)
    print(title_for_db)
    # Validate required entities from NLP
    # Determine teacher information
    # Priority: input_data.teacher_id > nlp_teacher_name
    final_teacher_id = input_data.teacher_id

    db_conn = None
    try:
        # Call the existing service function with extracted parameters
        generated_content, _ = await generate_initial_teaching_plan_service(
            initial_outline=teaching_outline,
            style_tone=style_tone,
            output_structure=output_structure
        )

        if not generated_content:
            # This specific error message might be better if it came from the service itself
            raise HTTPException(status_code=500, detail="Failed to generate teaching plan content from LLM service. Check service logs.")

        # Determine title for saving (use NLP extracted, then input_data's query as basis, or default)
        title_to_save = title_for_db 
        
        # Database operations
        MYSQL_DB_NAME = os.environ.get("MYSQL_DB")
        if not MYSQL_DB_NAME:
            print("API WARNING: MYSQL_DB environment variable not set. Cannot save teaching plan.")
            # Return the generated content without saving if DB is not configured
            return TeachingPlanOutput(
                title=title_to_save, 
                generated_plan_content=generated_content,
                error_message="Plan generated but not saved; MYSQL_DB not configured."
            )

        db_conn = get_mysql_connection(db_name=MYSQL_DB_NAME)
        if not db_conn:
            print("API ERROR: Failed to connect to the database for saving teaching plan.")
            # Return generated content but indicate save failure
            return TeachingPlanOutput(
                title=title_to_save,
                generated_plan_content=generated_content,
                error_message="Plan generated but failed to connect to DB for saving."
            )

        # If after all that, final_teacher_id is still None, it will be saved as such if DB allows.
        
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
            # save_teaching_plan would have printed an error
            return TeachingPlanOutput(
                title=title_to_save,
                generated_plan_content=generated_content,
                error_message="Plan generated but failed to save to database. Check server logs."
            )
            
    except HTTPException as he: # Includes NLP validation errors re-raised as HTTPException
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
async def create_assessment_endpoint(input_data: AssessmentNLInput = Body(..., examples={
    "kanto_quiz_request": {
        "summary": "Kanto Region Quiz",
        "value": {"query": "Generate an assessment for Prof. Oak based on the Kanto region Pokedex, Gym Leaders, and Elite Four. Include 2 multiple-choice and 1 short-answer. Title it 'Kanto Basics Quiz'."}
    },
    "history_assessment_with_teacher_id": {
        "summary": "History Assessment with Teacher ID",
        "value": {"query": "Create an assessment on World War 1 causes for 10th grade. Focus on short answer questions, maybe 3 of them.", "teacher_id": 7}
    }
})):
    """
    Endpoint to generate and save an assessment from a natural language query.
    - Processes the `query` using NLP to extract `teaching_plan_content`, `teacher_name`, 
      `question_preferences`, and `title_for_db`.
    - `teacher_id` can be optionally provided in the input.
    - The extracted information is used to generate and save the assessment.
    """
    if not input_data.query or not input_data.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    ASSESSMENT_GEN_SYSTEM_PROMPT = """
You are an AI assistant that extracts parameters for generating an assessment from a user's query.
The user wants to generate an assessment. Extract the following entities:
- "teaching_plan_content" (string, required): The core content, topic, or summary of material the assessment should cover.
- "question_preferences" (object, optional): A dictionary specifying desired question types and counts (e.g., {{"multiple-choice": 3, "short-answer": 2}}).
- "title_for_db" (string, optional): A specific title for saving the assessment.

If "teaching_plan_content" is missing, state that in the "error" field of your JSON response.
Output MUST be a JSON object.
"""
    parsed_entities_dict = await parse_query_with_llm(input_data.query, ASSESSMENT_GEN_SYSTEM_PROMPT)

    if "error" in parsed_entities_dict:
        raise HTTPException(status_code=400, detail=f"NLP processing error: {parsed_entities_dict['error']} - Details: {parsed_entities_dict.get('details', 'N/A')}")

    teaching_plan_content = parsed_entities_dict.get("teaching_plan_content")
    if not teaching_plan_content:
        raise HTTPException(status_code=400, detail="NLP could not extract required entity 'teaching_plan_content' from query.")

    nlp_teacher_name = parsed_entities_dict.get("teacher_name")
    nlp_title_for_db = parsed_entities_dict.get("title_for_db")
    
    question_preferences_raw = parsed_entities_dict.get("question_preferences")
    question_preferences = {} # Default for AssessmentInput model
    if isinstance(question_preferences_raw, str):
        try:
            question_preferences = json.loads(question_preferences_raw)
            if not isinstance(question_preferences, dict):
                raise HTTPException(status_code=400, detail="NLP extracted 'question_preferences' but it was not a valid dictionary structure after parsing string.")
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail=f"NLP extracted 'question_preferences' as a string, but it was not valid JSON: {question_preferences_raw}")
    elif isinstance(question_preferences_raw, dict):
        question_preferences = question_preferences_raw

    assessment_service_input = AssessmentInput(
        teaching_plan_content=teaching_plan_content,
        teacher_id=input_data.teacher_id, # Pass through if provided
        teacher_name=nlp_teacher_name, # Pass through if extracted
        question_preferences=question_preferences, # Pass parsed or default
        title_for_db=nlp_title_for_db # Pass through if extracted
    )
    
    # Determine teacher information for DB saving (similar to teaching plan endpoint)
    final_teacher_id = input_data.teacher_id 
    # effective_teacher_name will be nlp_teacher_name or a default if needed by get_or_create_teacher
    # The AssessmentInput model itself carries teacher_name and teacher_id to the service.
    # The service layer or this endpoint's DB part will resolve it.

    db_conn = None
    try:
        # generate_assessment_service takes AssessmentInput
        generated_content, rag_keywords = await generate_assessment_service(assessment_service_input)

        if not generated_content:
            error_detail = "Failed to generate assessment content from LLM service."
            if rag_keywords: # Assuming rag_keywords is a list of strings
                error_detail += f" RAG keywords extracted were: {', '.join(rag_keywords)}."
            raise HTTPException(status_code=500, detail=error_detail)

        # Determine title for saving
        # Use title from NLP if present, else use title from original input (if it was there), else generate default
        title_to_save = nlp_title_for_db if nlp_title_for_db else f"Assessment on '{teaching_plan_content[:30]}...' by {nlp_teacher_name or 'System'}"
        
        MYSQL_DB_NAME = os.environ.get("MYSQL_DB")
        if not MYSQL_DB_NAME:
            print("API WARNING: MYSQL_DB environment variable not set. Cannot save assessment.")
            return AssessmentOutput(
                title=title_to_save,
                generated_assessment_content=generated_content,
                error_message="Assessment generated but not saved; MYSQL_DB not configured."
            )

        db_conn = get_mysql_connection(db_name=MYSQL_DB_NAME)
        if not db_conn:
            print("API ERROR: Failed to connect to the database for saving assessment.")
            return AssessmentOutput(
                title=title_to_save,
                generated_assessment_content=generated_content,
                error_message="Assessment generated but failed to connect to DB for saving."
            )

        if not final_teacher_id and nlp_teacher_name: # If no teacher_id from input, try to use name from NLP
            if db_conn and db_conn.is_connected():
                final_teacher_id = get_or_create_teacher(db_conn, nlp_teacher_name)
                if not final_teacher_id:
                    print(f"API WARNING: Could not get or create teacher '{nlp_teacher_name}'. Assessment will be saved without specific teacher linkage if schema allows.")
            else: # This case should be hit if MYSQL_DB_NAME was not set or connection failed
                 print("API WARNING: No DB connection to get/create teacher for assessment. Assessment will not be associated with a teacher.")
        
        # If still no final_teacher_id (e.g., no input_id, no nlp_name, or get_or_create_teacher failed),
        # it will be saved with NULL teacher_id if DB schema allows.

        assessment_id = save_assessment(
            db_conn,
            title_to_save, # Use the determined title
            generated_content,
            final_teacher_id # May be None
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
                error_message="Assessment generated but failed to save to database. Check server logs."
            )

    except HTTPException as he:
        raise he
    except Exception as e:
        print(f"API ERROR: An unexpected error occurred in /assessments/generate/ endpoint: {e}")
        raise HTTPException(status_code=500, detail=f"An internal server error occurred: {str(e)}")
    finally:
        if db_conn and db_conn.is_connected():
            print("Closing DB connection for /assessments/generate/ endpoint.")
            db_conn.close()

@router.post(
    "/student-assessments/evaluate-answers/",
    response_model=Message, # Changed from List[StudentAssessmentEvaluationOutput]
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
async def evaluate_student_answers(input_data: StudentAssessmentNLInput = Body(..., examples={
    "john_doe_assessment_submission": {
        "summary": "John Doe's answers via query",
        "value": {
            "query": "John Doe answered assessment 45. For Q1, he said 'Paris'. For Q2, his answer was 'Water is H2O'.",
            "assessment_id": 45,
            "student_name": "John Doe" # Can also be extracted by LLM if student_name_from_query is used
        }
    },
    "jane_doe_with_student_id": {
        "summary": "Jane Doe's answers with student ID",
        "value": {
            "query": "For assessment 101, my answers are: Question Alpha was 'A', Question Beta was 'B'.",
            "assessment_id": 101,
            "student_id": 777,
            "student_name": "Jane Doe" # student_name is still good for confirmation or if service needs it
        }
    }
})):
    """
    Endpoint to evaluate a student's assessment answers, where answers are provided in a natural language query.
    - The `query` field should contain the student's answers in text.
    - `assessment_id` is mandatory and provided directly.
    - `student_id` and/or `student_name` are provided directly. The service input model `StudentAssessmentInput` requires `student_name`.
    - NLP is used to extract the list of answers from the `query`.
    """
    if not input_data.query or not input_data.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
    if not input_data.assessment_id: # Should be caught by Pydantic, but explicit check is fine
        raise HTTPException(status_code=422, detail="assessment_id is required.")

    ASSESSMENT_EVAL_SYSTEM_PROMPT = """
You are an AI assistant that extracts a student's answers for an assessment from a user query.
The user wants to evaluate a student's assessment. Extract the following entities from the main query text:
- "answers" (array of objects, required): A list of the student's answers. Each object in the array MUST have:
    - "question_identifier" (string, required): The identifier of the question (e.g., "Question 1", "1a", "Section A Q1").
    - "student_answer_text" (string, required): The text of the student's answer.
  Example format for answers field: [{"question_identifier": "Q1", "student_answer_text": "Paris"}, {"question_identifier": "Q2", "student_answer_text": "Water"}]

The user will also provide student_id (optional), student_name (optional direct input), and assessment_id (required direct input) separately from the main query text. Your primary focus for extraction from the query text is the "answers" list.
If the "answers" list cannot be extracted or is malformed (e.g., missing identifiers or text), state that in the "error" field of your JSON.
Output MUST be a JSON object.
"""
    parsed_entities_dict = await parse_query_with_llm(input_data.query, ASSESSMENT_EVAL_SYSTEM_PROMPT)

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
        if s_ans is None or not isinstance(s_ans, str): # Allow empty string, but must be present and string
            raise HTTPException(status_code=400, detail=f"Missing or invalid 'student_answer_text' in extracted answer item at index {i}.")
        parsed_answers_for_service.append(StudentAssessmentAnswerItem(question_identifier=q_id.strip(), student_answer_text=s_ans))
    
    # Determine student_name (StudentAssessmentInput model requires it)
    final_student_name = input_data.student_name
    # Removed: student_name_from_query is no longer extracted by NLP for this endpoint.
    # if not final_student_name:
    #     final_student_name = parsed_entities_dict.get("student_name_from_query") 
    
    if not final_student_name: # If still no name, and student_id also wasn't given directly (Pydantic would catch if student_name was required on NLInput and not given)
         # StudentAssessmentInput requires student_name. If student_id is also missing, it's an issue.
         # If student_id IS present, we technically have an identifier, but service input needs name.
         # This logic assumes the API caller should provide student_name in StudentAssessmentNLInput if it's not extractable by LLM.
        if not input_data.student_id: # No student_id and no student_name at all
            raise HTTPException(status_code=400, detail="Student identification failed: NLP did not find student name in query, and no student_name or student_id was provided in the input fields.")
        else: # student_id is present, but student_name is not. StudentAssessmentInput needs student_name.
            raise HTTPException(status_code=400, detail="Student name is required for evaluation. Please provide 'student_name' directly in the input, or ensure it can be extracted from the query if only 'student_id' is given.")


    student_assessment_service_input = StudentAssessmentInput(
        student_id=input_data.student_id,
        student_name=final_student_name,
        assessment_id=input_data.assessment_id,
        answers=parsed_answers_for_service
    )

    try:
        evaluation_results = await evaluate_student_assessment_answers_service(student_assessment_service_input)
        
        # Check if the service returned an error condition (e.g., DB not configured, overall processing error)
        if evaluation_results and any(result.error_message for result in evaluation_results if result.question_identifier == "Overall Error"):
            # If there's an overall error, it's likely a 500, but the service might have specific error messages.
            # The service currently returns a list even for a single overall error.
            overall_error = next((res.error_message for res in evaluation_results if res.question_identifier == "Overall Error"), "Service error")
            raise HTTPException(status_code=500, detail=overall_error)
        
        # Check if any individual answer failed to save (though service might raise before this)
        # Log warning if needed, but the response will be a generic success message.
        if evaluation_results and any(result.error_message and not result.answer_id for result in evaluation_results):
             print("API WARNING: Some answers may not have been saved successfully during the evaluation process. Check service logs for details.")
             # The endpoint will still return the generic success message as per requirements.

        return Message(message="Student assessment answers processed and saved successfully.")
            
    except HTTPException as he:
        raise he
    except ValueError as ve: # Catch specific errors like "Assessment not found" or "Student not found"
        raise HTTPException(status_code=404, detail=str(ve))
    except Exception as e:
        print(f"API ERROR: An unexpected error occurred in /student-assessments/evaluate-answers/ endpoint: {e}")
        # Log the full error e for server-side debugging
        raise HTTPException(status_code=500, detail=f"An internal server error occurred: {str(e)}")

# --- Practice Question Endpoints ---

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
async def generate_practice_questions_endpoint(input_data: PracticeQuestionNLInput = Body(..., examples={
    "python_lists": {
        "summary": "Python list questions",
        "value": {"query": "I want to practice python list comprehensions"}
    },
    "photosynthesis_with_prefs_and_student": {
        "summary": "Photosynthesis with preferences and student",
        "value": {"query": "Make two multiple-choice and one short-answer question about photosynthesis.", "student_id": 101, "student_name": "Alice"}
    },
    "calculus_student_name_only": {
        "summary": "Calculus questions for a named student",
        "value": {"query": "Generate some calculus derivative problems for Bob.", "student_name": "Bob"}
    }
})):
    """
    Endpoint to generate practice questions from a natural language query.
    - Processes the `query` using NLP to extract `practice_topic` and `question_preferences`.
    - `student_id` and `student_name` can be optionally provided in the input for personalization.
    - The extracted information is used to call the practice question generation service.
    """
    if not input_data.query or not input_data.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    PRACTICE_QUESTION_SYSTEM_PROMPT = """
You are an AI assistant that extracts parameters for generating practice questions from a user's query.
The user wants to generate practice questions. Extract the following entities:
- "practice_topic" (string, required): The specific topic for the practice questions (e.g., "Python list comprehensions", "photosynthesis").
- "question_preferences" (object, optional): A dictionary specifying the desired types and counts of questions. For example: {{"multiple-choice": 2, "short-answer": 1, "programming": 1}}. If not specified, it's okay to omit.

If "practice_topic" is missing, clearly state that in the "error" field of your JSON response.
Output MUST be a JSON object.
"""
    parsed_entities_dict = await parse_query_with_llm(input_data.query, PRACTICE_QUESTION_SYSTEM_PROMPT)

    if "error" in parsed_entities_dict:
        raise HTTPException(status_code=400, detail=f"NLP processing error: {parsed_entities_dict['error']} - Details: {parsed_entities_dict.get('details', 'N/A')}")

    practice_topic = parsed_entities_dict.get("practice_topic")
    if not practice_topic:
        raise HTTPException(status_code=400, detail="NLP could not extract required entity 'practice_topic' from query.")

    question_preferences_raw = parsed_entities_dict.get("question_preferences")
    question_preferences = {} # Default to empty dict, PracticeQuestionsInput model will apply its own default
    if isinstance(question_preferences_raw, str):
        try:
            question_preferences = json.loads(question_preferences_raw)
            if not isinstance(question_preferences, dict):
                raise HTTPException(status_code=400, detail="NLP extracted 'question_preferences' but it was not a valid dictionary structure after parsing string.")
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail=f"NLP extracted 'question_preferences' as a string, but it was not valid JSON: {question_preferences_raw}")
    elif isinstance(question_preferences_raw, dict):
        question_preferences = question_preferences_raw
    
    # Prepare input for the service call using the original PracticeQuestionsInput model
    practice_questions_service_input = PracticeQuestionsInput(
        practice_topic=practice_topic,
        question_preferences=question_preferences, # Let Pydantic model handle default if it's {}
        student_id=input_data.student_id,
        student_name=input_data.student_name
    )

    try:
        result = await generate_practice_questions_service(practice_questions_service_input)
        # PracticeQuestionsOutput is designed to carry an error message if one occurs.
        return result
    except HTTPException as he: # Should not be strictly necessary if service doesn't raise HTTPException
        raise he
    except Exception as e:
        # Log the exception e for server-side debugging
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
async def get_practice_feedback_endpoint(input_data: PracticeFeedbackNLInput = Body(..., examples={
    "feedback_request_short_answer": {
        "summary": "Feedback for a short answer",
        "value": {
            "student_query_answer": "A list comprehension is a way to make lists.",
            "student_id": 101,
            "catalog_id": 201,
            "question_text": "What is a list comprehension in Python?",
            "model_answer": "A concise way to create lists based on existing lists.",
            "question_type": "Short-Answer"
        }
    },
    "feedback_request_programming": {
        "summary": "Feedback for a programming question answer",
        "value": {
            "student_query_answer": "def solve():\n  return 42",
            "student_id": 102,
            "catalog_id": 205,
            "question_text": "Write a Python function that returns the number 42.",
            "model_answer": "def get_the_answer():\n  return 42",
            "question_type": "Programming"
        }
    }
})):
    """
    Endpoint to get feedback on a student's natural language answer to a practice question.
    - All fields in `PracticeFeedbackNLInput` are required.
    - The `student_query_answer` field contains the student's free-form text answer.
    - This data is then used to construct the input for the feedback generation service.
    """
    # Pydantic performs validation for presence of required fields based on PracticeFeedbackNLInput model.
    # No explicit 'if not all(...)' check needed here as Pydantic handles it.
    # If any field in PracticeFeedbackNLInput is missing, FastAPI returns a 422 error automatically.

    # Prepare input for the service call using the original PracticeFeedbackInput model
    feedback_service_input = PracticeFeedbackInput(
        student_id=input_data.student_id,
        catalog_id=input_data.catalog_id,
        question_text=input_data.question_text,
        model_answer=input_data.model_answer, # Ensure this matches field in PracticeFeedbackInput
        student_answer=input_data.student_query_answer, # Mapping student_query_answer to student_answer
        question_type=input_data.question_type
    )

    try:
        result = await get_practice_feedback_service(feedback_service_input)
        # PracticeFeedbackOutput is designed to carry an error message if one occurs.
        return result
    except HTTPException as he: # Should not be strictly necessary if service doesn't raise HTTPException
        raise he
    except Exception as e:
        # Log the exception e for server-side debugging
        print(f"API ERROR: An unexpected error occurred in /practice-questions/feedback/ endpoint: {e}")
        raise HTTPException(status_code=500, detail=f"An internal server error occurred: {str(e)}")

# --- NLP Unified Query Endpoint --- # REMOVED
# The /natural-query/ endpoint and its implementation 
# async def natural_language_query_endpoint(...)
# have been removed.


# --- Endpoint to get student performance for a specific assessment ---
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
    """
    Endpoint to fetch all student performance data for a given assessment ID.
    - `assessment_id`: Path parameter specifying the assessment.
    - Returns a list of `StudentPerformanceDetail` objects.
    - Handles cases like assessment not found by raising HTTPException (expected from service).
    """
    try:
        performance_details = await get_student_assessment_performance_service(assessment_id=assessment_id)
        # If service returns empty list for a valid assessment_id with no submissions, it's a valid 200.
        # If assessment_id itself is invalid/not found, service should raise HTTPException.
        return performance_details
    except HTTPException as he:
        raise he # Re-raise HTTPExceptions (like 404, 501 from service) to let FastAPI handle them
    except Exception as e:
        print(f"API ERROR: An unexpected error occurred in /assessments/{assessment_id}/student-performance/ endpoint: {e}")
        # Log the full error e for server-side debugging
        raise HTTPException(status_code=500, detail=f"An internal server error occurred: {str(e)}")