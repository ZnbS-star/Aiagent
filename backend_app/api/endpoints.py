from fastapi import APIRouter, HTTPException, Body, Depends
from backend_app.models import (
    StudentQuestionInput, StudentQuestionOutput, 
    TeachingPlanInput, TeachingPlanOutput,
    AssessmentInput, AssessmentOutput,
    StudentAssessmentInput, StudentAssessmentEvaluationOutput,
    PracticeQuestionsInput, PracticeQuestionsOutput,
    PracticeFeedbackInput, PracticeFeedbackOutput, # Added
    Message 
)
from backend_app.services import (
    process_student_question_service, 
    generate_initial_teaching_plan_service,
    generate_assessment_service,
    evaluate_student_assessment_answers_service,
    generate_practice_questions_service,
    get_practice_feedback_service # Added
)
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
    summary="Generate and Save an Initial Teaching Plan",
    description="Generates a first draft of a teaching plan using RAG and an LLM, then saves it to the database.",
    responses={
        200: {"description": "Teaching plan generated and saved successfully."},
        400: {"model": Message, "description": "Bad Request (e.g., missing required fields)"},
        500: {"model": Message, "description": "Internal Server Error / LLM or RAG failure"}
    }
)
async def create_initial_teaching_plan(input_data: TeachingPlanInput = Body(..., examples={
    "biology_plan": {
        "summary": "Basic Biology Plan",
        "description": "Generate a plan for High School Biology by Dr. Smith.",
        "value": {
            "teacher_name": "Dr. Smith",
            "subject": "High School Biology",
            "teaching_outline": "Unit 1: Introduction to Cells. Unit 2: Genetics. Unit 3: Ecology."
        }
    },
    "physics_plan_with_style": {
        "summary": "Physics Plan with Style",
        "description": "Generate a physics plan with a specific style.",
        "value": {
            "teacher_name": "Prof. Curie",
            "subject": "AP Physics 1",
            "teaching_outline": "Kinematics, Dynamics, Circular Motion & Gravitation, Energy, Momentum",
            "style_tone": "Inquiry-based and hands-on, with real-world examples."
        }
    }
})):
    """
    Endpoint to generate an initial teaching plan.
    - Requires `teacher_name`, `subject`, and `teaching_outline`.
    - Requires `subject` and `teaching_outline`.
    - `teacher_name` or `teacher_id` should be provided if the plan needs to be associated with a teacher.
    - `title_for_db` is optional; if not provided, a default title will be generated.
    """
    # Basic input validation
    if not input_data.subject or not input_data.teaching_outline:
        raise HTTPException(status_code=400, detail="Subject and teaching_outline are required.")
    if not input_data.teacher_id and not input_data.teacher_name:
        print("Warning: Neither teacher_id nor teacher_name provided. Plan will be saved without teacher association if DB allows, or fail if teacher_id is mandatory.")
        # Depending on DB schema for teaching_plans.teacher_id (NULL or NOT NULL)

    final_teacher_id = input_data.teacher_id
    db_conn = None
    
    try:
        generated_content, rag_snippets_used = await generate_initial_teaching_plan_service(
            teacher_name=input_data.teacher_name if input_data.teacher_name else "Unknown Teacher", # Service expects a name
            subject=input_data.subject,
            initial_outline=input_data.teaching_outline,
            style_tone=input_data.style_tone,
            output_structure=input_data.output_structure
        )

        if not generated_content:
            raise HTTPException(status_code=500, detail="Failed to generate teaching plan content from LLM. Check service logs.")

        # Determine title for saving
        title_to_save = input_data.title_for_db if input_data.title_for_db else f"Initial Plan for {input_data.subject} by {input_data.teacher_name or 'System'}"

        # Save to database
        MYSQL_DB_NAME = os.environ.get("MYSQL_DB")
        if not MYSQL_DB_NAME:
            print("API WARNING: MYSQL_DB environment variable not set. Cannot save teaching plan.")
            # Return the generated content without saving if DB is not configured
            return TeachingPlanOutput(
                title=title_to_save, 
                subject=input_data.subject,
                generated_plan_content=generated_content,
                error_message="Plan generated but not saved; MYSQL_DB not configured."
            )

        db_conn = get_mysql_connection(db_name=MYSQL_DB_NAME)
        if not db_conn:
            print("API ERROR: Failed to connect to the database for saving teaching plan.")
            # Return generated content but indicate save failure
            return TeachingPlanOutput(
                title=title_to_save,
                subject=input_data.subject,
                generated_plan_content=generated_content,
                error_message="Plan generated but failed to connect to DB for saving."
            )

        # Get/Create teacher_id if not provided directly
        if not final_teacher_id and input_data.teacher_name:
            if db_conn and db_conn.is_connected(): # Ensure connection before this call
                final_teacher_id = get_or_create_teacher(db_conn, input_data.teacher_name)
                if not final_teacher_id:
                    print(f"API WARNING: Could not get or create teacher '{input_data.teacher_name}'. Plan will be saved without specific teacher linkage if schema allows.")
            else:
                print("API WARNING: No DB connection to get/create teacher. Plan will not be associated with a teacher.")
        
        plan_id = save_teaching_plan( # This function handles its own cursor and commit/rollback
            db_conn, # Pass the connection
            input_data.subject, 
            title_to_save, 
            generated_content, 
            final_teacher_id
        )

        if plan_id:
            return TeachingPlanOutput(
                teaching_plan_id=plan_id,
                title=title_to_save, 
                subject=input_data.subject,
                generated_plan_content=generated_content,
                teacher_id=final_teacher_id
            )
        else:
            # save_teaching_plan would have printed an error
            return TeachingPlanOutput(
                title=title_to_save,
                subject=input_data.subject,
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
    summary="Generate and Save an Assessment",
    description="Generates assessment questions based on teaching plan content (using RAG & LLM) and saves it to the database.",
    responses={
        200: {"description": "Assessment generated and saved successfully."},
        400: {"model": Message, "description": "Bad Request (e.g., missing required fields)"},
        500: {"model": Message, "description": "Internal Server Error / LLM or RAG failure"}
    }
)
async def create_assessment_endpoint(input_data: AssessmentInput = Body(..., examples={
    "general_assessment": {
        "summary": "General Assessment",
        "description": "Generate an assessment for a given teaching plan.",
        "value": {
            "teacher_name": "Prof. Oak",
            "teaching_plan_content": "The Kanto region is home to many Pokemon. Key concepts include Pokedex, Gym Leaders, and the Elite Four.",
            "question_preferences": {"multiple-choice": 2, "short-answer": 1},
            "title_for_db": "Kanto Region Basics Quiz"
        }
    }
})):
    """
    Endpoint to generate and save an assessment.
    - Requires `teaching_plan_content`.
    - `teacher_name` or `teacher_id` should be provided for association.
    - `question_preferences` and `title_for_db` are optional.
    """
    if not input_data.teaching_plan_content:
        raise HTTPException(status_code=400, detail="teaching_plan_content is required.")

    final_teacher_id = input_data.teacher_id
    db_conn = None

    try:
        generated_content, rag_keywords = await generate_assessment_service(input_data)

        if not generated_content:
            error_detail = "Failed to generate assessment content from LLM. Check service logs."
            if rag_keywords:
                error_detail += f" Keywords extracted for RAG were: {rag_keywords}"
            raise HTTPException(status_code=500, detail=error_detail)

        title_to_save = input_data.title_for_db if input_data.title_for_db else f"Assessment for Teacher {input_data.teacher_name or final_teacher_id or 'Unknown'}"

        # Save to database
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

        if not final_teacher_id and input_data.teacher_name:
            if db_conn.is_connected(): # Ensure connection before this call
                final_teacher_id = get_or_create_teacher(db_conn, input_data.teacher_name)
                if not final_teacher_id:
                    print(f"API WARNING: Could not get or create teacher '{input_data.teacher_name}'. Assessment will be saved without specific teacher linkage if schema allows.")
            else: # Should not happen if initial db_conn check passed, but as safeguard
                 print("API WARNING: No DB connection to get/create teacher for assessment. Assessment will not be associated with a teacher.")


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
    response_model=List[StudentAssessmentEvaluationOutput], # Returns a list of evaluations
    summary="Evaluate Student's Answers for an Assessment",
    description="Receives a student's answers for a specific assessment, uses an LLM to evaluate each answer against the assessment content (and RAG context if available), and saves the evaluation to the database.",
    responses={
        200: {"description": "Student answers evaluated and saved successfully."},
        400: {"model": Message, "description": "Bad Request (e.g., invalid input, missing fields)"},
        404: {"model": Message, "description": "Not Found (e.g., assessment ID not found)"},
        500: {"model": Message, "description": "Internal Server Error / LLM or DB failure"}
    }
)
async def evaluate_student_answers(input_data: StudentAssessmentInput = Body(..., examples={
    "john_doe_quiz1_answers": {
        "summary": "John Doe's Quiz 1 Answers",
        "description": "Submit John Doe's answers for assessment ID 1.",
        "value": {
            "student_name": "John Doe",
            "assessment_id": 1,
            "answers": [
                {"question_identifier": "Question 1", "student_answer_text": "Paris is the capital of France."},
                {"question_identifier": "Question 2", "student_answer_text": "Photosynthesis uses sunlight, water, and CO2 to produce glucose and oxygen."}
            ]
        }
    },
    "jane_doe_exam_part1": {
        "summary": "Jane Doe's Exam Part 1",
        "description": "Submit Jane's answers using her student ID for assessment ID 5.",
        "value": {
            "student_id": 102,
            "student_name": "Jane Doe", # student_name can be optional if ID is given, but good for cross-check
            "assessment_id": 5,
            "answers": [
                {"question_identifier": "Section A Q1a", "student_answer_text": "The mitochondria is the powerhouse of the cell."},
                {"question_identifier": "Section B Q2", "student_answer_text": "for i in range(10): print(i)"}
            ]
        }
    }
})):
    """
    Endpoint to evaluate and save a student's answers for a given assessment.
    - Requires `assessment_id` and a list of `answers` (each with `question_identifier` and `student_answer_text`).
    - Requires either `student_id` or `student_name`.
    - The service layer handles fetching assessment content, LLM evaluation, and saving.
    """
    if not input_data.answers:
        raise HTTPException(status_code=400, detail="No answers provided for evaluation.")
    if not input_data.student_id and not input_data.student_name:
        raise HTTPException(status_code=400, detail="Either student_id or student_name is required.")

    try:
        evaluation_results = await evaluate_student_assessment_answers_service(input_data)
        
        # Check if the service returned an error condition (e.g., DB not configured, overall processing error)
        if evaluation_results and any(result.error_message for result in evaluation_results if result.question_identifier == "Overall Error"):
            # If there's an overall error, it's likely a 500, but the service might have specific error messages.
            # The service currently returns a list even for a single overall error.
            overall_error = next((res.error_message for res in evaluation_results if res.question_identifier == "Overall Error"), "Service error")
            raise HTTPException(status_code=500, detail=overall_error)
        
        # Check if any individual answer failed to save (though service might raise before this)
        if evaluation_results and any(result.error_message and not result.answer_id for result in evaluation_results):
             print("API WARNING: Some answers may not have been saved successfully. Check response details.")
             # Still return 200 but with error messages in the items.

        return evaluation_results
            
    except HTTPException as he:
        raise he
    except ValueError as ve: # Catch specific errors like "Assessment not found" or "Student not found"
        raise HTTPException(status_code=404, detail=str(ve))
    except Exception as e:
        print(f"API ERROR: An unexpected error occurred in /student-assessments/evaluate-answers/ endpoint: {e}")
        # Log the full error e for server-side debugging
        raise HTTPException(status_code=500, detail=f"An internal server error occurred: {str(e)}")
