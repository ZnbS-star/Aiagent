import pytest
from unittest.mock import AsyncMock, patch
from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend_app.api.endpoints import router # Assuming router is directly importable
from backend_app.models import (
    StudentQuestionInput, StudentQuestionOutput, # Kept for /student-qa/ tests if any, or other uses.
    TeachingPlanNLInput, TeachingPlanOutput,
    AssessmentNLInput, AssessmentInput, AssessmentOutput,
    StudentAssessmentNLInput, StudentAssessmentInput, StudentAssessmentEvaluationOutput, StudentAssessmentAnswerItem,
    PracticeQuestionNLInput, PracticeQuestionsInput, PracticeQuestionsOutput, PracticeQuestionItem,
    PracticeFeedbackNLInput, PracticeFeedbackInput, PracticeFeedbackOutput,
    Message # For error responses
)
import json # For specific tests like NLP outputting stringified JSON

# Setup FastAPI app for testing
app = FastAPI()
app.include_router(router, prefix="/api")
client = TestClient(app)

# Path for parse_query_with_llm mock in endpoints module
PARSE_QUERY_PATH = "backend_app.api.endpoints.parse_query_with_llm"

# --- Test Cases for /teaching-plans/generate-initial/ ---
@patch("backend_app.api.endpoints.save_teaching_plan", return_value=1) # Mock DB save
@patch("backend_app.api.endpoints.get_or_create_teacher", new_callable=AsyncMock) # Mock teacher lookup
@patch("backend_app.api.endpoints.get_mysql_connection") # Mock DB connection
@patch("backend_app.api.endpoints.generate_initial_teaching_plan_service", new_callable=AsyncMock)
@patch(PARSE_QUERY_PATH, new_callable=AsyncMock)
def test_create_initial_teaching_plan_nl_success(
    mock_parse_nlp, mock_generate_plan_serv, mock_db_conn_get, mock_get_teacher, mock_save_plan
):
    mock_parse_nlp.return_value = {
        "subject": "History", "teaching_outline": "Ancient Rome", "teacher_name": "Prof. Minerva",
        "style_tone": "Engaging", "output_structure": "Weekly", "title_for_db": "Ancient Rome Plan"
    }
    mock_generate_plan_serv.return_value = ("Detailed plan about Rome", ["rome_snippet1"])
    mock_get_teacher.return_value = 1 # teacher_id

    nl_input = TeachingPlanNLInput(query="Create a history plan about Ancient Rome for Prof. Minerva, make it engaging and weekly. Title it 'Ancient Rome Plan'.")
    response = client.post("/api/teaching-plans/generate-initial/", json=nl_input.model_dump())

    assert response.status_code == 200
    response_data = response.json()
    assert response_data["subject"] == "History"
    assert response_data["generated_plan_content"] == "Detailed plan about Rome"
    assert response_data["title"] == "Ancient Rome Plan"
    mock_generate_plan_serv.assert_called_once()
    # Check that service was called with data from NLP
    args, kwargs = mock_generate_plan_serv.call_args
    assert kwargs['subject'] == "History"
    assert kwargs['initial_outline'] == "Ancient Rome"
    assert kwargs['teacher_name'] == "Prof. Minerva"


@patch(PARSE_QUERY_PATH, new_callable=AsyncMock)
def test_create_initial_teaching_plan_nlp_error(mock_parse_nlp):
    mock_parse_nlp.return_value = {"error": "NLP failed miserably"}
    nl_input = TeachingPlanNLInput(query="some query")
    response = client.post("/api/teaching-plans/generate-initial/", json=nl_input.model_dump())
    assert response.status_code == 400
    assert "NLP processing error: NLP failed miserably" in response.json()["detail"]

@patch(PARSE_QUERY_PATH, new_callable=AsyncMock)
def test_create_initial_teaching_plan_nlp_missing_entities(mock_parse_nlp):
    mock_parse_nlp.return_value = {"subject": "Math"} # Missing teaching_outline
    nl_input = TeachingPlanNLInput(query="Math plan")
    response = client.post("/api/teaching-plans/generate-initial/", json=nl_input.model_dump())
    assert response.status_code == 400
    assert "NLP could not extract required information from query: teaching_outline missing" in response.json()["detail"]

def test_create_initial_teaching_plan_empty_query():
    response = client.post("/api/teaching-plans/generate-initial/", json={"query": ""})
    assert response.status_code == 400
    assert response.json()["detail"] == "Query cannot be empty."

# --- Test Cases for /practice-questions/generate/ (Refactored for NLP) ---

@patch("backend_app.api.endpoints.generate_practice_questions_service", new_callable=AsyncMock)
@patch(PARSE_QUERY_PATH, new_callable=AsyncMock)
def test_generate_practice_questions_nl_success(mock_parse_nlp, mock_service):
    mock_parse_nlp.return_value = {
        "practice_topic": "Python lists",
        "question_preferences": {"multiple-choice": 2, "short-answer": 1}
    }
    mock_service_response = PracticeQuestionsOutput(
        generated_questions=[PracticeQuestionItem(question_type="mcq", question_text="Q1?", model_answer="A1")]
    )
    mock_service.return_value = mock_service_response

    nl_input = PracticeQuestionNLInput(query="Generate 2 mcq and 1 short answer on python lists", student_id=123)
    response = client.post("/api/practice-questions/generate/", json=nl_input.model_dump())

    assert response.status_code == 200
    assert response.json() == mock_service_response.model_dump()

    mock_parse_nlp.assert_called_once()
    # Service called with data from NLP + direct input
    expected_service_input = PracticeQuestionsInput(
        practice_topic="Python lists",
        question_preferences={"multiple-choice": 2, "short-answer": 1},
        student_id=123,
        student_name=None # Assuming not provided in this NLInput example
    )
    mock_service.assert_called_once_with(expected_service_input)

@patch(PARSE_QUERY_PATH, new_callable=AsyncMock)
def test_generate_practice_questions_nl_nlp_error(mock_parse_nlp):
    mock_parse_nlp.return_value = {"error": "NLP broke"}
    nl_input = PracticeQuestionNLInput(query="bad query")
    response = client.post("/api/practice-questions/generate/", json=nl_input.model_dump())
    assert response.status_code == 400
    assert "NLP processing error: NLP broke" in response.json()["detail"]

@patch(PARSE_QUERY_PATH, new_callable=AsyncMock)
def test_generate_practice_questions_nl_missing_topic(mock_parse_nlp):
    mock_parse_nlp.return_value = {"question_preferences": {"multiple-choice": 1}} # Missing practice_topic
    nl_input = PracticeQuestionNLInput(query="one mcq")
    response = client.post("/api/practice-questions/generate/", json=nl_input.model_dump())
    assert response.status_code == 400
    assert "NLP could not extract required entity 'practice_topic' from query" in response.json()["detail"]

@patch(PARSE_QUERY_PATH, new_callable=AsyncMock)
def test_generate_practice_questions_nl_invalid_prefs_json_string(mock_parse_nlp):
    mock_parse_nlp.return_value = {
        "practice_topic": "Python",
        "question_preferences": "{'mcq':1" # Malformed JSON string
    }
    nl_input = PracticeQuestionNLInput(query="python questions with bad prefs")
    response = client.post("/api/practice-questions/generate/", json=nl_input.model_dump())
    assert response.status_code == 400
    assert "NLP extracted 'question_preferences' as a string, but it was not valid JSON" in response.json()["detail"]

def test_generate_practice_questions_nl_empty_query():
    response = client.post("/api/practice-questions/generate/", json={"query": ""})
    assert response.status_code == 400
    assert response.json()["detail"] == "Query cannot be empty."


# --- Test Cases for /practice-questions/feedback/ (Updated for NLInput) ---

@patch("backend_app.api.endpoints.get_practice_feedback_service", new_callable=AsyncMock)
def test_get_practice_feedback_nl_success(mock_service):
    mock_response_data = PracticeFeedbackOutput(correctness_assessment="Correct", detailed_feedback="Well done!")
    mock_service.return_value = mock_response_data

    nl_input_data = PracticeFeedbackNLInput(
        student_query_answer="The answer is 42.",
        student_id=1, catalog_id=101, question_text="What is the meaning of life?",
        model_answer="42", question_type="short-answer"
    )
    response = client.post("/api/practice-questions/feedback/", json=nl_input_data.model_dump())

    assert response.status_code == 200
    assert response.json() == mock_response_data.model_dump()

    expected_service_input = PracticeFeedbackInput(
        student_id=1, catalog_id=101, question_text="What is the meaning of life?",
        model_answer="42", student_answer="The answer is 42.", question_type="short-answer"
    )
    mock_service.assert_called_once_with(expected_service_input)


def test_get_practice_feedback_nl_validation_error():
    # PracticeFeedbackNLInput has all fields as required.
    response = client.post("/api/practice-questions/feedback/", json={"student_id": 1, "catalog_id": 101}) # Missing many fields
    assert response.status_code == 422


# --- Test Cases for /assessments/generate/ (NLP Integrated) ---
@patch("backend_app.api.endpoints.save_assessment", return_value=1)
@patch("backend_app.api.endpoints.get_or_create_teacher", new_callable=AsyncMock)
@patch("backend_app.api.endpoints.get_mysql_connection")
@patch("backend_app.api.endpoints.generate_assessment_service", new_callable=AsyncMock)
@patch(PARSE_QUERY_PATH, new_callable=AsyncMock)
def test_create_assessment_nl_success(
    mock_parse_nlp, mock_generate_assessment_serv, mock_db_conn_get, mock_get_teacher, mock_save_assessment
):
    mock_parse_nlp.return_value = {
        "teaching_plan_content": "WW1 history", "teacher_name": "Prof. History",
        "question_preferences": {"short-answer": 5}, "title_for_db": "WW1 Quiz"
    }
    mock_generate_assessment_serv.return_value = ("Generated assessment on WW1", ["ww1_keyword"])
    mock_get_teacher.return_value = 2 # teacher_id

    nl_input = AssessmentNLInput(query="Create a WW1 quiz for Prof. History, 5 short answers, title 'WW1 Quiz'.")
    response = client.post("/api/assessments/generate/", json=nl_input.model_dump())

    assert response.status_code == 200
    response_data = response.json()
    assert response_data["generated_assessment_content"] == "Generated assessment on WW1"
    assert response_data["title"] == "WW1 Quiz"

    expected_service_input = AssessmentInput(
        teaching_plan_content="WW1 history", teacher_id=None, teacher_name="Prof. History",
        question_preferences={"short-answer": 5}, title_for_db="WW1 Quiz"
    )
    mock_generate_assessment_serv.assert_called_once_with(expected_service_input)

# --- Test Cases for /student-assessments/evaluate-answers/ (NLP Integrated) ---
@patch("backend_app.api.endpoints.evaluate_student_assessment_answers_service", new_callable=AsyncMock)
@patch(PARSE_QUERY_PATH, new_callable=AsyncMock)
def test_evaluate_student_answers_nl_success(mock_parse_nlp, mock_eval_service):
    mock_parse_nlp.return_value = {
        "answers": [{"question_identifier": "Q1", "student_answer_text": "Answer A"}],
        "student_name_from_query": "Dynamo" # Fallback if not in direct input
    }
    mock_service_response = [
        StudentAssessmentEvaluationOutput(
            assessment_id=1, question_identifier="Q1", student_id=123, student_answer_text="Answer A",
            llm_assessed_correctness="Correct", llm_evaluation_feedback="Good job!"
        )
    ]
    mock_eval_service.return_value = mock_service_response

    nl_input = StudentAssessmentNLInput(
        query="Student Dynamo's answer for Q1 was 'Answer A'",
        assessment_id=1,
        student_id=123
        # student_name not provided directly, relying on NLP extraction or it will fail validation if strict
    )
    # The endpoint logic for student_name: uses input_data.student_name first, then NLP's.
    # If StudentAssessmentInput strictly requires student_name, and input_data.student_name is None,
    # then nlp_student_name must be extracted.
    # The test above has student_name_from_query = "Dynamo".
    # If input_data.student_name was provided, it would override.
    # Let's assume the endpoint correctly resolves student_name to "Dynamo".

    # Adjusting input to ensure student_name is explicitly handled as per endpoint logic
    # If student_name is None in NLInput, and NLP extracts it as "Dynamo"
    # then final_student_name becomes "Dynamo"

    response = client.post("/api/student-assessments/evaluate-answers/", json=nl_input.model_dump())

    assert response.status_code == 200
    assert response.json() == [item.model_dump() for item in mock_service_response]

    expected_service_input = StudentAssessmentInput(
        student_id=123, student_name="Dynamo", assessment_id=1,
        answers=[StudentAssessmentAnswerItem(question_identifier="Q1", student_answer_text="Answer A")]
    )
    mock_eval_service.assert_called_once_with(expected_service_input)


@patch(PARSE_QUERY_PATH, new_callable=AsyncMock)
def test_evaluate_student_answers_nl_nlp_missing_answers(mock_parse_nlp):
    mock_parse_nlp.return_value = {"student_name_from_query": "Eve"} # Missing "answers"
    nl_input = StudentAssessmentNLInput(query="Eve took test 3", assessment_id=3, student_name="Eve")
    response = client.post("/api/student-assessments/evaluate-answers/", json=nl_input.model_dump())
    assert response.status_code == 400
    assert "NLP could not extract a valid list of answers" in response.json()["detail"]

@patch(PARSE_QUERY_PATH, new_callable=AsyncMock)
def test_evaluate_student_answers_nl_nlp_malformed_answers(mock_parse_nlp):
    mock_parse_nlp.return_value = {
        "answers": [{"question_id": "Q1", "text": "Bad format"}], # Wrong keys
        "student_name_from_query": "Frank"
    }
    nl_input = StudentAssessmentNLInput(query="Frank's answers...", assessment_id=4, student_name="Frank")
    response = client.post("/api/student-assessments/evaluate-answers/", json=nl_input.model_dump())
    assert response.status_code == 400
    assert "Missing or invalid 'question_identifier'" in response.json()["detail"]

def test_evaluate_student_answers_nl_empty_query():
    response = client.post("/api/student-assessments/evaluate-answers/", json={"query": "", "assessment_id":1, "student_name":"Test"})
    assert response.status_code == 400
    assert response.json()["detail"] == "Query cannot be empty."

# --- Test Cases for /student-qa/ (Should remain largely unchanged) ---
# Add existing or new tests for /student-qa/ if they are not there.
# For this task, assuming they exist and are functional.
# Example:
@patch("backend_app.api.endpoints.process_student_question_service", new_callable=AsyncMock)
def test_student_question_answer_success(mock_service):
    mock_response = StudentQuestionOutput(student_question="What is AI?", llm_answer="AI is complex.")
    mock_service.return_value = mock_response
    response = client.post("/api/student-qa/", json={"question": "What is AI?"})
    assert response.status_code == 200
    assert response.json()["llm_answer"] == "AI is complex."

# (Keep other existing tests for /student-qa/ if any)
```
