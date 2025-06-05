import pytest
from unittest.mock import AsyncMock, patch
from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend_app.api.endpoints import router # Assuming router is directly importable
from backend_app.models import (
    PracticeQuestionsInput, PracticeQuestionsOutput, PracticeQuestionItem,
    PracticeFeedbackInput, PracticeFeedbackOutput,
    NaturalLanguageQueryInput, NLQueryResponse,
    StudentQuestionOutput # For NLQueryResponse.service_response example
)

# Setup FastAPI app for testing
app = FastAPI()
app.include_router(router, prefix="/api") # Assuming a prefix like in a real app
client = TestClient(app)

# --- Test Cases for /practice-questions/generate/ ---

@patch("backend_app.api.endpoints.generate_practice_questions_service", new_callable=AsyncMock)
def test_generate_practice_questions_success(mock_service):
    mock_response_data = PracticeQuestionsOutput(
        generated_questions=[PracticeQuestionItem(question_type="mcq", question_text="What is Python?", model_answer="A snake.")]
    )
    mock_service.return_value = mock_response_data

    input_data = PracticeQuestionsInput(practice_topic="Python basics", question_preferences={"mcq": 1})
    response = client.post("/api/practice-questions/generate/", json=input_data.model_dump())

    assert response.status_code == 200
    assert response.json() == mock_response_data.model_dump()
    mock_service.assert_called_once_with(input_data)

@patch("backend_app.api.endpoints.generate_practice_questions_service", new_callable=AsyncMock)
def test_generate_practice_questions_service_error(mock_service):
    mock_response_data = PracticeQuestionsOutput(generated_questions=[], error_message="LLM is tired")
    mock_service.return_value = mock_response_data

    input_data = PracticeQuestionsInput(practice_topic="Advanced Quantum Physics")
    response = client.post("/api/practice-questions/generate/", json=input_data.model_dump())

    assert response.status_code == 200 # Error is in the response body
    assert response.json()["error_message"] == "LLM is tired"
    mock_service.assert_called_once_with(input_data)

def test_generate_practice_questions_validation_error():
    response = client.post("/api/practice-questions/generate/", json={"question_preferences": {"mcq": 1}}) # Missing practice_topic
    assert response.status_code == 422 # FastAPI's validation error

# --- Test Cases for /practice-questions/feedback/ ---

@patch("backend_app.api.endpoints.get_practice_feedback_service", new_callable=AsyncMock)
def test_get_practice_feedback_success(mock_service):
    mock_response_data = PracticeFeedbackOutput(correctness_assessment="Correct", detailed_feedback="Well done!")
    mock_service.return_value = mock_response_data

    input_data = PracticeFeedbackInput(
        student_id=1, catalog_id=101, question_text="Is sky blue?",
        model_answer="Yes", student_answer="Yes", question_type="short-answer"
    )
    response = client.post("/api/practice-questions/feedback/", json=input_data.model_dump())

    assert response.status_code == 200
    assert response.json() == mock_response_data.model_dump()
    mock_service.assert_called_once_with(input_data)

@patch("backend_app.api.endpoints.get_practice_feedback_service", new_callable=AsyncMock)
def test_get_practice_feedback_service_error(mock_service):
    mock_response_data = PracticeFeedbackOutput(correctness_assessment="Undetermined", detailed_feedback="", error_message="Feedback LLM failed")
    mock_service.return_value = mock_response_data

    input_data = PracticeFeedbackInput(
        student_id=1, catalog_id=101, question_text="Is sky blue?",
        model_answer="Yes", student_answer="Yes", question_type="short-answer"
    )
    response = client.post("/api/practice-questions/feedback/", json=input_data.model_dump())

    assert response.status_code == 200
    assert response.json()["error_message"] == "Feedback LLM failed"
    mock_service.assert_called_once_with(input_data)

def test_get_practice_feedback_validation_error():
    response = client.post("/api/practice-questions/feedback/", json={"student_id": 1}) # Missing many fields
    assert response.status_code == 422

# --- Test Cases for /natural-query/ ---

@patch("backend_app.api.endpoints.process_natural_language_query", new_callable=AsyncMock)
def test_natural_query_success_routes_to_service(mock_nlp_service):
    # Example: NLP service routes to student_question
    mock_sq_output = StudentQuestionOutput(student_question="What is life?", llm_answer="42")
    mock_nl_response = NLQueryResponse(
        status="Success",
        original_query="What is life?",
        detected_intent="student_question",
        service_response=mock_sq_output
    )
    mock_nlp_service.return_value = mock_nl_response

    input_data = NaturalLanguageQueryInput(query="What is life?")
    response = client.post("/api/natural-query/", json=input_data.model_dump())

    assert response.status_code == 200
    # Pydantic models need careful comparison if they contain nested models
    # Convert response.json() to NLQueryResponse for easier comparison if needed,
    # or compare field by field if simple.
    response_json = response.json()
    assert response_json["status"] == "Success"
    assert response_json["detected_intent"] == "student_question"
    # Ensure service_response is correctly serialized and deserialized
    assert response_json["service_response"]["student_question"] == mock_sq_output.student_question
    assert response_json["service_response"]["llm_answer"] == mock_sq_output.llm_answer

    mock_nlp_service.assert_called_once_with(input_data)

@patch("backend_app.api.endpoints.process_natural_language_query", new_callable=AsyncMock)
def test_natural_query_clarification_needed(mock_nlp_service):
    mock_nl_response = NLQueryResponse(
        status="ClarificationNeeded",
        message="Missing topic for practice questions.",
        original_query="I want to practice."
    )
    mock_nlp_service.return_value = mock_nl_response

    input_data = NaturalLanguageQueryInput(query="I want to practice.")
    response = client.post("/api/natural-query/", json=input_data.model_dump())

    assert response.status_code == 200
    assert response.json()["status"] == "ClarificationNeeded"
    assert response.json()["message"] == "Missing topic for practice questions."
    mock_nlp_service.assert_called_once_with(input_data)

@patch("backend_app.api.endpoints.process_natural_language_query", new_callable=AsyncMock)
def test_natural_query_nlp_processing_error(mock_nlp_service):
    mock_nlp_service.side_effect = Exception("Major NLP meltdown") # Simulate unexpected error in NLP handler

    input_data = NaturalLanguageQueryInput(query="This will cause an error.")
    response = client.post("/api/natural-query/", json=input_data.model_dump())

    assert response.status_code == 500
    assert "An internal server error occurred" in response.json()["detail"]
    # The exact message might be "An internal server error occurred: Major NLP meltdown"
    # or generic depending on FastAPI's exception handling verbosity.
    mock_nlp_service.assert_called_once_with(input_data)

def test_natural_query_empty_query():
    # The endpoint itself has validation for empty query string
    response = client.post("/api/natural-query/", json={"query": "  "}) # Empty or whitespace query
    # This is caught by the endpoint's own validation `if not input_data.query or not input_data.query.strip():`
    assert response.status_code == 400
    assert response.json()["detail"] == "Query cannot be empty."

def test_natural_query_validation_error_no_query_field():
    # FastAPI validation for missing 'query' field altogether
    response = client.post("/api/natural-query/", json={})
    assert response.status_code == 422 # Unprocessable Entity
```
