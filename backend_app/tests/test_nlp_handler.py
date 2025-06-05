import pytest
from unittest.mock import AsyncMock, patch, MagicMock
import json
import os

from backend_app.models import (
    NaturalLanguageQueryInput, NLQueryResponse,
    StudentQuestionInput, StudentQuestionOutput,
    TeachingPlanInput, TeachingPlanOutput,
    AssessmentInput, AssessmentOutput, StudentAssessmentAnswerItem,
    StudentAssessmentInput, StudentAssessmentEvaluationOutput,
    PracticeQuestionsInput, PracticeQuestionItem, PracticeQuestionsOutput,
    PracticeFeedbackInput, PracticeFeedbackOutput
)
from backend_app.nlp_handler import process_natural_language_query
# Import service functions only for type hinting if needed, they will be mocked.

# Path to the nlp_handler's LLM object for patching
# Assuming nlp_handler.llm is the instance of ChatZhipuAI
NLP_HANDLER_LLM_CHAIN_PATH = "backend_app.nlp_handler.chain" # Patching the chain directly after it's constructed from prompt|llm|parser
NLP_HANDLER_LLM_OBJECT_PATH = "backend_app.nlp_handler.llm" # For checking if LLM is None

# --- Test Cases ---

@pytest.mark.asyncio
@patch(NLP_HANDLER_LLM_CHAIN_PATH) # Patching the combined chain
async def test_process_query_practice_questions_success(mock_llm_chain):
    # Arrange
    mock_llm_chain.ainvoke = AsyncMock(return_value={
        "intent": "generate_practice_questions",
        "entities": {"practice_topic": "Python lists", "student_id": "123", "question_preferences": {"multiple-choice": 1}}
    })

    expected_service_input = PracticeQuestionsInput(
        practice_topic="Python lists",
        student_id=123,
        question_preferences={"multiple-choice": 1}
    )
    mock_service_response = PracticeQuestionsOutput(
        generated_questions=[PracticeQuestionItem(question_type="mc", question_text="Q1?", model_answer="A1")]
    )

    with patch("backend_app.nlp_handler.generate_practice_questions_service", AsyncMock(return_value=mock_service_response)) as mock_service:
        nl_input = NaturalLanguageQueryInput(query="Generate practice questions on Python lists for student 123, one mcq.")

        # Act
        response = await process_natural_language_query(nl_input)

        # Assert
        mock_llm_chain.ainvoke.assert_called_once()
        mock_service.assert_called_once_with(expected_service_input)
        assert response.status == "Success"
        assert response.message == "Practice questions generated."
        assert response.detected_intent == "generate_practice_questions"
        assert response.extracted_entities == {"practice_topic": "Python lists", "student_id": "123", "question_preferences": {"multiple-choice": 1}}
        assert response.service_response == mock_service_response

@pytest.mark.asyncio
@patch(NLP_HANDLER_LLM_CHAIN_PATH)
async def test_process_query_student_question_success(mock_llm_chain):
    mock_llm_chain.ainvoke = AsyncMock(return_value={
        "intent": "student_question",
        "entities": {"question": "What is FastAPI?", "student_id": "456"}
    })
    expected_service_input = StudentQuestionInput(question="What is FastAPI?", student_id=456)
    mock_service_response = StudentQuestionOutput(student_question="What is FastAPI?", llm_answer="It's a framework.")

    with patch("backend_app.nlp_handler.process_student_question_service", AsyncMock(return_value=mock_service_response)) as mock_service:
        nl_input = NaturalLanguageQueryInput(query="What is FastAPI? My ID is 456")
        response = await process_natural_language_query(nl_input)

        mock_service.assert_called_once_with(expected_service_input)
        assert response.status == "Success"
        assert response.message == "Student question answered."
        assert response.service_response == mock_service_response

@pytest.mark.asyncio
@patch(NLP_HANDLER_LLM_CHAIN_PATH)
async def test_process_query_teaching_plan_success(mock_llm_chain):
    mock_llm_chain.ainvoke = AsyncMock(return_value={
        "intent": "generate_teaching_plan",
        "entities": {"subject": "History", "teaching_outline": "WW2", "teacher_name": "Prof X"}
    })
    # Service generate_initial_teaching_plan_service returns (generated_content, rag_snippets_used)
    mock_service_return = ("Plan content here", ["snippet1"])
    expected_service_response_model = TeachingPlanOutput(
        title="Plan for History", subject="History", generated_plan_content="Plan content here", teacher_id=None
    )

    with patch("backend_app.nlp_handler.generate_initial_teaching_plan_service", AsyncMock(return_value=mock_service_return)) as mock_service:
        nl_input = NaturalLanguageQueryInput(query="Create a teaching plan for Prof X on WW2 in History.")
        response = await process_natural_language_query(nl_input)

        # Assert service call arguments (order might matter if not kwargs)
        mock_service.assert_called_once()
        call_args = mock_service.call_args[1] # Get kwargs
        assert call_args['subject'] == "History"
        assert call_args['initial_outline'] == "WW2"
        assert call_args['teacher_name'] == "Prof X"

        assert response.status == "Success"
        assert response.service_response == expected_service_response_model

@pytest.mark.asyncio
@patch(NLP_HANDLER_LLM_CHAIN_PATH)
async def test_process_query_get_practice_feedback_success(mock_llm_chain):
    entities = {
        "student_id": "123", "catalog_id": "1", "question_text": "Q?",
        "model_answer": "A", "student_answer": "B", "question_type": "MCQ"
    }
    mock_llm_chain.ainvoke = AsyncMock(return_value={"intent": "get_practice_feedback", "entities": entities})
    expected_input = PracticeFeedbackInput(**{k: int(v) if k in ["student_id", "catalog_id"] else v for k, v in entities.items()})
    mock_response = PracticeFeedbackOutput(correctness_assessment="Incorrect", detailed_feedback="...")

    with patch("backend_app.nlp_handler.get_practice_feedback_service", AsyncMock(return_value=mock_response)) as mock_service:
        nl_input = NaturalLanguageQueryInput(query="feedback for student 123, question 1...")
        response = await process_natural_language_query(nl_input)
        mock_service.assert_called_once_with(expected_input)
        assert response.status == "Success"
        assert response.service_response == mock_response

@pytest.mark.asyncio
@patch(NLP_HANDLER_LLM_CHAIN_PATH)
async def test_process_query_generate_assessment_success(mock_llm_chain):
    entities = {"teaching_plan_content": "Photosynthesis details", "teacher_name": "Dr. Green"}
    mock_llm_chain.ainvoke = AsyncMock(return_value={"intent": "generate_assessment", "entities": entities})

    # service returns (generated_content, rag_keywords_list)
    mock_service_gen_content = "Assessment Content Here"
    mock_service_rag_keywords = ["photo", "synth"]

    expected_service_input = AssessmentInput(
        teaching_plan_content=entities["teaching_plan_content"],
        teacher_name=entities["teacher_name"],
        question_preferences={"multiple-choice": 2, "short-answer": 2, "programming": 0} # Default
    )
    expected_output_model = AssessmentOutput(
        generated_assessment_content=mock_service_gen_content,
        title=None, # Title is optional, not set by default in handler if not in entities
        teacher_id=None # teacher_id is optional, not set by default if not in entities
    )

    with patch("backend_app.nlp_handler.generate_assessment_service", AsyncMock(return_value=(mock_service_gen_content, mock_service_rag_keywords))) as mock_service:
        nl_input = NaturalLanguageQueryInput(query="generate assessment for Dr. Green on Photosynthesis details")
        response = await process_natural_language_query(nl_input)

        mock_service.assert_called_once_with(expected_service_input)
        assert response.status == "Success"
        assert response.service_response == expected_output_model

@pytest.mark.asyncio
@patch(NLP_HANDLER_LLM_CHAIN_PATH)
async def test_process_query_evaluate_assessment_success(mock_llm_chain):
    entities = {
        "student_name": "Alice", "assessment_id": "10",
        "answers": [{"question_identifier": "Q1", "student_answer_text": "Ans1"}]
    }
    mock_llm_chain.ainvoke = AsyncMock(return_value={"intent": "evaluate_assessment", "entities": entities})
    expected_input = StudentAssessmentInput(
        student_name="Alice", assessment_id=10, student_id=None, # student_id is optional if name is there
        answers=[StudentAssessmentAnswerItem(question_identifier="Q1", student_answer_text="Ans1")]
    )
    mock_response_item = StudentAssessmentEvaluationOutput(assessment_id=10, question_identifier="Q1", student_id=0, student_answer_text="Ans1", llm_assessed_correctness="Correct", llm_evaluation_feedback="Good") # student_id might be filled by service
    mock_service_eval_output = [mock_response_item]


    with patch("backend_app.nlp_handler.evaluate_student_assessment_answers_service", AsyncMock(return_value=mock_service_eval_output)) as mock_service:
        nl_input = NaturalLanguageQueryInput(query="evaluate Alice's answers for assessment 10...")
        response = await process_natural_language_query(nl_input)
        mock_service.assert_called_once_with(expected_input)
        assert response.status == "Success"
        assert response.service_response == mock_service_eval_output


@pytest.mark.asyncio
@patch(NLP_HANDLER_LLM_CHAIN_PATH)
async def test_process_query_clarification_needed_missing_entity(mock_llm_chain):
    mock_llm_chain.ainvoke = AsyncMock(return_value={
        "intent": "generate_practice_questions",
        "entities": {"student_id": "123"} # Missing practice_topic
    })
    nl_input = NaturalLanguageQueryInput(query="Generate questions for student 123")
    response = await process_natural_language_query(nl_input)
    assert response.status == "ClarificationNeeded"
    assert "Missing required entity: 'practice_topic'" in response.message

@pytest.mark.asyncio
@patch(NLP_HANDLER_LLM_CHAIN_PATH)
async def test_process_query_clarification_needed_invalid_type(mock_llm_chain):
    mock_llm_chain.ainvoke = AsyncMock(return_value={
        "intent": "generate_practice_questions",
        "entities": {"practice_topic": "Math", "student_id": "abc"} # Invalid student_id
    })
    nl_input = NaturalLanguageQueryInput(query="Practice Math for student abc")
    response = await process_natural_language_query(nl_input)
    assert response.status == "ClarificationNeeded"
    assert "Invalid student_id format: 'abc'" in response.message

@pytest.mark.asyncio
@patch(NLP_HANDLER_LLM_CHAIN_PATH)
async def test_process_query_clarification_needed_malformed_answers(mock_llm_chain):
    mock_llm_chain.ainvoke = AsyncMock(return_value={
        "intent": "evaluate_assessment",
        "entities": {
            "student_name": "Bob", "assessment_id": "1",
            "answers": [{"question_id": "Q1"}] # Missing student_answer_text, wrong key "question_id"
        }
    })
    nl_input = NaturalLanguageQueryInput(query="Evaluate Bob's answers for assessment 1")
    response = await process_natural_language_query(nl_input)
    assert response.status == "ClarificationNeeded"
    assert "Missing or invalid 'question_identifier'" in response.message # Or similar based on validation order

@pytest.mark.asyncio
@patch(NLP_HANDLER_LLM_CHAIN_PATH)
async def test_process_query_llm_parse_error(mock_llm_chain):
    # Simulate JsonOutputParser raising an error or chain returning non-parseable string
    # For JsonOutputParser, it usually expects the LLM to output a JSON string.
    # If the LLM output is not a string, or not valid JSON, it might raise.
    # Or if the chain's parser component fails.
    # Here, we make ainvoke itself raise JSONDecodeError, as if parser failed.
    mock_llm_chain.ainvoke = AsyncMock(side_effect=json.JSONDecodeError("mock error", "doc", 0))

    nl_input = NaturalLanguageQueryInput(query="Any query")
    response = await process_natural_language_query(nl_input)

    assert response.status == "LLMOutputParsingError"
    assert "Failed to parse the structured response" in response.message

@pytest.mark.asyncio
@patch(NLP_HANDLER_LLM_CHAIN_PATH)
async def test_process_query_unknown_intent(mock_llm_chain):
    mock_llm_chain.ainvoke = AsyncMock(return_value={"intent": "unknown_intent", "entities": {}})
    nl_input = NaturalLanguageQueryInput(query="Tell me a joke")
    response = await process_natural_language_query(nl_input)
    assert response.status == "UnknownIntent"
    assert "Could not understand the request intent" in response.message

@pytest.mark.asyncio
@patch(NLP_HANDLER_LLM_CHAIN_PATH)
async def test_process_query_service_error(mock_llm_chain):
    mock_llm_chain.ainvoke = AsyncMock(return_value={
        "intent": "generate_practice_questions",
        "entities": {"practice_topic": "Error Topic"}
    })
    mock_service_response = PracticeQuestionsOutput(generated_questions=[], error_message="Service exploded")

    with patch("backend_app.nlp_handler.generate_practice_questions_service", AsyncMock(return_value=mock_service_response)) as mock_service:
        nl_input = NaturalLanguageQueryInput(query="practice error topic")
        response = await process_natural_language_query(nl_input)

        assert response.status == "ServiceError"
        assert response.message == "Service exploded"
        assert response.service_response == mock_service_response


# Test for LLM not initialized (simpler patch)
@pytest.mark.asyncio
@patch(NLP_HANDLER_LLM_OBJECT_PATH, None) # Patch the llm object in nlp_handler to be None
async def test_process_query_llm_not_initialized():
    nl_input = NaturalLanguageQueryInput(query="Any query")
    response = await process_natural_language_query(nl_input)
    assert response.status == "Failure"
    assert "NLP processing service not available" in response.message

# To test the os.environ part, one might need to use monkeypatch fixture from pytest
# and carefully manage when nlp_handler is imported or reloaded.
# For example:
# @pytest.mark.asyncio
# def test_llm_initialization_failure_due_to_missing_key(monkeypatch):
#     monkeypatch.delenv("ZHIPUAI_API_KEY", raising=False)
#     # Need to reload nlp_handler or ensure it's imported *after* monkeypatching
#     # This can be tricky with how Python caches imports.
#     # A common way is to import within the test or use importlib.reload
#     import importlib
#     from backend_app import nlp_handler # Assuming nlp_handler is importable this way
#     importlib.reload(nlp_handler) # Reload to re-evaluate llm = None based on missing key

#     assert nlp_handler.llm is None # Check if llm object itself became None

#     # Then call the function that uses this llm object
#     nl_input = NaturalLanguageQueryInput(query="test query")
#     response = await nlp_handler.process_natural_language_query(nl_input)
#     assert response.status == "Failure"
#     assert "NLP processing service not available" in response.message
#     # Remember to restore env var if other tests need it, or run this in a separate process.
#     # monkeypatch usually handles cleanup automatically.
```
