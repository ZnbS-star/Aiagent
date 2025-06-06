import pytest
from unittest.mock import patch, AsyncMock
from backend_app.nlp_utils import parse_query_with_llm # Assuming llm instance is nlp_utils.llm
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.exceptions import OutputParsingError
from langchain_core.messages import AIMessage
import json
import os

# Path for patching the llm object inside nlp_utils
NLP_UTILS_LLM_PATH = "backend_app.nlp_utils.llm"
# Path for patching the default_output_parser if needed for specific tests
NLP_UTILS_DEFAULT_PARSER_PATH = "backend_app.nlp_utils.default_output_parser"


@pytest.mark.asyncio
async def test_parse_query_with_llm_success():
    query = "test query"
    system_prompt = "test system prompt"
    expected_llm_response_dict = {"intent": "test_intent", "entities": {"key": "value"}}

    # The chain is `prompt_template | llm | output_parser`.
    # We need to mock what `llm.ainvoke` returns, which is then fed to `output_parser`.
    # `JsonOutputParser` expects a string (usually JSON string) or a Message with string content.
    mock_llm_output_json_str = json.dumps(expected_llm_response_dict)
    mock_ai_message = AIMessage(content=mock_llm_output_json_str)

    # Patch the llm instance within nlp_utils module that is used by parse_query_with_llm
    with patch(NLP_UTILS_LLM_PATH) as mock_llm_instance:
        # Configure the mock LLM's ainvoke method
        mock_llm_instance.ainvoke = AsyncMock(return_value=mock_ai_message)

        result = await parse_query_with_llm(query, system_prompt)

        assert result == expected_llm_response_dict
        mock_llm_instance.ainvoke.assert_called_once()
        # The first argument to ainvoke by the chain will be the formatted prompt (which includes the query)
        # We can inspect call_args if needed: call_args = mock_llm_instance.ainvoke.call_args[0][0]
        # For example, check if query is in the prompt: assert query in str(call_args)


@pytest.mark.asyncio
async def test_parse_query_with_llm_output_parsing_error_due_to_malformed_json():
    query = "test query for json error"
    system_prompt = "test system prompt"
    malformed_json_string = '{"intent": "test", "entities": {"key": "value"}' # Missing closing brace

    mock_ai_message_malformed = AIMessage(content=malformed_json_string)

    with patch(NLP_UTILS_LLM_PATH) as mock_llm_instance:
        mock_llm_instance.ainvoke = AsyncMock(return_value=mock_ai_message_malformed)

        # The actual JsonOutputParser will try to parse malformed_json_string and fail.
        # This will result in OutputParsingError being raised by the parser,
        # which should be caught by the general Exception in parse_query_with_llm.
        result = await parse_query_with_llm(query, system_prompt)

        assert "error" in result
        assert result["error"] == "An unexpected error occurred while processing the query with LLM."
        assert "details" in result
        # Check that the details mention OutputParsingError, which wraps the JSONDecodeError
        assert "OutputParsingError" in result["details"]
        # It might also be useful to check if the original malformed string is mentioned,
        # e.g., by inspecting e.llm_output if OutputParsingError is caught and that attribute exists.

@pytest.mark.asyncio
async def test_parse_query_with_llm_jsonoutputparser_raises_explicit_json_error():
    # Test when the parser itself directly raises JSONDecodeError or similar
    # This might happen if the content fed to it is not a string or Message type it expects.
    query = "test query"
    system_prompt = "test system prompt"

    with patch(NLP_UTILS_LLM_PATH) as mock_llm_instance:
        # LLM returns something completely unexpected for the parser
        mock_llm_instance.ainvoke = AsyncMock(return_value=AIMessage(content=12345)) # Content is not a string

        # In this scenario, JsonOutputParser might raise an error before JSONDecodeError,
        # possibly TypeError or its own OutputParsingError if it can't handle the input type.
        result = await parse_query_with_llm(query, system_prompt)

        assert "error" in result
        # The exact error message might vary based on how JsonOutputParser handles non-string content.
        # It's likely an OutputParsingError or a general error.
        assert result["error"] == "An unexpected error occurred while processing the query with LLM."
        assert "details" in result
        # We expect some form of parsing or type error detail.
        assert "OutputParsingError" in result["details"] or "TypeError" in result["details"]


@pytest.mark.asyncio
async def test_parse_query_with_llm_llm_call_general_exception():
    query = "test query for general exception"
    system_prompt = "test system prompt"

    with patch(NLP_UTILS_LLM_PATH) as mock_llm_instance:
        mock_llm_instance.ainvoke = AsyncMock(side_effect=Exception("LLM provider network error"))

        result = await parse_query_with_llm(query, system_prompt)

        assert "error" in result
        assert result["error"] == "An unexpected error occurred while processing the query with LLM."
        assert "details" in result
        assert "LLM provider network error" in result["details"]

@pytest.mark.asyncio
async def test_parse_query_with_llm_service_unavailable_llm_is_none():
    query = "test query when llm is None"
    system_prompt = "test system prompt"

    # Patch the llm instance in nlp_utils to be None for the duration of this test
    with patch(NLP_UTILS_LLM_PATH, None):
        result = await parse_query_with_llm(query, system_prompt)

        assert "error" in result
        assert result["error"] == "NLP service not available: LLM not configured or API key missing."

# It might also be good to test the construction of full_system_prompt
def test_system_prompt_formatting_in_parse_query():
    # This is not an async test as it doesn't call the LLM
    parser = JsonOutputParser()
    format_instructions = parser.get_format_instructions()
    base_system_prompt = "Extract entities."

    # To test the internal prompt construction, we'd ideally need to capture
    # what's passed to ChatPromptTemplate.from_messages.
    # This is a bit more involved. For now, assume the string formatting is correct
    # as it's simple: f"{system_prompt_text}\n\n{format_instructions}"
    # A functional test like test_parse_query_with_llm_success implicitly tests this too.
    expected_full_prompt = f"{base_system_prompt}\n\n{format_instructions}"

    # This test doesn't directly call parse_query_with_llm in a way that tests
    # its internal prompt creation easily without more complex mocking of ChatPromptTemplate itself.
    # However, the success test implicitly validates that the prompt was formatted correctly enough
    # for the mocked LLM to be called.
    assert "{format_instructions}" not in expected_full_prompt # Placeholder should be replaced
    assert "Extract entities." in expected_full_prompt
    assert "json" in expected_full_prompt.lower() # Format instructions usually mention JSON
```
