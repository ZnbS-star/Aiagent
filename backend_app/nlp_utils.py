import os
from typing import Dict, Any
from langchain_community.chat_models import ChatZhipuAI # Or another LLM like ChatTongyi if preferred
from langchain_core.prompts import ChatPromptTemplate
# SystemMessagePromptTemplate and HumanMessagePromptTemplate are not strictly needed
# if using SystemMessage and HumanMessage directly with ChatPromptTemplate.from_messages
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.messages import SystemMessage, HumanMessage
import json # For potential fallback parsing or error handling

# Initialize LLM and Parser (module scope)
llm = None
# The ChatZhipuAI client will attempt to load ZHIPUAI_API_KEY from environment by default if not provided.
# This explicit check is to allow the system to be aware if NLP capabilities are disabled due to missing key.
if os.environ.get("ZHIPUAI_API_KEY"):
    try:
        llm = ChatZhipuAI(model="glm-4", temperature=0.1) # Low temperature for deterministic parsing
        print("INFO: ChatZhipuAI LLM initialized successfully for nlp_utils.")
    except Exception as e:
        print(f"ERROR: Failed to initialize ChatZhipuAI LLM in nlp_utils: {e}. NLP features may be limited.")
        llm = None # Ensure llm is None if initialization fails
else:
    print("WARNING: ZHIPUAI_API_KEY not found in environment. NLP features in nlp_utils will be disabled.")

default_output_parser = JsonOutputParser()

async def parse_query_with_llm(
    query: str,
    system_prompt_text: str,
    output_parser: JsonOutputParser = default_output_parser # Allow passing a different parser if needed
) -> Dict[str, Any]:
    """
    Processes a natural language query using an LLM to extract structured information
    based on a given system prompt and JSON output format.

    Args:
        query (str): The natural language query from the user.
        system_prompt_text (str): The system prompt text that instructs the LLM on how to
                                  process the query and what entities to extract. It should
                                  NOT include format_instructions placeholder initially.
        output_parser (JsonOutputParser, optional): An instance of JsonOutputParser.
                                                    Defaults to a module-level default instance.

    Returns:
        Dict[str, Any]: A dictionary containing the parsed JSON output from the LLM,
                        or an error dictionary if processing fails.
                        An error dictionary will have an "error" key with a message.
    """
    if not llm:
        return {"error": "NLP service not available: LLM not configured or API key missing."}

    format_instructions = output_parser.get_format_instructions()

    # Ensure system_prompt_text doesn't already contain the placeholder, then append.
    # This makes the function easier to use as user doesn't need to remember to add {format_instructions}.
    # A more robust way might be to check if "{format_instructions}" is already in system_prompt_text.
    # For now, we assume it's not and append it.
    full_system_prompt = f"{system_prompt_text}\n\n{format_instructions}"

    prompt_template = ChatPromptTemplate.from_messages([
        SystemMessage(content=full_system_prompt),
        HumanMessage(content="{user_query}") # Placeholder for the actual user query
    ])

    chain = prompt_template | llm | output_parser

    try:
        # The input to ainoke should match the input variables in the HumanMessagePromptTemplate,
        # which is "user_query" in this case.
        response_json = await chain.ainvoke({"user_query": query})
        return response_json  # This should be a dictionary parsed by JsonOutputParser

    except json.JSONDecodeError as e:
        # This exception might be raised by JsonOutputParser if the LLM output is not valid JSON.
        # However, JsonOutputParser often tries to recover or might wrap the error.
        # It's good to have it, but often the error might come as a langchain_core.exceptions.OutputParsingError.
        print(f"LLM output JSON parsing error in parse_query_with_llm: {e}")
        # Attempting to access e.text might not be standard for json.JSONDecodeError directly from parser
        # The JsonOutputParser might include the problematic text in its own exception type.
        return {"error": "Failed to parse LLM response as JSON.", "details": str(e)}

    except Exception as e:
        # This will catch other errors, including langchain_core.exceptions.OutputParsingError
        # if JsonOutputParser itself raises it due to malformed JSON from LLM.
        error_type = type(e).__name__
        print(f"Error during LLM query processing in parse_query_with_llm ({error_type}): {e}")

        # If it's an OutputParsingError from Langchain, it might contain the original LLM output.
        # Example: if isinstance(e, langchain_core.exceptions.OutputParsingError):
        # original_llm_output = e.llm_output if hasattr(e, 'llm_output') else "Not available"

        return {"error": "An unexpected error occurred while processing the query with LLM.",
                "details": f"{error_type}: {str(e)}"}

# Example usage (for testing purposes, not part of the module's regular execution path):
# if __name__ == '__main__':
#     import asyncio
#     async def main_test():
#         if not llm:
#             print("LLM not initialized, cannot run example.")
#             return

#         test_query = "I want to practice Algebra 1 on topics like linear equations and quadratics. My student ID is 789."
#         test_system_prompt = """
# You are an AI assistant helping to understand student requests for a learning platform.
# Extract the following information from the student's query:
# - "intent": Should be "generate_practice_questions".
# - "entities": A dictionary containing:
#   - "practice_topic" (string, required): The topic for practice.
#   - "student_id" (integer, optional): The ID of the student.
#   - "sub_topics" (list of strings, optional): Specific sub-topics mentioned.
# """
#         # Expected output (conceptual):
#         # {
#         #   "intent": "generate_practice_questions",
#         #   "entities": {
#         #     "practice_topic": "Algebra 1",
#         #     "student_id": 789,
#         #     "sub_topics": ["linear equations", "quadratics"]
#         #   }
#         # }

#         parsed_result = await parse_query_with_llm(test_query, test_system_prompt)
#         print("\n--- Example LLM Parsing Result ---")
#         print(json.dumps(parsed_result, indent=2))

#     asyncio.run(main_test())
```
