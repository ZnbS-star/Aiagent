import os
from typing import Dict, Any
from langchain_community.chat_models import ChatZhipuAI 
from langchain_core.output_parsers import JsonOutputParser
llm = None
if os.environ.get("ZHIPUAI_API_KEY"):
    try:
        llm = ChatZhipuAI(model="glm-4",temperature=0.0) # Low temperature for deterministic parsing
        print("INFO: ChatZhipuAI LLM initialized successfully for nlp_utils.")
    except Exception as e:
        print(f"ERROR: Failed to initialize ChatZhipuAI LLM in nlp_utils: {e}. NLP features may be limited.")
        llm = None # Ensure llm is None if initialization fails
else:
    print("WARNING: ZHIPUAI_API_KEY not found in environment. NLP features in nlp_utils will be disabled.")

default_output_parser = JsonOutputParser()

async def parse_query_with_llm(
    full_prompt_text: str,
    output_parser: JsonOutputParser = default_output_parser
) -> Dict[str, Any]:
    if not llm:
        return {"error": "NLP service not available: LLM not configured or API key missing."}

    format_instructions = output_parser.get_format_instructions()
    
    prompt_with_format = f"{full_prompt_text}\n\n{format_instructions}"
    chain = llm | output_parser

    try:
        response_json = await chain.ainvoke(prompt_with_format)
        return response_json
    
    except Exception as e:
        error_type = type(e).__name__

        print(f"--- FAILED PROMPT --- \n{prompt_with_format}\n--- END FAILED PROMPT ---")
        print(f"Error during LLM query processing in parse_query_with_llm ({error_type}): {e}")
        
        return {"error": "An unexpected error occurred while processing the query with LLM.", 
                "details": f"{error_type}: {str(e)}"}