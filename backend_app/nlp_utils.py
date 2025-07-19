import os
from typing import Dict, Any, Literal, Optional
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
    
async def validate_content_with_llm(
    source_of_truth: str,
    generated_content: str,
    validation_type: Literal['relevance', 'accuracy'],
    context_topic: Optional[str] = None
) -> bool:
    if not llm:
        print("VALIDATOR ERROR: LLM not available, validation skipped, returning False.")
        return False
    if not source_of_truth or not generated_content:
        return False

    print(f"--- Running Validator (Type: {validation_type}) ---")

    if validation_type == 'relevance':
        if not context_topic:
            print("VALIDATOR WARNING: Relevance check needs a context_topic. Skipping.")
            return True # Default to True if no topic is provided for relevance
        system_prompt = "你是一位严谨的逻辑分析师。你的唯一任务是判断一段文本是否与指定主题相关。请只回答'Yes'或'No'。"
        human_prompt = (
            f"已知主题是：'{context_topic}'。\n\n"
            f"请判断以下这段生成的内容是否与这个主题紧密相关？\n\n"
            f"--- 生成的内容 ---\n{generated_content}\n--- 内容结束 ---\n\n"
            f"判断结果 (只回答 'Yes' 或 'No'):"
        )
    elif validation_type == 'accuracy':
        system_prompt = "你是一位严谨的事实核查员。你的唯一任务是判断一个陈述是否与提供的原文信息一致且不矛盾。请只回答'Yes'或'No'。"
        human_prompt = (
            f"这是权威的原文信息：\n\n--- 原文 ---\n{source_of_truth}\n--- 原文结束 ---\n\n"
            f"请判断以下这个陈述，其内容是否完全基于上述原文，并且没有与原文相悖或添加原文没有的信息？\n\n"
            f"--- 待核查的陈述 ---\n{generated_content}\n--- 陈述结束 ---\n\n"
            f"判断结果 (只回答 'Yes' 或 'No'):"
        )
    else:
        return False
        
    try:
        validator_llm = ChatZhipuAI(model="glm-4", temperature=0.0) # Use a precise model for validation
        response = await validator_llm.ainvoke(f"{system_prompt}\n{human_prompt}")
        
        result_text = response.content.strip().lower()
        print(f"Validator LLM raw response: '{result_text}'")
        
        # Check for a clear "yes" signal. Be strict.
        if "yes" in result_text and "no" not in result_text:
            print("--- Validation PASSED ---")
            return True
        else:
            print("--- Validation FAILED ---")
            return False
            
    except Exception as e:
        print(f"VALIDATOR ERROR: An exception occurred during validation: {e}")
        return False