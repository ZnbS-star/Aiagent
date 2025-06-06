import os
import sys
from typing import List

# Adjust path to import from root directory and database_utils
# This is a common way to handle imports from a parent directory in a sub-directory app
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from backend_app.models import StudentQuestionInput, StudentQuestionOutput
from backend_app.database_utils import get_mysql_connection, get_or_create_student,get_or_create_teacher # Assuming student ID might be used later
from backend_app.student_qa import (
    search_knowledge_base_for_answer, 
    construct_student_qa_prompt, 
    get_llm_response_to_student
)
from backend_app.models import (
    AssessmentInput, AssessmentOutput, 
    StudentAssessmentInput, StudentAssessmentEvaluationOutput,
    PracticeQuestionsInput, PracticeQuestionItem, PracticeQuestionsOutput,
    PracticeFeedbackInput, PracticeFeedbackOutput, StudentPerformanceDetail # Added StudentPerformanceDetail
)

from fastapi import HTTPException # Added for placeholder service

from backend_app.assessment_generator import ( # Used by generate_assessment_service
        extract_keywords_with_llm,
        perform_rag_search,
        construct_assessment_prompt,
        generate_assessment_with_llm
    )
from backend_app.assessment_evaluation import ( # Used by evaluate_student_assessment_answers_service
        get_assessment_content_by_id, 
        construct_evaluation_prompt,
        get_llm_evaluation_for_answer, 
        parse_llm_evaluation
    )
from backend_app.practice_assistant import ( # Used by generate_practice_questions_service
        search_knowledge_for_practice_topic,
        construct_practice_question_prompt,
        get_llm_practice_questions,
        parse_questions_and_answers,
        construct_feedback_prompt as pa_construct_feedback_prompt, # Alias to avoid name clash
        get_llm_feedback_on_answer as pa_get_llm_feedback_on_answer,
        parse_feedback_and_correctness as pa_parse_feedback_and_correctness
    )

from backend_app.database_utils import (
        save_student_assessment_answer, 
        get_student_history_summary, 
        save_practice_question_to_catalog,
        save_practice_attempt, # Added for feedback service
        get_student_performance_for_assessment # Added for new service
    )

# Embeddings and DB
from langchain_community.embeddings import ZhipuAIEmbeddings
from langchain_chroma import Chroma
# LLMs
from langchain_community.chat_models import ChatZhipuAI
from langchain_community.chat_models.tongyi import ChatTongyi # For assessment generation
# Prompts & Parsers
from langchain_core.messages import SystemMessage, HumanMessage


CHROMA_PERSIST_DIR = 'chroma_db_zhipu' # From student_qa.py

def _get_zhipuai_api_key():
    api_key = os.environ.get("ZHIPUAI_API_KEY")
    if api_key is None:
        print("Warning: ZHIPUAI_API_KEY not found in environment. Using default key for service layer.")
        # In a real app, this might raise an error or use a config service
    return api_key

# --- Service Functions ---

async def process_student_question_service(input_data: StudentQuestionInput) -> StudentQuestionOutput:
    """
    Service layer function to process a student's question using RAG and LLM.
    """
    print(f"SERVICE: Processing student question: '{input_data.question}'")
    api_key = _get_zhipuai_api_key()
    rag_snippets = []
    llm_answer = "Could not determine an answer."
    error_message = None
    try:

        try:
            embeddings = ZhipuAIEmbeddings() # Assumes API key is in env or handled by class
        except Exception as e:
            print(f"SERVICE ERROR: Failed to initialize embeddings model: {e}")
            return StudentQuestionOutput(
                student_question=input_data.question,
                rag_context=None,
                llm_answer="Error: Could not initialize embeddings model.",
                error_message=f"Failed to initialize embeddings model: {e}"
            )

        rag_snippets = search_knowledge_base_for_answer(
            student_question=input_data.question,
            embeddings_model_instance=embeddings,
            vector_store_dir=CHROMA_PERSIST_DIR,
            top_k=10 # Default or from input_data if added
        )
        print(f"SERVICE: RAG search retrieved {len(rag_snippets)} snippets.")

        # 3. Construct Prompt for LLM
        prompt_components = construct_student_qa_prompt(
            student_question=input_data.question,
            rag_snippets=rag_snippets
        )
        print("SERVICE: Prompt constructed.")

        # 4. Get LLM Response
        # get_llm_response_to_student expects the API key to be passed directly
        llm_response = get_llm_response_to_student(
            system_prompt=prompt_components["system_message"],
            human_prompt=prompt_components["human_message"], 
        )

        if llm_response:
            llm_answer = llm_response
            print("SERVICE: LLM response received.")
        else:
            llm_answer = "Failed to get a response from the LLM."
            error_message = "LLM did not provide an answer."
            print("SERVICE ERROR: LLM did not provide an answer.")
            
    except Exception as e:
        print(f"SERVICE ERROR: An unexpected error occurred: {e}")
        error_message = f"An unexpected error occurred: {str(e)}"
        # Ensure llm_answer reflects error if it happened before LLM call
        if llm_answer == "Could not determine an answer.": # Check if it's still the initial default
            llm_answer = "An error occurred while processing the question."

    return StudentQuestionOutput(
        student_question=input_data.question,
        rag_context=rag_snippets if rag_snippets else None,
        llm_answer=llm_answer,
        error_message=error_message
    )



def _get_dashscope_api_key():
    api_key = os.environ.get("DASHSCOPE_API_KEY")
    if api_key is None:
        print("Warning: DASHSCOPE_API_KEY not found in environment. Using default key for Dashscope/Tongyi services.")
    return api_key

async def generate_initial_teaching_plan_service(

    initial_outline: str, # Used for RAG query and as initial_human_task
    style_tone:str,
    output_structure:str
    # LLM and Embeddings will be initialized inside, using env vars for keys
) -> tuple[str , List[str] ]: # Returns (plan_content, rag_snippets_used) or (None, None)
    zhipuai_api_key = _get_zhipuai_api_key() # Uses the existing helper
    # A. RAG Search Logic
    retrieved_rag_snippets = []
    try:
        print(f"SERVICE: Performing RAG search for query: '{initial_outline[:100]}...'")
        if not os.path.exists(CHROMA_PERSIST_DIR):
            print(f"SERVICE WARNING: Chroma DB directory '{CHROMA_PERSIST_DIR}' not found. Proceeding without RAG context.")
        else:
            embeddings_for_rag = ZhipuAIEmbeddings() 
            vector_store = Chroma(
                persist_directory=CHROMA_PERSIST_DIR,
                embedding_function=embeddings_for_rag
            )
            rag_docs = vector_store.similarity_search(initial_outline, k=10)
            if rag_docs:
                retrieved_rag_snippets = [doc.page_content for doc in rag_docs]
                print(f"SERVICE: Retrieved {len(retrieved_rag_snippets)} snippets from RAG.")
            else:
                print("SERVICE: No relevant snippets found from RAG.")
    except Exception as e:
        print(f"SERVICE ERROR during RAG search: {e}. Proceeding without RAG context.")

    system_message = (
        f"你是一位经验丰富的教师，你的任务是根据提供的大纲和补充材料，撰写一份详细的教案初稿。需注重内容的清晰性、准确性，并全面覆盖要点。最后输出语言是中文"
    )
    
    human_message_parts = [
        f"教师提供的初始大纲是:\n{initial_outline}\n"
    ]
    if retrieved_rag_snippets:
        human_message_parts.append("考虑以下来自源材料的相关摘录，以获取更多背景或细节:")
        for i, snippet in enumerate(retrieved_rag_snippets):
            human_message_parts.append(f"--- Snippet {i+1} ---\n{snippet}\n--- End Snippet {i+1} ---")
    else:
        human_message_parts.append("(No supplementary materials from RAG were available or retrieved.)")
    human_message_parts.append(f"输出语言风格要求：{style_tone}")
    human_message_parts.append(f"输出结构要求:{output_structure}")
    human_message_content = "\n".join(human_message_parts)

    # C. LLM Call for Plan Generation
    try:
        print("SERVICE: Initializing LLM for teaching plan generation (ChatZhipuAI)...")
        # Using ZhipuAI (glm-4) as it's generally good for generation tasks.
        # API key is passed directly or picked from env by ChatZhipuAI
        llm_for_plan = ChatZhipuAI(model="glm-4", temperature=0.7, api_key=zhipuai_api_key) 

        messages = [
            SystemMessage(content=system_message),
            HumanMessage(content=human_message_content)
        ]
        
        print("SERVICE: Generating teaching plan via LLM...")
        ai_message = await llm_for_plan.ainvoke(messages) # Use await for async
        generated_plan_content = ai_message.content

        if generated_plan_content and generated_plan_content.strip():
            print("SERVICE: Teaching plan generated successfully.")
            return generated_plan_content, retrieved_rag_snippets
        else:
            print("SERVICE ERROR: LLM returned empty content for teaching plan.")
            return None, retrieved_rag_snippets 
    except Exception as e:
        print(f"SERVICE ERROR during LLM call for teaching plan generation: {e}")
        return None, retrieved_rag_snippets

async def generate_assessment_service(
    input_data: AssessmentInput,
) -> tuple[str , List[str] ]: # Returns (generated_assessment_content, keywords_for_rag_info)
    """
    Service to generate assessment questions based on teaching plan content,
    using keyword extraction, RAG, and an LLM.
    """
    print(f"SERVICE: Initiating assessment generation for teacher_id: {input_data.teacher_id or input_data.teacher_name}")

    zhipuai_api_key = _get_zhipuai_api_key() # For RAG embeddings
    dashscope_api_key = _get_dashscope_api_key() # For Tongyi LLM (keywords & assessment gen)

    # Ensure API keys are in os.environ for Langchain components that might expect it
    if "ZHIPUAI_API_KEY" not in os.environ and zhipuai_api_key:
        os.environ["ZHIPUAI_API_KEY"] = zhipuai_api_key
    if "DASHSCOPE_API_KEY" not in os.environ and dashscope_api_key:
        os.environ["DASHSCOPE_API_KEY"] = dashscope_api_key

    extracted_keywords = []
    retrieved_rag_snippets = []
    generated_assessment_content = None

    try:
        # 1. Keyword Extraction (using ChatTongyi as per assessment_generator.py)
        # assessment_generator.py initializes ChatTongyi inside its functions or main block.
        # Here, we need an instance or to call a helper.
        # Let's assume extract_keywords_with_llm can take an initialized LLM or initialize one.
        # For simplicity, let's initialize it here.
        print("SERVICE: Initializing LLM for keyword extraction (ChatTongyi)...")
        keyword_llm = ChatTongyi(temperature=0.5) # API key from env
        
        extracted_keywords = extract_keywords_with_llm(
            input_data.teaching_plan_content, 
            keyword_llm # Pass the instance
        )
        print(f"SERVICE: Extracted keywords: {extracted_keywords}")

        # 2. RAG Search (if keywords were extracted)
        if extracted_keywords:
            print("SERVICE: Initializing embeddings for RAG (ZhipuAIEmbeddings)...")
            embeddings_for_rag = ZhipuAIEmbeddings() # API key from env
            if os.path.exists(CHROMA_PERSIST_DIR):
                retrieved_rag_snippets = perform_rag_search(
                    extracted_keywords, 
                    embeddings_for_rag, 
                    CHROMA_PERSIST_DIR
                )
                print(f"SERVICE: RAG search retrieved {len(retrieved_rag_snippets)} snippets.")
            else:
                print(f"SERVICE WARNING: Chroma DB directory '{CHROMA_PERSIST_DIR}' not found. Proceeding without RAG.")
        
        # 3. Construct Prompt for Assessment Generation
        # assessment_generator.construct_assessment_prompt expects: 
        # (teaching_plan_content, retrieved_rag_snippets, question_preferences)
        assessment_prompt_components = construct_assessment_prompt(
            input_data.teaching_plan_content,
            retrieved_rag_snippets,
            input_data.question_preferences
        )
        print("SERVICE: Assessment prompt constructed.")

        # 4. Generate Assessment with LLM (using ChatTongyi as per assessment_generator.py)
        # assessment_generator.generate_assessment_with_llm expects system_prompt, human_prompt
        # It initializes ChatTongyi internally.
        generated_assessment_content = generate_assessment_with_llm(
             assessment_prompt_components["system_message"],
             assessment_prompt_components["human_message"]
        )

        if generated_assessment_content and generated_assessment_content.strip():
            print("SERVICE: Assessment content generated successfully.")
        else:
            print("SERVICE ERROR: LLM returned empty content for assessment.")
            generated_assessment_content = None # Ensure it's None if empty

    except Exception as e:
        print(f"SERVICE ERROR during assessment generation pipeline: {e}")
        # Return None for content, but keywords might still be useful for debugging/partial success
        return None, extracted_keywords 

    return generated_assessment_content, extracted_keywords

async def evaluate_student_assessment_answers_service(
    input_data: StudentAssessmentInput
) -> List[StudentAssessmentEvaluationOutput]:
    """
    Service to evaluate student's answers for a given assessment, save them, and return evaluations.
    """
    print(f"SERVICE: Initiating evaluation for student_id: {input_data.student_id or input_data.student_name} on assessment_id: {input_data.assessment_id}")

    zhipuai_api_key = _get_zhipuai_api_key() # For ChatZhipuAI used in assessment_evaluation.py
    # Ensure API key is in os.environ for Langchain components
    if "ZHIPUAI_API_KEY" not in os.environ and zhipuai_api_key:
        os.environ["ZHIPUAI_API_KEY"] = zhipuai_api_key

    results: List[StudentAssessmentEvaluationOutput] = []
    db_conn = None
    
    MYSQL_DB_NAME = os.environ.get("MYSQL_DB")
    try:
        db_conn = get_mysql_connection(db_name=MYSQL_DB_NAME)
        if not db_conn:
            raise Exception("Failed to connect to the database.")

        # 1. Get/Create Student ID
        actual_student_id = input_data.student_id
        if not actual_student_id:
            if not input_data.student_name: # Should be caught by Pydantic model if student_name is mandatory when id is not
                raise ValueError("Student name or ID is required.")
            actual_student_id = get_or_create_student(db_conn, input_data.student_name)
            if not actual_student_id:
                raise Exception(f"Failed to get or create student: {input_data.student_name}")
        
        # 2. Fetch Assessment Content
        # get_assessment_content_by_id is synchronous, if service is async, this might need await asyncio.to_thread
        assessment_data = get_assessment_content_by_id(db_conn, input_data.assessment_id)
        if not assessment_data or not assessment_data.get("content"):
            raise ValueError(f"Could not retrieve content for assessment ID {input_data.assessment_id}.")
        assessment_content = assessment_data["content"]

        # 3. Loop through answers, evaluate, display, and save
        for answer_item in input_data.answers:
            question_id_str = answer_item.question_identifier
            student_ans_text = answer_item.student_answer_text
            
            print(f"SERVICE: Evaluating answer for Q: {question_id_str}, Student: {actual_student_id}")

            # Construct prompt
            eval_prompt_components = construct_evaluation_prompt(
                assessment_content, question_id_str, student_ans_text
            )
            
            # Get LLM evaluation (this is a sync call from assessment_evaluation.py)
            # If this service is async, this should be:
            # raw_llm_evaluation = await get_llm_evaluation_for_answer(...)
            # For now, assuming get_llm_evaluation_for_answer can be called directly if it's not async
            # Or, we make this service function synchronous if its callees are sync.
            # Let's assume for now direct call works or we'd refactor the callee to be async.
            raw_llm_evaluation = get_llm_evaluation_for_answer(
                eval_prompt_components["system_message"],
                eval_prompt_components["human_message"],
                zhipuai_api_key 
            )

            parsed_eval = parse_llm_evaluation(raw_llm_evaluation)
            
            # Save the evaluated answer
            answer_db_id = save_student_assessment_answer(
                db_conn,
                input_data.assessment_id,
                question_id_str,
                actual_student_id,
                student_ans_text,
                parsed_eval["llm_evaluation_feedback"],
                parsed_eval["llm_assessed_correctness"]
            )
            
            results.append(StudentAssessmentEvaluationOutput(
                answer_id=answer_db_id if answer_db_id else None,
                assessment_id=input_data.assessment_id,
                question_identifier=question_id_str,
                student_id=actual_student_id,
                student_answer_text=student_ans_text,
                llm_assessed_correctness=parsed_eval["llm_assessed_correctness"],
                llm_evaluation_feedback=parsed_eval["llm_evaluation_feedback"],
                error_message=None if answer_db_id else "Failed to save this answer."
            ))
            
    except Exception as e:
        print(f"SERVICE ERROR in evaluate_student_assessment_answers_service: {e}")
        # Append a general error object if the whole process fails mid-way
        # This might duplicate if an error was already added for DB config.
        # A more robust error handling would check if results already contains an error.
        if not results or results[-1].error_message != "Database not configured.":
             results.append(StudentAssessmentEvaluationOutput(
                assessment_id=input_data.assessment_id, 
                question_identifier="Overall Error", 
                student_id=input_data.student_id or 0,
                student_answer_text="N/A",
                llm_assessed_correctness="Error",
                llm_evaluation_feedback=str(e),
                error_message=str(e)
            ))
    finally:
        if db_conn and db_conn.is_connected():
            db_conn.close()
            print("SERVICE: DB connection closed for evaluate_student_assessment_answers_service.")
            
    return results

async def generate_practice_questions_service(
    input_data: PracticeQuestionsInput
) -> PracticeQuestionsOutput:
    """
    Service to generate practice questions, incorporating RAG and student history,
    and save them to the catalog.
    """
    print(f"SERVICE: Generating practice questions for topic: {input_data.practice_topic}")
    
    zhipuai_api_key = _get_zhipuai_api_key()
    dashscope_api_key = _get_dashscope_api_key() # For ChatTongyi (question generation)

    # Ensure API keys are in os.environ for Langchain components
    if "ZHIPUAI_API_KEY" not in os.environ and zhipuai_api_key:
        os.environ["ZHIPUAI_API_KEY"] = zhipuai_api_key
    if "DASHSCOPE_API_KEY" not in os.environ and dashscope_api_key:
        os.environ["DASHSCOPE_API_KEY"] = dashscope_api_key

    db_conn = None
    student_id = input_data.student_id
    history_summary = "No specific student performance history provided."
    retrieved_rag_snippets = []
    generated_q_items: List[PracticeQuestionItem] = []
    
    MYSQL_DB_NAME = os.environ.get("MYSQL_DB")
    if not MYSQL_DB_NAME:
        return PracticeQuestionsOutput(generated_questions=[], error_message="Database not configured.")

    try:
        db_conn = get_mysql_connection(db_name=MYSQL_DB_NAME)
        if not db_conn:
            raise Exception("Failed to connect to the database.")
     

        # 3. Get Student History Summary (if student_id is available)
        if student_id:
            summary = get_student_history_summary(db_conn, student_id)
            if summary: history_summary = summary # Use default if summary is empty/None
            print(f"SERVICE: Student history summary: {history_summary}")
        
        # 4. RAG Search
        embeddings_for_rag = ZhipuAIEmbeddings()
        if os.path.exists(CHROMA_PERSIST_DIR):
            retrieved_rag_snippets = search_knowledge_for_practice_topic(
                input_data.practice_topic, embeddings_for_rag, CHROMA_PERSIST_DIR
            )
        else:
            print(f"SERVICE WARNING: Chroma DB directory '{CHROMA_PERSIST_DIR}' not found. No RAG context.")

        # 5. Construct Prompt for LLM
        prompt_components = construct_practice_question_prompt(
            input_data.practice_topic,
            input_data.question_preferences,
            student_history_summary=history_summary,
            retrieved_context_snippets=retrieved_rag_snippets
        )

        # 6. Generate Practice Questions (uses ChatTongyi via assessment_generator import)
        raw_generated_questions = get_llm_practice_questions(
            prompt_components["system_message"],
            prompt_components["human_message"]
            # API key for ChatTongyi is handled by its instantiation if in env
        )

        # 7. Parse and Save Questions to Catalog
        if raw_generated_questions:
            parsed_list = parse_questions_and_answers(raw_generated_questions)
            if parsed_list:
                for item in parsed_list:
                    concepts = [input_data.practice_topic] # Simple concept tagging
                    catalog_id = save_practice_question_to_catalog(
                        db_conn,
                        item["question"],
                        item["question_type"],
                        item["model_answer"],
                        concepts_list=concepts
                        # No teacher_id here as per final schema for practice_questions_catalog
                    )
                    if catalog_id:
                        generated_q_items.append(PracticeQuestionItem(
                            catalog_id=catalog_id,
                            question_type=item["question_type"],
                            question_text=item["question"],
                            model_answer=item["model_answer"]
                        ))
                    else:
                        print(f"SERVICE WARNING: Failed to save a generated question to catalog: {item['question'][:50]}")
                if not generated_q_items: # All questions failed to save
                     return PracticeQuestionsOutput(generated_questions=[], error_message="Generated questions but failed to save any to catalog.")
            else: # Parsing failed
                return PracticeQuestionsOutput(generated_questions=[], error_message="Failed to parse generated questions from LLM.")
        else: # LLM returned no questions
            return PracticeQuestionsOutput(generated_questions=[], error_message="LLM failed to generate practice questions.")
            
    except Exception as e:
        print(f"SERVICE ERROR in generate_practice_questions_service: {e}")
        return PracticeQuestionsOutput(generated_questions=[], error_message=f"An unexpected error occurred: {str(e)}")
    finally:
        if db_conn and db_conn.is_connected():
            db_conn.close()
            print("SERVICE: DB connection closed for generate_practice_questions_service.")
            
    return PracticeQuestionsOutput(generated_questions=generated_q_items)

async def get_practice_feedback_service(
    input_data: PracticeFeedbackInput
) -> PracticeFeedbackOutput:
    """
    Service to get LLM feedback on a student's practice answer and save the attempt.
    """
    print(f"SERVICE: Getting feedback for student {input_data.student_id} on catalog_id {input_data.catalog_id}")

    dashscope_api_key = _get_dashscope_api_key() # For ChatTongyi (feedback)
    if "DASHSCOPE_API_KEY" not in os.environ and dashscope_api_key:
        os.environ["DASHSCOPE_API_KEY"] = dashscope_api_key

    db_conn = None
    MYSQL_DB_NAME = os.environ.get("MYSQL_DB")
    if not MYSQL_DB_NAME:
        return PracticeFeedbackOutput(correctness_assessment="Error", detailed_feedback="Database not configured.", error_message="Database not configured.")

    try:
        # 1. Construct prompt for LLM feedback
        # pa_construct_feedback_prompt expects: 
        # (practice_question_text, model_answer_text, student_answer_text, question_type_str)
        feedback_prompt_components = pa_construct_feedback_prompt(
            input_data.question_text,
            input_data.model_answer,
            input_data.student_answer,
            input_data.question_type
        )

        # 2. Get LLM feedback (uses ChatTongyi via practice_assistant import)
        raw_llm_feedback = pa_get_llm_feedback_on_answer(
            feedback_prompt_components["system_message"],
            feedback_prompt_components["human_message"]
            # API key for ChatTongyi is handled by its instantiation if in env
        )

        if not raw_llm_feedback:
            return PracticeFeedbackOutput(correctness_assessment="Error", detailed_feedback="LLM failed to provide feedback.", error_message="LLM feedback generation failed.")

        # 3. Parse feedback
        parsed_feedback = pa_parse_feedback_and_correctness(raw_llm_feedback)
        
        # 4. Save practice attempt
        db_conn = get_mysql_connection(db_name=MYSQL_DB_NAME)
        if not db_conn:
            raise Exception("Failed to connect to database to save practice attempt.")

        attempt_id = save_practice_attempt(
            db_conn,
            input_data.student_id,
            input_data.catalog_id,
            input_data.student_answer,
            parsed_feedback["correctness"],
            parsed_feedback["detailed_feedback"]
        )

        if not attempt_id:
            return PracticeFeedbackOutput(
                correctness_assessment=parsed_feedback["correctness"],
                detailed_feedback=parsed_feedback["detailed_feedback"],
                error_message="Feedback generated but failed to save practice attempt."
            )

        return PracticeFeedbackOutput(
            attempt_id=attempt_id,
            correctness_assessment=parsed_feedback["correctness"],
            detailed_feedback=parsed_feedback["detailed_feedback"]
        )

    except Exception as e:
        print(f"SERVICE ERROR in get_practice_feedback_service: {e}")
        return PracticeFeedbackOutput(correctness_assessment="Error", detailed_feedback=str(e), error_message=f"An unexpected error occurred: {str(e)}")
    finally:
        if db_conn and db_conn.is_connected():
            db_conn.close()
            print("SERVICE: DB connection closed for get_practice_feedback_service.")

# --- Service to get student performance details for an assessment ---
async def get_student_assessment_performance_service(assessment_id: int) -> List[StudentPerformanceDetail]:
    """
    Service to retrieve all student performance details for a specific assessment.
    """
    print(f"SERVICE: Call received for get_student_assessment_performance_service with assessment_id: {assessment_id}")
    
    db_conn = None
    performance_details: List[StudentPerformanceDetail] = []
    
    MYSQL_DB_NAME = os.environ.get("MYSQL_DB")
    if not MYSQL_DB_NAME:
        print("SERVICE ERROR: MYSQL_DB environment variable not set. Cannot fetch performance details.")
        # Depending on desired behavior, could raise 500 or return empty with logged error.
        # Raising 500 as it's a configuration issue preventing service operation.
        raise HTTPException(status_code=500, detail="Database configuration error.")

    try:
        db_conn = get_mysql_connection(db_name=MYSQL_DB_NAME)
        if not db_conn:
            print("SERVICE ERROR: Failed to connect to the database.")
            raise HTTPException(status_code=500, detail="Failed to connect to the database.")

        raw_performance_data = get_student_performance_for_assessment(db_conn, assessment_id)

        if not raw_performance_data:
            # This isn't necessarily an error; it could be a valid assessment with no submissions yet.
            print(f"SERVICE: No performance data found for assessment_id: {assessment_id}. Returning empty list.")
            return []

        # Map dictionary results to StudentPerformanceDetail Pydantic models
        for row in raw_performance_data:
            # Pydantic will validate types. If a datetime object is not directly returned
            # by connector for submission_timestamp and is a string, it might need parsing.
            # Assuming the connector provides Python datetime objects for TIMESTAMP columns.
            try:
                performance_details.append(StudentPerformanceDetail(**row))
            except Exception as pydantic_err: # Catch potential Pydantic validation errors
                print(f"SERVICE ERROR: Pydantic validation error for row {row}: {pydantic_err}")
                # Decide how to handle: skip this row, or raise an error for the whole request.
                # For now, let's skip problematic rows and log.
                continue 
        
        print(f"SERVICE: Successfully retrieved and mapped {len(performance_details)} performance records for assessment_id: {assessment_id}")

    except HTTPException as he: # Re-raise HTTPExceptions from connection attempts
        raise he
    except Exception as e:
        print(f"SERVICE ERROR: An unexpected error occurred in get_student_assessment_performance_service: {e}")
        # Log the full error e for server-side debugging
        # For a teacher-facing endpoint, returning a 500 is appropriate for unexpected issues.
        raise HTTPException(status_code=500, detail=f"An internal server error occurred while fetching performance data: {str(e)}")
    finally:
        if db_conn and db_conn.is_connected():
            db_conn.close()
            print(f"SERVICE: DB connection closed for get_student_assessment_performance_service (assessment_id: {assessment_id}).")
            
    return performance_details
