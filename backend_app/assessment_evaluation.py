import sys # For potential sys.exit()
import os # For environment variables
import mysql.connector # For mysql.connector.Error
from langchain_community.chat_models import ChatZhipuAI # For LLM evaluation
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import sys # For potential sys.exit()
import os # For environment variables
from backend_app.database_utils import get_mysql_connection, get_or_create_student, save_student_assessment_answer # For database interaction
import mysql.connector # For mysql.connector.Error
from langchain_community.chat_models import ChatZhipuAI # For LLM evaluation
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.embeddings import ZhipuAIEmbeddings # For RAG
from langchain_chroma import Chroma         # For RAG
# import re # Might be needed for more complex parsing in get_student_answers

def get_assessment_id_to_evaluate():
    """Prompts the teacher to enter the ID of the assessment to evaluate."""
    while True:
        try:
            assessment_id_str = input("Enter the ID of the assessment you want to evaluate: ").strip()
            if not assessment_id_str:
                print("Assessment ID cannot be empty.")
                continue
            assessment_id = int(assessment_id_str)
            if assessment_id > 0:
                return assessment_id
            else:
                print("Assessment ID must be a positive integer.")
        except ValueError:
            print("Invalid input. Please enter a valid integer ID.")

def get_student_name_for_evaluation():
    """Prompts the teacher for the student's name."""
    student_name = input("Enter the name of the student whose assessment you are evaluating: ").strip()
    if not student_name:
        print("Student name cannot be empty. Please try again.")
        # Recursive call or loop if strict validation is needed here
        # For now, just return it, caller might handle empty.
    return student_name

def get_student_answers_for_assessment_questions(assessment_content, student_name):
    """
    Parses assessment questions and prompts the teacher to input the student's answers.

    Args:
        assessment_content (str): The string content of the assessment, containing questions.
        student_name (str): The name of the student, for context in prompts.

    Returns:
        list: A list of dictionaries, each with 'question_identifier' and 'student_answer_text'.
              Returns empty list if no questions parsed or issues occur.
    """
    print(f"\n--- Entering Answers for {student_name} ---")
    student_answers_collected = []

    # Simplified parsing: Assumes questions are separated by "--- Question" or similar,
    # or just numbered lines. For a more robust solution, the assessment generation
    # would need to use clear markers like [START_QUESTION_X]...[END_QUESTION_X]
    # For now, let's assume questions are identifiable blocks.
    # We'll split by a generic "Question X" or numbered list pattern if possible,
    # or treat the whole content as a single block needing manual question identification.

    # This is a placeholder for more robust question parsing.
    # Let's assume for now the teacher will be shown blocks of text and will identify
    # the question they are providing an answer for.
    # A simple approach: Split by common question delimiters like "Q:", "Question:", "问题：".
    # This will be very rough.
    
    # For now, let's try to split by lines that start with "Type: " as that was in assessment_generator
    # This assumes the LLM generating assessments follows that format.
    
    # A very basic split by "Type: " which might indicate start of new question block
    # More robust parsing would use the [START_QUESTION] markers if they are reliably generated.
    # Given the current assessment_generator.py does not enforce those markers for individual questions
    # within the *assessment content block itself* (it uses them for the overall LLM output format),
    # we need a simpler strategy here or update assessment_generator's prompt.

    # Let's assume, for this phase, questions are manually identified by the teacher based on the printed content.
    # The teacher will be shown the full assessment content and then prompted for each answer.
    # We will need a way to identify which question the answer belongs to.
    
    print("\nAssessment Content:")
    print("-" * 30)
    print(assessment_content)
    print("-" * 30)
    print(f"\nPlease provide {student_name}'s answers for the questions in the assessment above.")

    num_questions_answered = 0
    while True:
        num_questions_answered += 1
        question_identifier = input(f"\nEnter identifier for question {num_questions_answered} (e.g., 'Q1', '1', 'Section A Q2'): ").strip()
        if not question_identifier:
            retry = input("Question identifier cannot be empty. Try again? (y/n): ").lower()
            if retry != 'y':
                num_questions_answered -=1 # Correct the count
                break
            num_questions_answered -=1 # Correct the count
            continue

        print(f"Enter {student_name}'s answer for question '{question_identifier}'. Press Ctrl+D (or Ctrl+Z then Enter on Windows) for multi-line input.")
        student_answer_text = sys.stdin.read().strip()
        
        student_answers_collected.append({
            "question_identifier": question_identifier,
            "student_answer_text": student_answer_text
        })

        another = input("Add another answer for a different question? (y/n, default y): ").lower()
        if another == 'n':
            break
            
    if not student_answers_collected:
        print("No answers were collected.")
    else:
        print(f"\nCollected {len(student_answers_collected)} answers for {student_name}.")
        
    return student_answers_collected

def get_assessment_content_by_id(db_conn, assessment_id):
    """
    Retrieves the content and other details of a specific assessment from the database.

    Args:
        db_conn: Active MySQL database connection.
        assessment_id (int): The ID of the assessment to retrieve.

    Returns:
        dict: A dictionary containing assessment details (id, title, content, teacher_id, created_at)
              or None if not found or an error occurs.
    """
    if not db_conn:
        print("No database connection provided to get_assessment_content_by_id.")
        return None
    
    cursor = db_conn.cursor(dictionary=True) # Use dictionary cursor
    try:
        cursor.execute("SELECT id, title, content, teacher_id, created_at FROM assessments WHERE id = %s", (assessment_id,))
        assessment_data = cursor.fetchone()
        if assessment_data:
            return assessment_data
        else:
            print(f"No assessment found with ID: {assessment_id}")
            return None
    except mysql.connector.Error as err:
        print(f"Error retrieving assessment with ID {assessment_id}: {err}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred in get_assessment_content_by_id: {e}")
        return None
    finally:
        cursor.close()

def get_rag_context_for_grading(question_text, embeddings_model_instance, vector_store_dir, top_k=2): # top_k=2 for concise context
    """
    Performs RAG search using question text to get context for grading.
    Args:
        question_text (str): The text of the question being graded.
        embeddings_model_instance: An initialized ZhipuAIEmbeddings instance.
        vector_store_dir (str): Directory of the Chroma vector store.
        top_k (int): Number of top documents to retrieve.
    Returns:
        list: List of page_content strings of retrieved documents, or empty list.
    """
    if not question_text:
        print("No question text provided for RAG search.")
        return []

    print(f"\nPerforming RAG search for grading context related to: '{question_text[:100]}...' (top_k={top_k})")

    try:
        if not os.path.exists(vector_store_dir):
            print(f"Warning: Chroma DB directory '{vector_store_dir}' not found. Cannot perform RAG search for grading.")
            return []

        vector_store = Chroma(
            persist_directory=vector_store_dir,
            embedding_function=embeddings_model_instance
        )
        
        retrieved_docs = vector_store.similarity_search(question_text, k=top_k) # Using similarity_search
        
        retrieved_contents = []
        if retrieved_docs:
            print(f"Retrieved {len(retrieved_docs)} RAG snippets for grading context.")
            for doc in retrieved_docs:
                retrieved_contents.append(doc.page_content)
        else:
            print("No relevant RAG snippets found for grading context.")
        return retrieved_contents
    except Exception as e:
        print(f"An error occurred during RAG search for grading: {e}")
        return []

def construct_evaluation_prompt(assessment_full_content, question_identifier, student_answer_text, rag_context_snippets=None):
    """
    Constructs system and human messages for LLM-based evaluation of a student's answer
    to a specific question within a larger assessment, potentially with RAG context.

    Args:
        assessment_full_content (str): The entire text content of the assessment.
        question_identifier (str): The identifier for the specific question being evaluated.
        student_answer_text (str): The student's submitted answer.
        rag_context_snippets (list, optional): List of relevant text snippets from RAG. Defaults to None.

    Returns:
        dict: A dictionary containing "system_message" and "human_message".
    """
    system_message = (
        "You are an expert AI teaching assistant and evaluator. Your role is to evaluate a student's answer "
        "to a specific question within the provided assessment content. "
        "First, locate the question identified by the 'Question Identifier' within the 'Full Assessment Content'. "
        "Then, find the corresponding model answer if provided within the assessment. "
        "Finally, evaluate the 'Student's Submitted Answer' against the identified question and its model answer.\n\n"
        "Provide your evaluation in two parts:\n"
        "1.  **Correctness Assessment:** Start your response *exactly* with 'Correctness: [status]', where [status] is one of: Correct, Partially Correct, Incorrect, or Cannot Determine (if the question or model answer is unclear from the context).\n"
        "2.  **Detailed Feedback:** Following the 'Correctness:' line, provide a detailed explanation for your assessment. "
        "Explain any errors, misconceptions, or omissions in the student's answer. "
        "Highlight positive aspects if any. Suggest improvements or clarifications. "
        "Be constructive and encouraging."
    )

    human_message_parts = []
    human_message_parts.append("### Full Assessment Content ###")
    human_message_parts.append(assessment_full_content)
    
    human_message_parts.append(f"\n### Question to Evaluate (Identifier) ###")
    human_message_parts.append(question_identifier)
    
    human_message_parts.append("\n### Student's Submitted Answer for the above question ###")
    human_message_parts.append(student_answer_text if student_answer_text.strip() else "(No answer provided by student for this question)")

    if rag_context_snippets:
        human_message_parts.append("\n### Relevant Context from Knowledge Base (for your reference during evaluation) ###")
        for i, snippet in enumerate(rag_context_snippets):
            human_message_parts.append(f"--- Context Snippet {i+1} ---\n{snippet}\n--- End Snippet {i+1} ---")
        human_message_parts.append("\nNote: Use these context snippets to verify factual accuracy, understand expected depth, or identify nuances in the student's answer where applicable.")
    else:
        human_message_parts.append("\n(No additional context from knowledge base was retrieved for this question.)")

    human_message_parts.append("\n### Your Evaluation Task ###")
    human_message_parts.append(
        "Based on the full assessment content, the specific question identified, the student's answer, and any provided context snippets, "
        "please provide your structured evaluation (Correctness Assessment line first, then Detailed Feedback)."
    )

    human_message_content = "\n\n".join(human_message_parts)

    return {
        "system_message": system_message,
        "human_message": human_message_content
    }

def get_llm_evaluation_for_answer(system_prompt, human_prompt, llm_api_key):
    """
    Sends the student's answer and context to an LLM for evaluation.

    Args:
        system_prompt (str): The system message for the LLM evaluator.
        human_prompt (str): The human message containing assessment, question, and student answer.

    Returns:
        str: The LLM's raw evaluation string, or None if an error occurs.
    """
    try:
        # Using ChatZhipuAI for evaluation. Temperature might be lower for more deterministic eval.
        # Assuming ChatZhipuAI defaults to reading ZHIPUAI_API_KEY from environment if api_key is not provided.
        llm = ChatZhipuAI(temperature=0.4)
    except Exception as e:
        print(f"Error initializing LLM (ChatZhipuAI) for evaluation. Ensure ZHIPUAI_API_KEY is set in environment. Details: {e}")
        return None

    prompt_template = ChatPromptTemplate.from_messages([
        ("system", "{system_message_var}"),
        ("human", "{human_message_var}")
    ])
    output_parser = StrOutputParser()
    chain = prompt_template | llm | output_parser

    print("\nRequesting LLM evaluation of the student's answer...")
    try:
        response = chain.invoke({
            "system_message_var": system_prompt,
            "human_message_var": human_prompt
        })
        return response
    except Exception as e:
        print(f"An error occurred during LLM evaluation: {e}")
        return None

def parse_llm_evaluation(llm_evaluation_str):
    """
    Parses the LLM's evaluation string to extract structured correctness and detailed feedback.

    Args:
        llm_evaluation_str (str): The raw string output from the LLM.

    Returns:
        dict: A dictionary with keys 'llm_assessed_correctness' and 'llm_evaluation_feedback'.
              Defaults to "Undetermined" and the original string if parsing fails.
    """
    assessed_correctness = "Undetermined" # Default status
    evaluation_feedback = llm_evaluation_str # Default to the full string

    if not llm_evaluation_str or not llm_evaluation_str.strip():
        return {
            "llm_assessed_correctness": assessed_correctness,
            "llm_evaluation_feedback": "No evaluation content received from LLM."
        }

    # Attempt to parse the "Correctness: [status]" line
    first_line_end_index = llm_evaluation_str.find('\n')
    first_line = llm_evaluation_str[:first_line_end_index if first_line_end_index != -1 else len(llm_evaluation_str)].strip()

    prefix = "Correctness: "
    if first_line.startswith(prefix):
        status = first_line[len(prefix):].strip()
        # Basic validation against common expected statuses. Can be expanded.
        # The prompt asks for: Correct, Partially Correct, Incorrect, or Cannot Determine
        valid_statuses = ["Correct", "Partially Correct", "Incorrect", "Cannot Determine"]
        if status in valid_statuses:
            assessed_correctness = status
            if first_line_end_index != -1:
                evaluation_feedback = llm_evaluation_str[first_line_end_index + 1:].strip()
            else: # Only one line was returned, and it was the status line
                evaluation_feedback = "" 
            print(f"Parsed LLM Correctness Assessment: {assessed_correctness}") # For debugging
        else:
            print(f"Warning: LLM returned an unexpected status '{status}'. Using raw feedback for details.")
            # Keep defaults: assessed_correctness="Undetermined", evaluation_feedback=llm_evaluation_str
    else:
        print("Warning: LLM evaluation output did not start with 'Correctness:'. Using raw feedback for details.")
        # Keep defaults

    return {
        "llm_assessed_correctness": assessed_correctness,
        "llm_evaluation_feedback": evaluation_feedback
    }

# if __name__ == '__main__':
#     # Test functions (requires manual input)
#     assessment_id = get_assessment_id_to_evaluate()
#     print(f"Assessment ID to evaluate: {assessment_id}")

#     student_name = get_student_name_for_evaluation()
#     print(f"Student name: {student_name}")

#     # Dummy assessment content for testing get_student_answers_for_assessment_questions
#     dummy_assessment_content = """
# Type: Multiple-Choice
# Question: What is the capital of France?
# A) London
# B) Berlin
# C) Paris
# D) Madrid
# Correct Answer: C

# Type: Short-Answer
# Question: Explain the concept of photosynthesis in one sentence.
# Model Answer: Photosynthesis is the process by which green plants use sunlight, water, and carbon dioxide to create their own food and release oxygen.
#     """
#     answers = get_student_answers_for_assessment_questions(dummy_assessment_content, student_name)
#     print("\nCollected Answers:")
#     for ans in answers:
#         print(f"  Q ID: {ans['question_identifier']}, Answer: '{ans['student_answer_text'][:50]}...'")

if __name__ == '__main__':
    print("--- Student Assessment Evaluation Tool ---")

    # API Key Setup: Ensure ZHIPUAI_API_KEY is set in the environment.
    # ChatZhipuAI and ZhipuAIEmbeddings are expected to pick it up automatically.
    if not os.environ.get("ZHIPUAI_API_KEY"):
        print("Critical Error: ZHIPUAI_API_KEY not found in environment. This key is required for LLM evaluation and RAG embeddings. Please set it and retry.")
        sys.exit(1)
    # Ensure DASHSCOPE_API_KEY is also checked if any DashScope clients were to be used. For now, only ZhipuAI is used.
    if not os.environ.get("DASHSCOPE_API_KEY"):
        print("Warning: DASHSCOPE_API_KEY not found in environment. This might be an issue if other LLM providers are used elsewhere or in future.")

    # MySQL Database connection details
    MYSQL_DB_NAME = os.environ.get("MYSQL_DB")
    if not MYSQL_DB_NAME:
        print("Error: MYSQL_DB environment variable not set. Exiting.")
        sys.exit(1)

    db_conn = None
    try:
        db_conn = get_mysql_connection(db_name=MYSQL_DB_NAME)
        if not db_conn:
            print("Failed to connect to the database. Exiting.")
            sys.exit(1)
        
        # Initialize Embeddings Model for RAG
        try:
            embeddings = ZhipuAIEmbeddings() # API key from os.environ
        except Exception as e:
            print(f"Error initializing ZhipuAIEmbeddings: {e}. RAG context for grading will be unavailable.")
            embeddings = None # Allow script to continue without RAG if embeddings fail

        CHROMA_PERSIST_DIR = 'chroma_db_zhipu' # Define Chroma DB path

        # 1. Get Assessment ID
        assessment_id = get_assessment_id_to_evaluate()
        
        # 2. Retrieve Assessment Content
        assessment_data = get_assessment_content_by_id(db_conn, assessment_id)
        if not assessment_data or not assessment_data.get("content"):
            print(f"Could not retrieve content for assessment ID {assessment_id}. Exiting.")
            sys.exit(1)
        
        assessment_content = assessment_data["content"]
        print(f"\nSuccessfully retrieved assessment: '{assessment_data.get('title', 'Untitled Assessment')}'")

        # 3. Get Student Name and ID
        student_name = get_student_name_for_evaluation()
        if not student_name:
            print("No student name provided. Exiting.")
            sys.exit(1)
        
        student_id = get_or_create_student(db_conn, student_name)
        if not student_id:
            print(f"Could not get or create student '{student_name}'. Exiting.")
            sys.exit(1)
        print(f"Evaluating for student: {student_name} (ID: {student_id})")

        # 4. Get Student's Answers for each question in the assessment
        student_answers_list = get_student_answers_for_assessment_questions(assessment_content, student_name)

        if not student_answers_list:
            print("No answers were provided by the teacher for evaluation. Exiting.")
            sys.exit(1)
            
        # 5. Loop through answers, evaluate, display, and save
        for answer_item in student_answers_list:
            question_id_str = answer_item["question_identifier"]
            student_ans_text = answer_item["student_answer_text"]

            print(f"\n--- Evaluating Answer for Question: {question_id_str} ---")

            # Get RAG context for the current question identifier
            rag_grading_snippets = []
            if embeddings: # Only attempt RAG if embeddings initialized
                # Using question_id_str (e.g., "Question 1") as proxy for question_text for RAG.
                # This is a simplification; ideally, the actual question text would be used here.
                rag_grading_snippets = get_rag_context_for_grading(
                    question_id_str, # Or ideally, the actual text of this question if parsed out
                    embeddings,
                    CHROMA_PERSIST_DIR
                )
            
            eval_prompt_components = construct_evaluation_prompt(
                assessment_content, 
                question_id_str, 
                student_ans_text,
                rag_context_snippets=rag_grading_snippets # Pass RAG snippets
            )
            
            # ZHIPUAI_API_KEY is no longer passed as an argument.
            # get_llm_evaluation_for_answer will use the environment variable.
            raw_llm_evaluation = get_llm_evaluation_for_answer(
                eval_prompt_components["system_message"],
                eval_prompt_components["human_message"]
            )

            parsed_evaluation = parse_llm_evaluation(raw_llm_evaluation)
            correctness = parsed_evaluation["llm_assessed_correctness"]
            feedback_details = parsed_evaluation["llm_evaluation_feedback"]

            print(f"\nLLM Correctness Assessment: {correctness}")
            print("LLM Detailed Feedback:")
            print(feedback_details if feedback_details else "(No detailed feedback provided or parsing issue)")

            save_student_assessment_answer(
                db_conn,
                assessment_id,
                question_id_str,
                student_id,
                student_ans_text,
                feedback_details, 
                correctness       
            )
            print("-" * 30)

        print("\n--- Evaluation Process Complete ---")

    except KeyboardInterrupt:
        print("\nUser interrupted the process. Exiting.")
    except Exception as e:
        print(f"An unexpected error occurred in the main evaluation flow: {e}")
    finally:
        if db_conn and db_conn.is_connected():
            db_conn.close()
            print("\nDatabase connection closed.")