import os # For environment variables & path checks
os.environ["ZHIPUAI_API_KEY"] = "3573c5d116ae476eac14e7c61faffaab.LYbGSlVQ3qRNDcfR"
os.environ["DASHSCOPE_API_KEY"] ="sk-36c2e0675a6d4d1b9a528c4b79f9f400"
from langchain_community.embeddings import ZhipuAIEmbeddings # For RAG
from langchain_chroma import Chroma         # For RAG
from langchain_community.chat_models.tongyi import ChatTongyi # For LLM question generation
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from backend_app.database_utils import (
    get_mysql_connection, 
    get_or_create_student, 
    save_practice_question_to_catalog, 
    save_practice_attempt,
    get_student_history_summary
)
import sys # Added for sys.exit() on critical DB error
import re # Already present, just ensuring

def get_practice_topic():
    """
    Prompts the user to enter a topic or keywords for practice.
    """
    print("\nWhat topic or keywords would you like to practice today?")
    topic = input("Enter topic/keywords: ")
    return topic.strip()

def get_practice_question_preferences():
    """
    Asks the user for their preferences on practice question types and counts.
    Returns a dictionary like {"multiple-choice": 2, "programming": 1}.
    """
    preferences = {}
    available_types = ["multiple-choice", "short-answer", "programming"] # As specified

    print("\n--- Specify Desired Practice Question Types and Counts ---")
    for type_name in available_types:
        include_type = input(f"Include {type_name} practice questions? (y/n, default n): ").lower()
        if include_type == 'y':
            while True:
                try:
                    num_questions_str = input(f"How many {type_name} questions? (enter a number, e.g., 1-3): ")
                    num_questions = int(num_questions_str)
                    if num_questions > 0:
                        preferences[type_name] = num_questions
                        break
                    else:
                        print("Please enter a positive number.")
                except ValueError:
                    print("Invalid input. Please enter a number.")
    return preferences

def search_knowledge_for_practice_topic(topic_keywords, embeddings_model_instance, vector_store_dir, top_k=10):
    """
    Performs a RAG search using practice topic keywords against a Chroma vector store.

    Args:
        topic_keywords (str): Keywords describing the practice topic.
        embeddings_model_instance: An initialized ZhipuAIEmbeddings instance.
        vector_store_dir (str): The directory of the Chroma vector store.
        top_k (int): The number of top documents to retrieve for context.

    Returns:
        list: A list of strings, where each string is the page_content of a retrieved document.
              Returns an empty list if no documents are found or an error occurs.
    """
    if not topic_keywords:
        print("No topic keywords provided for RAG search.")
        return []

    print(f"\nSearching knowledge base for practice materials related to: '{topic_keywords}' (top_k={top_k})...")

    try:
        vector_store = Chroma(
            persist_directory=vector_store_dir,
            embedding_function=embeddings_model_instance
        )
        
        retrieved_docs_with_scores = vector_store.similarity_search_with_score(topic_keywords, k=top_k)
        
        retrieved_contents = []
        if retrieved_docs_with_scores:
            print(f"Retrieved {len(retrieved_docs_with_scores)} relevant snippets from the knowledge base for practice context.")
            for doc, score in retrieved_docs_with_scores:
                retrieved_contents.append(doc.page_content)
                # print(f"  Score: {score:.4f} - Snippet: {doc.page_content[:100]}...") # Optional for debugging
        else:
            print("No relevant snippets found in the knowledge base for this practice topic.")
        
        return retrieved_contents
    except Exception as e:
        print(f"An error occurred during RAG search for practice materials: {e}")
        print("Ensure the vector store is correctly set up and ZHIPUAI_API_KEY is valid for embeddings.")
        return []

def construct_practice_question_prompt(practice_topic, question_preferences, student_history_summary=None,retrieved_context_snippets=None):
    """
    Constructs system and human messages for LLM-based practice question generation.

    Args:
        practice_topic (str): The topic for which practice questions are needed.
        question_preferences (dict): Dictionary specifying types and counts of questions.
                                     e.g., {"multiple-choice": 3, "programming": 1}
        retrieved_context_snippets (list, optional): List of relevant text snippets from RAG.
                                                    Defaults to None.

    Returns:
        dict: A dictionary containing "system_message" and "human_message".
    """
    system_message = (
        "You are an expert in creating diverse educational practice questions for various topics. "
        "Your task is to generate a set of practice questions based on the provided topic, "
        "any relevant context snippets, and the specified question types and quantities.\n"
        "For multiple-choice questions, provide 3-4 plausible options and clearly indicate the correct answer.\n"
        "For short-answer questions, provide a concise model answer or key points to cover.\n"
        "For programming questions, provide a clear problem statement. If the question implies a specific language (e.g., Python, Java), use that. "
        "If not, you can assume Python or provide a language-agnostic pseudocode problem. "
        "Provide an example solution or key evaluation criteria for programming questions.\n"
        "Ensure questions are relevant to the topic and any provided context."
    )

    human_message_parts = []
    human_message_parts.append(f"### Practice Topic ###\n{practice_topic}")

    if retrieved_context_snippets:
        human_message_parts.append("\n### Relevant Context Snippets (for reference) ###")
        for i, snippet in enumerate(retrieved_context_snippets):
            human_message_parts.append(f"--- Snippet {i+1} ---\n{snippet}\n--- End Snippet {i+1} ---")
        human_message_parts.append("\nNote: Use these snippets to inform the questions, ensuring they are grounded in this context where appropriate.")
    else:
        human_message_parts.append("\n(No specific context snippets provided; generate general questions for the topic.)")
    
    # Add student history summary if available and meaningful
    if student_history_summary and \
       "no specific areas of difficulty" not in student_history_summary.lower() and \
       "no recent incorrect" not in student_history_summary.lower() and \
       "could not retrieve" not in student_history_summary.lower() and \
       "history is sparse" not in student_history_summary.lower() and \
       "did not have specific concepts tagged" not in student_history_summary.lower() and \
       "no specific recurring concepts" not in student_history_summary.lower() and \
       "unexpected error occurred while summarizing" not in student_history_summary.lower():
        human_message_parts.append("\n### Student's Recent Performance Summary (for personalization) ###")
        human_message_parts.append(student_history_summary)
        human_message_parts.append("\nBased on this summary, please try to generate some questions that target the student's identified areas of difficulty or related foundational concepts. However, also ensure variety in the questions and do not focus exclusively on these areas. Maintain the requested question types and quantities below.")
    else:
        human_message_parts.append("\n(No specific student performance history provided to guide question generation focus for this round, or student is performing well. Please generate questions based on the provided content and general preferences.)")

    human_message_parts.append("\n### Requested Practice Questions ###")
    if not question_preferences:
        human_message_parts.append("Please generate a diverse set of 2-3 practice questions suitable for the topic, including at least one conceptual and one practical (if applicable).")
    else:
        requested_specific_types = False
        for q_type, num in question_preferences.items():
            if q_type == "multiple-choice":
                human_message_parts.append(f"- Generate {num} multiple-choice question(s).")
                requested_specific_types = True
            elif q_type == "short-answer":
                human_message_parts.append(f"- Generate {num} short-answer question(s).")
                requested_specific_types = True
            elif q_type == "programming":
                human_message_parts.append(f"- Generate {num} programming question(s)/exercise(s).")
                requested_specific_types = True
        
        if not requested_specific_types:
             human_message_parts.append("No specific recognized question types selected. Please generate a balanced mix of 2-3 questions for the topic.")

    human_message_parts.append("\n### Output Formatting Instructions ###")
    human_message_parts.append(
        "Please format your response as follows, with each question and its model answer clearly demarcated:\n\n"
        "[START_QUESTION]\n"
        "Type: [Question Type: Multiple-Choice/Short-Answer/Programming]\n"
        "Question: [Your question content here]\n"
        "[If Multiple-Choice, add options like:\n"
        "A) Option 1\n"
        "B) Option 2\n"
        "C) Option 3\n"
        "D) Option 4]\n"
        "[END_QUESTION]\n"
        "[START_MODEL_ANSWER]\n"
        "[Your model answer or solution here. For Multiple-Choice, clearly state the correct option, e.g., Correct Answer: A)]\n"
        "[END_MODEL_ANSWER]\n\n"
        "Repeat this structure for each question generated. Ensure no other text outside these blocks for each Q&A pair."
    )
    # The following specific instructions are now covered by the block structure, but kept for reinforcement if needed, or can be removed.
    # human_message_parts.append("- For each question, clearly indicate its type (e.g., 'Type: Multiple-Choice').") # Covered
    # human_message_parts.append("- For multiple-choice, label options (A, B, C, D) and state the correct answer (e.g., 'Correct Answer: A').") # Covered
    human_message_parts.append("- For short-answer, provide a 'Model Answer:' or 'Key Points:' within the [START_MODEL_ANSWER] block.") # Adapted
    human_message_parts.append("- For programming questions, provide 'Problem Statement:' (as part of Question content), and 'Example Solution (Python or relevant language):' or 'Key Evaluation Criteria:' within the [START_MODEL_ANSWER] block.") # Adapted
    human_message_parts.append("- Ensure questions are directly relevant to the practice topic and informed by any provided context snippets.")

    human_message_content = "\n\n".join(human_message_parts)

    return {
        "system_message": system_message,
        "human_message": human_message_content
    }

def construct_feedback_prompt(practice_question_text, model_answer_text, student_answer_text, question_type_str):
    """
    Constructs system and human messages for generating feedback on a student's answer.

    Args:
        practice_question_text (str): The text of the practice question.
        model_answer_text (str): The model answer/solution.
        student_answer_text (str): The student's submitted answer.
        question_type_str (str): The type of the question (e.g., "Multiple-Choice", "Programming").

    Returns:
        dict: A dictionary containing "system_message" and "human_message".
    """

    system_message = (
        "You are an expert AI teaching assistant. Your role is to provide detailed, constructive, and encouraging "
        "feedback on a student's answer to a practice question. You will be given the original question, "
        "the model answer, and the student's answer. Analyze the student's response thoroughly."
    )

    human_message_parts = []
    human_message_parts.append(f"### Original Question (Type: {question_type_str}) ###")
    human_message_parts.append(practice_question_text)
    
    human_message_parts.append("\n### Model Answer/Solution ###")
    human_message_parts.append(model_answer_text)
    
    human_message_parts.append("\n### Student's Submitted Answer ###")
    human_message_parts.append(student_answer_text if student_answer_text.strip() else "(No answer provided by student)")

    human_message_parts.append("\n### Your Feedback Task ###")
    human_message_parts.append(
        "**IMPORTANT: Start your entire feedback response with a single line formatted *exactly* as follows: "
        "'Overall Assessment: [assessment]', where [assessment] is one of these exact strings: "
        "Correct, Partially Correct, Incorrect. Do not add any other text or explanation before this line.**"
    )
    human_message_parts.append("\nFollowing that initial assessment line, please evaluate the student's answer based on the model answer. Structure your subsequent feedback clearly. Specifically, please include the following sections in your feedback:")
    human_message_parts.append("1.  **Overall Assessment Details (Explanation):** After the initial assessment line, you can elaborate here on why it's correct, partially correct, or incorrect.")
    human_message_parts.append("2.  **Positive Aspects (if any):** Mention any parts of the student's answer that are on the right track, even if the overall answer isn't perfect.")
    human_message_parts.append("3.  **Error Analysis & Misconceptions:** If the answer is not fully correct, clearly explain any errors, misconceptions, or omissions. Pinpoint where the student went wrong (error localization).")
    human_message_parts.append("4.  **Suggestions for Correction/Improvement:** Provide specific, actionable advice on how the student can correct their answer or improve their understanding.")
    
    if question_type_str.lower() == "programming":
        human_message_parts.append("5.  **Programming Specifics (if applicable):**")
        human_message_parts.append("    - Analyze the student's code for correctness (bugs, logical errors).")
        human_message_parts.append("    - Comment on code style, efficiency, and adherence to best practices if relevant.")
        human_message_parts.append("    - Suggest specific improvements or alternative coding approaches if beneficial.")
    
    human_message_parts.append("\nRemember to maintain an encouraging and supportive tone throughout your feedback after the initial assessment line.")

    human_message_content = "\n".join(human_message_parts)

    return {
        "system_message": system_message,
        "human_message": human_message_content
    }

def parse_feedback_and_correctness(llm_feedback_str):
    """
    Parses the LLM feedback to extract correctness assessment and detailed feedback.

    Args:
        llm_feedback_str (str): The raw feedback string from the LLM.

    Returns:
        dict: A dictionary with "correctness" and "detailed_feedback".
              Returns default values if parsing fails.
    """
    correctness = "Undetermined" # Default if parsing fails
    detailed_feedback = llm_feedback_str # Default to full string

    if not llm_feedback_str or not llm_feedback_str.strip():
        return {"correctness": correctness, "detailed_feedback": "No feedback content received."}

    # Try to parse the specific first line: "Overall Assessment: [status]"
    first_line_end_index = llm_feedback_str.find('\n')
    first_line = llm_feedback_str[:first_line_end_index if first_line_end_index != -1 else len(llm_feedback_str)].strip()

    assessment_prefix = "Overall Assessment: "
    if first_line.startswith(assessment_prefix):
        status_str = first_line[len(assessment_prefix):].strip()
        # Validate against expected statuses
        valid_statuses = ["Correct", "Partially Correct", "Incorrect"]
        if status_str in valid_statuses:
            correctness = status_str
            # The rest of the string is detailed feedback
            if first_line_end_index != -1:
                detailed_feedback = llm_feedback_str[first_line_end_index + 1:].strip()
            else: # Should not happen if status_str was parsed, but as a fallback
                detailed_feedback = "" # No more content after the first line
            print(f"Parsed correctness: {correctness}") # For debugging
        else:
            print(f"Warning: Parsed status '{status_str}' is not one of {valid_statuses}. Using raw feedback.")
            # Keep default correctness="Undetermined" and full feedback string
    else:
        print("Warning: LLM feedback did not start with 'Overall Assessment:'. Using raw feedback.")
        # Keep default correctness="Undetermined" and full feedback string

    return {"correctness": correctness, "detailed_feedback": detailed_feedback}

def get_llm_feedback_on_answer(system_prompt, human_prompt):
    """
    Sends the practice question, model answer, and student answer to the LLM for feedback.
    Assumes DASHSCOPE_API_KEY is set in the environment for ChatTongyi.

    Args:
        system_prompt (str): The system message for the LLM.
        human_prompt (str): The human message for the LLM.

    Returns:
        str: The LLM's feedback, or None if an error occurs.
    """
    try:
        # Using ChatTongyi, consistent with get_llm_practice_questions
        # Temperature might be set slightly higher for more descriptive feedback if desired.
        llm = ChatTongyi(temperature=0.7) 
    except Exception as e:
        print(f"Error initializing LLM (ChatTongyi) for feedback. Ensure DASHSCOPE_API_KEY is set. Details: {e}")
        return None

    prompt_template = ChatPromptTemplate.from_messages([
        ("system", "{system_message_var}"),
        ("human", "{human_message_var}")
    ])
    output_parser = StrOutputParser()
    chain = prompt_template | llm | output_parser

    print("\nRequesting feedback on student's answer from LLM...")
    try:
        response = chain.invoke({
            "system_message_var": system_prompt,
            "human_message_var": human_prompt
        })
        return response
    except Exception as e:
        print(f"An error occurred during LLM interaction for feedback: {e}")
        return None

def get_llm_practice_questions(system_prompt, human_prompt):
    """
    Sends the prompt to the LLM to generate practice questions and returns the response.
    Assumes DASHSCOPE_API_KEY is set in the environment for ChatTongyi.

    Args:
        system_prompt (str): The system message for the LLM.
        human_prompt (str): The human message for the LLM.

    Returns:
        str: The LLM's response (generated practice questions), or None if an error occurs.
    """
    try:
        # Using ChatTongyi as it's used for question generation in assessment_generator.py
        # Ensure DASHSCOPE_API_KEY is set in the environment.
        llm = ChatTongyi(temperature=0.6) # Temperature can be tuned for creativity vs. precision
    except Exception as e:
        print(f"Error initializing LLM (ChatTongyi). Ensure DASHSCOPE_API_KEY is set. Details: {e}")
        return None

    prompt_template = ChatPromptTemplate.from_messages([
        ("system", "{system_message_var}"),
        ("human", "{human_message_var}")
    ])
    output_parser = StrOutputParser()
    chain = prompt_template | llm | output_parser

    print("\nRequesting practice questions from LLM...")
    try:
        response = chain.invoke({
            "system_message_var": system_prompt,
            "human_message_var": human_prompt
        })
        return response
    except Exception as e:
        print(f"An error occurred during LLM interaction for practice question generation: {e}")
        return None

def parse_questions_and_answers(llm_response_str):
    parsed_qa_pairs = []
    if not llm_response_str or not llm_response_str.strip():
        return parsed_qa_pairs

    # Regex to find blocks, more robust than simple splitting if LLM adds extra newlines
    # This regex captures the content between the start and end tags for question and answer.
    # It assumes that a question block is always followed by an answer block.
    qa_blocks = re.findall(r"\[START_QUESTION\](.*?)\[END_QUESTION\]\s*\[START_MODEL_ANSWER\](.*?)\[END_MODEL_ANSWER\]", llm_response_str, re.DOTALL)

    for q_content, a_content in qa_blocks:
        question_full = q_content.strip() # Includes Type line
        model_answer = a_content.strip()
        
        # Extract question type if it's consistently formatted
        q_type = "Unknown"
        question_text_only = question_full # Default to full content if Type not found
        
        type_match = re.search(r"Type:\s*(.*)", question_full)
        if type_match:
            q_type = type_match.group(1).strip()
            # Remove the type line from the question content for cleaner display
            # This replaces only the first occurrence of the "Type: ..." line
            question_text_only = re.sub(r"Type:\s*.*\n?", "", question_full, count=1).strip()


        parsed_qa_pairs.append({
            "question_type": q_type, # Store type if found
            "question": question_text_only, # Store cleaned question text
            "model_answer": model_answer
        })

    if not parsed_qa_pairs and llm_response_str.strip(): 
        print("Warning: Could not parse any Q&A pairs from LLM response using regex. LLM output might not conform to expected format. Displaying raw output as a fallback can be implemented if desired.")
        # Fallback: (Currently not adding raw output to list, returns empty as per instruction)
        pass

    return parsed_qa_pairs

if __name__ == "__main__":
    print("--- Interactive Practice Session Setup ---")

    MYSQL_DB_NAME = "Aiagent"
    if not MYSQL_DB_NAME:
        print("Error: MYSQL_DB environment variable not set. Cannot connect to the database.")
        sys.exit(1) # Use sys.exit for cleaner exit on critical errors

    CHROMA_PERSIST_DIR = 'chroma_db_zhipu' # Consistent path for ChromaDB
    db_conn = None # Initialize db_conn to ensure it's in scope for finally
    teacher_id = None # Initialize teacher_id

    
    try:
        # Establish DB connection
        db_conn = get_mysql_connection(db_name=MYSQL_DB_NAME)
        if not db_conn:
            print("Failed to connect to the database. Exiting.")
            sys.exit(1)

        # Get Student Information
        student_name = input("Enter your student name: ").strip()
        if not student_name:
            print("Student name is required. Exiting.")
            sys.exit(1)
        student_id = get_or_create_student(db_conn, student_name)
        if not student_id:
            print(f"Could not get or create student '{student_name}'. Exiting.")
            sys.exit(1)
        print(f"Practice session for student: {student_name} (ID: {student_id})")

        # 1. Get practice topic from user
        practice_topic = get_practice_topic()
        if not practice_topic:
            print("No practice topic provided. Exiting.")
            exit()

        # 2. Get student's practice history summary
        history_summary = get_student_history_summary(db_conn, student_id)
        print("\n--- Student Practice History Summary ---")
        print(history_summary if history_summary else "No history summary available or no specific issues found.")

        # 2. Get question preferences from user
        question_prefs = get_practice_question_preferences()

        # 3. Perform RAG search for context (optional, depends on topic)
        retrieved_snippets = []
        try:
            embeddings = ZhipuAIEmbeddings() # API key is picked from env
            if os.path.exists(CHROMA_PERSIST_DIR):
                retrieved_snippets = search_knowledge_for_practice_topic(practice_topic, embeddings, CHROMA_PERSIST_DIR)
            else:
                print(f"Knowledge base directory '{CHROMA_PERSIST_DIR}' not found. Proceeding without RAG context.")
        except Exception as e:
            print(f"Error during RAG setup or search: {e}. Proceeding without RAG context.")
            
        # 4. Construct the prompt for the LLM
        prompt_components = construct_practice_question_prompt(practice_topic, question_prefs, history_summary,retrieved_snippets)
        
        # 5. Get practice questions from LLM
        # get_llm_practice_questions expects the API key to be handled by ChatTongyi via environment or its own config
        generated_questions = get_llm_practice_questions(
            prompt_components["system_message"],
            prompt_components["human_message"]
        )

        # 6. Parse, Save, and Display generated practice questions, then start interactive session
        parsed_questions_with_ids = []
        if generated_questions:
            parsed_qa_from_llm = parse_questions_and_answers(generated_questions)
            if parsed_qa_from_llm:
                print(f"\nSaving {len(parsed_qa_from_llm)} generated questions to catalog...")
                for qa_item in parsed_qa_from_llm:
                    # For now, using the practice_topic as a single concept.
                    # This could be enhanced by LLM-assisted concept extraction from question text.
                    concepts = [practice_topic] if practice_topic else []
                    catalog_id = save_practice_question_to_catalog(
                        db_conn,
                        qa_item['question'],
                        qa_item['question_type'],
                        qa_item['model_answer'],
                        concepts_list=concepts
                    )
                    if catalog_id:
                        # Store catalog_id with the question for saving attempts
                        qa_item['catalog_id'] = catalog_id 
                        parsed_questions_with_ids.append(qa_item)
                    else:
                        print(f"Warning: Failed to save question to catalog: {qa_item['question'][:50]}...")
            else: # Parsing failed but we have raw output
                print("\n--- Raw LLM Output (Parsing Failed, Not Saved) ---")
                print(generated_questions)
        
        if not parsed_questions_with_ids:
            print("No practice questions were successfully generated or saved. Exiting practice session.")
        else:
            print(f"\n--- Starting Interactive Practice Session ({len(parsed_questions_with_ids)} questions) ---")
            for i, qa_pair in enumerate(parsed_questions_with_ids):
                print(f"\n--- Question {i+1}/{len(parsed_questions_with_ids)} (Type: {qa_pair['question_type']}) ---")
                print(qa_pair['question'])
                
                student_answer = input("\nYour answer: ")
                
                feedback_prompt_components = construct_feedback_prompt(
                    qa_pair['question'],
                    qa_pair['model_answer'],
                    student_answer,
                    qa_pair['question_type']
                )
                
                llm_feedback = get_llm_feedback_on_answer(
                    feedback_prompt_components["system_message"],
                    feedback_prompt_components["human_message"]
                )

                parsed_feedback_data = parse_feedback_and_correctness(llm_feedback)
                correctness_assessment = parsed_feedback_data["correctness"]
                detailed_llm_feedback = parsed_feedback_data["detailed_feedback"]

                print("\n--- AI Feedback ---")
                # Print the structured Overall Assessment first, then detailed feedback
                print(f"Overall Assessment: {correctness_assessment}")
                if detailed_llm_feedback: # Check if there is any detailed feedback after parsing
                    print("\n--- Detailed Feedback ---")
                    print(detailed_llm_feedback)
                elif llm_feedback and correctness_assessment == "Undetermined": 
                    # This means parsing failed and we have the raw feedback in detailed_llm_feedback
                    print(detailed_llm_feedback) # Print raw if parsing failed to get status
                elif not llm_feedback: # LLM call itself failed
                    print("Sorry, could not get feedback at this time.")
                
                # Save the practice attempt
                if qa_pair.get('catalog_id'): # Ensure we have a catalog_id
                    save_practice_attempt(
                        db_conn,
                        student_id,
                        qa_pair['catalog_id'],
                        student_answer,
                        correctness_assessment, # Use parsed correctness
                        detailed_llm_feedback if detailed_llm_feedback else (llm_feedback if llm_feedback else "No feedback generated.") # Save detailed or raw if detailed is empty
                    )
                else:
                    print("Warning: Could not save practice attempt as question was not cataloged.")

                show_model_answer = input("\nShow model answer/solution? (y/n, default y): ").lower()
                if show_model_answer != 'n':
                    print("\n--- Model Answer/Solution ---")
                    print(qa_pair['model_answer'])
                
                print("-" * 40)

                if i < len(parsed_questions_with_ids) - 1:
                    next_action = input("Press Enter for the next question, or type 's' to stop this set: ").lower()
                    if next_action == 's':
                        print("Stopping current set of practice questions.")
                        break 
                else:
                    print("End of this set of practice questions.")
            
            print("\nPractice session finished.")

    except KeyboardInterrupt:
        print("\nUser interrupted the process. Exiting.")
    except Exception as e:
        print(f"An unexpected error occurred in the main practice session flow: {e}")
    finally:
        if db_conn and db_conn.is_connected():
            db_conn.close()
            print("\nDatabase connection closed.")
