import sys
import os # Ensure os is imported
# API keys ZHIPUAI_API_KEY and DASHSCOPE_API_KEY should be set in the environment.
# Client libraries (ChatTongyi, ZhipuAIEmbeddings) are expected to pick them up.
MYSQL_DB_NAME = "Aiagent"
from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.embeddings import ZhipuAIEmbeddings # For RAG
from langchain_chroma import Chroma         # For RAG
from backend_app.database_utils import get_mysql_connection, save_assessment, get_or_create_teacher

def extract_keywords_with_llm(teaching_plan_content, llm_instance, max_keywords=10):
    keyword_extraction_prompt_text = (
        f"Please analyze the following teaching plan content and extract the {max_keywords} most important "
        "keywords or key concepts that would be suitable for querying a database to find relevant "
        "supplementary materials. Return these keywords as a single comma-separated string. "
        "For example: keyword1, keyword2, keyword3\n\n"
        "Teaching Plan Content:\n"
        "---BEGIN CONTENT---\n"
        "{plan_content}"
        "\n---END CONTENT---\n\n"
        "Comma-separated keywords:"
    )
    
    keyword_prompt_template = ChatPromptTemplate.from_messages([
        ("system", "You are an expert in analyzing educational content and extracting key terms for database queries."),
        ("human", keyword_extraction_prompt_text) 
    ])
    output_parser = StrOutputParser()
    keyword_extraction_chain = keyword_prompt_template | llm_instance | output_parser

    try:
        response = keyword_extraction_chain.invoke({"plan_content": teaching_plan_content})
        if response:
            # Remove potential surrounding quotes if LLM adds them
            response = response.strip().strip('"').strip("'")
            keywords = [keyword.strip() for keyword in response.split(',') if keyword.strip()]
            # Limit to max_keywords just in case LLM gives more, though prompt asks for a specific number
            keywords = keywords[:max_keywords] 
            if keywords:
                print(f"Extracted keywords: {keywords}")
                return keywords
            else:
                print("LLM did not return keywords in expected format.")
                return []
        else:
            print("LLM returned no response for keyword extraction.")
            return []
    except Exception as e:
        print(f"An error occurred during keyword extraction: {e}")
        return []

def perform_rag_search(keywords_list, embeddings_model_instance, vector_store_dir, top_k=10):
    if not keywords_list:
        print("No keywords provided for RAG search.")
        return []
    query = " ".join(keywords_list) # Combine keywords into a single query string
    try:
        vector_store = Chroma(
            persist_directory=vector_store_dir,
            embedding_function=embeddings_model_instance
        )
        retrieved_docs_with_scores = vector_store.similarity_search_with_score(query, k=top_k)
        retrieved_contents = []
        if retrieved_docs_with_scores:
            for doc, score in retrieved_docs_with_scores:
                retrieved_contents.append(doc.page_content)
        else:
            print("No relevant documents found in the knowledge base for the keywords.")
        return retrieved_contents
    except Exception as e:
        print(f"An error occurred during RAG search: {e}")
        print("Ensure the vector store exists at the specified directory and ZHIPUAI_API_KEY is valid for embeddings.")
        return []
    
def get_teaching_plan_content():
    print("Please paste the teaching plan content below. Press Ctrl+D (or Ctrl+Z then Enter on Windows) when you're done.")
    teaching_plan_content = sys.stdin.read()
    return teaching_plan_content

def get_question_preferences():
    preferences = {}
    available_types = ["multiple-choice", "short-answer", "programming"]
    print("\n--- Specify Desired Question Types and Counts ---")
    for type_name in available_types:
        include_type = input(f"Include {type_name} questions? (y/n, default n): ").lower()
        if include_type == 'y':
            while True:
                try:
                    num_questions_str = input(f"How many {type_name} questions? (enter a number): ")
                    num_questions = int(num_questions_str)
                    if num_questions > 0:
                        preferences[type_name] = num_questions
                        break
                    else:
                        print("Please enter a positive number.")
                except ValueError:
                    print("Invalid input. Please enter a number.")
    return preferences

def construct_assessment_prompt(teaching_plan_content, retrieved_rag_snippets, question_preferences):
    # System Message
    system_message = (
        "You are an expert in designing educational assessments for various subjects, "
        "including conceptual questions and practical programming exercises. "
        "Your task is to generate a set of assessment questions based on the provided teaching material "
        "and the specified question types and quantities. "
        "For multiple-choice questions, provide 3-4 options and indicate the correct answer. "
        "For short-answer questions, provide a concise model answer. "
        "For programming questions, provide a clear problem statement, any necessary boilerplate or context, "
        "and an example solution or key grading criteria/rubric. "
        "Ensure questions are relevant to the teaching content."
    )

    # Human Message Construction
    human_message_parts = []
    human_message_parts.append("### Provided Teaching Plan Content ###")
    human_message_parts.append(teaching_plan_content)

    # Add RAG snippets if available
    if retrieved_rag_snippets: # Check if the list is not empty
        human_message_parts.append("\n### Relevant Excerpts from Source Textbooks (for additional context) ###")
        for i, snippet in enumerate(retrieved_rag_snippets):
            human_message_parts.append(f"--- Snippet {i+1} ---\n{snippet}\n--- End Snippet {i+1} ---")
        human_message_parts.append("\nNote: These excerpts are for context. Generate questions PRIMARILY based on the Teaching Plan Content provided above, using these excerpts for clarification or detail where appropriate.")
    else:
        human_message_parts.append("\n(No additional textbook excerpts were retrieved or provided for context.)")
    
    human_message_parts.append("\n### Requested Assessment Questions ###")

    if not question_preferences:
        human_message_parts.append("Please generate a diverse set of 3-5 questions suitable for this content, including at least one programming question if applicable.")
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
        
        if not requested_specific_types: # Handles cases where preferences dict might exist but be empty or contain unhandled types
             human_message_parts.append("No specific recognized question types selected. Please generate a balanced mix of 3-5 questions based on the content.")


    human_message_parts.append("\n### Important Instructions ###")
    human_message_parts.append("- For each question, clearly indicate its type (e.g., 'Type: Multiple-Choice').")
    human_message_parts.append("- For multiple-choice, label options (A, B, C, D) and state the correct answer (e.g., 'Correct Answer: A').")
    human_message_parts.append("- For short-answer, provide a 'Model Answer:'.")
    human_message_parts.append("- For programming questions, provide 'Problem Statement:', 'Example Solution (Python/JavaScript or relevant language):', and if applicable, 'Key Grading Criteria:'.")
    human_message_parts.append("- Ensure all questions directly assess understanding or application of the provided teaching plan content.")

    human_message_content = "\n".join(human_message_parts)

    return {
        "system_message": system_message,
        "human_message": human_message_content
    }

def generate_assessment_with_llm(system_prompt, human_prompt):
    try:
        # Assuming DASHSCOPE_API_KEY is in the environment
        llm = ChatTongyi(temperature=0.7)
    except Exception as e:
        print(f"Error initializing LLM (ChatTongyi). Ensure DASHSCOPE_API_KEY is set correctly. Details: {e}")
        return None

    prompt_template = ChatPromptTemplate.from_messages([
        ("system", "{system_message_var}"),
        ("human", "{human_message_var}")
    ])
    output_parser = StrOutputParser()
    chain = prompt_template | llm | output_parser

    print("\nSending request to LLM for assessment generation...")
    try:
        response = chain.invoke({
            "system_message_var": system_prompt,
            "human_message_var": human_prompt
        })
        return response
    except Exception as e:
        print(f"An error occurred during LLM interaction: {e}")
        return None

def display_assessment_output(llm_response_str):
    print("\n--- Generated Assessment Content ---")
    if llm_response_str and llm_response_str.strip():
        print(llm_response_str)
    else:
        print("No assessment content was generated, or an error occurred.")

if __name__ == "__main__":

    print("--- Assessment Generation Tool ---")
    db_conn = get_mysql_connection(db_name=MYSQL_DB_NAME)
    teacher_name = input("Enter your teacher name: ").strip()
    teacher_id = get_or_create_teacher(db_conn, teacher_name)
    try:
        try:
            keyword_llm_instance = ChatTongyi(temperature=0.7) # Slightly lower temp for more factual keywords
        except Exception as e:
            print(f"Error initializing LLM for keyword extraction. Ensure API keys are set. Details: {e}")
            exit()

        # 1. Get teaching plan content
        teaching_content = get_teaching_plan_content()
        
        retrieved_rag_snippets = [] # Default to empty list
        if teaching_content.strip(): # Only proceed if there's content
            extracted_keywords = extract_keywords_with_llm(teaching_content, keyword_llm_instance)

            if extracted_keywords:
                try:
                    # Initialize embeddings model for RAG
                    embeddings = ZhipuAIEmbeddings() 
                    
                    # Define Chroma DB path (as used in other scripts)
                    CHROMA_PERSIST_DIR = 'chroma_db_zhipu' # Ensure this is consistent
                    if not os.path.exists(CHROMA_PERSIST_DIR):
                         print(f"Warning: Chroma DB directory '{CHROMA_PERSIST_DIR}' not found. RAG search will be skipped.")
                    else:
                         retrieved_rag_snippets = perform_rag_search(extracted_keywords, embeddings, CHROMA_PERSIST_DIR)
                except Exception as e:
                    print(f"Error during RAG processing (embeddings or search): {e}")
                    # Continue without RAG snippets if this part fails
            else:
                print("No keywords extracted. Skipping RAG search.")
        else:
            print("No teaching plan content provided. Skipping keyword extraction and RAG search.")
            # exit() was here before, but now we let construct_assessment_prompt handle empty teaching_content if necessary
            # If teaching_content is empty, construct_assessment_prompt will receive it as such.

        # 2. Get user preferences for questions
        question_prefs = get_question_preferences()

        # 3. Construct the prompt
        prompt_components = construct_assessment_prompt(teaching_content, retrieved_rag_snippets, question_prefs)
        system_msg = prompt_components["system_message"]
        human_msg = prompt_components["human_message"]

        # 4. Generate assessment with LLM
        assessment_output = generate_assessment_with_llm(system_msg, human_msg)

        # 5. Display the output
        display_assessment_output(assessment_output)

        if assessment_output and assessment_output.strip(): # Check if there's content to save
            save_choice = input("\nDo you want to save these assessment questions to the database? (y/n, default n): ").lower()
            if save_choice == 'y':
                print("\nSaving assessment questions...")
                title_for_db = input("Enter a title for these assessment questions: ").strip()

                if not title_for_db:
                    print("Title is required to save. Assessment not saved.")
                elif not db_conn or not db_conn.is_connected():
                    print("No database connection available. Assessment not saved.")
                else:
                    success = save_assessment(db_conn, title_for_db, assessment_output, teacher_id)
                    if success:
                        print(f"Assessment '{title_for_db}' saved successfully.")
                    else:
                        print("Failed to save assessment.")
            else:
                print("Assessment questions not saved.")
        else:
            print("No assessment content generated or content was empty. Nothing to save.")

    except KeyboardInterrupt:
        print("\nUser interrupted the process. Exiting.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
