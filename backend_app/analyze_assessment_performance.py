# analyze_assessment_performance.py
import os
import sys # For sys.exit()
from database_utils import get_mysql_connection, get_assessment_question_stats
# Ensure mysql.connector is available if database_utils uses its Error class directly in exceptions caught here.

def display_stats(stats_list):
    if not stats_list:
        print("No performance statistics to display.")
        return

    print("\n--- Assessment Performance Statistics ---")
    for stats in stats_list:
        print(f"\nQuestion Identifier: {stats['question_identifier']}")
        print(f"  Total Attempts: {stats['total_attempts']}")
        correct_count = stats.get('Correct', 0)
        partially_correct_count = stats.get('Partially Correct', 0)
        incorrect_count = stats.get('Incorrect', 0)
        
        accuracy = 0
        if stats['total_attempts'] > 0:
            # Define how accuracy is calculated (e.g., only 'Correct' or 'Correct' + 'Partially Correct'/2)
            accuracy = (correct_count / stats['total_attempts']) * 100 
        
        print(f"  Correct Answers: {correct_count} ({accuracy:.2f}% accuracy - based on 'Correct' only)")
        if 'Partially Correct' in stats and stats['Partially Correct'] > 0 : # Only print if exists and > 0
             print(f"  Partially Correct Answers: {partially_correct_count}")
        if 'Incorrect' in stats and stats['Incorrect'] > 0: # Only print if exists and > 0
             print(f"  Incorrect Answers: {incorrect_count}")
        
        # Print any other statuses found
        for status, count in stats.items():
            if status not in ['question_identifier', 'total_attempts', 'Correct', 'Partially Correct', 'Incorrect'] and count > 0:
                print(f"  {status}: {count}")
    print("---------------------------------------")

if __name__ == "__main__":
    # API Key Setup (Prophylactic, as this script primarily uses DB)
    # Ensuring environment is consistent for any underlying library expectations.
    if "MYSQL_HOST" not in os.environ or \
       "MYSQL_USER" not in os.environ or \
       "MYSQL_PASSWORD" not in os.environ or \
       "MYSQL_DB" not in os.environ:
        print("Error: MySQL environment variables (MYSQL_HOST, MYSQL_USER, MYSQL_PASSWORD, MYSQL_DB) not set.")
        print("Please set them before running the script.")
        sys.exit(1)
    
    # These are not directly used by this script but are part of the project's common setup.
    # Ensuring environment is consistent for any underlying library expectations.
    # It's expected that DASHSCOPE_API_KEY and ZHIPUAI_API_KEY are set in the environment
    # if any imported modules or future functionalities require them.
    if "DASHSCOPE_API_KEY" not in os.environ:
        print("Warning: DASHSCOPE_API_KEY not found in environment. This might be an issue if needed by imports.")
    if "ZHIPUAI_API_KEY" not in os.environ:
        print("Warning: ZHIPUAI_API_KEY not found in environment. This might be an issue if needed by imports.")
    db_conn = None

    print("--- Assessment Performance Analyzer ---")
    try:
        db_name = os.environ.get("MYSQL_DB")
        db_conn = get_mysql_connection(db_name=db_name)
        if not db_conn:
            print("Failed to connect to the database. Exiting.")
            sys.exit(1)

        assessment_id_str = input("Enter the Assessment ID to analyze: ").strip()
        if not assessment_id_str.isdigit():
            print("Invalid Assessment ID. Please enter a number.")
            sys.exit(1)
        assessment_id = int(assessment_id_str)

        question_identifier_input = input("Enter specific Question Identifier (e.g., 'Question 1') (optional, press Enter to skip): ").strip()
        question_identifier = question_identifier_input if question_identifier_input else None
        
        stats_data = get_assessment_question_stats(db_conn, assessment_id, question_identifier)
        display_stats(stats_data)

    except KeyboardInterrupt:
        print("\nUser interrupted. Exiting.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    finally:
        if db_conn and db_conn.is_connected():
            db_conn.close()
            print("\nDatabase connection closed.")

