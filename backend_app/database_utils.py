import mysql.connector
import os
from datetime import datetime # To potentially use for created_at if not using DB default
import json # For storing concepts_list as JSON
from collections import Counter

def get_mysql_connection(db_name=None):
    """Establishes a connection to the MySQL server.
    Connects to a specific database if db_name is provided, otherwise connects to the server.
    """
    try:
        host = "localhost"
        user = "root"
        password = "Zws112535!"
        
        if not user or not password:
            print("Error: MYSQL_USER and MYSQL_PASSWORD environment variables must be set.")
            return None

        connection_params = {
            'host': host,
            'user': user,
            'password': password
        }
        if db_name:
            connection_params['database'] = db_name
        
        conn = mysql.connector.connect(**connection_params)
        # print(f"Successfully connected to MySQL (Database: {db_name if db_name else 'Server only'}).")
        return conn
    except mysql.connector.Error as err:
        print(f"Error connecting to MySQL: {err}")
        return None

def init_db():
    """Initializes the database and creates tables if they don't exist."""
    db_name = "Aiagent"
    
    conn_to_db = get_mysql_connection(db_name=db_name)
    if not conn_to_db:
        return

    cursor = conn_to_db.cursor()
    try:
        # Teachers, Teaching Plans, Assessments
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS teachers (
            teacher_id INT AUTO_INCREMENT PRIMARY KEY,
            teacher_name VARCHAR(255) UNIQUE NOT NULL
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
        """)
        print("Table 'teachers' ensured.")

        cursor.execute("""
        CREATE TABLE IF NOT EXISTS teaching_plans (
            id INT AUTO_INCREMENT PRIMARY KEY,
            teacher_id INT NULL,
            subject VARCHAR(255),
            title VARCHAR(255) NOT NULL,
            content TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (teacher_id) REFERENCES teachers(teacher_id) ON DELETE SET NULL
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
        """)
        print("Table 'teaching_plans' (with teacher_id) ensured.")

        cursor.execute("""
        CREATE TABLE IF NOT EXISTS assessments (
            id INT AUTO_INCREMENT PRIMARY KEY,
            teacher_id INT NULL,
            title VARCHAR(255) NOT NULL,
            content TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (teacher_id) REFERENCES teachers(teacher_id) ON DELETE SET NULL
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
        """)
        print("Table 'assessments' (with teacher_id) ensured.")

        # Students, Practice Questions Catalog, Practice Attempts
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS students (
            student_id INT AUTO_INCREMENT PRIMARY KEY,
            student_name VARCHAR(255) UNIQUE NOT NULL
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
        """)
        print("Table 'students' ensured.")

        cursor.execute("""
        CREATE TABLE IF NOT EXISTS practice_questions_catalog (
            catalog_id INT AUTO_INCREMENT PRIMARY KEY,
            question_text TEXT NOT NULL,
            question_type VARCHAR(50),
            model_answer TEXT,
            concepts_covered TEXT, 
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
        """)
        print("Table 'practice_questions_catalog' (with concepts_covered, no teacher_id) ensured.")
        # Remove teacher_id from practice_questions_catalog if it was added by an ALTER statement before
        # This is a bit tricky as simply dropping a column that might not exist can error.
        # A safer way is to check if it exists first.
        cursor.execute("SHOW COLUMNS FROM practice_questions_catalog LIKE 'teacher_id'")
        if cursor.fetchone():
            # Need to drop FK first if it exists
            try:
                cursor.execute("ALTER TABLE practice_questions_catalog DROP FOREIGN KEY fk_teacher_catalog")
                print("Foreign key 'fk_teacher_catalog' dropped from 'practice_questions_catalog'.")
            except mysql.connector.Error as fk_err:
                if fk_err.errno == 1091: # Can't DROP 'fk_teacher_catalog'; check that column/key exists
                    print("Foreign key 'fk_teacher_catalog' not found or already dropped.")
                else:
                    raise # Re-raise other errors
            cursor.execute("ALTER TABLE practice_questions_catalog DROP COLUMN teacher_id")
            print("Column 'teacher_id' dropped from 'practice_questions_catalog'.")
        else:
            print("Column 'teacher_id' does not exist in 'practice_questions_catalog', no action needed.")


        # Ensure old concept tables are dropped
        cursor.execute("DROP TABLE IF EXISTS practice_question_to_concept_linking;")
        print("Table 'practice_question_to_concept_linking' dropped if existed.")
        cursor.execute("DROP TABLE IF EXISTS question_concepts;")
        print("Table 'question_concepts' dropped if existed.")


        cursor.execute("""
        CREATE TABLE IF NOT EXISTS practice_attempts (
            attempt_id INT AUTO_INCREMENT PRIMARY KEY,
            student_id INT NOT NULL,
            catalog_id INT NOT NULL,
            student_answer TEXT,
            correctness_assessment VARCHAR(50), 
            llm_feedback TEXT,
            attempt_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (student_id) REFERENCES students(student_id) ON DELETE CASCADE,
            FOREIGN KEY (catalog_id) REFERENCES practice_questions_catalog(catalog_id) ON DELETE CASCADE
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
        """)
        print("Table 'practice_attempts' ensured.")

        cursor.execute("""
        CREATE TABLE IF NOT EXISTS student_assessment_answers (
            answer_id INT AUTO_INCREMENT PRIMARY KEY,
            assessment_id INT NOT NULL,
            question_identifier TEXT NOT NULL, 
            student_id INT NOT NULL,
            student_answer_text TEXT,
            llm_evaluation_feedback TEXT,
            llm_assessed_correctness VARCHAR(50), 
            submission_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (assessment_id) REFERENCES assessments(id) ON DELETE CASCADE,
            FOREIGN KEY (student_id) REFERENCES students(student_id) ON DELETE CASCADE
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
        """)
        print("Table 'student_assessment_answers' ensured.")
        
        conn_to_db.commit()
        print("Database initialization complete (schema finalized).")
    except mysql.connector.Error as err:
        print(f"Error during table creation/modification: {err}")
        conn_to_db.rollback()
    finally:
        cursor.close()
        conn_to_db.close()

def get_student_performance_for_assessment(db_conn, assessment_id: int) -> list:
    """
    Retrieves all student performance details for a specific assessment.

    Args:
        db_conn: Active MySQL database connection.
        assessment_id (int): The ID of the assessment.

    Returns:
        list: A list of dictionaries, where each dictionary represents a student's answer
              and evaluation details. Returns an empty list if no data or an error occurs.
    """
    if not db_conn:
        print("No database connection provided to get_student_performance_for_assessment.")
        return []

    results = []
    cursor = None  # Initialize cursor to None for finally block
    try:
        cursor = db_conn.cursor(dictionary=True)
        sql = """
            SELECT
                saa.answer_id,
                saa.assessment_id,
                saa.student_id,
                s.student_name,
                saa.question_identifier,
                saa.student_answer_text,
                saa.llm_evaluation_feedback,
                saa.llm_assessed_correctness,
                saa.submission_timestamp
            FROM
                student_assessment_answers saa
            LEFT JOIN
                students s ON saa.student_id = s.student_id
            WHERE
                saa.assessment_id = %s
            ORDER BY
                s.student_name, saa.submission_timestamp;
        """
        cursor.execute(sql, (assessment_id,))
        results = cursor.fetchall()
        if not results:
            print(f"No performance data found for assessment_id: {assessment_id}")
            return [] # Return empty list if no records found, which is not an error

    except mysql.connector.Error as err:
        print(f"Error retrieving student performance for assessment_id {assessment_id}: {err}")
        return [] # Return empty list on error
    except Exception as e:
        print(f"An unexpected error occurred in get_student_performance_for_assessment: {e}")
        return [] # Return empty list on unexpected error
    finally:
        if cursor:
            cursor.close()
            
    return results

def save_teaching_plan(db_conn,title, content, teacher_id=None):
    """Saves a teaching plan to the database."""
    if not db_conn:
        print("No database connection provided to save_teaching_plan.")
        return False
    
    cursor = db_conn.cursor()
    sql = "INSERT INTO teaching_plans (teacher_id, subject, title, content) VALUES (%s, %s, %s, %s)"
    val = (teacher_id,title, content)
    try:
        cursor.execute(sql, val)
        db_conn.commit()
        print(f"Teaching plan '{title}' (Teacher ID: {teacher_id}) saved successfully. Last inserted ID: {cursor.lastrowid}")
        return cursor.lastrowid # Return the ID of the inserted row
    except mysql.connector.Error as err:
        print(f"Error saving teaching plan '{title}' (Teacher ID: {teacher_id}): {err}")
        db_conn.rollback()
        return False
    finally:
        cursor.close()

def save_assessment(db_conn, title, content, teacher_id=None):
    """Saves an assessment to the database."""
    if not db_conn:
        print("No database connection provided to save_assessment.")
        return False
        
    cursor = db_conn.cursor()
    sql = "INSERT INTO assessments (teacher_id, title, content) VALUES (%s, %s, %s)"
    val = (teacher_id, title, content)
    try:
        cursor.execute(sql, val)
        db_conn.commit()
        print(f"Assessment '{title}' (Teacher ID: {teacher_id}) saved successfully. Last inserted ID: {cursor.lastrowid}")
        return cursor.lastrowid # Return the ID of the inserted row
    except mysql.connector.Error as err:
        print(f"Error saving assessment '{title}' (Teacher ID: {teacher_id}): {err}")
        db_conn.rollback()
        return False
    finally:
        cursor.close()

def get_or_create_teacher(db_conn, teacher_name):
    if not db_conn: return None
    cursor = db_conn.cursor()
    try:
        cursor.execute("SELECT teacher_id FROM teachers WHERE teacher_name = %s", (teacher_name,))
        result = cursor.fetchone()
        if result:
            return result[0] # teacher_id
        else:
            cursor.execute("INSERT INTO teachers (teacher_name) VALUES (%s)", (teacher_name,))
            db_conn.commit()
            print(f"Teacher '{teacher_name}' created with ID: {cursor.lastrowid}")
            return cursor.lastrowid
    except mysql.connector.Error as err:
        print(f"Error in get_or_create_teacher for '{teacher_name}': {err}")
        db_conn.rollback()
        return None
    finally:
        cursor.close()

def get_or_create_student(db_conn, student_name):
    if not db_conn: return None
    cursor = db_conn.cursor()
    try:
        # Check if student exists
        cursor.execute("SELECT student_id FROM students WHERE student_name = %s", (student_name,))
        result = cursor.fetchone()
        if result:
            return result[0] # student_id
        else:
            # Create student
            cursor.execute("INSERT INTO students (student_name) VALUES (%s)", (student_name,))
            db_conn.commit()
            print(f"Student '{student_name}' created with ID: {cursor.lastrowid}")
            return cursor.lastrowid
    except mysql.connector.Error as err:
        print(f"Error in get_or_create_student for '{student_name}': {err}")
        db_conn.rollback()
        return None
    finally:
        cursor.close()


def save_practice_question_to_catalog(db_conn, question_text, question_type, model_answer, concepts_list=None): # Removed teacher_id from params
    if not db_conn: return None
    cursor = db_conn.cursor()
    try:
        concepts_covered_str = None
        if concepts_list:
            valid_concepts = [str(c).strip() for c in concepts_list if c and str(c).strip()]
            if valid_concepts:
                concepts_covered_str = json.dumps(valid_concepts)

        sql_question = "INSERT INTO practice_questions_catalog (question_text, question_type, model_answer, concepts_covered) VALUES (%s, %s, %s, %s)"
        val_question = (question_text, question_type, model_answer, concepts_covered_str)
        cursor.execute(sql_question, val_question)
        db_conn.commit()
        catalog_id = cursor.lastrowid
        print(f"Practice question saved to catalog with ID: {catalog_id} (concepts: {concepts_covered_str})") # Updated print
        return catalog_id
    except mysql.connector.Error as err:
        print(f"Error in save_practice_question_to_catalog: {err}") # Updated print
        db_conn.rollback()
        return None
    finally:
        cursor.close()

def save_practice_attempt(db_conn, student_id, catalog_id, student_answer, correctness_assessment, llm_feedback):
    if not db_conn: return None
    cursor = db_conn.cursor()
    try:
        sql = ("INSERT INTO practice_attempts (student_id, catalog_id, student_answer, correctness_assessment, llm_feedback) "
               "VALUES (%s, %s, %s, %s, %s)")
        val = (student_id, catalog_id, student_answer, correctness_assessment, llm_feedback)
        cursor.execute(sql, val)
        db_conn.commit()
        print(f"Practice attempt for student {student_id} on question {catalog_id} saved.")
        return cursor.lastrowid
    except mysql.connector.Error as err:
        print(f"Error saving practice attempt: {err}")
        db_conn.rollback()
        return None
    finally:
        cursor.close()

def get_student_history_summary(db_conn, student_id, recent_attempts_limit=20, incorrect_focus_limit=5):
    """
    Retrieves and summarizes a student's practice history, focusing on concepts
    from incorrectly or partially correctly answered questions.

    Args:
        db_conn: Active MySQL database connection.
        student_id (int): The ID of the student.
        recent_attempts_limit (int): How many recent attempts to consider.
        incorrect_focus_limit (int): Max number of top struggled concepts to highlight.

    Returns:
        str: A summary string for the LLM, or a string indicating no specific issues found.
    """
    if not db_conn:
        return "Could not retrieve practice history due to database connection issue."

    cursor = db_conn.cursor(dictionary=True) # Use dictionary cursor for easier column access
    history_summary = ""
    
    try:
        # Fetch recent attempts, prioritizing incorrect/partially correct ones
        # We fetch more than incorrect_focus_limit initially to get a good sample of concepts
        sql = """
        SELECT pa.correctness_assessment, pqc.concepts_covered
        FROM practice_attempts pa
        JOIN practice_questions_catalog pqc ON pa.catalog_id = pqc.catalog_id
        WHERE pa.student_id = %s 
          AND pa.correctness_assessment IN ('Incorrect', 'Partially Correct')
        ORDER BY pa.attempt_timestamp DESC
        LIMIT %s;
        """
        cursor.execute(sql, (student_id, recent_attempts_limit))
        attempts = cursor.fetchall()

        if not attempts:
            return "No recent incorrect or partially correct answers found in history. Student is doing well or history is sparse!"

        struggled_concepts = []
        for attempt in attempts:
            if attempt['concepts_covered']:
                try:
                    # concepts_covered is stored as a JSON string list
                    concepts = json.loads(attempt['concepts_covered'])
                    if isinstance(concepts, list): # Ensure it's a list
                        struggled_concepts.extend(concepts)
                except json.JSONDecodeError:
                    print(f"Warning: Could not parse concepts_covered JSON: {attempt['concepts_covered']}")
        
        if not struggled_concepts:
            return "Recent incorrect answers did not have specific concepts tagged, or concepts were unparsable."

        concept_counts = Counter(struggled_concepts)
        most_common_struggles = concept_counts.most_common(incorrect_focus_limit)

        if not most_common_struggles:
            return "No specific recurring concepts identified in recent incorrect answers."

        summary_parts = ["Student has recently shown difficulty with questions covering these concepts:"]
        for concept, count in most_common_struggles:
            summary_parts.append(f"- '{concept}' (appeared in {count} recent incorrect/partially correct answers)")
        
        history_summary = " ".join(summary_parts)
        
        # Optional: Add info about correctly answered concepts if desired (more complex query)

    except mysql.connector.Error as err:
        print(f"Error retrieving student history: {err}")
        history_summary = "Could not retrieve detailed practice history due to a database error."
    except Exception as e:
        print(f"An unexpected error occurred in get_student_history_summary: {e}")
        history_summary = "An unexpected error occurred while summarizing practice history."
    finally:
        cursor.close()
        
    return history_summary if history_summary else "No specific areas of difficulty noted in recent history."

def get_assessment_question_stats(db_conn, assessment_id, question_identifier=None):
    """
    Retrieves performance statistics for questions in a given assessment.

    Args:
        db_conn: Active MySQL database connection.
        assessment_id (int): The ID of the assessment to analyze.
        question_identifier (str, optional): Specific question identifier to filter by.
                                                If None, stats for all questions in the assessment.

    Returns:
        list: A list of dictionaries, where each dict contains stats for a question.
              Example: [{'question_identifier': 'Q1', 'total_attempts': 10, 'correct': 5, ...}]
              Returns an empty list if no data or an error occurs.
    """
    if not db_conn:
        print("No database connection provided to get_assessment_question_stats.")
        return []

    cursor = db_conn.cursor(dictionary=True)
    results = []
    
    try:
        base_sql = """
            SELECT 
                question_identifier, 
                llm_assessed_correctness, 
                COUNT(*) as count
            FROM student_assessment_answers
            WHERE assessment_id = %s
        """
        params = [assessment_id]

        if question_identifier:
            base_sql += " AND question_identifier = %s"
            params.append(question_identifier)
        
        base_sql += " GROUP BY question_identifier, llm_assessed_correctness ORDER BY question_identifier, llm_assessed_correctness;"
        
        cursor.execute(base_sql, tuple(params))
        raw_stats = cursor.fetchall()

        if not raw_stats:
            print(f"No student answers found for assessment ID {assessment_id}" + (f" and question '{question_identifier}'." if question_identifier else "."))
            return []

        # Process raw_stats into a more structured list
        # Group by question_identifier
        stats_by_question = {}
        for row in raw_stats:
            qid = row['question_identifier']
            if qid not in stats_by_question:
                stats_by_question[qid] = {
                    'question_identifier': qid,
                    'total_attempts': 0,
                    'Correct': 0,
                    'Partially Correct': 0,
                    'Incorrect': 0,
                    'Not Evaluated': 0 # Or other statuses
                }
            status = row['llm_assessed_correctness']
            count = row['count']
            if status in stats_by_question[qid]:
                stats_by_question[qid][status] += count
            else: # Handle unexpected status values
                stats_by_question[qid][status] = count 
            stats_by_question[qid]['total_attempts'] += count
        
        results = list(stats_by_question.values())

    except mysql.connector.Error as err:
        print(f"Error retrieving assessment stats: {err}")
    finally:
        cursor.close()
            
    return results

def save_student_assessment_answer(db_conn, assessment_id, question_identifier, student_id, 
                                     student_answer_text, llm_evaluation_feedback, llm_assessed_correctness):
    """Saves a student's answer and its LLM evaluation to the database."""
    if not db_conn:
        print("No database connection provided to save_student_assessment_answer.")
        return False
    
    cursor = db_conn.cursor()
    sql = ("""
        INSERT INTO student_assessment_answers 
        (assessment_id, question_identifier, student_id, student_answer_text, 
         llm_evaluation_feedback, llm_assessed_correctness) 
        VALUES (%s, %s, %s, %s, %s, %s)
        """)
    val = (assessment_id, question_identifier, student_id, student_answer_text, 
           llm_evaluation_feedback, llm_assessed_correctness)
    try:
        cursor.execute(sql, val)
        db_conn.commit()
        print(f"Student answer for assessment {assessment_id}, question '{question_identifier}', student {student_id} saved. ID: {cursor.lastrowid}")
        return cursor.lastrowid
    except mysql.connector.Error as err: # Ensure mysql.connector is imported
        print(f"Error saving student assessment answer: {err}")
        db_conn.rollback()
        return False
    finally:
        cursor.close()