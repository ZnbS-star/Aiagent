import os 
import re
from typing import Any, Dict, List, Optional
import mysql.connector
from mysql.connector import Error 
import json 
from collections import Counter

def get_mysql_connection(db_name=None):
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
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS activity_log (
        log_id INT AUTO_INCREMENT PRIMARY KEY,
        user_id INT NOT NULL,
        user_role VARCHAR(20) NOT NULL COMMENT 'e.g., teacher, student, admin',
        activity_type VARCHAR(255) NOT NULL COMMENT 'e.g., GENERATE_TEACHING_PLAN, SUBMIT_PRACTICE_ANSWER',
        details JSON NULL COMMENT 'Optional details, e.g., {"plan_id": 123}',
        activity_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
        """)
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

        cursor.execute("SHOW COLUMNS FROM practice_questions_catalog LIKE 'teacher_id'")
        if cursor.fetchone():

            try:
                cursor.execute("ALTER TABLE practice_questions_catalog DROP FOREIGN KEY fk_teacher_catalog")
                print("Foreign key 'fk_teacher_catalog' dropped from 'practice_questions_catalog'.")
            except mysql.connector.Error as fk_err:
                if fk_err.errno == 1091: 
                    print("Foreign key 'fk_teacher_catalog' not found or already dropped.")
                else:
                    raise 
            cursor.execute("ALTER TABLE practice_questions_catalog DROP COLUMN teacher_id")
            print("Column 'teacher_id' dropped from 'practice_questions_catalog'.")
        else:
            print("Column 'teacher_id' does not exist in 'practice_questions_catalog', no action needed.")


        
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

    if not db_conn:
        print("No database connection provided to get_student_performance_for_assessment.")
        return []

    results = []
    cursor = None  
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
            return [] 

    except mysql.connector.Error as err:
        print(f"Error retrieving student performance for assessment_id {assessment_id}: {err}")
        return [] 
    except Exception as e:
        print(f"An unexpected error occurred in get_student_performance_for_assessment: {e}")
        return [] 
    finally:
        if cursor:
            cursor.close()
            
    return results

def save_teaching_plan(db_conn, title, content, teacher_id=None, subject=None):
    """Saves a teaching plan to the database."""
    if not db_conn:
        print("No database connection provided to save_teaching_plan.")
        return None
    
    cursor = db_conn.cursor()
    # <--- 修改SQL和参数
    sql = "INSERT INTO teaching_plans (teacher_id, title, content, subject) VALUES (%s, %s, %s, %s)"
    val = (teacher_id, title, content, subject)
    try:
        cursor.execute(sql, val)
        db_conn.commit()
        print(f"Teaching plan '{title}' (Subject: {subject}, Teacher ID: {teacher_id}) saved. ID: {cursor.lastrowid}")
        return cursor.lastrowid
    except mysql.connector.Error as err:
        print(f"Error saving teaching plan '{title}': {err}")
        db_conn.rollback()
        return None
    finally:
        cursor.close()

# --- 4. 修改 save_assessment 函数 (同时完成问题/答案分离) ---
def save_assessment(db_conn, title, questions_text, answers_text, teacher_id=None, subject=None):
    """Saves an assessment to the database with questions and answers separated."""
    if not db_conn:
        print("No database connection provided to save_assessment.")
        return None
        
    cursor = db_conn.cursor()
    # <--- 修改SQL和参数
    sql = "INSERT INTO assessments (teacher_id, title, questions_text, answers_text, subject) VALUES (%s, %s, %s, %s, %s)"
    val = (teacher_id, title, questions_text, answers_text, subject)
    try:
        cursor.execute(sql, val)
        db_conn.commit()
        print(f"Assessment '{title}' (Subject: {subject}, Teacher ID: {teacher_id}) saved. ID: {cursor.lastrowid}")
        return cursor.lastrowid 
    except mysql.connector.Error as err:
        print(f"Error saving assessment '{title}': {err}")
        db_conn.rollback()
        return None
    finally:
        cursor.close()

def get_or_create_teacher(db_conn, teacher_name):
    if not db_conn: return None
    cursor = db_conn.cursor()
    try:
        cursor.execute("SELECT teacher_id FROM teachers WHERE teacher_name = %s", (teacher_name,))
        result = cursor.fetchone()
        if result:
            return result[0] 
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
        
        cursor.execute("SELECT student_id FROM students WHERE student_name = %s", (student_name,))
        result = cursor.fetchone()
        if result:
            return result[0] 
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


def save_practice_question_to_catalog(db_conn, question_text,model_answer, concepts_list=None): 
    if not db_conn: return None
    cursor = db_conn.cursor()
    try:
        concepts_covered_str = None
        if concepts_list:
            valid_concepts = [str(c).strip() for c in concepts_list if c and str(c).strip()]
            if valid_concepts:
                concepts_covered_str = json.dumps(valid_concepts)

        sql_question = "INSERT INTO practice_questions_catalog (question_text, model_answer, concepts_covered) VALUES (%s, %s, %s)"
        val_question = (question_text,model_answer, concepts_covered_str)
        cursor.execute(sql_question, val_question)
        db_conn.commit()
        catalog_id = cursor.lastrowid
        print(f"Practice question saved to catalog with ID: {catalog_id} (concepts: {concepts_covered_str})") 
        return catalog_id
    except mysql.connector.Error as err:
        print(f"Error in save_practice_question_to_catalog: {err}") 
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

    if not db_conn:
        return "Could not retrieve practice history due to database connection issue."

    cursor = db_conn.cursor(dictionary=True) 
    history_summary = ""
    
    try:

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

                    concepts = json.loads(attempt['concepts_covered'])
                    if isinstance(concepts, list):
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
                    'Not Evaluated': 0 
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

def get_practice_question_details_by_id(db_conn: mysql.connector.connection.MySQLConnection, catalog_id: int) -> Optional[Dict[str, Any]]:
    if not db_conn:
        print("DB_UTILS ERROR: No database connection provided to get_practice_question_details_by_id.")
        return None


    result_dict = {}

    try:

        cursor = db_conn.cursor(dictionary=True)
        

        query = """
            SELECT 
                question_text, 
                model_answer
            FROM 
                practice_questions_catalog 
            WHERE 
                catalog_id = %s
        """
        

        cursor.execute(query, (catalog_id,))
        

        result = cursor.fetchone()
        
        if result:
            print(f"DB_UTILS INFO: Successfully fetched details for practice question with catalog_id: {catalog_id}")

            result_dict = result
        else:
            print(f"DB_UTILS WARNING: No practice question found with catalog_id: {catalog_id}")
            return None 

    except mysql.connector.Error as err:
        print(f"DB_UTILS ERROR: Failed to fetch practice question details for catalog_id {catalog_id}. Error: {err}")

        return None
    finally:

        if 'cursor' in locals() and cursor:
            cursor.close()
            
    return result_dict

async def get_student_for_auth(student_name: str) -> Optional[Dict[str, Any]]:
    # '''Fetches student details for authentication from the students table.'''
    db_conn = None
    MYSQL_DB_NAME = os.environ.get("MYSQL_DB")
    try:
        db_conn = get_mysql_connection(db_name=MYSQL_DB_NAME)
        if db_conn is None:
            print(f"DB ERROR: Failed to connect to database for get_student_for_auth (student: {student_name}).")
            return None
        
        cursor = db_conn.cursor(dictionary=True)
        # Assumes student_name is the unique identifier for login, and you've added hashed_password
        query = "SELECT student_id, student_name, hashed_password FROM students WHERE student_name = %s"
        cursor.execute(query, (student_name,))
        student_record = cursor.fetchone()
        
        if student_record:
            return student_record # Contains student_id, student_name, hashed_password
        return None
    except Error as e:
        print(f"DB ERROR: Error fetching student {student_name} for auth: {e}")
        return None
    finally:
        if db_conn and db_conn.is_connected():
            if 'cursor' in locals() and cursor: # ensure cursor exists
                cursor.close()


async def save_student_registration(student_name: str, hashed_password: str) -> Optional[Dict[str, Any]]:
    db_conn = None
    MYSQL_DB_NAME = os.environ.get("MYSQL_DB")
    try:
        db_conn = get_mysql_connection(db_name=MYSQL_DB_NAME)
        if db_conn is None:
            print(f"DB ERROR: Failed to connect to database for save_student_registration (student: {student_name}).")
            return None

        cursor = db_conn.cursor()
        # Assumes student_name is unique. If it can conflict, an explicit check or error handling for unique constraint is needed.
        # This also assumes you have added `hashed_password` column to `students` table.
        query = "INSERT INTO students (student_name, hashed_password) VALUES (%s, %s)"
        values = (student_name, hashed_password)
        cursor.execute(query, values)
        db_conn.commit()
        
        if cursor.lastrowid:
            return {"student_id": cursor.lastrowid, "student_name": student_name}
        return None
    except Error as e:
        print(f"DB ERROR: Error saving student registration for {student_name}: {e}")
        if db_conn: # Check if db_conn was successfully assigned
            db_conn.rollback()
        return None
    finally:
        if db_conn and db_conn.is_connected():
            if 'cursor' in locals() and cursor: # ensure cursor exists
                cursor.close()

async def get_teacher_for_auth(teacher_name: str) -> Optional[Dict[str, Any]]:
    db_conn = None
    MYSQL_DB_NAME = os.environ.get("MYSQL_DB")
    try:
        db_conn = get_mysql_connection(db_name=MYSQL_DB_NAME)
        if db_conn is None:
            print(f"DB ERROR: Failed to connect to database for get_teacher_for_auth (teacher: {teacher_name}).")
            return None
        
        cursor = db_conn.cursor(dictionary=True)
        query = "SELECT teacher_id, teacher_name, hashed_password FROM teachers WHERE teacher_name = %s"
        cursor.execute(query, (teacher_name,))
        teacher_record = cursor.fetchone()
        
        if teacher_record:
            return teacher_record 
        return None
    except Error as e:
        print(f"DB ERROR: Error fetching teacher {teacher_name} for auth: {e}")
        return None
    finally:
        if db_conn and db_conn.is_connected():
            if 'cursor' in locals() and cursor: # ensure cursor exists
                cursor.close()

async def save_teacher_registration(teacher_name: str, hashed_password: str) -> Optional[Dict[str, Any]]:
    db_conn = None
    MYSQL_DB_NAME = os.environ.get("MYSQL_DB")


    try:
        db_conn = get_mysql_connection(db_name=MYSQL_DB_NAME)
        if db_conn is None:
            print(f"DB ERROR: Failed to connect to database for save_teacher_registration (teacher: {teacher_name}).")
            return None

        cursor = db_conn.cursor()
        query = "INSERT INTO teachers (teacher_name, hashed_password) VALUES (%s, %s)"
        values = (teacher_name, hashed_password)
        cursor.execute(query, values)
        db_conn.commit()
        
        if cursor.lastrowid:
            return {"teacher_id": cursor.lastrowid, "teacher_name": teacher_name}
        return None
    except Error as e:
        print(f"DB ERROR: Error saving teacher registration for {teacher_name}: {e}")
        if db_conn: 
            db_conn.rollback()
        return None
    finally:
        if db_conn and db_conn.is_connected():
            if 'cursor' in locals() and cursor: 
                cursor.close()


def get_teaching_plan_by_id(db_conn, plan_id: int) -> Optional[dict]:
    """Retrieves a single teaching plan by its ID."""
    if not db_conn or not plan_id:
        return None
    
    cursor = db_conn.cursor(dictionary=True) # 使用字典游标，方便按列名获取
    try:
        cursor.execute("SELECT * FROM teaching_plans WHERE id = %s", (plan_id,))
        plan = cursor.fetchone()
        if plan:
            print(f"Successfully retrieved teaching plan with ID: {plan_id}")
            return plan
        else:
            print(f"No teaching plan found with ID: {plan_id}")
            return None
    except mysql.connector.Error as err:
        print(f"Error retrieving teaching plan with ID {plan_id}: {err}")
        return None
    finally:
        cursor.close()


def get_all_assessment_for_student_view(db_conn: mysql.connector.connection.MySQLConnection) -> List[Dict[str, Any]]:
    if not db_conn:
        print("DB_UTILS ERROR: No database connection provided.")
        return []
    cursor = None
    try:
        cursor = db_conn.cursor(dictionary=True)
        query = """
            SELECT 
                a.id,
                a.title
            FROM 
                published_assessments pa
            JOIN 
                assessments a ON pa.assessment_id = a.id
            WHERE
                pa.is_active = TRUE
            ORDER BY 
                pa.published_at DESC;
        """
        cursor.execute(query)
        raw_results = cursor.fetchall()
        return raw_results
    except mysql.connector.Error as err:
        print(f"DB_UTILS ERROR: Failed to fetch published practice questions list. Error: {err}")
        return []
    finally:
        if cursor:
            cursor.close()


def get_assessment_details_by_id(db_conn: mysql.connector.connection.MySQLConnection, assessment_id: int) -> Optional[Dict[str, Any]]:

    if not db_conn:
        print("DB_UTILS ERROR: No database connection provided.")
        return None
    try:
        cursor = db_conn.cursor(dictionary=True)
        query = """
            SELECT 
                id,
                title,
                questions_text AS content, 
                answers_text,             
                subject
            FROM 
                assessments 
            WHERE 
                id = %s
        """
        cursor.execute(query, (assessment_id,))
        result = cursor.fetchone()
        return result
    except mysql.connector.Error as err:
        print(f"DB_UTILS ERROR: Failed to fetch assessment details for ID {assessment_id}. Error: {err}")
        return None
    finally:
        if 'cursor' in locals() and cursor:
            cursor.close()

def get_question_identifiers_from_assessment(db_conn, assessment_id: int) -> list[str]:

    if not db_conn:
        return []
    
    cursor = db_conn.cursor(dictionary=True)
    try:
        cursor.execute("SELECT questions_text FROM assessments WHERE id = %s", (assessment_id,))
        result = cursor.fetchone()
        if not result or not result.get('questions_text'):
            return []
        
        content = result['questions_text']
        
        # 使用正则表达式匹配 "题目x" 或 "Question x" 等模式
        # 这个正则表达式可以根据你生成题目的格式进行微调
        # 它会寻找 "题目" 或 "Question" 开头，后面跟数字和冒号或点
        identifiers = re.findall(r'(?:题目|Question)\s*(\d+)[\s:：.]', content)
        
        # 将匹配到的数字格式化为 "题目X"
        formatted_identifiers = [f"题目{num}" for num in identifiers]

        # 如果上面的正则没匹配到，可以尝试一个更通用的，比如匹配每一行的开头
        if not formatted_identifiers:
             # 匹配以 "数字." 或 "数字、" 开头的行作为题目标识
             lines = content.split('\n')
             for line in lines:
                 match = re.match(r'^\s*(\d+[.、])', line)
                 if match:
                     formatted_identifiers.append(match.group(1).strip())

        print(f"DB_UTILS: Found identifiers for assessment {assessment_id}: {formatted_identifiers}")
        return formatted_identifiers
        
    except mysql.connector.Error as err:
        print(f"Error getting question identifiers: {err}")
        return []
    finally:
        cursor.close()

def publish_assessment(db_conn, assessment_id: int, teacher_id: int) -> bool:
    if not db_conn: return False
    cursor = db_conn.cursor()
    sql = "INSERT IGNORE INTO published_assessments (assessment_id, teacher_id) VALUES (%s, %s)"
    try:
        cursor.execute(sql, (assessment_id, teacher_id))
        db_conn.commit()
        # cursor.rowcount会返回受影响的行数，如果是1则表示插入成功，0表示已存在
        return cursor.rowcount > 0
    except mysql.connector.Error as err:
        print(f"Error publishing assessment {assessment_id}: {err}")
        db_conn.rollback()
        return False
    finally:
        cursor.close()


def get_assessments_by_teacher_id(db_conn, teacher_id: int) -> List[Dict[str, Any]]:
    if not db_conn: return []
    cursor = db_conn.cursor(dictionary=True)
    results = []
    try:
        sql = """
            SELECT a.id, a.title, a.subject, a.created_at
            FROM assessments a
            LEFT JOIN published_assessments pa ON a.id = pa.assessment_id
            WHERE a.teacher_id = %s and pa.publish_id IS NULL
            ORDER BY a.created_at DESC;
        """
        cursor.execute(sql, (teacher_id,))
        results = cursor.fetchall()
    except mysql.connector.Error as err:
        print(f"Error fetching assessments for teacher {teacher_id}: {err}")
    finally:
        cursor.close()
    return results

def get_aggregated_student_performance(db_conn, assessment_id: int) -> list:
    """
    Retrieves aggregated performance data for each student on a specific assessment.
    """
    if not db_conn:
        print("No database connection provided to get_aggregated_student_performance.")
        return []

    cursor = None
    try:
        cursor = db_conn.cursor(dictionary=True)
        # This query groups by student and counts the occurrences of each correctness status.
        sql = """
            SELECT
                saa.student_id,
                s.student_name,
                COUNT(saa.answer_id) AS total_answered,
                SUM(CASE WHEN saa.llm_assessed_correctness = 'Correct' THEN 1 ELSE 0 END) AS correct_count,
                SUM(CASE WHEN saa.llm_assessed_correctness = 'Partially Correct' THEN 1 ELSE 0 END) AS partially_correct_count,
                SUM(CASE WHEN saa.llm_assessed_correctness = 'Incorrect' THEN 1 ELSE 0 END) AS incorrect_count
            FROM
                student_assessment_answers saa
            JOIN
                students s ON saa.student_id = s.student_id
            WHERE
                saa.assessment_id = %s
            GROUP BY
                saa.student_id, s.student_name
            ORDER BY
                s.student_name;
        """
        cursor.execute(sql, (assessment_id,))
        results = cursor.fetchall()
        if not results:
            print(f"No performance data found for assessment_id: {assessment_id}")
        return results
    except mysql.connector.Error as err:
        print(f"Error aggregating student performance for assessment_id {assessment_id}: {err}")
        return []
    finally:
        if cursor:
            cursor.close()

def log_activity(db_conn, user_id: int, user_role: str, activity_type: str, details: Optional[Dict] = None):
    """Logs a user activity to the activity_log table."""
    if not db_conn: return
    cursor = db_conn.cursor()
    details_json = json.dumps(details) if details else None
    try:
        sql = "INSERT INTO activity_log (user_id, user_role, activity_type, details) VALUES (%s, %s, %s, %s)"
        cursor.execute(sql, (user_id, user_role, activity_type, details_json))
        db_conn.commit()
    except mysql.connector.Error as err:
        print(f"Error logging activity: {err}")
        db_conn.rollback()
    finally:
        cursor.close()

def get_users_by_role(db_conn, role: str, page: int = 1, page_size: int = 10, search: Optional[str] = None) -> Dict[str, Any]:
    if role not in ['student', 'teacher']:
        return {"total": 0, "users": []}

    table_name = "students" if role == 'student' else 'teachers'
    id_col = "student_id" if role == 'student' else 'teacher_id'
    name_col = "student_name" if role == 'student' else 'teacher_name'
    
    offset = (page - 1) * page_size
    
    cursor = db_conn.cursor(dictionary=True)
    
    # --- 构建查询 ---
    count_query = f"SELECT COUNT(*) as total FROM {table_name}"
    data_query = f"SELECT {id_col} as id, {name_col} as username FROM {table_name}"
    
    params = []
    where_clauses = []

    if search:
        where_clauses.append(f"{name_col} LIKE %s")
        params.append(f"%{search}%")

    if where_clauses:
        query_suffix = " WHERE " + " AND ".join(where_clauses)
        count_query += query_suffix
        data_query += query_suffix

    data_query += f" ORDER BY id  LIMIT %s OFFSET %s"
    
    try:
        # --- 增加调试打印 ---
        print(f"Executing count query: {count_query} with params: {tuple(params)}")
        cursor.execute(count_query, tuple(params))
        total_count_result = cursor.fetchone()
        
        # --- 增加调试打印 ---
        if total_count_result:
            print(f"Total count from DB: {total_count_result['total']}")
            total_count = total_count_result['total']
        else:
            print("Count query returned nothing.")
            total_count = 0
            
        # --- 增加调试打印 ---
        print(f"Executing data query: {data_query} with params: {tuple(params + [page_size, offset])}")
        cursor.execute(data_query, tuple(params + [page_size, offset]))
        users = cursor.fetchall()
        print(f"Users fetched from DB: {users}")
        
        return {"total": total_count, "users": users}

    except mysql.connector.Error as err:
        print(f"Error getting users by role: {err}")
        return {"total": 0, "users": []}
    finally:
        cursor.close()

def admin_create_user(db_conn, username: str, hashed_password: str, role: str) -> Optional[int]:
    """
    管理员在后台创建用户。
    """
    if role not in ['student', 'teacher']:
        return None

    table_name = "students" if role == 'student' else 'teachers'
    name_col = "student_name" if role == 'student' else 'teacher_name'
    
    sql = f"INSERT INTO {table_name} ({name_col}, hashed_password) VALUES (%s, %s)"
    
    cursor = db_conn.cursor()
    try:
        cursor.execute(sql, (username, hashed_password))
        db_conn.commit()
        return cursor.lastrowid
    except mysql.connector.Error as err:
        print(f"Error creating user by admin: {err}")
        db_conn.rollback()
        return None
    finally:
        cursor.close()

def admin_update_user_password(db_conn, user_id: int, new_hashed_password: str, role: str) -> bool:
    """
    管理员重置用户密码。
    """
    if role not in ['student', 'teacher']:
        return False
        
    table_name = "students" if role == 'student' else 'teachers'
    id_col = "student_id" if role == 'student' else 'teacher_id'

    sql = f"UPDATE {table_name} SET hashed_password = %s WHERE {id_col} = %s"
    
    cursor = db_conn.cursor()
    try:
        cursor.execute(sql, (new_hashed_password, user_id))
        db_conn.commit()
        return cursor.rowcount > 0 # 返回是否成功更新了行
    except mysql.connector.Error as err:
        print(f"Error updating user password by admin: {err}")
        db_conn.rollback()
        return False
    finally:
        cursor.close()

def admin_delete_user(db_conn, user_id: int, role: str) -> bool:
    """
    管理员删除用户。
    """
    if role not in ['student', 'teacher']:
        return False

    table_name = "students" if role == 'student' else 'teachers'
    id_col = "student_id" if role == 'student' else 'teacher_id'

    sql = f"DELETE FROM {table_name} WHERE {id_col} = %s"
    
    cursor = db_conn.cursor()
    try:
        cursor.execute(sql, (user_id,))
        db_conn.commit()
        return cursor.rowcount > 0 # 返回是否成功删除了行
    except mysql.connector.Error as err:
        print(f"Error deleting user by admin: {err}")
        db_conn.rollback()
        return False
    finally:
        cursor.close()



def get_teacher_resource_detail(db_conn, resource_type: str, resource_id: int) -> Optional[Dict[str, Any]]:

    details = None
    if resource_type == 'teaching_plans':
        plan = get_teaching_plan_by_id(db_conn, resource_id)
        if plan:
            # 统一输出格式
            details = {
                "id": plan.get('id'),
                "title": plan.get('title'),
                "subject": plan.get('subject'),
                "full_content": plan.get('content', '')
            }
    elif resource_type == 'assessments':
        assessment = get_assessment_details_by_id(db_conn, resource_id)
        if assessment:
            questions = assessment.get('content', '') 
            answers = assessment.get('answers_text', '')
            details = {
                "id": assessment.get('id'),
                "title": assessment.get('title'), 
                "subject": assessment.get('subject'),
                "full_content": f"{questions}\n\n---参考答案与解析---\n\n{answers}"
            }
    return details

def get_all_subjects(db_conn) -> List[str]:
    """获取系统中所有不重复的学科列表。"""
    cursor = db_conn.cursor()
    # 从两个表中分别查询学科，然后合并去重
    query = """
        (SELECT DISTINCT subject FROM teaching_plans WHERE subject IS NOT NULL AND subject != '')
        UNION
        (SELECT DISTINCT subject FROM assessments WHERE subject IS NOT NULL AND subject != '')
    """
    try:
        cursor.execute(query)
        # fetchall() 返回的是元组列表，例如 [('数学',), ('物理',)]
        # 我们需要将其转换为字符串列表
        subjects = [row[0] for row in cursor.fetchall()]
        return subjects
    except mysql.connector.Error as err:
        print(f"Error getting all subjects: {err}")
        return []
    finally:
        cursor.close()

def get_unified_resources_by_subject(db_conn, subject: str, page: int = 1, page_size: int = 10, search: Optional[str] = None) -> Dict[str, Any]:

    offset = (page - 1) * page_size
    cursor = db_conn.cursor(dictionary=True)

    plans_query = """
        SELECT
            tp.id,
            tp.title,
            tp.created_at,
            tp.subject,
            t.teacher_name AS creator,
            'teaching_plans' AS resource_type
        FROM teaching_plans AS tp
        LEFT JOIN teachers t ON tp.teacher_id = t.teacher_id
    """
    
    # 考核部分
    assessments_query = """
        SELECT
            a.id,
            a.title,
            a.created_at,
            a.subject,
            t.teacher_name AS creator,
            'assessments' AS resource_type
        FROM assessments AS a
        LEFT JOIN teachers t ON a.teacher_id = t.teacher_id
    """
    
    # 基础的统一视图查询
    unified_view_query = f"({plans_query}) UNION ALL ({assessments_query})"

    # --- 在统一视图上进行筛选、搜索和分页 ---
    
    # WHERE 条件
    where_clauses = ["subject = %s"] # 按学科筛选是基本条件
    params = [subject]

    if search:

        where_clauses.append("title LIKE %s")
        params.append(f"%{search}%")


    final_query_suffix = " WHERE " + " AND ".join(where_clauses)
    

    count_query = f"SELECT COUNT(*) as total FROM ({unified_view_query}) AS unified_resources" + final_query_suffix
    

    data_query = f"SELECT * FROM ({unified_view_query}) AS unified_resources" + final_query_suffix + " ORDER BY created_at  LIMIT %s OFFSET %s"

    try:

        cursor.execute(count_query, tuple(params))
        total_count_result = cursor.fetchone()
        total_count = total_count_result['total'] if total_count_result else 0
        

        cursor.execute(data_query, tuple(params + [page_size, offset]))
        resources = cursor.fetchall()
        
        return {"total": total_count, "resources": resources}

    except mysql.connector.Error as err:
        print(f"Error getting unified resources for subject '{subject}': {err}")
        return {"total": 0, "resources": []}
    finally:
        cursor.close()

def get_activity_stats(db_conn, period: str) -> List[Dict]:
    if period == 'daily':
        interval_clause = "activity_timestamp >= CURDATE()"
    elif period == 'weekly':
        # WEEK(NOW()) 和 WEEK(activity_timestamp) 确保是在本周
        interval_clause = "YEARWEEK(activity_timestamp, 1) = YEARWEEK(NOW(), 1)"
    else:
        return []

    sql = f"""
        SELECT 
            user_role,
            activity_type,
            COUNT(*) as count
        FROM activity_log
        WHERE {interval_clause}
        GROUP BY user_role, activity_type
        ORDER BY user_role, count DESC;
    """
    cursor = db_conn.cursor(dictionary=True)
    try:
        cursor.execute(sql)
        return cursor.fetchall()
    except mysql.connector.Error as err:
        print(f"Error getting activity stats for period '{period}': {err}")
        return []
    finally:
        cursor.close()

def get_teacher_content_creation_stats(db_conn) -> List[Dict]:
    """
    从 activity_log 表中统计每位教师创建和修改教案/考核的次数。
    """
    activity_types = (
        'GENERATE_TEACHING_PLAN', 
        'REFINE_TEACHING_PLAN',
        'GENERATE_ASSESSMENT',
        'REFINE_ASSESSMENT'
    )
    
    # --- 【核心修正点】 ---
    # 1. 动态生成占位符字符串，例如 " (%s, %s, %s, %s) "
    placeholders = ', '.join(['%s'] * len(activity_types))
    
    # 2. 将动态生成的占位符嵌入到 SQL 查询中
    sql = f"""
        SELECT
            t.teacher_id,
            t.teacher_name,
            al.activity_type,
            COUNT(al.log_id) as count
        FROM activity_log al
        JOIN teachers t ON al.user_id = t.teacher_id
        WHERE al.user_role = 'teacher' AND al.activity_type IN ({placeholders})
        GROUP BY t.teacher_id, t.teacher_name, al.activity_type
        ORDER BY t.teacher_id;
    """
    
    cursor = db_conn.cursor(dictionary=True)
    try:
        # 3. 将 activity_types 元组直接作为参数传递，驱动会自动解包
        cursor.execute(sql, activity_types)
        return cursor.fetchall()
    except mysql.connector.Error as err:
        print(f"Error getting teacher content creation stats: {err}")
        return []
    finally:
        cursor.close()

def get_low_performing_subjects(db_conn, limit: int = 0) -> List[Dict]:
    sql = """
        SELECT
            a.subject,
            AVG(CASE 
                WHEN saa.llm_assessed_correctness = 'Correct' THEN 100
                WHEN saa.llm_assessed_correctness = 'Partially Correct' THEN 50
                ELSE 0 
            END) as average_score,
            COUNT(DISTINCT saa.student_id) as student_count,
            COUNT(saa.answer_id) as total_answers
        FROM assessments a
        JOIN student_assessment_answers saa ON a.id = saa.assessment_id
        WHERE a.subject IS NOT NULL AND a.subject != ''
        GROUP BY a.subject
        HAVING total_answers > 0 -- 只统计有一定作答量的学科
        ORDER BY average_score ASC
        LIMIT %s;
    """
    cursor = db_conn.cursor(dictionary=True)
    try:
        cursor.execute(sql, (limit,))
        return cursor.fetchall()
    except mysql.connector.Error as err:
        print(f"Error getting low performing subjects: {err}")
        return []
    finally:
        cursor.close()

def get_daily_accuracy_trend(db_conn) -> List[Dict]:
    """
    【注意】这个函数现在是独立的，它自己管理数据库连接。
    这是为了解决一个顽固的连接过早关闭的问题。
    """
    
    # 1. 在函数内部创建连接
    conn = None
    cursor = None
    MYSQL_DB_NAME = os.environ.get("MYSQL_DB")

    try:
        conn = get_mysql_connection(db_name=MYSQL_DB_NAME)
        if not conn:
            print("Error: get_daily_accuracy_trend could not establish its own DB connection.")
            return []

        sql = """
            SELECT 
                DATE(attempt_timestamp) as date,
                AVG(
                    CASE
                        WHEN correctness_assessment LIKE '%_%%' THEN CAST(REPLACE(correctness_assessment, '%%', '') AS DECIMAL(10,2))
                        WHEN correctness_assessment = 'Correct' THEN 100.0
                        WHEN correctness_assessment = 'Partially Correct' THEN 50.0
                        WHEN correctness_assessment = 'Incorrect' THEN 0.0
                        ELSE NULL 
                    END
                ) as average_accuracy
            FROM practice_attempts
            WHERE correctness_assessment IS NOT NULL
            GROUP BY DATE(attempt_timestamp)
            HAVING AVG(CASE ... END) IS NOT NULL 
            ORDER BY date 
            LIMIT 30;
        """
        sql = sql.replace("CASE ... END", """
                    CASE
                        WHEN correctness_assessment LIKE '%_%%' THEN CAST(REPLACE(correctness_assessment, '%%', '') AS DECIMAL(10,2))
                        WHEN correctness_assessment = 'Correct' THEN 100.0
                        WHEN correctness_assessment = 'Partially Correct' THEN 50.0
                        WHEN correctness_assessment = 'Incorrect' THEN 0.0
                        ELSE NULL 
                    END
        """)

        cursor = conn.cursor(dictionary=True)
        cursor.execute(sql)
        results = cursor.fetchall()
        
        for row in results:
            if row['average_accuracy'] is not None:
                row['average_accuracy'] = float(row['average_accuracy'])
        
        return results

    except mysql.connector.Error as err:
        print(f"Error in self-contained get_daily_accuracy_trend: {err}")
        return []
    finally:
        # 2. 在函数结束时，关闭自己创建的所有资源
        if cursor:
            cursor.close()
        if conn and conn.is_connected():
            conn.close()

def get_all_practice_attempts_with_concepts(db_conn) -> List[Dict]:
    """
    获取所有包含知识点标签的练习作答记录。
    """
    sql = """
        SELECT 
            pqc.concepts_covered,
            pa.correctness_assessment
        FROM practice_attempts pa
        JOIN practice_questions_catalog pqc ON pa.catalog_id = pqc.catalog_id
        WHERE pqc.concepts_covered IS NOT NULL AND pqc.concepts_covered != '[]' AND pa.correctness_assessment IS NOT NULL
    """
    cursor = db_conn.cursor(dictionary=True)
    try:
        cursor.execute(sql)
        return cursor.fetchall()
    except mysql.connector.Error as err:
        print(f"Error getting all practice attempts with concepts: {err}")
        return []
    finally:
        if 'cursor' in locals() and cursor:
            cursor.close()

def get_published_assessments_by_teacher(db_conn, teacher_id: int) -> List[Dict[str, Any]]:
    if not db_conn: return []
    cursor = db_conn.cursor(dictionary=True)
    results = []
    try:
        sql = """
            SELECT 
                a.id, 
                a.title, 
                a.subject, 
                a.created_at
            FROM assessments a
            JOIN published_assessments pa ON a.id = pa.assessment_id
            WHERE a.teacher_id = %s
            ORDER BY pa.published_at DESC;
        """
        cursor.execute(sql, (teacher_id,))
        results = cursor.fetchall()
    except mysql.connector.Error as err:
        print(f"Error fetching published assessments for teacher {teacher_id}: {err}")
    finally:
        cursor.close()
    return results