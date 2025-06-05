from pydantic import BaseModel, Field
from typing import List, Optional, Dict

class StudentQuestionInput(BaseModel):
    question: str = Field(..., example="Explain the concept of photosynthesis.")
    student_id: Optional[int] = Field(None, example=123, description="Optional student ID for history tracking.")
    # top_k_rag: Optional[int] = Field(3, example=3, description="Number of RAG snippets to retrieve.") # Keep it simple for now

class StudentQuestionOutput(BaseModel):
    student_question: str
    rag_context: Optional[List[str]]
    llm_answer: str
    error_message: Optional[str] = None

class TeachingPlanInput(BaseModel):
    teacher_id: Optional[int] = Field(None, example=1, description="ID of the teacher requesting the plan.")
    teacher_name: Optional[str] = Field(None, example="Dr. Smith", description="Name of the teacher, to get/create teacher_id if ID not provided.")
    subject: str = Field(..., example="High School Biology")
    teaching_outline: str = Field(..., example="Week 1: Cell Structure, Week 2: Photosynthesis...")
    # Assuming prompt_engineering details like RAG context for plan generation are handled by the service layer
    # top_k_rag: Optional[int] = Field(3, example=3) # For RAG context to generate plan
    title_for_db: Optional[str] = Field(None, example="My Biology Q1 Plan", description="Optional title for saving the plan to the database.")
    style_tone: Optional[str] = Field(None, example="Engaging and interactive for 10th graders.")
    output_structure: Optional[str] = Field(
        "Please include: 1. Key Concepts; 2. Weekly Breakdown; 3. Example Activities; 4. Assessment Ideas.",
        example="1. Key Concepts; 2. Weekly Breakdown; 3. Example Activities; 4. Assessment Ideas."
    )

class TeachingPlanOutput(BaseModel):
    teaching_plan_id: Optional[int] = None # If saved
    title: str
    subject: str
    generated_plan_content: str
    teacher_id: Optional[int] = None
    error_message: Optional[str] = None

class AssessmentInput(BaseModel):
    teacher_id: Optional[int] = Field(None, example=1)
    teacher_name: Optional[str] = Field(None, example="Dr. Smith")
    teaching_plan_content: str = Field(..., example="Detailed content of a teaching plan on Photosynthesis...")
    question_preferences: Dict[str, int] = Field(
        default_factory=lambda: {"multiple-choice": 2, "short-answer": 2, "programming": 0},
        example={"multiple-choice": 3, "short-answer": 2}
    )
    title_for_db: Optional[str] = Field(None, example="Photosynthesis Quiz Chapter 1", description="Optional title for saving the assessment to the database.")
    # RAG for assessment generation is based on teaching_plan_content keywords, handled by service

class AssessmentOutput(BaseModel):
    assessment_id: Optional[int] = None # If saved
    title: Optional[str] = None # Title might be requested from user before saving
    generated_assessment_content: str
    teacher_id: Optional[int] = None
    error_message: Optional[str] = None

class StudentAssessmentAnswerItem(BaseModel):
    question_identifier: str = Field(..., example="Question 1")
    student_answer_text: str = Field(..., example="Paris is the capital of France.")

class StudentAssessmentInput(BaseModel):
    student_id: Optional[int] = Field(None, example=123)
    student_name: str = Field(..., example="John Doe") # Used if student_id is not provided
    assessment_id: int = Field(..., example=1)
    answers: List[StudentAssessmentAnswerItem]

class StudentAssessmentEvaluationOutput(BaseModel): # Renamed for clarity
    answer_id: Optional[int] = None # If saved
    assessment_id: int
    question_identifier: str
    student_id: int
    student_answer_text: str
    llm_assessed_correctness: str
    llm_evaluation_feedback: str
    error_message: Optional[str] = None

class PracticeQuestionsInput(BaseModel):
    student_id: Optional[int] = Field(None, example=123) # For history-based personalization
    student_name: Optional[str] = Field(None, example="Alice") # For history-based personalization if ID not known
    practice_topic: str = Field(..., example="Python list comprehensions")
    question_preferences: Dict[str, int] = Field(
        default_factory=lambda: {"multiple-choice": 1, "short-answer": 1, "programming": 1},
        example={"programming": 2, "short-answer": 1}
    )
    # RAG and history summary are fetched by the service layer

class PracticeQuestionItem(BaseModel): # For individual Q&A pairs
    catalog_id: Optional[int] = None # If saved to catalog
    question_type: str
    question_text: str # Renamed from 'question' for clarity vs. student's question
    model_answer: str

class PracticeQuestionsOutput(BaseModel):
    generated_questions: List[PracticeQuestionItem]
    error_message: Optional[str] = None

class PracticeFeedbackInput(BaseModel):
    student_id: int = Field(..., example=123)
    catalog_id: int = Field(..., example=101) # ID of the question from practice_questions_catalog
    question_text: str # The actual question text
    model_answer: str # The model answer
    student_answer: str
    question_type: str = Field(..., example="Programming")

class PracticeFeedbackOutput(BaseModel):
    attempt_id: Optional[int] = None # If attempt saved
    correctness_assessment: str
    detailed_feedback: str
    error_message: Optional[str] = None

# Generic message model for simple status updates or errors not fitting other models
class Message(BaseModel):
    message: str