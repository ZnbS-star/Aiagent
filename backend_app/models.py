from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Union


from datetime import datetime
class StudentPerformanceDetail(BaseModel):
    answer_id: int
    assessment_id: int
    student_id: int
    student_name: Optional[str] = None 
    question_identifier: str
    student_answer_text: Optional[str] = None
    llm_evaluation_feedback: Optional[str] = None
    llm_assessed_correctness: Optional[str] = None
    submission_timestamp: datetime

class StudentQuestionInput(BaseModel):
    question: str = Field(..., example="Explain the concept of photosynthesis.")
    student_id: Optional[int] = Field(None, example=123, description="Optional student ID for history tracking.")


class StudentQuestionOutput(BaseModel):
    student_question: str
    rag_context: Optional[List[str]]
    llm_answer: str
    error_message: Optional[str] = None

class TeachingPlanInput(BaseModel):
    teacher_id: Optional[int] = Field(None, example=1, description="ID of the teacher requesting the plan.")
    subject: str = Field(..., example="High School Biology")
    teaching_outline: str = Field(..., example="Week 1: Cell Structure, Week 2: Photosynthesis...")
    title_for_db: Optional[str] = Field(None, example="My Biology Q1 Plan", description="Optional title for saving the plan to the database.")
    style_tone: Optional[str] = Field(None, example="Engaging and interactive for 10th graders.")
    output_structure: Optional[str] = Field(
        "Please include: 1. Key Concepts; 2. Weekly Breakdown; 3. Example Activities; 4. Assessment Ideas.",
        example="1. Key Concepts; 2. Weekly Breakdown; 3. Example Activities; 4. Assessment Ideas."
    )

class TeachingPlanOutput(BaseModel):
    teaching_plan_id: Optional[int] = None 
    title: str
    generated_plan_content: str
    teacher_id: Optional[int] = None
    error_message: Optional[str] = None

class AssessmentInput(BaseModel):
    teacher_id: Optional[int] = Field(None, example=1)
    subject: Optional[str] = Field(None, example="Computer Science", description="The subject area.") 
    teaching_plan_content: str = Field(..., example="Content on Photosynthesis...")
    question_preferences: Dict[str, int] = Field(
        default_factory=dict,
        example={"multiple-choice": 3}
    )
    title_for_db: Optional[str] = Field(None, example="Photosynthesis Quiz")

class AssessmentOutput(BaseModel):
    assessment_id: Optional[int] = None 
    title: Optional[str] = None 
    generated_assessment_content: str
    teacher_id: Optional[int] = None
    subject: Optional[str] = None 
    error_message: Optional[str] = None

class StudentAssessmentAnswerItem(BaseModel):
    question_identifier: str = Field(..., example="Question 1")
    student_answer_text: str = Field(..., example="Paris is the capital of France.")

class StudentAssessmentInput(BaseModel):
    student_id: Optional[int] = Field(None, example=123)
    assessment_id: int = Field(..., example=1)
    answers: List[StudentAssessmentAnswerItem]

class StudentAssessmentEvaluationOutput(BaseModel): 
    answer_id: Optional[int] = None 
    assessment_id: int
    question_identifier: str
    student_id: int
    student_answer_text: str
    llm_assessed_correctness: str
    llm_evaluation_feedback: str
    error_message: Optional[str] = None

class PracticeQuestionsInput(BaseModel):
    student_id: Optional[int] = Field(None, example=123) 
    practice_topic: str = Field(..., example="Python list comprehensions")
    question_preferences: Dict[str, int] = Field(
        default_factory=lambda: {"multiple-choice": 1, "short-answer": 1, "programming": 1},
        example={"programming": 2, "short-answer": 1}
    )
    

class PracticeQuestionItem(BaseModel): 
    question_text: str 
    model_answer: str

class PracticeQuestionsOutput(BaseModel):
    generated_questions: List[PracticeQuestionItem]
    catalog_id: Optional[int] = None
    error_message: Optional[str] = None

class PracticeFeedbackInput(BaseModel):
    student_id: int = Field(..., example=123)
    catalog_id: int = Field(..., example=101) 
    student_answer: str


class FeedbackItem(BaseModel):
    question_identifier: str
    student_answer: str
    correctness: str
    feedback: str
class PracticeFeedbackOutput(BaseModel):
    attempt_id: int
    overall_comment: str  # 对学生本次作答的总体评价
    feedback_details: List[FeedbackItem] # 每个问题的详细反馈列表
    error_message: Optional[str] = None

class TeachingPlanNLInput(BaseModel):
    query: str = Field(..., example="Generate a high school biology teaching plan about cell division for Dr. Evo. Make it engaging.")
    teacher_id: int = Field(None, description="Optional teacher ID if known, e.g., from session.")

class PracticeQuestionNLInput(BaseModel):
    query: str = Field(..., example="Generate some python list comprehension questions, maybe 2 multiple choice and 1 programming.")
    student_id: Optional[int] = Field(None, description="Optional student ID for history personalization.")


class AssessmentNLInput(BaseModel):
    query: str = Field(..., example="Generate an assessment for Prof. Oak based on the Kanto region Pokedex, Gym Leaders, and Elite Four. Include 2 multiple-choice and 1 short-answer. Title it 'Kanto Basics Quiz'.")
    teacher_id: Optional[int] = Field(None, description="Optional teacher ID if known.")

class StudentAssessmentNLInput(BaseModel):
    query: str = Field(..., example="For assessment 12, student John Doe (id 77) answered: Q1 was 'Paris', Section B Q2 was 'The mitochondria is the powerhouse of the cell'.")
    student_id: Optional[int] = Field(None, description="Student ID, if known and can be extracted or provided separately.")
    assessment_id: int = Field(..., description="ID of the assessment being evaluated.")



class Message(BaseModel):
    message: str

class UserBase(BaseModel):
    username: str = Field(..., example="john_doe")

class UserCreate(UserBase):
    username: str = Field(..., example="john_doe")
    password: str = Field(..., example="securepassword123")
    repassword:str
    role: int

class UserLogin(BaseModel):
    username: str = Field(..., example="john_doe")
    password: str = Field(..., example="securepassword123")
    role: int

class TokenData(BaseModel):
    username: Optional[str] = None

class Token(BaseModel):
    access_token: str
    token_type: str
    username:str
    userid:int
    role:int


    class Config:
        orm_mode = True 


class ChatMessage(BaseModel):
    role: str  # e.g., "user", "assistant", "system"
    content: str

class RefineStudentQAInput(BaseModel):
    
    history: List[ChatMessage]
    new_query: str
    student_id: Optional[int] = None

class RefineTeachingPlanInput(BaseModel):
    base_teaching_plan_id: int = Field(None, description="The ID of the teaching plan being refined. If null, a new plan will be created based on history.")
    history: List[ChatMessage]
    new_query: str
    teacher_id: Optional[int] = None

class RefineAssessmentInput(BaseModel):
    base_assessment_id: Optional[int] = Field(None, description="The ID of the assessment being refined. If null, a new assessment will be created based on history.")
    history: List[ChatMessage]
    new_query: str
    teacher_id: Optional[int] = None

class PracticeQuestionListItem(BaseModel):
    id: int
    title: str = Field(..., description="A concise title for the practice set, derived from its concepts or content.")


class PracticeQuestionListOutput(BaseModel):
    questions: List[PracticeQuestionListItem]

# 新增：练习详情接口的响应模型
class PracticeQuestionDetailOutput(BaseModel):
    id: int
    content: str

class TeacherAssessmentListItem(BaseModel):
    id: int
    title: str
    subject: Optional[str] = None
    created_at: datetime

class TeacherAssessmentListOutput(BaseModel):
    assessments: List[TeacherAssessmentListItem]

class StudentPerformanceDetail(BaseModel):
    answer_id: int
    assessment_id: int
    student_id: int
    student_name: Optional[str] = None 
    question_identifier: str
    student_answer_text: Optional[str] = None
    llm_evaluation_feedback: Optional[str] = None
    llm_assessed_correctness: Optional[str] = None
    submission_timestamp: datetime

class StudentAssessmentSummary(BaseModel):
    student_id: int
    student_name: Optional[str] = None
    total_answered: int
    correct_count: int
    partially_correct_count: int
    incorrect_count: int
    accuracy: float = Field(..., example=87.5, description="Accuracy score, calculated as (Correct * 1 + Partially Correct * 0.5) / Total * 100")

class PublishAssessment(BaseModel):
    teacher_id:int
    assessment_id:int

class PracticeChatInput(BaseModel):
    student_id: int
    # 关键：前端需要传递整个对话历史
    history: List[ChatMessage]
    # 用户的最新一条消息
    new_query: str
    # 关键：前端需要告知当前正在回答的是哪个题库的题
    # 第一次生成时为None，之后应为返回的catalog_id
    active_catalog_id: Optional[int] = None

class PracticeChatOutput(BaseModel):
    # AI的文本回复，例如 "好的，我已经把题目改难了："
    assistant_response_text: str
    # 检测到的用户意图，方便前端调试或做特定UI处理
    intent_detected: str
    # 如果AI生成了新题目，这里会有内容
    new_questions: Optional[PracticeQuestionsOutput] = None
    # 如果AI给出了反馈，这里会有内容
    feedback: Optional[PracticeFeedbackOutput] = None
    # 任何可能发生的错误
    error_message: Optional[str] = None