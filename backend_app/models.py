from pydantic import BaseModel, Field
from typing import List, Literal, Optional, Dict, Union


from datetime import datetime,date
AdminResourceType = Literal['teaching_plans', 'assessments']
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
    question: str 
    student_id: Optional[int] 


class StudentQuestionOutput(BaseModel):
    student_question: str
    rag_context: Optional[List[str]]
    llm_answer: str
    error_message: Optional[str] = None

class TeachingPlanInput(BaseModel):
    teacher_id: Optional[int] 
    subject: str 
    teaching_outline: str 
    title_for_db: Optional[str] 
    style_tone: Optional[str] 
    output_structure: Optional[str] 

class TeachingPlanOutput(BaseModel):
    teaching_plan_id: Optional[int] = None 
    title: str
    generated_plan_content: str
    teacher_id: Optional[int] = None
    error_message: Optional[str] = None

class AssessmentInput(BaseModel):
    teacher_id: Optional[int] 
    subject: Optional[str] 
    teaching_plan_content: str 
    question_preferences: Dict[str, int] 
    title_for_db: Optional[str] 

class AssessmentOutput(BaseModel):
    assessment_id: Optional[int] = None 
    title: Optional[str] = None 
    generated_assessment_content: str
    teacher_id: Optional[int] = None
    subject: Optional[str] = None 
    error_message: Optional[str] = None

class StudentAssessmentAnswerItem(BaseModel):
    question_identifier: str 
    student_answer_text: str 

class StudentAssessmentInput(BaseModel):
    student_id: Optional[int] 
    assessment_id: int 
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
    student_id: Optional[int] 
    practice_topic: str 
    question_preferences: Dict[str, int] 
    

class PracticeQuestionItem(BaseModel): 
    question_text: str 
    model_answer: str



class PracticeFeedbackInput(BaseModel):
    student_id: int 
    catalog_id: int 
    student_answer: str





class TeachingPlanNLInput(BaseModel):
    query: str 
    teacher_id: int 

class PracticeQuestionNLInput(BaseModel):
    query: str 
    student_id: Optional[int] 


class AssessmentNLInput(BaseModel):
    query: str 
    teacher_id: Optional[int] 

class StudentAssessmentNLInput(BaseModel):
    query: str 
    student_id: Optional[int] 
    assessment_id: int 


class Message(BaseModel):
    message: str

class UserBase(BaseModel):
    username: str 

class UserCreate(UserBase):
    username: str 
    password: str 
    repassword:str
    role: int

class UserLogin(BaseModel):
    username: str 
    password: str 
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
    role: str  
    content: str

class RefineStudentQAInput(BaseModel):
    
    history: List[ChatMessage]
    new_query: str
    student_id: Optional[int] = None

class RefineTeachingPlanInput(BaseModel):
    base_teaching_plan_id: int 
    history: List[ChatMessage]
    new_query: str
    teacher_id: Optional[int] = None

class RefineAssessmentInput(BaseModel):
    base_assessment_id: Optional[int] 
    history: List[ChatMessage]
    new_query: str
    teacher_id: Optional[int] = None

class PracticeQuestionListItem(BaseModel):
    id: int
    title: str 


class PracticeQuestionListOutput(BaseModel):
    questions: List[PracticeQuestionListItem]


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
    accuracy: float 

class PublishAssessment(BaseModel):
    teacher_id:int
    assessment_id:int

class PracticeChatInput(BaseModel):
    student_id: int

    history: List[ChatMessage]

    new_query: str

    active_catalog_id: Optional[int] = None

class FeedbackItem(BaseModel):
    question_identifier: str
    student_answer: str
    correctness: str
    feedback: str

class PracticeFeedbackOutput(BaseModel):
    attempt_id: int
    overall_comment: str  
    feedback_details: List[FeedbackItem] #
    error_message: Optional[str] 

class PracticeQuestionsOutput(BaseModel):
    generated_questions: PracticeQuestionItem
    catalog_id: Optional[int] = None
    error_message: Optional[str] = None
class PracticeChatOutput(BaseModel):

    assistant_response_text: str

    intent_detected: str

    new_questions: Optional[PracticeQuestionsOutput] = None

    feedback: Optional[PracticeFeedbackOutput] = None

    error_message: Optional[str] = None

class AdminUserView(BaseModel):
    id: int
    username: str


class PaginatedUsersResponse(BaseModel):
    total: int
    users: List[AdminUserView]

class AdminCreateUserInput(BaseModel):
    username: str
    password: str
    role: Literal['student', 'teacher']

class AdminResetPasswordInput(BaseModel):
    new_password: str

class AdminResourceView(BaseModel):
    id: int
    title: Optional[str]
    creator: Optional[str]
    subject: Optional[str]
    created_at: datetime
    resource_type: str 

class PaginatedAdminResourcesResponse(BaseModel):
    total: int
    resources: List[AdminResourceView]


class AdminResourceDetailView(BaseModel):
    id: int
    title: Optional[str]
    subject: Optional[str]
    full_content: str

class ActivityStat(BaseModel):
    activity_type: str
    count: int

class UsageStats(BaseModel):
    teacher: List[ActivityStat]
    student: List[ActivityStat]

class DashboardUsageResponse(BaseModel):
    daily: UsageStats
    weekly: UsageStats

class TeacherEfficiencyStat(BaseModel):
    teacher_id: int
    teacher_name: str
    

    plans_created: int = 0
    assessments_created: int = 0
    

    plans_refined: int = 0
    assessments_refined: int = 0
    

    plan_efficiency_index: float 
    assessment_efficiency_index: float 

class SubjectPerformance(BaseModel):
    subject: str
    average_score: float
    student_count: int
    total_answers: int

class DailyAccuracy(BaseModel):
    date: date
    average_accuracy: Optional[float] = None

class ConceptStat(BaseModel):
    concept: str
    mastery_rate: float 
    total_attempts: int
    incorrect_attempts: int

class StudentEffectivenessResponse(BaseModel):
    accuracy_trend: List[DailyAccuracy]
    weakest_concepts: List[ConceptStat] 


class PublishedAssessmentInfo(BaseModel):
    id: int
    title: str
    subject: Optional[str] = None
    created_at: datetime


class QuestionAnalysisDetail(BaseModel):
    question_identifier: str = Field(..., description="The identifier of the question, e.g., '题目1'.")
    question_text: str = Field(..., description="The actual text of the question for context.")
    correct_rate: float = Field(..., description="The correct rate for this question (0.0 to 100.0).")
    main_knowledge_point: str = Field(..., description="The key knowledge point or skill tested by this question, inferred by the LLM.")
    common_errors: Optional[str] = Field(None, description="A summary of common student errors for this question, if identifiable.")


class AssessmentAnalysisOutput(BaseModel):
    assessment_title: str
    overall_summary: str = Field(..., description="A high-level summary of the class's performance on this assessment.")
    question_analysis: List[QuestionAnalysisDetail] = Field(..., description="A detailed analysis of the most problematic questions.")
    teaching_suggestions: List[str] = Field(..., description="Actionable teaching suggestions based on the analysis.")
    error_message: Optional[str] = None

class QuestionAnalysis(BaseModel):
    question_identifier: str = Field(..., description="问题的标识符, 例如 '题目1'")
    question_text: str = Field(..., description="该问题的实际文本，用于上下文展示")
    correct_rate: float = Field(..., description="该问题的正确率 (0.0 到 100.0)")
    main_knowledge_point: str = Field(..., description="LLM提炼的、该问题考察的核心知识点")
    common_error_analysis: str = Field(..., description="LLM根据题目和统计数据推测的常见错误原因或学生思维误区")



class AssessmentAnalysis(BaseModel):
    assessment_title: str = Field(..., description="被分析的考核的标题")
    overall_summary: str = Field(..., description="对班级整体表现的概括性总结，例如：整体掌握情况良好，但在...方面存在普遍困难。")
    strength_points: List[str] = Field(..., description="学生普遍掌握得比较好的知识点列表")
    weakness_points: List[str] = Field(..., description="学生普遍存在的薄弱知识点列表")
    problematic_questions: List[QuestionAnalysis] = Field(..., description="对错误率最高或最值得关注的几个问题的详细分析列表")
    teaching_suggestions: List[str] = Field(..., description="基于以上所有分析，给出的具体、可操作的教学建议列表")
    error_message: Optional[str] = None