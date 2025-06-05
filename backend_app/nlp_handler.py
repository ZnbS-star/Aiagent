import os
from typing import Union, Dict, List
from backend_app.models import (
    NaturalLanguageQueryInput, NLQueryResponse,
    StudentQuestionInput, StudentQuestionOutput,
    TeachingPlanInput, TeachingPlanOutput,
    AssessmentInput, AssessmentOutput,
    StudentAssessmentInput, StudentAssessmentEvaluationOutput, # Keep List for this one
    PracticeQuestionsInput, PracticeQuestionsOutput,
    PracticeFeedbackInput, PracticeFeedbackOutput
)
from backend_app.services import (
    process_student_question_service,
    generate_initial_teaching_plan_service,
    generate_assessment_service,
    evaluate_student_assessment_answers_service,
    generate_practice_questions_service,
    get_practice_feedback_service
)
from langchain_community.chat_models import ChatZhipuAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.messages import SystemMessage, HumanMessage
import json # For potential manual JSON fixing or error handling

# Initialize LLM and Parser
# Ensure ZHIPUAI_API_KEY is set in the environment.
llm = None
if os.environ.get("ZHIPUAI_API_KEY"):
    llm = ChatZhipuAI(model="glm-4", temperature=0.1)
else:
    print("CRITICAL ERROR: ZHIPUAI_API_KEY not set. NLP handler will not function.")
    # In a real application, you might raise an exception here or have a clearer fallback

output_parser = JsonOutputParser()

# System Prompt for Intent and Entity Extraction
SYSTEM_PROMPT_TEXT = """
You are an expert AI assistant that processes user queries to understand their intent and extract relevant information (entities).
Your goal is to convert the natural language query into a structured JSON object.

Output Format:
You MUST output a JSON object with two main keys: "intent" and "entities".
{format_instructions}

Possible Intents and their associated entities:

1.  "generate_practice_questions":
    Entities:
    - "practice_topic" (string, required): The topic for practice.
    - "student_id" (integer, optional): The ID of the student.
    - "student_name" (string, optional): The name of the student.
    - "question_preferences" (dict, optional): E.g., {{"multiple-choice": 2, "programming": 1}}

2.  "student_question":
    Entities:
    - "question" (string, required): The student's question.
    - "student_id" (integer, optional): The ID of the student.

3.  "generate_teaching_plan":
    Entities:
    - "teacher_name" (string, optional): Name of the teacher.
    - "teacher_id" (integer, optional): ID of the teacher.
    - "subject" (string, required): The subject of the plan.
    - "teaching_outline" (string, required): Keywords or outline for the plan.
    - "style_tone" (string, optional): Desired style.
    - "output_structure" (string, optional): Desired output structure.
    - "title_for_db" (string, optional): Optional title for saving.


4.  "get_practice_feedback":
    Entities:
    - "student_id" (integer, required): The ID of the student.
    - "catalog_id" (integer, required): ID of the question from practice_questions_catalog.
    - "question_text" (string, required): The actual question text.
    - "model_answer" (string, required): The model answer.
    - "student_answer" (string, required): The student's submitted answer.
    - "question_type" (string, required): E.g., "Programming", "Short-Answer".

5.  "generate_assessment":
    Entities:
    - "teacher_name" (string, optional): Name of the teacher.
    - "teacher_id" (integer, optional): ID of the teacher.
    - "teaching_plan_content" (string, required): Detailed content of a teaching plan.
    - "question_preferences" (dict, optional): E.g., {{"multiple-choice": 3, "short-answer": 2}}
    - "title_for_db" (string, optional): Optional title for saving.

6.  "evaluate_assessment":
    Entities:
    - "student_id" (integer, optional): Unique identifier for the student.
    - "student_name" (string, required if student_id is not provided, recommended otherwise): Name of the student.
    - "assessment_id" (integer, required): Unique identifier for the assessment being answered.
    - "answers" (array of objects, required): List of the student's answers. Each object in the array MUST have the following two keys:
        - "question_identifier" (string, required): Identifier for the question (e.g., "Question 1", "1a", "Section A Q1").
        - "student_answer_text" (string, required): The student's actual answer text for that question.
      Example for "answers" entity: `[{"question_identifier": "Q1", "student_answer_text": "The capital of France is Paris."}, {"question_identifier": "Q2", "student_answer_text": "Photosynthesis is how plants make food using sunlight."}]`

If the query is ambiguous or information is missing for a required entity for a clearly identifiable intent, set intent to "clarification_needed" and provide a message explaining what's missing in the "entities" field as a "message" key (e.g., entities: {{"message": "Missing practice_topic for generate_practice_questions intent."}}).
If the query does not match any known intent, set intent to "unknown_intent" and the "entities" can be an empty dictionary or contain any recognized parts of the query.

User Query: {{user_query}}

Your JSON Output:
"""

async def process_natural_language_query(nl_input: NaturalLanguageQueryInput) -> NLQueryResponse:
    if llm is None:
        return NLQueryResponse(
            status="Failure",
            message="NLP processing service not available due to missing LLM configuration (API key).",
            original_query=nl_input.query
        )

    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=SYSTEM_PROMPT_TEXT),
        HumanMessage(content="{user_query}")
    ])

    chain = prompt | llm | output_parser

    try:
        llm_response_json = await chain.ainvoke({
            "user_query": nl_input.query,
            "format_instructions": output_parser.get_format_instructions()
        })

        intent = llm_response_json.get("intent")
        entities = llm_response_json.get("entities", {})

        if intent == "generate_practice_questions":
            practice_topic = entities.get("practice_topic")
            if not practice_topic:
                return NLQueryResponse(
                    status="ClarificationNeeded",
                    message="Missing required entity: 'practice_topic' for generating practice questions.",
                    original_query=nl_input.query,
                    detected_intent=intent,
                    extracted_entities=entities
                )

            student_id = entities.get("student_id")
            if student_id is not None:
                try:
                    student_id = int(student_id)
                except ValueError:
                    return NLQueryResponse(status="ClarificationNeeded", message=f"Invalid student_id format: '{student_id}'. Must be an integer.", original_query=nl_input.query, detected_intent=intent, extracted_entities=entities)

            question_prefs_raw = entities.get("question_preferences", {})
            if isinstance(question_prefs_raw, str): # LLM might return JSON string
                try:
                    question_prefs = json.loads(question_prefs_raw)
                except json.JSONDecodeError:
                    question_prefs = {} # Default or error
            else:
                question_prefs = question_prefs_raw


            practice_input = PracticeQuestionsInput(
                practice_topic=practice_topic,
                student_id=student_id,
                student_name=entities.get("student_name"),
                question_preferences=question_prefs if question_prefs else {"multiple-choice": 1, "short-answer": 1, "programming": 1} # Ensure default if empty
            )
            service_response = await generate_practice_questions_service(practice_input)
            return NLQueryResponse(
                status="Success" if not service_response.error_message else "ServiceError",
                message=service_response.error_message if service_response.error_message else "Practice questions generated.",
                original_query=nl_input.query,
                detected_intent=intent,
                extracted_entities=entities,
                service_response=service_response
            )

        elif intent == "student_question":
            question = entities.get("question")
            if not question:
                return NLQueryResponse(
                    status="ClarificationNeeded",
                    message="Missing required entity: 'question' for student Q&A.",
                    original_query=nl_input.query,
                    detected_intent=intent,
                    extracted_entities=entities
                )

            student_id = entities.get("student_id")
            if student_id is not None:
                try:
                    student_id = int(student_id)
                except ValueError:
                     return NLQueryResponse(status="ClarificationNeeded", message=f"Invalid student_id format: '{student_id}'. Must be an integer.", original_query=nl_input.query, detected_intent=intent, extracted_entities=entities)

            student_qa_input = StudentQuestionInput(
                question=question,
                student_id=student_id
            )
            service_response = await process_student_question_service(student_qa_input)
            return NLQueryResponse(
                status="Success" if not service_response.error_message else "ServiceError",
                message=service_response.error_message if service_response.error_message else "Student question answered.",
                original_query=nl_input.query,
                detected_intent=intent,
                extracted_entities=entities,
                service_response=service_response
            )

        elif intent == "generate_teaching_plan":
            subject = entities.get("subject")
            teaching_outline = entities.get("teaching_outline")

            if not subject:
                return NLQueryResponse(
                    status="ClarificationNeeded",
                    message="Missing required entity: 'subject' for generating a teaching plan.",
                    original_query=nl_input.query,
                    detected_intent=intent,
                    extracted_entities=entities
                )
            if not teaching_outline:
                return NLQueryResponse(
                    status="ClarificationNeeded",
                    message="Missing required entity: 'teaching_outline' for generating a teaching plan.",
                    original_query=nl_input.query,
                    detected_intent=intent,
                    extracted_entities=entities
                )

            teacher_id = entities.get("teacher_id")
            if teacher_id is not None:
                try:
                    teacher_id = int(teacher_id)
                except ValueError:
                    return NLQueryResponse(status="ClarificationNeeded", message=f"Invalid teacher_id format: '{teacher_id}'. Must be an integer.", original_query=nl_input.query, detected_intent=intent, extracted_entities=entities)

            # Constructing TeachingPlanInput to pass to the service,
            # assuming generate_initial_teaching_plan_service can accept it or its fields.
            # Based on previous endpoint implementation, the service might take individual params.
            # Let's create the input object first, then decide how to call the service.
            teaching_plan_input_data = TeachingPlanInput(
                subject=subject,
                teaching_outline=teaching_outline,
                teacher_id=teacher_id,
                teacher_name=entities.get("teacher_name"),
                style_tone=entities.get("style_tone"),
                output_structure=entities.get("output_structure"),
                title_for_db=entities.get("title_for_db")
            )

            # The service generate_initial_teaching_plan_service in services.py
            # (as used in endpoints.py) takes individual arguments.
            generated_content, rag_snippets_used = await generate_initial_teaching_plan_service(
                teacher_name=teaching_plan_input_data.teacher_name if teaching_plan_input_data.teacher_name else "LLM Generated Plan", # Provide a default if None
                subject=teaching_plan_input_data.subject,
                initial_outline=teaching_plan_input_data.teaching_outline,
                style_tone=teaching_plan_input_data.style_tone,
                output_structure=teaching_plan_input_data.output_structure
                # Note: The service in endpoints.py also handles saving and teacher creation.
                # Here, we are just calling the core generation part.
                # The NLQueryResponse.service_response should ideally match one of the defined output models.
                # For now, we'll wrap what the service returns into a TeachingPlanOutput.
            )

            # The service returns (generated_content, rag_snippets_used)
            # We need to adapt this to fit into NLQueryResponse.service_response which expects a Pydantic model.
            # Let's assume the primary output is the generated_content.
            # The full TeachingPlanOutput model includes fields like teaching_plan_id, title, etc.,
            # which are typically populated when saving. The NLP handler might not save directly.
            # For now, we construct a simplified TeachingPlanOutput.

            # If the service directly returns a TeachingPlanOutput object, that would be ideal.
            # Assuming the current service call returns text content.
            # We need to create a TeachingPlanOutput object.
            service_response_model = TeachingPlanOutput(
                title=teaching_plan_input_data.title_for_db or f"Plan for {subject}", # Use provided title or generate one
                subject=subject,
                generated_plan_content=generated_content, # This is the main output from the service
                teacher_id=teacher_id, # If available
                # teacher_name not directly in TeachingPlanOutput, but could be added if needed
                error_message=None # Assuming success if we got here
            )
            # If generated_content is None or empty, it indicates an error from the service.
            if not generated_content:
                 service_response_model.error_message = "Failed to generate teaching plan content from LLM service."


            return NLQueryResponse(
                status="Success" if not service_response_model.error_message else "ServiceError",
                message=service_response_model.error_message if service_response_model.error_message else "Teaching plan generated.",
                original_query=nl_input.query,
                detected_intent=intent,
                extracted_entities=entities,
                service_response=service_response_model
            )

        elif intent == "get_practice_feedback":
            # Required entities: student_id, catalog_id, question_text, model_answer, student_answer, question_type
            required_str_fields = ["question_text", "model_answer", "student_answer", "question_type"]
            required_int_fields = ["student_id", "catalog_id"]
            missing_fields = []
            parsed_entities = {}

            for field in required_str_fields:
                value = entities.get(field)
                if not value:
                    missing_fields.append(field)
                else:
                    parsed_entities[field] = value

            for field in required_int_fields:
                value_str = entities.get(field)
                if not value_str: # Check if entity exists
                    missing_fields.append(field)
                    continue
                try:
                    parsed_entities[field] = int(value_str)
                except ValueError:
                    return NLQueryResponse(
                        status="ClarificationNeeded",
                        message=f"Invalid format for entity '{field}': '{value_str}'. Must be an integer.",
                        original_query=nl_input.query,
                        detected_intent=intent,
                        extracted_entities=entities
                    )

            if missing_fields:
                return NLQueryResponse(
                    status="ClarificationNeeded",
                    message=f"Missing required entities for 'get_practice_feedback': {', '.join(missing_fields)}.",
                    original_query=nl_input.query,
                    detected_intent=intent,
                    extracted_entities=entities
                )

            feedback_input_data = PracticeFeedbackInput(
                student_id=parsed_entities["student_id"],
                catalog_id=parsed_entities["catalog_id"],
                question_text=parsed_entities["question_text"],
                model_answer=parsed_entities["model_answer"],
                student_answer=parsed_entities["student_answer"], # Renamed from student_answer_text in previous model def
                question_type=parsed_entities["question_type"]
            )

            service_response = await get_practice_feedback_service(feedback_input_data)

            return NLQueryResponse(
                status="Success" if not service_response.error_message else "ServiceError",
                message=service_response.error_message if service_response.error_message else "Feedback provided.",
                original_query=nl_input.query,
                detected_intent=intent,
                extracted_entities=entities, # Return the original entities from LLM for transparency
                service_response=service_response
            )

        elif intent == "generate_assessment":
            teaching_plan_content = entities.get("teaching_plan_content")
            if not teaching_plan_content:
                return NLQueryResponse(
                    status="ClarificationNeeded",
                    message="Missing required entity: 'teaching_plan_content' for generating an assessment.",
                    original_query=nl_input.query,
                    detected_intent=intent,
                    extracted_entities=entities
                )

            teacher_id = entities.get("teacher_id")
            if teacher_id is not None:
                try:
                    teacher_id = int(teacher_id)
                except ValueError:
                    return NLQueryResponse(status="ClarificationNeeded", message=f"Invalid teacher_id format: '{teacher_id}'. Must be an integer.", original_query=nl_input.query, detected_intent=intent, extracted_entities=entities)

            question_prefs_raw = entities.get("question_preferences", {}) # Default to empty dict
            question_prefs = {}
            if isinstance(question_prefs_raw, str):
                try:
                    question_prefs = json.loads(question_prefs_raw)
                    if not isinstance(question_prefs, dict): # Ensure it's a dict after loading
                        question_prefs = {}
                        # Optionally, could return ClarificationNeeded if format is wrong after parsing
                except json.JSONDecodeError:
                    # Optional: return ClarificationNeeded if string is not valid JSON
                    # For now, defaults to empty dict, meaning default preferences will be used by service
                    print(f"Warning: Could not parse 'question_preferences' string: {question_prefs_raw}")
                    question_prefs = {}
            elif isinstance(question_prefs_raw, dict):
                question_prefs = question_prefs_raw
            # If not string or dict, it will default to empty dict by service input model if not provided

            assessment_input_data = AssessmentInput(
                teaching_plan_content=teaching_plan_content,
                teacher_id=teacher_id,
                teacher_name=entities.get("teacher_name"),
                question_preferences=question_prefs if question_prefs else {"multiple-choice": 2, "short-answer": 2, "programming": 0}, # Default from model
                title_for_db=entities.get("title_for_db")
            )

            # generate_assessment_service is expected to take AssessmentInput and return (str, list)
            generated_content, rag_keywords_list = await generate_assessment_service(assessment_input_data)

            if generated_content:
                service_output = AssessmentOutput(
                    generated_assessment_content=generated_content,
                    # Populate other fields of AssessmentOutput as available/relevant
                    # title and teacher_id can be passed from input if needed for response consistency
                    title=assessment_input_data.title_for_db,
                    teacher_id=assessment_input_data.teacher_id,
                    # assessment_id is not set here as it's not saved by this handler path directly
                )
                return NLQueryResponse(
                    status="Success",
                    message="Assessment generated successfully.",
                    original_query=nl_input.query,
                    detected_intent=intent,
                    extracted_entities=entities,
                    service_response=service_output
                )
            else:
                error_msg = "Failed to generate assessment content from LLM service."
                if rag_keywords_list: # Potentially include for debugging
                    error_msg += f" RAG keywords extracted: {rag_keywords_list}."
                return NLQueryResponse(
                    status="ServiceError",
                    message=error_msg,
                    original_query=nl_input.query,
                    detected_intent=intent,
                    extracted_entities=entities
                )

        elif intent == "evaluate_assessment":
            assessment_id_str = entities.get("assessment_id")
            answers_raw = entities.get("answers") # Expecting a list of dicts
            student_id_str = entities.get("student_id")
            student_name = entities.get("student_name")

            # Validate required fields for this intent
            if not assessment_id_str:
                return NLQueryResponse(status="ClarificationNeeded", message="Missing required entity: 'assessment_id'.", original_query=nl_input.query, detected_intent=intent, extracted_entities=entities)
            if not answers_raw:
                return NLQueryResponse(status="ClarificationNeeded", message="Missing required entity: 'answers' (list of student's answers).", original_query=nl_input.query, detected_intent=intent, extracted_entities=entities)
            if not student_name and not student_id_str: # Must have at least one student identifier
                 return NLQueryResponse(status="ClarificationNeeded", message="Missing required information: 'student_name' or 'student_id'.", original_query=nl_input.query, detected_intent=intent, extracted_entities=entities)

            # Validate student_name if student_id is not provided, or if model strictly requires it
            # StudentAssessmentInput requires student_name.
            if not student_name:
                return NLQueryResponse(status="ClarificationNeeded", message="Missing required entity: 'student_name'. This is required for evaluating assessments.", original_query=nl_input.query, detected_intent=intent, extracted_entities=entities)

            # Type conversion for assessment_id
            try:
                assessment_id = int(assessment_id_str)
            except ValueError:
                return NLQueryResponse(status="ClarificationNeeded", message=f"Invalid format for 'assessment_id': '{assessment_id_str}'. Must be an integer.", original_query=nl_input.query, detected_intent=intent, extracted_entities=entities)

            # Type conversion for student_id (if present)
            student_id = None
            if student_id_str:
                try:
                    student_id = int(student_id_str)
                except ValueError:
                    return NLQueryResponse(status="ClarificationNeeded", message=f"Invalid format for 'student_id': '{student_id_str}'. Must be an integer.", original_query=nl_input.query, detected_intent=intent, extracted_entities=entities)

            # Validate 'answers' structure
            if not isinstance(answers_raw, list) or not answers_raw: # Must be a non-empty list
                return NLQueryResponse(status="ClarificationNeeded", message="'answers' entity must be a non-empty list of answer objects.", original_query=nl_input.query, detected_intent=intent, extracted_entities=entities)

            parsed_answers_for_input_model = []
            for i, ans_item in enumerate(answers_raw):
                if not isinstance(ans_item, dict):
                    return NLQueryResponse(status="ClarificationNeeded", message=f"Each item in 'answers' list must be a dictionary. Item at index {i} is not.", original_query=nl_input.query, detected_intent=intent, extracted_entities=entities)

                q_identifier = ans_item.get("question_identifier")
                s_answer_text = ans_item.get("student_answer_text")

                if not q_identifier or not isinstance(q_identifier, str) or not q_identifier.strip():
                    return NLQueryResponse(status="ClarificationNeeded", message=f"Missing or invalid 'question_identifier' (must be a non-empty string) in answer item at index {i}.", original_query=nl_input.query, detected_intent=intent, extracted_entities=entities)
                if s_answer_text is None or not isinstance(s_answer_text, str): # Allow empty string for an answer, but type must be string
                     return NLQueryResponse(status="ClarificationNeeded", message=f"Missing or invalid 'student_answer_text' (must be a string) in answer item at index {i}.", original_query=nl_input.query, detected_intent=intent, extracted_entities=entities)

                parsed_answers_for_input_model.append(StudentAssessmentAnswerItem(question_identifier=q_identifier.strip(), student_answer_text=s_answer_text))

            # Create StudentAssessmentInput instance
            student_assessment_input_data = StudentAssessmentInput(
                student_id=student_id, # student_id is optional in the model if student_name is present
                student_name=student_name, # student_name is required in the model
                assessment_id=assessment_id,
                answers=parsed_answers_for_input_model
            )

            # Call the service
            service_response_list = await evaluate_student_assessment_answers_service(student_assessment_input_data)
            # service_response_list is List[StudentAssessmentEvaluationOutput]

            return NLQueryResponse(
                status="Success",
                message="Assessment answers evaluated successfully.",
                original_query=nl_input.query,
                detected_intent=intent,
                extracted_entities=entities, # Include original LLM entities for transparency
                service_response=service_response_list
            )

        elif intent == "clarification_needed":
            return NLQueryResponse(
                status="ClarificationNeeded",
                message=entities.get("message", "The query is ambiguous or missing information. Please provide more details."),
                original_query=nl_input.query,
                detected_intent=intent,
                extracted_entities=entities
            )

        elif intent == "unknown_intent":
            return NLQueryResponse(
                status="UnknownIntent",
                message="Could not understand the request intent from the query.",
                original_query=nl_input.query,
                detected_intent=intent, # Could be "unknown_intent" or what LLM thought
                extracted_entities=entities
            )

        else: # Other intents are defined in prompt but not yet implemented here
            return NLQueryResponse(
                status="PendingImplementation",
                message=f"Intent '{intent}' recognized but not yet implemented.",
                original_query=nl_input.query,
                detected_intent=intent,
                extracted_entities=entities
            )

    except json.JSONDecodeError as e: # Error parsing LLM's JSON output
        # Log the raw output from LLM if possible before parsing for debugging
        # This error means the LLM output was not valid JSON as expected by JsonOutputParser
        print(f"NLP Handler JSON Parsing Error: {e}. LLM output might not be valid JSON.")
        return NLQueryResponse(
            status="LLMOutputParsingError",
            message="Failed to parse the structured response from the language model. The output was not valid JSON.",
            original_query=nl_input.query,
            # Potentially include the raw LLM output string if it's safe and useful for debugging client-side,
            # or just log it server-side. For now, keeping it server-side.
        )
    except Exception as e:
        print(f"NLP Handler Error: An unexpected error occurred: {e}")
        # Log the full exception
        return NLQueryResponse(
            status="Failure",
            message=f"An unexpected error occurred during NLP processing: {str(e)}",
            original_query=nl_input.query
        )
