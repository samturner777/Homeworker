from fastapi import FastAPI, HTTPException, status
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from enum import Enum
from typing import List, Optional, Dict
import datetime
import time
import os
# ==========================================
# 1. ENUMS & PYDANTIC SCHEMAS
# ==========================================

class GradeLevel(str, Enum):
    EARLY_ELEMENTARY = "K-3 (Early Elementary)"
    LATE_ELEMENTARY = "4-5 (Late Elementary)"
    MIDDLE_SCHOOL = "6-8 (Middle School)"
    HIGH_SCHOOL = "9-12 (High School)"
    UNIVERSITY = "University / College"

class Subject(str, Enum):
    MATH = "Mathematics"
    SCIENCE = "Science"
    LANGUAGE_ARTS = "Language Arts / English"
    HISTORY = "History / Social Studies"
    COMPUTER_SCIENCE = "Computer Science"
    GENERAL = "General Studies"

class TaskStatus(str, Enum):
    NOT_STARTED = "Not Started"
    IN_PROGRESS = "In Progress"
    COMPLETED = "Completed"

# Request Models
class AssignmentCreate(BaseModel):
    title: str = Field(..., example="Counting Apples")
    subject: Subject = Field(..., example=Subject.MATH)
    description: str = Field(..., example="Learn basic addition using apple slices.")
    due_date: str = Field(..., example=str(datetime.date.today()))

class StudentCreate(BaseModel):
    name: str = Field(..., example="Leo")
    grade_level: GradeLevel = Field(..., example=GradeLevel.EARLY_ELEMENTARY)

class TutorRequest(BaseModel):
    student_id: int
    assignment_id: int
    question: str = Field(..., example="I don't know what 4 + 3 is!")

# Response Models
class AssignmentResponse(AssignmentCreate):
    id: int
    status: TaskStatus

class StudentResponse(BaseModel):
    id: int
    name: str
    grade_level: GradeLevel
    assignments: List[AssignmentResponse] = []

class TutorResponse(BaseModel):
    student_name: str
    grade_level: str
    system_prompt_used: str
    ai_response: str


# ==========================================
# 2. ADAPTIVE TUTORING ENGINE
# ==========================================

class AdaptiveTutorEngine:
    """Dynamically adjusts vocabulary, methodology, and guardrails based on grade tier."""
    
    @staticmethod
    def generate_system_prompt(student_name: str, grade_level: GradeLevel, subject: Subject) -> str:
        base_prompt = f"You are an expert educational tutor helping a student named {student_name} in {subject.value}. "
        
        level_rules = {
            GradeLevel.EARLY_ELEMENTARY: (
                "Use very simple words, short sentences, and fun emojis. "
                "Focus on direct encouragement and foundational concepts. Never give negative feedback. "
                "Explain things as if telling a fun story."
            ),
            GradeLevel.LATE_ELEMENTARY: (
                "Use clear, structured steps. Help the student understand *how* to find the answer "
                "rather than just giving it to them. Keep the tone friendly and supportive."
            ),
            GradeLevel.MIDDLE_SCHOOL: (
                "Strictly use the Socratic method. Ask guiding questions to lead the student to the answer. "
                "Break down complex problems into smaller, manageable chunks."
            ),
            GradeLevel.HIGH_SCHOOL: (
                "Maintain an academic and analytical tone. Help with critical thinking, advanced problem-solving, "
                "and structuring arguments. Do not do the work for them; provide frameworks and explanations."
            ),
            GradeLevel.UNIVERSITY: (
                "Adopt a rigorous, scholarly, and peer-like academic tone. Focus on research methodology, "
                "deep theoretical understanding, advanced debugging, and critical analysis. "
                "Assume high foundational knowledge."
            )
        }
        
        guardrails = (
            "\nCRITICAL ACADEMIC INTEGRITY RULE: Never output direct answers for homework or quizzes. "
            "Your goal is to guide the student to solve it themselves."
        )
        return base_prompt + level_rules[grade_level] + guardrails

    @staticmethod
    def get_simulated_response(grade_level: GradeLevel, student_name: str, question: str) -> str:
        simulated_responses = {
            GradeLevel.EARLY_ELEMENTARY: f"Hi {student_name}! 🌟 Let's look at this together! Instead of looking at the whole big number, what happens if we count just the first row of blocks? 🧱 Count them with me!",
            GradeLevel.LATE_ELEMENTARY: f"Great question, {student_name}! To tackle this problem, let's break it into two steps. What clue words do you see in your assignment that tell us what to do first?",
            GradeLevel.MIDDLE_SCHOOL: "I see what you're trying to do here! Instead of me giving you the direct answer, let's think about the underlying relationship. What happens to your primary variable when the condition changes?",
            GradeLevel.HIGH_SCHOOL: "To construct a strong solution here, let's analyze the framework. How would you challenge or test your initial hypothesis against the assignment requirements?",
            GradeLevel.UNIVERSITY: "Regarding your analytical approach: check the computational or theoretical constraints of your premise. Have you evaluated how an edge-case shift impacts your overall derivation?"
        }
        return simulated_responses.get(grade_level, "How can I guide you through this concept today?")


# ==========================================
# 3. FASTAPI APPLICATION & ROUTING
# ==========================================

app = FastAPI(
    title="OmniLearn AI Homework Assistant API",
    description="Adaptive tutoring API scaling from K-1 to University level.",
    version="1.0.0"
)

# Enable CORS for frontend interaction
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-Memory Database Simulation
db_students: Dict[int, Dict] = {}
db_assignments: Dict[int, Dict] = {}

# --- STATIC FRONTEND ROUTE ---
@app.get("/", tags=["Frontend"], summary="Serve Animated Landing Page")
async def serve_landing_page():
    return FileResponse("index.html")

# --- STUDENT ENDPOINTS ---
@app.post("/api/students", response_model=StudentResponse, status_code=status.HTTP_201_CREATED, tags=["Students"])
async def create_student(student: StudentCreate):
    student_id = int(time.time() * 1000)
    new_student = {
        "id": student_id,
        "name": student.name,
        "grade_level": student.grade_level,
        "assignments": []
    }
    db_students[student_id] = new_student
    return new_student

@app.get("/api/students/{student_id}", response_model=StudentResponse, tags=["Students"])
async def get_student(student_id: int):
    if student_id not in db_students:
        raise HTTPException(status_code=404, detail="Student not found.")
    return db_students[student_id]

# --- ASSIGNMENT ENDPOINTS ---
@app.post("/api/students/{student_id}/assignments", response_model=AssignmentResponse, status_code=status.HTTP_201_CREATED, tags=["Assignments"])
async def add_assignment(student_id: int, assignment: AssignmentCreate):
    if student_id not in db_students:
        raise HTTPException(status_code=404, detail="Student not found.")
    
    assign_id = int(time.time() * 1000) + 1
    new_assignment = {
        "id": assign_id,
        "title": assignment.title,
        "subject": assignment.subject,
        "description": assignment.description,
        "due_date": assignment.due_date,
        "status": TaskStatus.NOT_STARTED
    }
    db_assignments[assign_id] = new_assignment
    db_students[student_id]["assignments"].append(new_assignment)
    return new_assignment

# --- AI TUTOR ENGINE ENDPOINT ---
@app.post("/api/tutor/ask", response_model=TutorResponse, tags=["AI Tutor"])
async def ask_tutor(request: TutorRequest):
    if request.student_id not in db_students:
        raise HTTPException(status_code=404, detail="Student not found.")
    if request.assignment_id not in db_assignments:
        raise HTTPException(status_code=404, detail="Assignment not found.")
        
    student = db_students[request.student_id]
    assignment = db_assignments[request.assignment_id]
    
    # Generate grade-tailored prompt and response
    sys_prompt = AdaptiveTutorEngine.generate_system_prompt(
        student_name=student["name"],
        grade_level=student["grade_level"],
        subject=assignment["subject"]
    )
    ai_reply = AdaptiveTutorEngine.get_simulated_response(
        grade_level=student["grade_level"],
        student_name=student["name"],
        question=request.question
    )
    
    return TutorResponse(
        student_name=student["name"],
        grade_level=student["grade_level"].value,
        system_prompt_used=sys_prompt,
        ai_response=ai_reply
    )
