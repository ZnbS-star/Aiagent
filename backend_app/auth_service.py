import os
from typing import Optional, Dict, Any
from fastapi import HTTPException, status
from backend_app.database_utils import (
    get_teacher_for_auth, save_teacher_registration,
    get_student_for_auth, save_student_registration
)
from backend_app.security import get_password_hash, verify_password, create_access_token
from backend_app.models import UserCreate, UserLogin, Token

# --- Teacher Authentication Service ---

async def register_teacher_service(user_data: UserCreate) -> Dict[str, Any]:
    """处理教师注册的业务逻辑"""
    # 检查用户是否已存在
    existing_teacher = await get_teacher_for_auth(user_data.username)
    if existing_teacher:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="该教师用户名已被注册。",
        )
    
    # 哈希密码
    hashed_password = get_password_hash(user_data.password)
    
    # 保存新教师
    new_teacher = await save_teacher_registration(user_data.username, hashed_password)
    if not new_teacher:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="创建教师账户失败。",
        )
        
    return {"teacher_id": new_teacher["teacher_id"], "teacher_name": new_teacher["teacher_name"]}


async def login_teacher_service(form_data: UserLogin) -> Token:
    """处理教师登录的业务逻辑"""
    teacher = await get_teacher_for_auth(form_data.username)
    if not teacher or not verify_password(form_data.password, teacher["hashed_password"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="用户名或密码不正确。",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    access_token = create_access_token(
        data={"sub": teacher["teacher_name"], "role": "teacher"}
    )
    return Token(access_token=access_token, token_type="bearer")


# --- Student Authentication Service ---

async def register_student_service(user_data: UserCreate) -> Dict[str, Any]:
    """处理学生注册的业务逻辑"""
    existing_student = await get_student_for_auth(user_data.username)
    if existing_student:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="该学生用户名已被注册。",
        )
        
    hashed_password = get_password_hash(user_data.password)
    new_student = await save_student_registration(user_data.username, hashed_password)
    
    if not new_student:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="创建学生账户失败。",
        )
        
    return {"student_id": new_student["student_id"], "student_name": new_student["student_name"]}


async def login_student_service(form_data: UserLogin) -> Token:
    """处理学生登录的业务逻辑"""
    student = await get_student_for_auth(form_data.username)
    if not student or not verify_password(form_data.password, student["hashed_password"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="用户名或密码不正确。",
            headers={"WWW-Authenticate": "Bearer"},
        )
        
    access_token = create_access_token(
        data={"sub": student["student_name"], "role": "student"}
    )
    return Token(access_token=access_token, token_type="bearer")

ADMIN_USERNAME = os.environ.get("ADMIN_USERNAME")
ADMIN_PASSWORD = os.environ.get("ADMIN_PASSWORD")

async def login_admin_service(form_data: UserLogin) -> Token:
    if form_data.username != ADMIN_USERNAME or form_data.password != ADMIN_PASSWORD:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="管理员用户名或密码不正确。",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    access_token = create_access_token(
        data={"sub": ADMIN_USERNAME, "role": "admin"}
    )
    return Token(access_token=access_token, token_type="bearer")