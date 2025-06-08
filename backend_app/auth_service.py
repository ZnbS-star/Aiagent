import os
from typing import Optional, Dict, Any
from fastapi import HTTPException, status
from backend_app.database_utils import (
    get_teacher_for_auth, save_teacher_registration,
    get_student_for_auth, save_student_registration
)
from backend_app.security import get_password_hash, verify_password, create_access_token
from backend_app.models import UserCreate, UserLogin, Token

ADMIN_USERNAME = os.environ.get("ADMIN_USERNAME", "admin")
ADMIN_PASSWORD = os.environ.get("ADMIN_PASSWORD", "supersecretpassword")


async def unified_register_service(user_data: UserCreate) -> Dict[str, Any]:
    hashed_password = get_password_hash(user_data.password)
    if user_data.role ==3:
        existing_teacher = await get_teacher_for_auth(user_data.username)
        if existing_teacher:
            raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="该教师用户名已被注册。",
            )
        new_teacher = await save_teacher_registration(user_data.username, hashed_password)
        if not new_teacher:
            raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="创建教师账户失败。",
        )
        return {"teacher_id": new_teacher["teacher_id"], "teacher_name": new_teacher["teacher_name"]}
    elif user_data.role == 2:
        existing_student = await get_student_for_auth(user_data.username)
        if existing_student:
            raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="该学生用户名已被注册。",
            )
        new_student = await save_student_registration(user_data.username, hashed_password)
        if not new_student:
            raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="创建教师账户失败。",
        )
        return {"student_id": new_student["teacher_id"], "student_name": new_student["teacher_name"]}
    elif user_data.role == 1:
        # 管理员不能通过此接口注册
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="管理员账户不能通过公开接口注册。",
        )
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"无效的角色: '{user_data.role}'。有效角色为 'student' 或 'teacher'。",
        )

async def unified_login_service(form_data: UserLogin) -> Token:
    
    if form_data.role == 3:
        teacher = await get_teacher_for_auth(form_data.username)
        if not teacher or not verify_password(form_data.password, teacher["hashed_password"]):
            raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="用户名或密码不正确。",
            headers={"WWW-Authenticate": "Bearer"},
            )
    
    # 创建Token
        access_token = create_access_token(
        data={"sub": teacher["teacher_name"], "role": "teacher"}
    )
        return Token(access_token=access_token, token_type="bearer")
    elif form_data.role == 2:
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
    elif form_data.role == 1:
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
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"无效的角色: '{form_data.role}'。有效角色为 'student', 'teacher' 或 'admin'。",
        )