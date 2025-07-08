from fastapi import FastAPI
import uvicorn
from backend_app.api import endpoints # Import the router from endpoints.py
from fastapi.middleware.cors import CORSMiddleware
# --- FastAPI App Initialization ---
app = FastAPI()
origins = [
    "http://localhost",
    "http://localhost:8080",  # 假设你的Vue开发服务器运行在这个端口
    "http://localhost:5173",  # 假设你的Vite开发服务器运行在这个端口
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # 允许访问的源列表
    allow_credentials=True,  # 是否支持携带cookie
    allow_methods=["*"],  # 允许所有的请求方法 (GET, POST, PUT, DELETE, etc.)
    allow_headers=["*"],  # 允许所有的请求头
)
# --- Root Endpoint ---
@app.get("/", tags=["Root"])
async def read_root():
    """
    Root endpoint providing a welcome message and basic API information.
    """
    return {
        "message": "Welcome to the Teacher's AI Toolkit API!",
        "documentation_links": {
            "swagger_ui": "/docs",
            "redoc": "/redoc"
        },
        "version": app.version
    }

# --- Include API Routers ---
# All endpoints defined in backend_app.api.endpoints will be included here.
# The prefix makes all routes from that router start with /api/v1
app.include_router(endpoints.router, prefix="/api", tags=["Teacher Toolkit Features"])


# --- Optional: Add Uvicorn startup command info for local development ---
# This is for information when running this file directly (though usually uvicorn command is used).
if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8080) # This line is for direct execution test, not for prod.