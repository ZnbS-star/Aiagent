from fastapi import FastAPI
import uvicorn
from backend_app.api import endpoints 
from fastapi.middleware.cors import CORSMiddleware
app = FastAPI()
origins = [
    "http://localhost",
    "http://localhost:8080",  
    "http://localhost:5173",  
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  
    allow_credentials=True,  
    allow_methods=["*"],  
    allow_headers=["*"],  
    expose_headers=["Content-Disposition"]
)

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


app.include_router(endpoints.router, prefix="/api", tags=["Teacher Toolkit Features"])



if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8080) 