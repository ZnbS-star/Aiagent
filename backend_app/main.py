from fastapi import FastAPI
import uvicorn
from backend_app.api import endpoints # Import the router from endpoints.py
# --- FastAPI App Initialization ---
app = FastAPI()
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