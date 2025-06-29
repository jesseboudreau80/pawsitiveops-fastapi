from dotenv import load_dotenv
load_dotenv()  # Load environment variables before anything else

from fastapi import FastAPI
from backend.app.api.api_v1.endpoints.ask_sam import router as sam_router

app = FastAPI(
    title="PAWSitiveOps API",
    description="FastAPI backend for compliance and chat agents",
    version="0.1.0"
)

@app.get("/")
def root():
    return {"message": "PAWSitiveOps API is live!"}

app.include_router(sam_router, prefix="/api/v1", tags=["ask_sam"])

