from fastapi import APIRouter, Body
from backend.app.services.openai_handler import get_sam_response

router = APIRouter()

@router.post("/ask-sam")
def ask_sam(message: str = Body(..., embed=True)):
    response = get_sam_response(message)
    return {"reply": response}
