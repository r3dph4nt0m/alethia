from fastapi import FastAPI, APIRouter, HTTPException
from pydantic import BaseModel
from typing import List

app = FastAPI(title="Alethia Endangered Languages API")
user_router = APIRouter(prefix="/user", tags=["User Management"])
ai_router = APIRouter(prefix="/agent", tags=["Language Agent"])

app.include_router(user_router)
app.include_router(ai_router)
