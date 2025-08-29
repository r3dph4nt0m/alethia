from fastapi import FastAPI, APIRouter
import uvicorn

app = FastAPI(title="Alethia Endangered Languages API")
user_router = APIRouter(prefix="/user", tags=["User Management"])
ai_router = APIRouter(prefix="/agent", tags=["Language Agent"])

app.include_router(user_router) # user_router.py
app.include_router(ai_router) # ai_router.py

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)