import app
from fastapi import HTTPException
from pydantic import BaseModel
import pandas as pd
import os
import numpy as np

class User(BaseModel):
    id: int
    username: str
    email: str
    password: str

    def verify(self, username):
        all_users = pd.read_csv("data/user_data/users.csv")
        if self.username in all_users['username'].values:
            found_user_data = all_users[username]
            if found_user_data['password'] == self.password:
                return True
        return False

    def to_csv(self, filename="data/user_data/users.csv"):
        if not os.path.exists(os.path.dirname(filename)):
            os.makedirs(os.path.dirname(filename))
        user_data = pd.read_csv(filename) if os.path.exists(filename) else pd.DataFrame(columns=["id", "username", "email", "password"])
        if self.id == None:
            self.id = np.random.randint(1000, 9999)
        user_data = pd.concat([user_data, pd.DataFrame([self.dict()])], ignore_index=True)
        user_data.to_csv(filename, index=False)

@app.user_router.get("/signup")
def signup(user: User):
    try:
        user.to_csv()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return {"message": "User created successfully", "user": user}

@app.user_router.get("/login")
def login(user: User):
    successful_log = user.verify(user.username)
    if successful_log:
        return {"message": "Login successful", "user": user}
    else:
        raise HTTPException(status_code=401, detail="Invalid username or password")