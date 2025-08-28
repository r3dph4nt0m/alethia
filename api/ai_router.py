import app
import pandas as pd
from fastapi import HTTPException
from pydantic import BaseModel
import os
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

class AIRequest(BaseModel):
    prompt: str
    max_length: int = 512
    num_return_sequences: int = 1
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("model/language_classifier")

    def process_request(self):
        inputs = self.tokenizer.encode(self.prompt, return_tensors="pt")
        outputs = self.model.generate(inputs, max_length=self.max_length, num_return_sequences=self.num_return_sequences)
        return [self.tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
    
@app.ai_router.post("/generate")
def generate_text(request: AIRequest):
    try:
        responses = request.process_request()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return {"responses": responses}