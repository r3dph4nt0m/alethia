import app
import pandas as pd
from fastapi import HTTPException
from pydantic import BaseModel
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from src.backend.BackgroundsAgent import QueryForBackground, QueryForFlashcards

class AIRequest(BaseModel):
    prompt: str
    max_length: int = 512
    num_return_sequences: int = 1
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("model/language_classifier/checkpoint-49200")

    def process_request(self) -> list:
        inputs = self.tokenizer.encode(self.prompt, return_tensors="pt")
        output = self.model.generate(inputs, max_length=self.max_length, num_return_sequences=self.num_return_sequences)
        text = self.tokenizer.decode(output, skip_special_tokens=True)
        classification = text.split("Classification:")[-1].strip().split("\n")[0]
        translation = text.split("Translation:")[-1].strip().split("\n")[0]
        return [{"generated_text": text, "classification": classification, "translation": translation}]
    
@app.ai_router.post("/generate")
def generate_text(request: AIRequest):
    try:
        responses = request.process_request()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return {"responses": responses}
@app.ai_router.post("/info")
def get_info(query: str):
    try:
        background_agent = QueryForBackground(query)
        background_info = background_agent.get_background()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return {"background_info": background_info}

@app.ai_router.post("/flashcards")
def get_flashcards(subject: str, num_cards: int):
    try:
        flashcard_agent = QueryForFlashcards(subject, num_cards)
        flashcards = flashcard_agent.get_flashcards() #NOTE: This returns in JSON format
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return {"flashcards": flashcards}