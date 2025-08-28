from google import genai
from dotenv import load_dotenv
import os

client = genai.Client()

class QueryForBackground:
    def __init__(self, query):
        self.query = query
        self.personality = "You are a helpful history assistant that is most knowledgeable " \
        "in the field of endangered cultures like Ainu, Khasi, and Welsh. You excel in providing meaningful " \
        f"contextual information about this given prompt: {self.query}. " \
        "Ensure that your response is structured for ase of understanding and informative"

    def get_background(self):
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=self.personality
        )
        return response.text
    
class QueryForFlashcards:
    def __init__(self, subject, num_cards):
        self.personality = f"You are a helpful history teacher who aspires to better educate other people through flashcards for better practice." \
        f"Can you generate a set of {num_cards} flashcards with the subject of {subject}. " \
        "Ensure your response structure is in JSON format with 'question' and 'answer' keys for each flashcard."

    def get_flashcards(self):
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=self.personality
        )
        return response.text
