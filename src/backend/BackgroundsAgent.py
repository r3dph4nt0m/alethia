from google import genai
from dotenv import load_dotenv
import json

client = genai.Client()

class QueryForBackground: #Query Gemini 2.5 Flash for background information or historical context for subjects
    def __init__(self, query: str):
        self.query = query
        self.personality = "You are a helpful history assistant that is most knowledgeable " \
        "in the field of endangered cultures like Ainu, Khasi, and Welsh. You excel in providing meaningful " \
        f"contextual information about this given prompt: {self.query}. " \
        "Ensure that your response is structured for ase of understanding and informative"

    def get_background(self) -> str:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=self.personality
        )
        return response.text

class QueryForFlashcards: #Query Gemini 2.5 Flash to generate a set of flashcards to practice from in JSON format
    def __init__(self, subject: str, num_cards: int):
        self.personality = f"You are a helpful history teacher who aspires to better educate other people through flashcards for better practice." \
        f"Can you generate a set of {num_cards} flashcards with the subject of {subject}. " \
        "Ensure your response structure is in JSON format with 'question' and 'answer' keys for each flashcard." \
        "For example: [{\"question\": \"What is...\", \"answer\": \"It is...\"}, ...]"

    def get_flashcards(self) -> dict:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=self.personality
        )
        json_response = json.loads(response.text)
        return json_response

