import torch
import os
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import sys
import proxy_bypass

# Add the current directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

# Now import TrainingSession
from TrainingSession import TrainingSession
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try different model paths based on your checkpoint structure
possible_model_paths = [
    "/Users/jaceysimpson/Vscode/EndangeredCultures/model/language_classifier/checkpoint-49200", 
    "/Users/jaceysimpson/Vscode/EndangeredCultures/model/language_classifier/checkpoint-49000",
]

model_path = None
for path in possible_model_paths:
    logger.info(f"Checking model path: {path}")
    if os.path.exists(path) and os.path.exists(os.path.join(path, "config.json")):
        logger.info(f"✓ Found valid model at: {path}")
        model_path = path
        break

if not model_path:
    logger.error("No valid model found. Available checkpoints:")
    checkpoint_dir = "/Users/jaceysimpson/Vscode/EndangeredCultures/model/language_classifier"
    if os.path.exists(checkpoint_dir):
        for item in os.listdir(checkpoint_dir):
            logger.info(f"  - {item}")
    exit(1)

proxy_bypass._configure_proxy_bypass()
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

try:
    model = GPT2LMHeadModel.from_pretrained(model_path)
    logger.info(f"Successfully loaded model from: {model_path}")
except Exception as e:
    logger.error(f"Error loading model: {e}")
    exit(1)

model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def test_language_model():
    # Change to the correct directory for data loading
    os.chdir("/Users/jaceysimpson/Vscode/EndangeredCultures")  # Project root for data access
    
    training_params = TrainingSession()
    test_dataset = training_params.test_dataset

    logger.info(f"Testing on {len(test_dataset)} samples")
    
    # Test a few samples
    for i, batch in enumerate(test_dataset):
        if i >= 5:  # Test only first 5 samples
            break
            
        # Get the input text properly
        input_ids = batch['input_ids'].unsqueeze(0)  # Add batch dimension
        attention_mask = batch['attention_mask'].unsqueeze(0)
        
        # Truncate input to leave room for generation
        max_input_length = 400  # Leave room for 112 new tokens
        if input_ids.shape[1] > max_input_length:
            input_ids = input_ids[:, :max_input_length]
            attention_mask = attention_mask[:, :max_input_length]
        
        # Decode the input to see what we're testing
        input_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
        
        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids.to(device),
                attention_mask=attention_mask.to(device),
                max_new_tokens=50,  # Generate 50 new tokens instead of max_length
                num_return_sequences=1,
                pad_token_id=tokenizer.eos_token_id,
                do_sample=True,
                top_k=50,
                top_p=0.95,
                temperature=0.8,
                eos_token_id=tokenizer.eos_token_id,
            )
            
            generated_text = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
            for i in range(len(generated_text)):
                logger.info(f"Input {i+1}: {input_text}")
                logger.info(f"Generated {i+1}: {generated_text[i]}")
                logger.info("-" * 40)


def test_with_custom_prompts():
    """Test with shorter, more focused prompts"""
    logger.info("Testing with custom prompts...")
    
    test_prompts = [
        "Dialogue: E-iperusuy yakun aep poronno an na e yan",
        "Dialogue: Ka kmie bad u kpa jong ka ki bitar",
        "Dialogue: Gwell hwyr na hwyrach",
        "Dialogue: Sut mae'r tywydd heddiw?",
        "Dialogue: A-kor wakka poro no an na.",
        "Dialogue: Nga kwah ban leit sha tea.",
        "Dialogue: Beth wyt ti'n hoffi wneud?",
        "Dialogue: Ku-e rusuy aep poronno an na e yan."
    ]
    
    for i, prompt in enumerate(test_prompts):
        inputs = tokenizer.encode(prompt, return_tensors="pt")
        
        with torch.no_grad():
            outputs = model.generate(
                inputs.to(device),
                max_new_tokens=30,  # Generate 30 new tokens
                num_return_sequences=1,
                pad_token_id=tokenizer.eos_token_id,
                do_sample=True,
                top_k=50,
                top_p=0.95,
                temperature=0.8,
                eos_token_id=tokenizer.eos_token_id,
            )
            
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            new_text = generated_text[len(prompt):]
            
            logger.info(f"Prompt {i+1}: {prompt}")
            logger.info(f"Generated: {new_text}")
            logger.info("-" * 40)

if __name__ == "__main__":
    # Test with dataset samples
    test_language_model()
    
    # Test with custom prompts
    test_with_custom_prompts()