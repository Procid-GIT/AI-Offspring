import torch
from transformers import pipeline

def main():
    print("Loading AI model... This may take a moment.")
    
    # Use GPU if available
    device = 0 if torch.cuda.is_available() else -1
    
    # Load a pre-trained conversational AI model
    # Upgraded to medium model for better responses
    chatbot = pipeline("text-generation", model="microsoft/DialoGPT-small", device=device)
    
    print("AI Chatbot ready! Type 'quit' to exit.")
    print("You can ask me to generate code by saying 'Generate code for [task]'")
    
    conversation_history = []
    
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'quit':
            break
        
        # Add user input to conversation
        conversation_history.append(f"User: {user_input}")
        
        # Generate response
        if "generate code" in user_input.lower():
            # For code generation, use a larger code-specific model
            prompt = f"Write Python code for: {user_input.replace('generate code for', '').strip()}"
            code_generator = pipeline("text-generation", model="Salesforce/codegen-2B-mono", device=device)
            response = code_generator(prompt, max_length=300, num_return_sequences=1)[0]['generated_text']
        else:
            # Regular conversation
            full_prompt = " ".join(conversation_history[-5:])  # Keep last 5 exchanges
            response = chatbot(full_prompt, max_length=150, num_return_sequences=1)[0]['generated_text']
        
        print(f"AI: {response}")
        conversation_history.append(f"AI: {response}")

if __name__ == "__main__":
    main()
