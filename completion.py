# Import from standard library
import os
import cohere
import logging
import streamlit as st
from dotenv import load_dotenv

# Configure logger
logging.getLogger("complete").setLevel(logging.WARNING)


# Load environment variables
load_dotenv()


# Assign credentials from environment variable or streamlit secrets dict
# co = cohere.Client(os.getenv("COHERE_API_KEY")) or st.secrets["COHERE_API_KEY"]
COHERE_API_KEY = cohere.Client(os.getenv("COHERE_API_KEY")) or st.secrets["COHERE_API_KEY"]
cohere_client = cohere.Client(COHERE_API_KEY)

class Completion:

    def ___init___(self, ):
        pass

    @staticmethod
    def complete(prompt_input, max_tokens, temperature):
      
        """
        Call Cohere Completion with text prompt.
        Args:
            prompt: text prompt
            max_tokens: max number of tokens to generate
            temperature: temperature for generation
        Return: predicted response text
        """
       
        try:
            response = cohere_client.generate(  
                model = 'command-nightly',
                prompt = str(prompt_input),
                max_tokens = max_tokens,
                temperature = temperature)
            
            print("generated text:\n", response.generations[0].text)
            return response.generations[0].text

        
        except Exception as e:
            logging.error(f"Cohere API error: {e}")
            st.session_state.text_error = f"Cohere API error: {e}"
            print("Error:", e)
