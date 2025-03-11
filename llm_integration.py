import os
from langchain_openai import OpenAI  # Updated import
from dotenv import load_dotenv

load_dotenv()

def create_llm_provider(provider: str):
    """Creates an LLM provider instance based on configuration."""
    if provider.lower() == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        model = os.getenv("OPENAI_MODEL", "gpt-4o")
        if not api_key:
            raise ValueError("OPENAI_API_KEY must be set in .env for OpenAI provider")
        return OpenAI(api_key=api_key, model_name=model)  # Use model_name instead of model
    elif provider.lower() == "anthropic":
        # Note: Your original code imports Anthropic from langchain_community.llms, which may also be deprecated.
        # For simplicity, we'll skip this unless you're using Anthropic.
        raise NotImplementedError("Anthropic support not updated yet.")
    elif provider.lower() == "none":
        return None
    else:
        raise ValueError(f"Unsupported LLM provider: {provider}. Choose 'openai', 'anthropic', or 'none'.")