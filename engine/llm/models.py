"""Centralized LLM model management.

This module provides a single point of access for all LLM models used across
the application. Models are instantiated once and reused to avoid redundant
initialization.

Usage:
    from engine.llm.models import ModelManager
    
    # Get specific models
    gemini = ModelManager.get_gemini()
    groq = ModelManager.get_groq()
    ollama = ModelManager.get_ollama()
    
    # Or use the factory methods for specific use cases
    model = ModelManager.get_model_for_image_processing()
    model = ModelManager.get_model_for_text_extraction()
"""

from typing import Optional

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama


class ModelManager:
    """Singleton-like manager for LLM model instances.
    
    Provides lazy initialization and caching of models to ensure efficient
    resource usage across the application.
    """

    _gemini_instance: Optional[ChatGoogleGenerativeAI] = None
    _groq_instance: Optional[ChatGroq] = None
    _ollama_instance: Optional[ChatOllama] = None

    @classmethod
    def get_gemini(
        cls, model: str = "gemini-2.5-flash"
    ) -> ChatGoogleGenerativeAI:
        """Get or create Gemini model instance.
        
        Args:
            model: Model name. Default is gemini-2.5-flash
            
        Returns:
            ChatGoogleGenerativeAI instance
        """
        if cls._gemini_instance is None:
            cls._gemini_instance = ChatGoogleGenerativeAI(model=model)
        return cls._gemini_instance

    @classmethod
    def get_groq(
        cls, model: str = "llama-3.3-70b-versatile", temperature: float = 0.1
    ) -> ChatGroq:
        """Get or create Groq model instance.
        
        Args:
            model: Model name. Default is llama-3.3-70b-versatile
            temperature: Temperature parameter. Default is 0.1
            
        Returns:
            ChatGroq instance
        """
        if cls._groq_instance is None:
            cls._groq_instance = ChatGroq(model=model, temperature=temperature)
        return cls._groq_instance

    @classmethod
    def get_ollama(
        cls, model: str = "llama3:latest", temperature: float = 0.1
    ) -> ChatOllama:
        """Get or create Ollama model instance.
        
        Args:
            model: Model name. Default is llama3:latest
            temperature: Temperature parameter. Default is 0.1
            
        Returns:
            ChatOllama instance
        """
        if cls._ollama_instance is None:
            cls._ollama_instance = ChatOllama(model=model, temperature=temperature)
        return cls._ollama_instance

    @classmethod
    def get_model_for_image_processing(cls) -> ChatGoogleGenerativeAI:
        """Get the model best suited for image processing (Gemini).
        
        Returns:
            ChatGoogleGenerativeAI instance optimized for vision tasks
        """
        return cls.get_gemini()

    @classmethod
    def get_model_for_text_extraction(cls) -> ChatGroq:
        """Get the model best suited for text extraction and analysis.
        
        Returns:
            ChatGroq instance optimized for text tasks
        """
        return cls.get_groq()

    @classmethod
    def get_model_for_local_processing(cls) -> ChatOllama:
        """Get the model for local/offline processing.
        
        Returns:
            ChatOllama instance for local inference
        """
        return cls.get_ollama()

    @classmethod
    def reset(cls) -> None:
        """Reset all model instances (useful for testing)."""
        cls._gemini_instance = None
        cls._groq_instance = None
        cls._ollama_instance = None
