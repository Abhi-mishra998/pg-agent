#!/usr/bin/env python3
"""
LlamaClient - Ollama API Integration

Handles communication with Ollama API for Llama model inference.
Supports llama3.1:8b and other Ollama models.

This module is designed to work with or without Ollama installed.
When Ollama is not available, it gracefully falls back to KB-only mode.
"""

import json
import logging
import time
from typing import Any, Dict, List, Optional
from dataclasses import dataclass
from pathlib import Path

import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

import os


@dataclass
class OllamaResponse:
    """Represents a response from Ollama API."""
    text: str
    model: str
    created_at: str
    done: bool
    total_duration: Optional[int] = None
    load_duration: Optional[int] = None
    prompt_eval_count: Optional[int] = None
    eval_count: Optional[int] = None


class LlamaClient:
    """
    Client for interacting with Ollama API.
    
    Supports:
    - Text completion with llama3.1:8b
    - Streaming responses
    - Multiple prompt formats
    - Error handling and retries
    
    Gracefully falls back when Ollama is not available.
    """
    
    def __init__(
        self,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        timeout: Optional[int] = None,
        silent: bool = False  # Silent mode for reducing noise when unavailable
    ):
        """
        Initialize the Ollama client.
        
        Args:
            base_url: Ollama API base URL (default: http://localhost:11434)
            model: Model name (default: llama3.1:8b)
            timeout: Request timeout in seconds (default: 120)
            silent: If True, suppress warnings when Ollama unavailable
        """
        self.base_url = base_url or os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')
        self.model = model or os.getenv('OLLAMA_MODEL', 'llama3.1:8b')
        self.timeout = timeout or int(os.getenv('OLLAMA_TIMEOUT', '120'))
        self.silent = silent
        
        self.logger = logging.getLogger(__name__)
        self._session = requests.Session()
        self._session.headers.update({'Content-Type': 'application/json'})
        
        # Check availability on init
        self._available = self._check_availability()
        
    def _make_request(
        self,
        endpoint: str,
        data: Dict[str, Any],
        stream: bool = False
    ) -> requests.Response:
        """Make HTTP request to Ollama API."""
        url = f"{self.base_url}{endpoint}"
        
        self.logger.debug(f"Making request to {url}")
        self.logger.debug(f"Request data: {json.dumps(data, indent=2)}")
        
        response = self._session.post(
            url,
            json=data,
            timeout=self.timeout,
            stream=stream
        )
        
        response.raise_for_status()
        return response
    
    def _check_availability(self) -> bool:
        """
        Internal check for Ollama availability.
        Returns True if available, False otherwise.
        """
        try:
            response = self._session.get(
                f"{self.base_url}/api/tags",
                timeout=5
            )
            response.raise_for_status()
            data = response.json()
            models = [m['name'] for m in data.get('models', [])]
            
            if self.model in models or self.model.split(':')[0] in [m.split(':')[0] for m in models]:
                return True
            else:
                if not self.silent:
                    self.logger.warning(f"Model {self.model} not found in available models: {models}")
                return False
                
        except requests.exceptions.ConnectionError:
            if not self.silent:
                self.logger.warning("Ollama server not available (LLM disabled)")
            return False
        except Exception as e:
            if not self.silent:
                self.logger.warning(f"Ollama availability check failed: {e}")
            return False
    
    def is_available(self) -> bool:
        """
        Check if Ollama server is available.
        
        Returns:
            True if Ollama is running and model is available, False otherwise.
        """
        return getattr(self, '_available', False)
    
    def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        template: Optional[str] = None,
        context: Optional[List[int]] = None,
        stream: bool = False,
        options: Optional[Dict[str, Any]] = None
    ) -> OllamaResponse:
        """
        Generate text using the Ollama API.
        
        Args:
            prompt: The user prompt
            system: System prompt (optional)
            template: Prompt template (optional)
            context: Previous context tokens (optional)
            stream: Whether to stream the response
            options: Additional model options (temperature, top_k, etc.)
            
        Returns:
            OllamaResponse object with generated text
        """
        data: Dict[str, Any] = {
            "model": self.model,
            "prompt": prompt,
            "stream": stream,
            "options": options or {}
        }
        
        if system:
            data["system"] = system
        if template:
            data["template"] = template
        if context:
            data["context"] = context
            
        try:
            if stream:
                return self._generate_stream(data)
            else:
                return self._generate(data)
                
        except requests.exceptions.Timeout:
            self.logger.error("Request timed out")
            raise
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Request failed: {e}")
            raise
    
    def _generate(self, data: Dict[str, Any]) -> OllamaResponse:
        """Non-streaming generation."""
        response = self._make_request("/api/generate", data)
        result = response.json()
        
        self.logger.debug(f"Generation response: {json.dumps(result, indent=2)}")
        
        return OllamaResponse(
            text=result.get('response', ''),
            model=result.get('model', self.model),
            created_at=result.get('created_at', ''),
            done=result.get('done', False),
            total_duration=result.get('total_duration'),
            load_duration=result.get('load_duration'),
            prompt_eval_count=result.get('prompt_eval_count'),
            eval_count=result.get('eval_count')
        )
    
    def _generate_stream(self, data: Dict[str, Any]) -> OllamaResponse:
        """Streaming generation (returns aggregated response)."""
        response = self._make_request("/api/generate", data, stream=True)
        
        full_response = []
        
        for line in response.iter_lines():
            if line:
                try:
                    result = json.loads(line.decode('utf-8'))
                    if 'response' in result:
                        full_response.append(result['response'])
                        
                    if result.get('done', False):
                        break
                        
                except json.JSONDecodeError:
                    continue
        
        return OllamaResponse(
            text=''.join(full_response),
            model=self.model,
            created_at='',
            done=True
        )
    
    def chat(
        self,
        messages: List[Dict[str, str]],
        stream: bool = False,
        options: Optional[Dict[str, Any]] = None
    ) -> OllamaResponse:
        """
        Chat completion using the Ollama API.
        
        Args:
            messages: List of chat messages with 'role' and 'content'
            stream: Whether to stream the response
            options: Additional model options
            
        Returns:
            OllamaResponse object with generated text
        """
        data: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": stream,
            "options": options or {}
        }
        
        try:
            response = self._make_request("/api/chat", data, stream=stream)
            
            if stream:
                return self._chat_stream(response)
            else:
                result = response.json()
                return OllamaResponse(
                    text=result.get('message', {}).get('content', ''),
                    model=result.get('model', self.model),
                    created_at=result.get('created_at', ''),
                    done=result.get('done', False)
                )
                
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Chat request failed: {e}")
            raise
    
    def _chat_stream(self, response: requests.Response) -> OllamaResponse:
        """Handle streaming chat response."""
        full_response = []
        
        for line in response.iter_lines():
            if line:
                try:
                    result = json.loads(line.decode('utf-8'))
                    if 'message' in result and 'content' in result['message']:
                        full_response.append(result['message']['content'])
                    if result.get('done', False):
                        break
                except json.JSONDecodeError:
                    continue
        
        return OllamaResponse(
            text=''.join(full_response),
            model=self.model,
            created_at='',
            done=True
        )
    
    def embed(self, text: str) -> List[float]:
        """
        Generate embeddings for text.
        
        Args:
            text: Text to embed
            
        Returns:
            List of embedding values
        """
        data = {
            "model": self.model,
            "input": text
        }
        
        try:
            response = self._make_request("/api/embeddings", data)
            result = response.json()
            return result.get('embedding', [])
            
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Embedding request failed: {e}")
            raise
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model."""
        try:
            response = self._make_request("/api/show", {"name": self.model})
            return response.json()
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Failed to get model info: {e}")
            return {}
    
    def close(self):
        """Close the client session."""
        self._session.close()


def create_llama_client(
    model: Optional[str] = None,
    base_url: Optional[str] = None
) -> LlamaClient:
    """
    Factory function to create a LlamaClient.
    
    Args:
        model: Model name (optional)
        base_url: Base URL (optional)
        
    Returns:
        Configured LlamaClient instance
    """
    return LlamaClient(model=model, base_url=base_url)


if __name__ == '__main__':
    # Test the client
    logging.basicConfig(level=logging.INFO)
    
    client = LlamaClient()
    
    if client.is_available():
        print(f"✓ Connected to Ollama with model: {client.model}")
        
        # Test generation
        response = client.generate("What is 2+2?")
        print(f"✓ Test response: {response.text}")
    else:
        print("✗ Ollama server not available. Make sure Ollama is running.")
        print("  Run: ollama serve")

