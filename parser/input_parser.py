#!/usr/bin/env python3
"""
InputParser - CLI Argument Parsing Module

Handles parsing of CLI arguments and input file processing.
Supports multiple input formats (JSON, text, YAML).
"""

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from urllib.parse import urlparse

import yaml


class InputParser:
    """
    Parses CLI arguments and input files.
    
    Supports:
    - CLI argument parsing
    - JSON file parsing
    - YAML file parsing
    - Text file parsing
    - Raw string input
    - URL validation
    """
    
    def __init__(self):
        """Initialize the input parser."""
        self.logger = logging.getLogger(__name__)
        
    def parse_arguments(self, args: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Parse CLI arguments.
        
        Args:
            args: Command line arguments (uses sys.argv if None)
            
        Returns:
            Dictionary of parsed arguments
        """
        parser = argparse.ArgumentParser(
            description='pg-agent: AI-powered agent system',
            formatter_class=argparse.RawDescriptionHelpFormatter,
            add_help=False
        )
        
        # Core arguments
        parser.add_argument('--input', '-i', type=str,
                            help='Input file path or URL')
        parser.add_argument('--output', '-o', type=str, default='data/output/',
                            help='Output directory path')
        parser.add_argument('--interactive', '-I', action='store_true',
                            help='Run in interactive mode')
        parser.add_argument('--query', '-q', type=str,
                            help='Single query to process')
        
        # LLM arguments
        parser.add_argument('--model', '-m', type=str,
                            help='LLM model to use')
        parser.add_argument('--temperature', '-t', type=float, default=0.7,
                            help='Model temperature (0.0-1.0)')
        parser.add_argument('--max-tokens', type=int, default=4096,
                            help='Maximum tokens in response')
        
        # Logging arguments
        parser.add_argument('--log-level', '-L', type=str,
                            default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                            help='Logging level')
        parser.add_argument('--quiet', '-Q', action='store_true',
                            help='Suppress informational output')
        
        # Parse arguments
        parsed = parser.parse_args(args)
        
        # Convert to dictionary
        result = {
            'input': parsed.input,
            'output': parsed.output,
            'interactive': parsed.interactive,
            'query': parsed.query,
            'model': parsed.model,
            'temperature': parsed.temperature,
            'max_tokens': parsed.max_tokens,
            'log_level': parsed.log_level,
            'quiet': parsed.quiet
        }
        
        return result
    
    def parse_file(self, file_path: str) -> Dict[str, Any]:
        """
        Parse an input file.
        
        Args:
            file_path: Path to the input file
            
        Returns:
            Parsed data dictionary
        """
        path = Path(file_path)
        
        if not path.exists():
            raise FileNotFoundError(f"Input file not found: {file_path}")
        
        # Detect file type
        suffix = path.suffix.lower()
        
        self.logger.debug(f"Parsing file: {file_path} (type: {suffix})")
        
        if suffix == '.json':
            return self._parse_json(path)
        elif suffix in ['.yaml', '.yml']:
            return self._parse_yaml(path)
        elif suffix in ['.txt', '.md', '.text']:
            return self._parse_text(path)
        elif suffix == '.csv':
            return self._parse_csv(path)
        else:
            # Try JSON first, then YAML, then text
            return self._parse_auto(path)
    
    def _parse_json(self, path: Path) -> Dict[str, Any]:
        """Parse JSON file."""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.logger.debug(f"Parsed JSON: {len(str(data))} chars")
        return {'type': 'json', 'data': data, 'source': str(path)}
    
    def _parse_yaml(self, path: Path) -> Dict[str, Any]:
        """Parse YAML file."""
        with open(path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        
        self.logger.debug(f"Parsed YAML: {len(str(data))} chars")
        return {'type': 'yaml', 'data': data, 'source': str(path)}
    
    def _parse_text(self, path: Path) -> Dict[str, Any]:
        """Parse text file."""
        with open(path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        self.logger.debug(f"Parsed text: {len(content)} chars")
        return {
            'type': 'text',
            'data': content,
            'source': str(path),
            'lines': content.count('\n'),
            'words': len(content.split())
        }
    
    def _parse_csv(self, path: Path) -> Dict[str, Any]:
        """Parse CSV file."""
        import pandas as pd
        
        df = pd.read_csv(path)
        
        self.logger.debug(f"Parsed CSV: {len(df)} rows, {len(df.columns)} columns")
        return {
            'type': 'csv',
            'data': df.to_dict(orient='records'),
            'source': str(path),
            'columns': list(df.columns),
            'rows': len(df)
        }
    
    def _parse_auto(self, path: Path) -> Dict[str, Any]:
        """Auto-detect file type and parse."""
        # Try JSON first
        try:
            return self._parse_json(path)
        except (json.JSONDecodeError, UnicodeDecodeError):
            pass
        
        # Try YAML
        try:
            return self._parse_yaml(path)
        except yaml.YAMLError:
            pass
        
        # Fall back to text
        return self._parse_text(path)
    
    def parse_query(self, query: str) -> Dict[str, Any]:
        """
        Parse a raw query string or file path.
        
        If the input is a valid file path, reads and returns the file contents.
        Otherwise, treats the input as raw text.
        
        Args:
            query: Raw query string or file path
            
        Returns:
            Structured query data with type 'file' or 'query'
        """
        text = query.strip()
        
        # Check if file path is provided
        if Path(text).exists():
            with open(text, "r", encoding="utf-8") as f:
                content = f.read()
            
            self.logger.debug(f"Parsed file: {text} ({len(content)} chars)")
            return {
                "type": "file",
                "source": text,
                "data": content
            }
        
        # Fallback: raw text
        return {
            'type': 'query',
            'data': text,
            'length': len(text),
            'words': len(text.split())
        }
    
    def parse_url(self, url: str) -> Dict[str, Any]:
        """
        Parse and validate a URL.
        
        Args:
            url: URL string to parse
            
        Returns:
            URL components and validation result
        """
        parsed = urlparse(url)
        
        return {
            'type': 'url',
            'url': url,
            'scheme': parsed.scheme,
            'netloc': parsed.netloc,
            'path': parsed.path,
            'is_valid': bool(parsed.scheme and parsed.netloc),
            'source': url
        }
    
    def normalize_data(
        self,
        data: Any,
        target_format: str = 'json'
    ) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Normalize data to a consistent format.
        
        Args:
            data: Input data (dict, list, or string)
            target_format: Target format ('json' or 'dict')
            
        Returns:
            Normalized data
        """
        if isinstance(data, dict):
            return data
        elif isinstance(data, list):
            return {'items': data}
        elif isinstance(data, str):
            try:
                # Try JSON
                return json.loads(data)
            except json.JSONDecodeError:
                try:
                    # Try YAML
                    return yaml.safe_load(data)
                except yaml.YAMLError:
                    # Return as text
                    return {'text': data}
        else:
            return {'raw': str(data)}
    
    def validate_input(
        self,
        data: Any,
        required_fields: Optional[List[str]] = None
    ) -> tuple[bool, List[str]]:
        """
        Validate input data.
        
        Args:
            data: Input data to validate
            required_fields: List of required field names
            
        Returns:
            Tuple of (is_valid, list of error messages)
        """
        errors = []
        
        if required_fields:
            if isinstance(data, dict):
                for field in required_fields:
                    if field not in data:
                        errors.append(f"Missing required field: {field}")
            else:
                errors.append("Data must be a dictionary for field validation")
        
        return len(errors) == 0, errors


def create_input_parser() -> InputParser:
    """Factory function to create an InputParser."""
    return InputParser()


if __name__ == '__main__':
    # Test the parser
    logging.basicConfig(level=logging.DEBUG)
    
    parser = InputParser()
    
    # Test argument parsing
    args = parser.parse_arguments(['--input', 'test.json', '--log-level', 'DEBUG'])
    print(f"Parsed arguments: {args}")
    
    # Test query parsing
    query = parser.parse_query("What is the meaning of life?")
    print(f"Parsed query: {query}")
    
    # Test URL parsing
    url = parser.parse_url("https://api.example.com/data")
    print(f"Parsed URL: {url}")

