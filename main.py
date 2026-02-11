#!/usr/bin/env python3
"""
pg-agent - Main Entry Point

A Python-based agent system that integrates with LLM (Llama),
handles signal processing, parses inputs, validates agent outputs,
and generates reports.
"""

import argparse
import logging
import sys
import os
from pathlib import Path

from dotenv import load_dotenv

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Load environment variables
load_dotenv()

from parser.input_parser import InputParser
from agent.terminal_agent import TerminalAgent


def setup_logging(log_level: str = "INFO") -> logging.Logger:
    """Configure application logging."""
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)


def main():
    """Main entry point for pg-agent."""
    parser = argparse.ArgumentParser(
        description='pg-agent: AI-powered agent system with LLM integration',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 main.py                           # Interactive mode
  python3 main.py --input data/raw/input.txt --output data/output/
  python3 main.py --interactive             # Force interactive mode
  python3 main.py --query "Your question"   # Single query mode
        """
    )
    
    # Input/Output arguments
    parser.add_argument('--input', '-i', type=str,
                        help='Path to input file (raw data)')
    parser.add_argument('--output', '-o', type=str, default='data/output/',
                        help='Path to output directory (default: data/output/)')
    
    # Mode arguments
    parser.add_argument('--interactive', '-I', action='store_true',
                        help='Run in interactive terminal mode')
    parser.add_argument('--query', '-q', type=str,
                        help='Single query to process')
    
    # Server mode
    parser.add_argument('--serve', '-s', action='store_true',
                        help='Start the web server for Review UI (FastAPI)')
    
    # Logging arguments
    parser.add_argument('--log-level', '-L', type=str, default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                        help='Logging level (default: INFO)')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging(args.log_level)
    logger.info("Starting pg-agent...")
    
    # Initialize input parser
    input_parser = InputParser()
    
    # Create output directory if needed
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Server mode - start FastAPI web server
        if args.serve:
            logger.info("Starting Review UI web server...")
            import uvicorn
            from recommendations.api import app
            uvicorn.run(app, host="0.0.0.0", port=8000, log_level=args.log_level.lower())
            return
        
        # Mode selection
        if args.interactive or args.input is None:
            # Interactive terminal mode
            logger.info("Starting interactive terminal mode...")
            agent = TerminalAgent(
                input_parser=input_parser,
                output_dir=str(output_dir),
                log_level=args.log_level
            )
            agent.run()
            
        elif args.query:
            # Single query mode
            logger.info(f"Processing single query: {args.query}")
            agent = TerminalAgent(
                input_parser=input_parser,
                output_dir=str(output_dir),
                log_level=args.log_level
            )
            result = agent.process_query(args.query)
            print(f"\nResult: {result}")
            
        elif args.input:
            # Batch processing mode
            logger.info(f"Processing input file: {args.input}")
            input_data = input_parser.parse_file(args.input)
            agent = TerminalAgent(
                input_parser=input_parser,
                output_dir=str(output_dir),
                log_level=args.log_level
            )
            results = agent.process_input(input_data)
            print(f"\nProcessed {len(results)} items")
            
    except KeyboardInterrupt:
        logger.info("Application interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Application error: {e}", exc_info=True)
        sys.exit(1)
    
    logger.info("pg-agent completed successfully")


if __name__ == '__main__':
    main()

