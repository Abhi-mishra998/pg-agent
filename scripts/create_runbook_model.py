#!/usr/bin/env python3
"""
Create the db-ops-runbook Ollama model from Modelfile.

This script:
1. Reads rules from model/rules.md
2. Reads examples from model/examples.md
3. Creates a Modelfile with system prompt
4. Creates the model using ollama create

Usage:
    python scripts/create_runbook_model.py

Requirements:
    - Ollama must be installed and running
    - Model 'llama3.1:8b' must be available
"""

import os
import subprocess
import sys
import textwrap


def run_command(cmd: list, check: bool = True) -> tuple:
    """
    Run command and return success status and output.
    
    Args:
        cmd: Command to run as list of strings
        check: Whether to check for errors
        
    Returns:
        Tuple of (success, stdout, stderr)
    """
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )
        success = result.returncode == 0 if check else True
        return success, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return False, "", "Command timed out"
    except Exception as e:
        return False, "", str(e)


def check_ollama_available() -> bool:
    """Check if ollama CLI is available."""
    success, stdout, stderr = run_command(["which", "ollama"], check=False)
    return success or os.name != 'nt'  # On Windows, which might not work


def check_model_available(model_name: str) -> bool:
    """Check if the base model is available."""
    success, stdout, stderr = run_command(["ollama", "list"], check=False)
    if not success:
        return False
    
    # Check if model is in the list (exact match or as base)
    model_base = model_name.split(':')[0]
    for line in stdout.split('\n'):
        if model_base in line.lower():
            return True
    return False


def create_modelfile(rules_path: str, examples_path: str) -> str:
    """Create the Modelfile content."""
    # Read rules and examples
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    rules_file = os.path.join(project_root, rules_path)
    examples_file = os.path.join(project_root, examples_path)
    
    with open(rules_file, 'r') as f:
        rules = f.read()
    
    with open(examples_file, 'r') as f:
        examples = f.read()
    
    # Create Modelfile content
    modelfile = textwrap.dedent(f'''\
        FROM llama3.1:8b

        SYSTEM "
        You are a PostgreSQL DBA Runbook AI for enterprise systems.

        RULES:
        {rules}

        You must always respond in this format:
        Diagnosis:
        Evidence:
        Root Cause:
        Immediate Actions:
        Preventive Actions:
        CLI Commands:

        No additional sections are allowed.
        "

        SYSTEM "
        TRAINING EXAMPLES:
        {examples}
        "

        PARAMETER temperature 0.2
        PARAMETER top_p 0.9
        PARAMETER num_ctx 8192
    ''')
    
    return modelfile


def main():
    """Main entry point."""
    print("=" * 60)
    print("Creating db-ops-runbook Ollama Model")
    print("=" * 60)
    
    # Configuration
    base_model = "llama3.1:8b"
    model_name = "db-ops-runbook"
    rules_path = "model/rules.md"
    examples_path = "model/examples.md"
    modelfile_path = "/tmp/Modelfile.db-ops-runbook"
    
    # Step 1: Check if ollama is available
    print("\n[1/4] Checking Ollama installation...")
    if not check_ollama_available():
        print("ERROR: Ollama CLI not found.")
        print("Please install Ollama from: https://ollama.com")
        sys.exit(1)
    print("✓ Ollama CLI found")
    
    # Step 2: Check if base model is available
    print(f"\n[2/4] Checking if base model '{base_model}' is available...")
    if not check_model_available(base_model):
        print(f"WARNING: Base model '{base_model}' not found.")
        print(f"Pulling {base_model} (this may take several minutes)...")
        success, stdout, stderr = run_command(["ollama", "pull", base_model])
        if not success:
            print(f"ERROR: Failed to pull {base_model}")
            print(f"stderr: {stderr}")
            sys.exit(1)
        print(f"✓ {base_model} pulled successfully")
    else:
        print(f"✓ Base model '{base_model}' is available")
    
    # Step 3: Create Modelfile
    print(f"\n[3/4] Creating Modelfile...")
    try:
        modelfile_content = create_modelfile(rules_path, examples_path)
        with open(modelfile_path, 'w') as f:
            f.write(modelfile_content)
        print(f"✓ Modelfile created: {modelfile_path}")
    except Exception as e:
        print(f"ERROR: Failed to create Modelfile: {e}")
        sys.exit(1)
    
    # Step 4: Create model
    print(f"\n[4/4] Creating model '{model_name}'...")
    print("This may take a few minutes on first run...")
    success, stdout, stderr = run_command(
        ["ollama", "create", model_name, "-f", modelfile_path]
    )
    
    if not success:
        print(f"ERROR: Failed to create model")
        print(f"stderr: {stderr}")
        sys.exit(1)
    
    print(f"✓ Model '{model_name}' created successfully!")
    
    # Summary
    print("\n" + "=" * 60)
    print("SUCCESS!")
    print("=" * 60)
    print(f"\nModel '{model_name}' is now available.")
    print("\nUsage options:")
    print("\n1. With pg-agent:")
    print("   python main.py --interactive")
    print("   pg-agent> runbook 'Your question here'")
    print("\n2. Direct Ollama usage:")
    print(f"   ollama run {model_name}")
    print("\n3. Python usage:")
    print("   from agent.runbook_agent import RunbookAgent")
    print("   agent = RunbookAgent()  # Uses db-ops-runbook model")
    print("   result = agent.generate_from_query('Your question')")


if __name__ == "__main__":
    main()

