# AI Agent Tutorial

A Python tutorial project demonstrating how to build an AI agent using Google's Gemini API. This project showcases how an AI agent can interact with the file system through custom tools/functions, enabling it to read, write, and explore files autonomously.

## Overview

This project implements an AI agent that can:

- **Self-prompt** and make decisions in a loop
- **Use tools** to interact with the file system
- **Read files** and directory contents
- **Write files** within a controlled working directory
- **Execute Python files** with arguments
- **Execute tasks** autonomously based on user instructions

## Features

### Core Functions

The agent has access to four main tools:

1. **`get_files_info`** - Lists files and directories in a specified path

   - Returns file names, directory flags, and file sizes
   - Validates paths to ensure they're within the working directory

2. **`get_files_contents`** - Reads file contents

   - Supports reading files up to 1000 characters
   - Automatically truncates larger files with a notification
   - Validates file paths for security

3. **`write_file`** - Writes content to files

   - Creates parent directories if they don't exist
   - Validates paths to prevent directory traversal attacks
   - Returns success/error messages

4. **`run_python_file`** - Executes Python files
   - Runs Python scripts within the working directory
   - Supports command-line arguments
   - Returns stdout, stderr, and exit codes
   - Includes timeout protection (30 seconds)
   - Validates file paths and ensures files are Python scripts

### Calculator Example

The project includes a calculator application in the `calculator/` directory that demonstrates:

- Expression evaluation with operator precedence
- JSON-formatted output
- Error handling for invalid expressions

## Prerequisites

- Python 3.13 or higher
- Google Gemini API key
- `python-dotenv` package (for environment variable management)
- `google-genai` package (Google's Gemini API client)

## Installation

1. Clone or navigate to the project directory:

```bash
cd 06_AGENTIC_AI/aiAgentTutorial
```

2. Install dependencies (if using uv):

```bash
uv sync
```

Or install manually:

```bash
pip install python-dotenv google-genai
```

3. Create a `.env` file in the project root:

```env
API_KEY=your_gemini_api_key_here
```

## Usage

### Running the Main Agent

The main agent script accepts command-line arguments:

```bash
python main.py "<your_prompt>" [--verbose] [1]
```

**Arguments:**

- `prompt` (required): The instruction or question for the AI agent
- `--verbose` (optional): Show detailed token usage information
- `1` (optional): Set to `1` to actually make the API call (default: skip call)

**Examples:**

```bash
# Basic usage
python main.py "List all files in the calculator directory"

# With verbose output
python main.py "Read the contents of lorem.txt" --verbose 1

# Skip API call (for testing)
python main.py "What files are in the project?" --verbose
```

### Running the Calculator

The calculator example can be run independently:

```bash
cd calculator
python main.py "3 + 5 * 2"
```

## Project Structure

```
aiAgentTutorial/
├── main.py                 # Main agent script
├── functions/              # Agent tools/functions
│   ├── get_files_info.py  # List directory contents
│   ├── get_files_contents.py  # Read file contents
│   ├── write_file.py      # Write files
│   └── run_python_file.py # Execute Python files
├── calculator/            # Example calculator application
│   ├── main.py
│   ├── pkg/
│   │   ├── calculator.py  # Calculator logic
│   │   └── render.py      # Output formatting
│   └── tests.py          # Calculator tests
├── tests.py              # Unit tests for agent functions
├── pyproject.toml        # Project configuration
└── readme/               # Documentation
    └── README.md         # This file
```

## Testing

Run the test suite to verify all functions work correctly:

```bash
python tests.py
```

The tests cover:

- `TestGetFilesInfo`: Directory listing functionality
- `TestGetFilesContents`: File reading functionality
- `TestWriteFile`: File writing functionality
- `TestRunPythonFile`: Python file execution functionality

## Security Features

The agent functions include security measures:

- **Path validation**: All file operations are restricted to the working directory
- **Directory traversal prevention**: Prevents access to files outside the working directory
- **File size limits**: File reading is limited to 1000 characters to prevent memory issues
- **Execution timeout**: Python file execution is limited to 30 seconds to prevent infinite loops
- **File type validation**: Only Python files can be executed via run_python_file

## How It Works

1. **Agent Initialization**: The main script connects to Google's Gemini API using your API key
2. **Tool Registration**: The agent has access to file system tools (get_files_info, get_files_contents, write_file, run_python_file)
3. **Autonomous Execution**: The agent can:
   - Analyze user prompts
   - Decide which tools to use
   - Execute tool calls in sequence
   - Iterate based on results
4. **Response Generation**: The agent returns results and can provide verbose token usage information

## Configuration

The agent uses the `gemini-2.5-flash` model by default. You can modify the model in `main.py`:

```python
response = client.models.generate_content(
    model="gemini-2.5-flash",  # Change this to use a different model
    ...
)
```

## Error Handling

The project includes comprehensive error handling:

- Invalid file paths
- Missing directories
- File read/write errors
- API connection issues
- Malformed responses
