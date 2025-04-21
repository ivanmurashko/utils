# utils
There are several small utils and projects that I found useful

## List of Utilities

1. [Translator](#translator) - LaTeX file translator from Russian to English using OpenAI API

## Translator

A utility for translating LaTeX files from Russian to English using OpenAI's GPT models. It preserves LaTeX formatting while translating the content.

### Features

- Translates LaTeX files while preserving formatting
- Supports both single files and directories
- Concurrent processing with configurable number of workers
- Progress tracking and detailed statistics
- Cost calculation for API usage
- Support for multiple OpenAI models with automatic pricing
- Automatic rate limit handling with exponential backoff
- Configurable retry mechanism for failed requests

### Supported Models

- `gpt-4.1-mini`: Input $0.40/1M tokens, Output $1.60/1M tokens
- `gpt-4o`: Input $2.50/1M tokens, Output $10.00/1M tokens

### Installation

1. Clone the repository
2. Install required dependencies:
```bash
pip install openai aiofiles tqdm
```

### Usage

Basic usage:
```bash
python translator.py --input <input_path> --output <output_path> --api-key <your_api_key>
```

#### Required Arguments
- `--input`, `-i`: Input file or directory containing LaTeX files
- `--output`, `-o`: Output directory for translated files
- `--api-key`, `-k`: OpenAI API key

#### Optional Arguments
- `--workers`, `-w`: Maximum number of concurrent workers (default: 10)
- `--model`, `-m`: OpenAI model to use (choices: gpt-4.1-mini, gpt-4o, default: gpt-4.1-mini)
- `--input-cost`: Override input cost per token (defaults to model-specific pricing)
- `--output-cost`: Override output cost per token (defaults to model-specific pricing)
- `--max-retries`: Maximum number of retries for rate-limited requests (default: 5)
- `--initial-retry-delay`: Initial delay between retries in seconds (default: 2.0)
- `--max-retry-delay`: Maximum delay between retries in seconds (default: 60.0)
- `--verbose`, `-v`: Enable verbose logging

### Examples

1. Translate a single file:
```bash
python translator.py --input document.tex --output translated/ --api-key sk-...
```

2. Translate a directory with custom workers:
```bash
python translator.py --input papers/ --output translated/ --api-key sk-... --workers 5
```

3. Use GPT-4o model with custom retry settings:
```bash
python translator.py --input papers/ --output translated/ --api-key sk-... --model gpt-4o --max-retries 10 --initial-retry-delay 5.0
```

4. Enable verbose logging:
```bash
python translator.py --input papers/ --output translated/ --api-key sk-... --verbose
```

5. Override pricing:
```bash
python translator.py --input papers/ --output translated/ --api-key sk-... --input-cost 0.0000004 --output-cost 0.0000016
```

### Output

The utility provides:
- Progress bar showing processing status
- Detailed logging of operations
- Summary of processed files including:
  - Model used
  - Input and output costs per token
  - Total cost and time spent
  - Number of retries performed
  - List of rate-limited files
  - List of any failed files

### Rate Limit Handling

The utility automatically handles rate limits by:
- Detecting rate limit errors from the API
- Implementing exponential backoff with jitter
- Extracting wait time from error messages when available
- Tracking and reporting rate-limited files
- Allowing configuration of retry parameters

### Notes

- The utility only processes `.tex` files
- Non-LaTeX files are copied to the output directory without translation
- Hidden files and directories (starting with '.') are skipped
- The output directory is cleared before processing starts
- Pricing is automatically set based on the selected model unless overridden
- Rate limit handling uses exponential backoff to prevent overwhelming the API
