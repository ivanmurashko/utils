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
- Configurable OpenAI model and token costs

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
- `--model`, `-m`: OpenAI model to use (default: "gpt-4.1-mini")
- `--input-cost`: Cost per input token (default: 0.40 / 1_000_000)
- `--output-cost`: Cost per output token (default: 1.60 / 1_000_000)
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

3. Enable verbose logging:
```bash
python translator.py --input papers/ --output translated/ --api-key sk-... --verbose
```

### Output

The utility provides:
- Progress bar showing processing status
- Detailed logging of operations
- Summary of processed files
- Total cost and time spent
- List of any failed files

### Notes

- The utility only processes `.tex` files
- Non-LaTeX files are copied to the output directory without translation
- Hidden files and directories (starting with '.') are skipped
- The output directory is cleared before processing starts
