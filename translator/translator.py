#!/usr/bin/python3
import os
import shutil
import time
import asyncio
import aiofiles
import re
import logging
import argparse
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
from pathlib import Path
from openai import AsyncOpenAI, RateLimitError
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Pricing models
PRICING_MODELS = {
    "gpt-4.1-mini": {
        "input_cost": 0.40 / 1_000_000,
        "output_cost": 1.60 / 1_000_000
    },
    "gpt-4o": {
        "input_cost": 2.50 / 1_000_000,
        "output_cost": 10.00 / 1_000_000
    }
}

@dataclass
class TranslationResult:
    file_name: str
    time_spent: float
    cost: float
    success: bool
    error: Optional[str] = None
    retries: int = 0

@dataclass
class ProcessingStats:
    total_files: int
    processed_files: int
    total_cost: float
    total_time: float
    failed_files: List[str]
    rate_limited_files: List[str]
    total_retries: int

class FileProcessor:
    def __init__(
        self,
        api_key: str,
        max_workers: int = 10,
        model: str = "gpt-4.1-mini",
        input_cost_per_token: Optional[float] = None,
        output_cost_per_token: Optional[float] = None,
        max_retries: int = 5,
        initial_retry_delay: float = 2.0,
        max_retry_delay: float = 60.0
    ):
        self.api_key = api_key
        self.max_workers = max_workers
        self.model = model
        self.max_retries = max_retries
        self.initial_retry_delay = initial_retry_delay
        self.max_retry_delay = max_retry_delay
        
        # Set pricing based on model or provided values
        if model in PRICING_MODELS and (input_cost_per_token is None or output_cost_per_token is None):
            self.input_cost_per_token = PRICING_MODELS[model]["input_cost"]
            self.output_cost_per_token = PRICING_MODELS[model]["output_cost"]
        else:
            self.input_cost_per_token = input_cost_per_token or PRICING_MODELS["gpt-4.1-mini"]["input_cost"]
            self.output_cost_per_token = output_cost_per_token or PRICING_MODELS["gpt-4.1-mini"]["output_cost"]
        
        self.client = AsyncOpenAI(api_key=api_key)
        self.stats = ProcessingStats(0, 0, 0.0, 0.0, [], [], 0)

    def contains_russian(self, text: str) -> bool:
        """Check if the text contains Russian characters."""
        return bool(re.search(r'[а-яА-Я]', text))

    async def translate_with_retry(self, content: str) -> Tuple[str, float, float, int]:
        """Translate content using OpenAI API with retry mechanism."""
        retry_count = 0
        last_error = None
        
        while retry_count <= self.max_retries:
            try:
                start_time = time.time()
                response = await self.client.responses.create(
                    model=self.model,
                    instructions=(
                        "You are a translator that processes LaTeX files. "
                        "Translate all Russian text in the LaTeX document to English while preserving the formatting. "
                        "Do not add any introductory or summary text. "
                        "Do not wrap the output in markdown or code blocks (no triple backticks). "
                        "Return only the translated LaTeX content, nothing else. "
                        "If the content is already entirely in English, return it unchanged."
                    ),
                    input=content,
                )
                end_time = time.time()
                
                translated_content = response.output_text
                time_spent = end_time - start_time

                # Calculate cost
                input_tokens = response.usage.input_tokens
                output_tokens = response.usage.output_tokens
                total_cost = (
                    input_tokens * self.input_cost_per_token +
                    output_tokens * self.output_cost_per_token
                )

                return translated_content, time_spent, total_cost, retry_count
                
            except RateLimitError as e:
                last_error = e
                retry_count += 1
                self.stats.total_retries += 1
                
                # Extract wait time from error message if available
                wait_time = self.initial_retry_delay
                if "try again in" in str(e):
                    try:
                        wait_time = float(str(e).split("try again in ")[1].split("s")[0])
                    except (IndexError, ValueError):
                        pass
                
                # Calculate exponential backoff with jitter
                wait_time = min(
                    self.initial_retry_delay * (2 ** (retry_count - 1)),
                    self.max_retry_delay
                ) * (1 + 0.1 * (retry_count - 1))  # Add 10% jitter
                
                logger.warning(
                    f"Rate limit reached. Retry {retry_count}/{self.max_retries}. "
                    f"Waiting {wait_time:.2f} seconds..."
                )
                await asyncio.sleep(wait_time)
                
            except Exception as e:
                logger.error(f"Translation error: {str(e)}")
                raise
        
        # If we've exhausted all retries
        raise last_error or Exception("Maximum retries exceeded")

    async def process_single_file(self, input_file_path: str, output_file_path: str) -> TranslationResult:
        """Process a single file."""
        try:
            # Resolve symlink to the actual file path
            actual_file_path = os.path.realpath(input_file_path)
            
            async with aiofiles.open(actual_file_path, 'r', encoding='utf-8') as f:
                content = await f.read()
            
            if self.contains_russian(content):
                translated_content, time_spent, cost, retries = await self.translate_with_retry(content)
            else:
                translated_content = content
                time_spent = 0
                cost = 0
                retries = 0
            
            # Ensure the output directory exists
            os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
            
            async with aiofiles.open(output_file_path, 'w', encoding='utf-8') as f:
                await f.write(translated_content)
                
            return TranslationResult(
                file_name=os.path.basename(input_file_path),
                time_spent=time_spent,
                cost=cost,
                success=True,
                retries=retries
            )
        except RateLimitError as e:
            logger.error(f"Rate limit error processing file {input_file_path}: {str(e)}")
            self.stats.rate_limited_files.append(os.path.basename(input_file_path))
            return TranslationResult(
                file_name=os.path.basename(input_file_path),
                time_spent=0,
                cost=0,
                success=False,
                error=f"Rate limit error: {str(e)}",
                retries=self.max_retries
            )
        except Exception as e:
            logger.error(f"Error processing file {input_file_path}: {str(e)}")
            return TranslationResult(
                file_name=os.path.basename(input_file_path),
                time_spent=0,
                cost=0,
                success=False,
                error=str(e)
            )

    async def process_directory(self, input_folder: str, output_folder: str) -> None:
        """Process all files in the input directory."""
        os.makedirs(output_folder, exist_ok=True)
        
        # Collect all tasks
        tasks = []
        for root, _, files in os.walk(input_folder):
            if os.path.basename(root).startswith('.'):
                continue

            for file in files:
                if file.startswith('.'):
                    continue
                
                input_file_path = os.path.join(root, file)
                relative_path = os.path.relpath(input_file_path, input_folder)
                output_file_path = os.path.join(output_folder, relative_path)

                if file.endswith('.tex'):
                    tasks.append(self.process_single_file(input_file_path, output_file_path))
                else:
                    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
                    shutil.copy(input_file_path, output_file_path)
                    logger.info(f"Copied file: {file}")

        self.stats.total_files = len(tasks)
        
        # Process tasks concurrently with semaphore
        semaphore = asyncio.Semaphore(self.max_workers)
        
        async def process_with_semaphore(task):
            async with semaphore:
                return await task

        # Run tasks with progress bar
        with tqdm(total=len(tasks), desc="Processing files") as pbar:
            for task in asyncio.as_completed([process_with_semaphore(t) for t in tasks]):
                result = await task
                self._update_stats(result)
                self._log_result(result)
                pbar.update(1)

    async def process_files(self, input_folder: str, output_folder: str) -> None:
        """Process all files in the input folder."""
        self.stats = ProcessingStats(0, 0, 0.0, 0.0, [], [], 0)
        
        # Resolve symlink to the actual input folder path
        actual_input_folder = os.path.realpath(input_folder)

        # Check if actual_input_folder is a file or directory
        if os.path.isfile(actual_input_folder):
            # Process single file
            output_file_path = os.path.join(output_folder, os.path.basename(actual_input_folder))
            os.makedirs(output_folder, exist_ok=True)
            
            result = await self.process_single_file(actual_input_folder, output_file_path)
            self._update_stats(result)
            self._log_result(result)
        else:
            # Process directory
            await self.process_directory(actual_input_folder, output_folder)

        self._log_summary()

    def _update_stats(self, result: TranslationResult) -> None:
        """Update processing statistics."""
        self.stats.processed_files += 1
        if result.success:
            self.stats.total_cost += result.cost
            self.stats.total_time += result.time_spent
        else:
            self.stats.failed_files.append(result.file_name)

    def _log_result(self, result: TranslationResult) -> None:
        """Log the result of processing a file."""
        if result.success:
            retry_info = f", Retries: {result.retries}" if result.retries > 0 else ""
            logger.info(
                f"Processed file: {result.file_name}, "
                f"Time: {result.time_spent:.2f}s, "
                f"Cost: ${result.cost:.4f}"
                f"{retry_info}"
            )
        else:
            logger.error(
                f"Failed to process file: {result.file_name}, "
                f"Error: {result.error}"
            )

    def _log_summary(self) -> None:
        """Log the summary of processing."""
        logger.info("\nProcessing Summary:")
        logger.info(f"Model: {self.model}")
        logger.info(f"Input cost per token: ${self.input_cost_per_token:.8f}")
        logger.info(f"Output cost per token: ${self.output_cost_per_token:.8f}")
        logger.info(f"Total files processed: {self.stats.processed_files}/{self.stats.total_files}")
        logger.info(f"Total cost: ${self.stats.total_cost:.4f}")
        logger.info(f"Total time: {self.stats.total_time:.2f} seconds")
        logger.info(f"Total retries: {self.stats.total_retries}")
        if self.stats.rate_limited_files:
            logger.warning(f"Rate limited files: {', '.join(self.stats.rate_limited_files)}")
        if self.stats.failed_files:
            logger.warning(f"Failed files: {', '.join(self.stats.failed_files)}")

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Translate LaTeX files from Russian to English using OpenAI API.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--input', '-i',
        required=True,
        help='Input file or directory containing LaTeX files'
    )
    
    parser.add_argument(
        '--output', '-o',
        required=True,
        help='Output directory for translated files'
    )
    
    parser.add_argument(
        '--api-key', '-k',
        required=True,
        help='OpenAI API key'
    )
    
    parser.add_argument(
        '--workers', '-w',
        type=int,
        default=10,
        help='Maximum number of concurrent workers'
    )
    
    parser.add_argument(
        '--model', '-m',
        choices=list(PRICING_MODELS.keys()),
        default='gpt-4.1-mini',
        help='OpenAI model to use for translation'
    )
    
    parser.add_argument(
        '--input-cost',
        type=float,
        help='Override input cost per token (defaults to model-specific pricing)'
    )
    
    parser.add_argument(
        '--output-cost',
        type=float,
        help='Override output cost per token (defaults to model-specific pricing)'
    )
    
    parser.add_argument(
        '--max-retries',
        type=int,
        default=5,
        help='Maximum number of retries for rate-limited requests'
    )
    
    parser.add_argument(
        '--initial-retry-delay',
        type=float,
        default=2.0,
        help='Initial delay between retries in seconds'
    )
    
    parser.add_argument(
        '--max-retry-delay',
        type=float,
        default=60.0,
        help='Maximum delay between retries in seconds'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    return parser.parse_args()

async def main():
    args = parse_arguments()
    
    # Set logging level based on verbose flag
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Remove output folder if it exists
    if os.path.exists(args.output):
        shutil.rmtree(args.output)

    processor = FileProcessor(
        api_key=args.api_key,
        max_workers=args.workers,
        model=args.model,
        input_cost_per_token=args.input_cost,
        output_cost_per_token=args.output_cost,
        max_retries=args.max_retries,
        initial_retry_delay=args.initial_retry_delay,
        max_retry_delay=args.max_retry_delay
    )
    
    await processor.process_files(args.input, args.output)

if __name__ == '__main__':
    asyncio.run(main())
