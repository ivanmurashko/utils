#!/usr/bin/python3
import os
import shutil
import sys
import time
import asyncio
import aiofiles
import re
import logging
import argparse
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
from pathlib import Path
from openai import AsyncOpenAI
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class TranslationResult:
    file_name: str
    time_spent: float
    cost: float
    success: bool
    error: Optional[str] = None

@dataclass
class ProcessingStats:
    total_files: int
    processed_files: int
    total_cost: float
    total_time: float
    failed_files: List[str]

class FileProcessor:
    def __init__(
        self,
        api_key: str,
        max_workers: int = 10,
        model: str = "gpt-4.1-mini",
        input_cost_per_token: float = 0.40 / 1_000_000,
        output_cost_per_token: float = 1.60 / 1_000_000
    ):
        self.api_key = api_key
        self.max_workers = max_workers
        self.model = model
        self.input_cost_per_token = input_cost_per_token
        self.output_cost_per_token = output_cost_per_token
        self.client = AsyncOpenAI(api_key=api_key)
        self.stats = ProcessingStats(0, 0, 0.0, 0.0, [])

    def contains_russian(self, text: str) -> bool:
        """Check if the text contains Russian characters."""
        return bool(re.search(r'[а-яА-Я]', text))

    async def translate(self, content: str) -> Tuple[str, float, float]:
        """Translate content using OpenAI API."""
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

            return translated_content, time_spent, total_cost
        except Exception as e:
            logger.error(f"Translation error: {str(e)}")
            raise

    async def process_file(self, input_file_path: str, output_file_path: str) -> TranslationResult:
        """Process a single file."""
        try:
            # Resolve symlink to the actual file path
            actual_file_path = os.path.realpath(input_file_path)
            
            async with aiofiles.open(actual_file_path, 'r', encoding='utf-8') as f:
                content = await f.read()
            
            if self.contains_russian(content):
                translated_content, time_spent, cost = await self.translate(content)
            else:
                translated_content = content
                time_spent = 0
                cost = 0
            
            # Ensure the output directory exists
            os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
            
            async with aiofiles.open(output_file_path, 'w', encoding='utf-8') as f:
                await f.write(translated_content)
                
            return TranslationResult(
                file_name=os.path.basename(input_file_path),
                time_spent=time_spent,
                cost=cost,
                success=True
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

    async def process_files(self, input_folder: str, output_folder: str) -> None:
        """Process all files in the input folder."""
        self.stats = ProcessingStats(0, 0, 0.0, 0.0, [])
        
        # Check if input_folder is a file or directory
        if os.path.isfile(input_folder):
            # Process single file
            output_file_path = os.path.join(output_folder, os.path.basename(input_folder))
            os.makedirs(output_folder, exist_ok=True)
            
            result = await self.process_file(input_folder, output_file_path)
            self._update_stats(result)
            self._log_result(result)
        else:
            # Process directory
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
                        tasks.append(self.process_file(input_file_path, output_file_path))
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
            logger.info(
                f"Processed file: {result.file_name}, "
                f"Time: {result.time_spent:.2f}s, "
                f"Cost: ${result.cost:.4f}"
            )
        else:
            logger.error(
                f"Failed to process file: {result.file_name}, "
                f"Error: {result.error}"
            )

    def _log_summary(self) -> None:
        """Log the summary of processing."""
        logger.info("\nProcessing Summary:")
        logger.info(f"Total files processed: {self.stats.processed_files}/{self.stats.total_files}")
        logger.info(f"Total cost: ${self.stats.total_cost:.4f}")
        logger.info(f"Total time: {self.stats.total_time:.2f} seconds")
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
        default='gpt-4.1-mini',
        help='OpenAI model to use for translation'
    )
    
    parser.add_argument(
        '--input-cost',
        type=float,
        default=0.40 / 1_000_000,
        help='Cost per input token'
    )
    
    parser.add_argument(
        '--output-cost',
        type=float,
        default=1.60 / 1_000_000,
        help='Cost per output token'
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
        output_cost_per_token=args.output_cost
    )
    
    await processor.process_files(args.input, args.output)

if __name__ == '__main__':
    asyncio.run(main())
