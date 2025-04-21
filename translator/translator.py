#!/usr/bin/python3
import os
import shutil
import sys
import openai
import time
import concurrent.futures
import re

class FileProcessor:
    def __init__(self, api_key, max_workers=10):
        self.api_key = api_key
        self.max_workers = max_workers

    def contains_russian(self, text):
        # Check for any Russian characters in the text
        return bool(re.search(r'[а-яА-Я]', text))

    def translate(self, content):
        openai.api_key = self.api_key
        start_time = time.time()
        client = openai.OpenAI(api_key=self.api_key)

        response = client.responses.create(
            model="gpt-4.1-mini",
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

        # Calculate time spent
        time_spent = end_time - start_time

        # Accurate cost calculation
        input_tokens = response.usage.input_tokens
        output_tokens = response.usage.output_tokens

        input_cost = input_tokens * 0.40 / 1_000_000  # $0.40 per 1M input tokens
        output_cost = output_tokens * 1.60 / 1_000_000  # $1.60 per 1M output tokens

        total_cost = input_cost + output_cost

        return translated_content, time_spent, total_cost  # Join lines back into a single string

    def process_file(self, input_file_path, output_file_path):
        # Resolve symlink to the actual file path
        actual_file_path = os.path.realpath(input_file_path)
        
        with open(actual_file_path, 'r') as f:
            content = f.read()
        
        if self.contains_russian(content):
            translated_content, time_spent, cost = self.translate(content)
        else:
            translated_content = content
            time_spent = 0
            cost = 0
        
        # Ensure the output directory exists
        os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
        
        with open(output_file_path, 'w') as f:
            f.write(translated_content)
        return os.path.basename(input_file_path), time_spent, cost

    def process_files(self, input_folder, output_folder):
        total_cost = 0
        total_time_spent = 0  # Initialize total time spent

        # Check if input_folder is a file or directory
        if os.path.isfile(input_folder):
            # Process only the single file
            file = os.path.basename(input_folder)
            output_file_path = os.path.join(output_folder, file)

            os.makedirs(output_folder, exist_ok=True)  # Create output folder if it doesn't exist

            file_info = self.process_file(input_folder, output_file_path)
            print(f"Processed file: {file_info[0]}, Time spent: {file_info[1]:.2f} seconds, Cost: ${file_info[2]:.4f}")
            total_cost += file_info[2]
            total_time_spent += file_info[1]  # Add to total time spent
        else:
            # Process all files in the directory
            os.makedirs(output_folder, exist_ok=True)  # Create output directory if it doesn't exist
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                future_to_file = {}
                for root, dirs, files in os.walk(input_folder):
                    # Skip directories that start with a dot
                    if os.path.basename(root).startswith('.'):
                        continue

                    
                    for file in files:
                        if file.startswith('.'):
                            continue  # Skip files that start with a dot
                        
                        input_file_path = os.path.join(root, file)
                        # Preserve the folder structure in the output path
                        relative_path = os.path.relpath(input_file_path, input_folder)
                        output_file_path = os.path.join(output_folder, relative_path)

                        if file.endswith('.tex'):
                            future = executor.submit(self.process_file, input_file_path, output_file_path)
                            future_to_file[future] = file
                        else:
                            os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
                            shutil.copy(input_file_path, output_file_path)
                            print(f"Copy file: {file}")

                for future in concurrent.futures.as_completed(future_to_file):
                    file_name = future_to_file[future]
                    try:
                        file_info = future.result()
                        print(f"Processed file: {file_info[0]}, Time spent: {file_info[1]:.2f} seconds, Cost: ${file_info[2]:.4f}")
                        total_cost += file_info[2]
                        total_time_spent += file_info[1]  # Add to total time spent
                    except Exception as e:
                        print(f"Error processing file {file_name}: {e}")

        # Print total cost and total time spent at the end
        print(f"Total cost for processing: ${total_cost:.4f}")
        print(f"Total time spent: {total_time_spent:.2f} seconds")

if __name__ == '__main__':
    if len(sys.argv) < 4 or len(sys.argv) > 5:
        print("Usage: python file_processor.py <input_folder> <output_folder> <api_key> [<max_workers>]")
        sys.exit(1)

    input_folder = sys.argv[1]
    output_folder = sys.argv[2]
    api_key = sys.argv[3]
    max_workers = int(sys.argv[4]) if len(sys.argv) == 5 else 10

    # Remove output folder if it exists
    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)

    processor = FileProcessor(api_key, max_workers)
    processor.process_files(input_folder, output_folder)
