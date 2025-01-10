import subprocess
import os
import re
import tarfile
import requests
import concurrent.futures
import openai  # openai library to use GPT API
from bs4 import BeautifulSoup
import logging
import shutil
import json
import time

from airflow import Dataset
from airflow.decorators import dag, task
from pendulum import datetime
import arxiv
from datetime import timedelta


# Default args for DAG
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# Define DAG
@dag(
    'arxiv_to_korean_pipeline',
    default_args=default_args,
    description='Fetch and translate ArXiv AI papers to Korean',
    schedule_interval=timedelta(days=1),
    start_date=datetime(2025, 1, 1),
    catchup=False,
)

def arxiv_pipeline():

    @task
    # Functions for pipeline tasks
    def fetch_arxiv_papers():
        query = '(cat:cs.AI OR cat:cs.LG OR cat:cs.CV OR cat:cs.CL)'
        search = arxiv.Search(
            query=query,
            max_results=100,
            sort_by=arxiv.SortCriterion.SubmittedDate
        )
        
        papers = []
        for result in search.results():
            paper = {
                'title': result.title,
                'abstract': result.summary,
                'url': result.entry_id,
                'date': result.published.date(),
                'authors': [author.name for author in result.authors]
            }
            papers.append(paper)
        
        return papers

    

    @task
    def filter_papers(papers, keywords=None):
        if keywords is None:
            keywords = ["transformer", "GAN", "self-supervised learning"]
        
        filtered_papers = []
        for paper in papers:
            if any(keyword.lower() in (paper['title'] + paper['abstract']).lower() for keyword in keywords):
                filtered_papers.append(paper)
        return filtered_papers[0]

    @task
    def log():
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    @task
    def extract_arxiv_id(url: str) -> str:
        """Extracting arXiv ID from URL"""
        logging.debug(f"Extracting arXiv ID from URL: {url}")
        arxiv_id = url.split('/')[-1] if 'arxiv.org' in url else url
        logging.debug(f"Extracted arXiv ID: {arxiv_id}")
        return arxiv_id

    @task
    def remove_latex_commands(text: str) -> str:
        # Auto-replacement of CJK* related
        text = re.sub(r'\\begin{CJK\*}\{.*?\}\{.*?\}', '', text)
        text = re.sub(r'\\end{CJK\*}', '', text)

        return text

    @task
    def translate_text(text: str, paper_info: dict, chunk_size: int, target_language: str = "Korean") -> str:
        """Translates text using GPT API while preserving LaTeX structure and formatting."""
        paper_title = paper_info.get('title', '')
        paper_abstract = paper_info.get('abstract', '')

        cleaned_text = remove_latex_commands(text)
        logging.debug("Sending translation request to GPT API.")

        retry_attempts = 3  # Number of retry attempts
        for attempt in range(retry_attempts):
            try:
                response = openai.ChatCompletion.create(
                    model="gpt-4o-mini",
                    response_format={"type": "json_object"},
                    messages=[
                        {"role": "system", "content": f"""
                            You are an AI assistant specialized in translating academic papers in LaTeX format to {target_language}. Your task is to translate the content accurately while preserving the LaTeX structure and formatting. Pay close attention to technical terms and follow these guidelines meticulously.
                            Translation Instructions:

                            1. Translate the main content into {target_language}, preserving the structure and flow of the original text. Use an academic and formal tone appropriate for scholarly publications in {target_language}. Do not insert any arbitrary line breaks in the translated content.

                            2. Technical Terms:
                            a. Retain well-known technical terms, product names, or specialized concepts (e.g., Few-Shot Learning) in English.
                            b. Do not translate examples, especially if they contain technical content or are essential for context.

                            3. LaTeX Commands:
                            - Do not translate LaTeX commands, functions, environments, or specific LaTeX-related keywords (e.g., \section{{}}, \begin{{}}, \end{{}}, \cite{{}}, \ref{{}}, or TikZ syntax such as /tikz/fill, /tikz/draw, etc.) into {target_language}. They must be output exactly as they are.
                            - Only translate the provided text without making any additional modifications.

                            4. Citation and Reference Keys:
                            - Ensure all citation keys within \\cite{{}} and reference keys within \\ref{{}} remain unchanged. Do not translate or modify these keys.

                            5. URLs and DOIs:
                            - Do not translate URLs, DOIs, or any other links. Keep them in their original form.

                            6. Mathematical Equations and Formulas:
                            - Maintain all mathematical equations and formulas as they are in the original LaTeX. Do not translate code or LaTeX mathematical notation.

                            7. Names:
                            - Do not translate author names, personal names, or any other individual names. Keep these in their original English form.

                            8. Consistency:
                            - Ensure consistent terminology throughout the translation.

                            9. Protection of LaTeX Commands:
                                - Preserve line breaks (\\\\) and other formatting commands exactly as they appear in the original text.

                            10. Avoid Misleading Translations:
                                - Do not translate technical terms, product names, specialized concepts, examples, or personal names where translation could lead to a loss of meaning or context.

                            11. JSON Structure:
                                - Translate the content line by line, providing the translation in a JSON structure.
                                For example:
                                ```json
                                translate : {{
                                    lines: [
                                    "Translated Line 1",
                                    "Translated Line 2"
                                ]}}
                                ```

                            ### Paper Info:
                            - Title : {paper_title}
                            - Abstract : {paper_abstract}
                            
                            ### Response Example:
                            #INPUT:
                            ["\\n", "\\documentclass{{article}} % For LaTeX2e\\n", "\\usepackage{{colm2024_conference}}\\n", "\\n", "\\usepackage{{microtype}}\\n"]
                            #OUTPUT:
                            {{"translate": {{"lines": ["\\n", "\\documentclass{{article}} % For LaTeX2e\\n", "\\usepackage{{colm2024_conference}}", "\\n", "\\usepackage{{microtype}}\\n"]}}}}
                            
                            ### VERY IMPORTANT
                            - DO NOT translate comments starting with "%" by arbitrarily merging them.
                            - DO NOT break a sentence into multiple paragraphs.
                            - Output the translated result in JSON format, without any other explanation.
                            - It should be translated and output in the same form as the unconditional input.
                            - Translate and output even single characters like '\\n', '{{', '/', '%', etc.
                            """
                        },
                        {"role": "user", "content": f"{cleaned_text}"}
                    ]
                )

                translated_content = response.choices[0].message['content']
                translation_result = json.loads(translated_content)
                translation_lines = translation_result["translate"]['lines']
                translated_line_count = len(translation_lines)

                if translated_line_count != chunk_size:
                    time.sleep(1)  # Optional: wait before retrying
                    continue  # Retry the translation

                return ''.join(translation_lines)

            except Exception as error:
                logging.error(f"Error during translation attempt {attempt + 1}: {error}")
                if attempt == retry_attempts - 1:
                    raise  # Re-raise the last exception after exhausting retries

        raise Exception("Translation failed after multiple attempts.")


    @task
    def add_custom_font_to_tex(tex_file_path: str, font_name: str = "Noto Sans KR", mono_font_name: str = "Noto Sans KR"):
        """Adding user designated font to the files"""
        logging.info(f"Adding custom font '{font_name}' to TeX file: {tex_file_path}")
        remove_cjk_related_lines(tex_file_path)
        font_setup = rf"""
            \usepackage{{kotex}}
            \usepackage{{xeCJK}}
            \setCJKmainfont{{{font_name}}}
            \setCJKmonofont{{{mono_font_name}}}
            \xeCJKsetup{{CJKspace=true}}
            """
        try:
            with open(tex_file_path, 'r+', encoding='utf-8') as file:
                lines = file.readlines()
                for i, line in enumerate(lines):
                    if line.startswith(r'\documentclass'):
                        lines.insert(i + 1, font_setup)
                        break
                file.seek(0)
                file.writelines(lines)
            logging.debug("Custom font added successfully.")
        except Exception as e:
            logging.error(f"Failed to add custom font: {e}")
            raise

    @task
    def remove_cjk_related_lines(tex_file_path: str):
        """Removing CJK related packages and settings from text files"""
        logging.info(f"Removing CJK related lines from TeX file: {tex_file_path}")

        cjk_related_keywords = [
            r'\usepackage{CJKutf8}',
            r'\usepackage{kotex}',
            r'\begin{CJK}',
            r'\end{CJK}',
            r'\CJKfamily',
            r'\CJK@',
            r'\CJKrmdefault',
            r'\CJKsfdefault',
            r'\CJKttdefault',
        ]

        try:
            with open(tex_file_path, 'r+', encoding='utf-8') as file:
                lines = file.readlines()
                new_lines = []
                for line in lines:
                    if not any(keyword in line for keyword in cjk_related_keywords):
                        new_lines.append(line)

                file.seek(0)
                file.writelines(new_lines)
                file.truncate()

            logging.debug("CJK related lines removed successfully.")
        except Exception as e:
            logging.error(f"Failed to remove CJK related lines: {e}")
            raise

    
    @task
    def process_and_translate_tex_files(directory: str, paper_info: dict, read_lines: int = 30,
                                        target_language: str = "Korean", max_parallel_tasks: int = 8):
        """Processes .tex files by splitting them into chunks and translating them in parallel, ensuring error handling."""
        logging.info(f"Processing and translating lines in .tex files in directory: {directory}")

        file_line_chunks = []
        total_chunks = 0

        # Split files into chunks of lines and save to a list
        for root, _, files in os.walk(directory):
            for file in files:
                if file.endswith(".tex"):
                    file_path = os.path.join(root, file)
                    original_file_path = file_path + "_original"
                    logging.info(f"Reading file: {file_path}")
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            lines = f.readlines()

                        # Save the original file with a different name
                        with open(original_file_path, 'w', encoding='utf-8') as original_f:
                            original_f.writelines(lines)

                        # Split the lines into safe chunks
                        chunks = chunk_lines_safely(lines, read_lines)

                        for idx, chunk in enumerate(chunks):
                            file_line_chunks.append((file_path, idx, chunk))
                        total_chunks += len(chunks)

                        # Save the modified file after removing comments or making other changes
                        with open(file_path, 'w', encoding='utf-8') as f:
                            f.writelines(lines)

                    except Exception as e:
                        logging.error(f"Error reading or writing file {file_path}: {e}")

        if total_chunks == 0:
            logging.warning("No lines to translate.")
            return

        completed_chunks = 0

        # Translate each chunk in parallel
        def translate_chunk(file_chunk_info):
            nonlocal completed_chunks
            file_path, chunk_idx, chunk = file_chunk_info
            try:
                # Format the chunks as a list of dictionaries
                formatted_chunk = [line for idx, line in enumerate(chunk)]
                translated_text = translate_text(json.dumps(formatted_chunk), paper_info, len(chunk), target_language)
            except Exception as e:
                logging.error(f"Error translating chunk in file {file_path}: {e}")
                translated_text = ''.join(chunk)  # Return original text in case of an error

            completed_chunks += 1
            progress = (completed_chunks / total_chunks) * 100
            logging.info(f"Translation progress: {progress:.2f}% completed.")
            return (file_path, chunk_idx, translated_text)

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_parallel_tasks) as executor:
            translated_pairs = list(executor.map(translate_chunk, file_line_chunks))

        # Reassemble and save the translated content by file
        file_contents = {}
        for file_path, chunk_idx, translated_chunk in translated_pairs:
            if file_path not in file_contents:
                file_contents[file_path] = []
            file_contents[file_path].append((chunk_idx, translated_chunk))

        for file_path, chunks in file_contents.items():
            # Sort chunks by their original index
            sorted_chunks = sorted(chunks, key=lambda x: x[0])
            translated_content = ''.join(chunk for _, chunk in sorted_chunks)
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(translated_content)
                logging.info(f"File translated and saved: {file_path}")
            except Exception as e:
                logging.error(f"Error writing translated content to {file_path}: {e}")


    @task
    def chunk_lines_safely(lines, lines_per_chunk):
        """
        Safely splits the given lines into chunks of a specified number of lines,
        excluding lines that are only newline characters.

        Args:
        - lines: List of all lines in the text.
        - lines_per_chunk: Number of lines to include in each chunk.

        Returns:
        - A list of chunks, where each chunk is a list of lines.
        """
        chunks = []
        current_chunk = []
        current_line_count = 0

        for line in lines:

            current_chunk.append(line)
            current_line_count += 1

            # If the specified number of lines is reached, save the current chunk and start a new one
            if current_line_count >= lines_per_chunk:
                chunks.append(current_chunk)
                current_chunk = []
                current_line_count = 0

        # If there are remaining lines, add them as the last chunk
        if current_chunk:
            chunks.append(current_chunk)

        return chunks


    @task
    def extract_tar_gz(tar_file_path: str, extract_to: str):
        """Extracting tar.gz file to designated directory"""
        logging.info(f"Extracting tar.gz file: {tar_file_path} to {extract_to}")

        try:
            with tarfile.open(tar_file_path, 'r:gz') as tar_ref:
                tar_ref.extractall(path=extract_to)
            logging.debug("Extraction completed successfully.")
        except Exception as e:
            logging.error(f"Failed to extract tar.gz file: {e}")
            raise

    @task
    def find_main_tex_file(directory: str) -> str:
        """Finding main .tex file that includes 'documentclass'"""
        logging.info(f"Searching for main .tex file in directory: {directory}")

        candidate_files = []
        for root, _, files in os.walk(directory):
            for file in files:
                if file.endswith(".tex") and "_original" not in file:
                    candidate_files.append(os.path.join(root, file))

        main_candidates = []
        for file in candidate_files:
            try:
                with open(file, 'r', encoding='utf-8') as f:
                    contents = f.read()
                    # Looking for file with \documentclass
                    if r'\documentclass' in contents:
                        # To verify main file, check packages and environment settings
                        if any(keyword in contents for keyword in [r'\begin{document}', r'\usepackage', r'\title', r'\author']):
                            logging.debug(f"Main candidate .tex file found: {file}")
                            main_candidates.append(file)
            except Exception as e:
                logging.error(f"Failed to read file {file}: {e}")

        # Return the very first file from main
        if main_candidates:
            logging.debug(f"Selected main .tex file: {main_candidates[0]}")
            return main_candidates[0]

        # If there's no main file candidate, transform the largest .tex file
        if candidate_files:
            main_tex = max(candidate_files, key=os.path.getsize, default=None)
            logging.debug(f"No clear main file found, selected by size: {main_tex}")
            return main_tex

        logging.warning("No .tex files found.")
        return None


    @task
    def compile_main_tex(directory: str, arxiv_id: str, font_name: str = "Noto Sans KR"):
        """Compile main .tex file to create PDF"""
        logging.info(f"Compiling main .tex file in directory: {directory}")

        main_tex_path = find_main_tex_file(directory)
        if main_tex_path:
            add_custom_font_to_tex(main_tex_path, font_name)
            compile_tex_to_pdf(main_tex_path, arxiv_id, compile_twice=True)
        else:
            logging.error("Main .tex file not found. Compilation aborted.")


    @task
    def compile_tex_to_pdf(tex_file_path: str, arxiv_id: str, compile_twice: bool = True):
        """Compile .txt file into .pdf"""
        logging.info(f"Compiling TeX file to PDF: {tex_file_path}")

        tex_dir = os.path.dirname(tex_file_path)
        tex_file = os.path.basename(tex_file_path)

        try:
            for _ in range(2 if compile_twice else 1):
                result = subprocess.run(
                    ['xelatex', '-interaction=nonstopmode',tex_file],
                    cwd=tex_dir,
                    encoding='utf-8'
                )
                logging.info(f"xelatex output: {result.stdout}")
                logging.info(f"xelatex errors: {result.stderr}")

            output_pdf = os.path.join(tex_dir, tex_file.replace(".tex", ".pdf"))
            if os.path.exists(output_pdf):
                current_dir = os.getcwd()
                final_pdf_path = os.path.join(current_dir, f"{arxiv_id}.pdf")
                os.rename(output_pdf, final_pdf_path)
                logging.info(f"PDF compiled and saved as: {final_pdf_path}")
            else:
                logging.error("PDF output not found after compilation.")
        except Exception as e:
            logging.error(f"Failed to compile TeX file: {e}")
            raise


    @task
    def download_arxiv_intro_and_tex(arxiv_id: str, download_dir: str, target_language: str = "Korean",
                                    font_name: str = "Noto Sans KR"):
        """Download and translate arXiv info and text file"""
        logging.info(f"Downloading and processing arXiv paper: {arxiv_id}")

        arxiv_url = f"https://export.arxiv.org/api/query?id_list={arxiv_id}"

        try:
            response = requests.get(arxiv_url)
            response.raise_for_status()
        except requests.RequestException as e:
            logging.error(f"Failed to fetch arXiv metadata: {e}")
            raise

        soup = BeautifulSoup(response.content, 'xml')
        entry = soup.find('entry')
        if not entry:
            logging.error("ArXiv entry not found.")
            raise ValueError("ArXiv entry not found.")

        paper_info = {
            "title": entry.find('title').text,
            "abstract": entry.find('summary').text
        }
        logging.debug(f"Paper info: {paper_info}")

        tar_url = f"https://arxiv.org/src/{arxiv_id}"
        tar_file_path = os.path.join(download_dir, f"{arxiv_id}.tar.gz")

        os.makedirs(download_dir, exist_ok=True)

        try:
            with requests.get(tar_url, stream=True) as r:
                r.raise_for_status()
                with open(tar_file_path, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
            logging.info(f"Downloaded tar.gz file: {tar_file_path}")
        except requests.RequestException as e:
            logging.error(f"Failed to download arXiv source tarball: {e}")
            raise

        extract_to = os.path.join(download_dir, arxiv_id)

        # delete if arxiv_id exists already
        if os.path.exists(extract_to):
            logging.info(f"Existing directory found: {extract_to}. Deleting it.")
            shutil.rmtree(extract_to)

        os.makedirs(extract_to, exist_ok=True)

        extract_tar_gz(tar_file_path, extract_to)
        process_and_translate_tex_files(extract_to, paper_info, target_language=target_language)
        compile_main_tex(extract_to, arxiv_id, font_name)

    @task
    def translate_abstract():
        # Settings for GPT API Request
        openai.api_key = 'YOUR_API_KEY_HERE'  # Your OpenAI API Key


    # Pipeline
    raw_data = fetch_arxiv_papers()
    filtered_data = filter_papers(raw_data)
    translate_abstract()
    arxiv_id = extract_arxiv_id(filtered_data['url'])
    download_dir = 'arxiv_downloads'
    download_arxiv_intro_and_tex(arxiv_id, download_dir, target_language="Korean", font_name="Noto Sans KR")


arxiv_dag = arxiv_pipeline()
