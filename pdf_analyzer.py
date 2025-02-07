import os
import sys
from dotenv import load_dotenv
import PyPDF2
import google.generativeai as genai
import fitz  # PyMuPDF for better PDF handling
import io
from PIL import Image
import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import print as rprint
import json
import datetime

# Load environment variables from .env file
load_dotenv()

# Configure the Gemini API using the GEMINI_API_KEY from .env
gemini_api_key = os.getenv("GEMINI_API_KEY")
if not gemini_api_key:
    print("Error: GEMINI_API_KEY not found in environment variables.")
    sys.exit(1)

genai.configure(api_key=gemini_api_key)

console = Console()
app = typer.Typer()

def extract_text_and_images_from_pdf(pdf_path):
    """Extracts both text and images from a PDF file using PyMuPDF."""
    text_content = ""
    images = []
    try:
        doc = fitz.open(pdf_path)
        for page_num in range(len(doc)):
            page = doc[page_num]
            text_content += page.get_text() + "\n"
            
            # Extract images
            image_list = page.get_images(full=True)
            for img_index, img in enumerate(image_list):
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                
                # Convert to PIL Image
                image = Image.open(io.BytesIO(image_bytes))
                images.append(image)
                
        doc.close()
    except Exception as e:
        console.print(f"[red]Error reading PDF file: {e}[/red]")
        sys.exit(1)
    return text_content, images


class PDFAnalyzer:
    """Enhanced PDF analyzer with multiple Gemini models for different tasks."""
    def __init__(self, mode="standard"):
        self.mode = mode
        
        if mode == "quick":
            # Use Gemini 1.5 Flash for everything in quick mode
            self.text_model = genai.GenerativeModel("gemini-1.5-flash")
            self.vision_model = self.text_model
        else:
            # Use most powerful models in standard mode
            self.text_model = genai.GenerativeModel("gemini-2.0-flash-exp")  # Latest Gemini 2.0
            self.vision_model = self.text_model  # 2.0 supports multimodal
            self.thinking_model = genai.GenerativeModel("gemini-2.0-flash-thinking-exp-1219")  # For complex reasoning
        
        # Common models for both modes
        self.embedding_model = genai.GenerativeModel("text-embedding-004")
        self.aqa_model = genai.GenerativeModel("aqa")
        
        # Chunk size optimization based on model context windows
        self.chunk_size = 8192 if mode == "quick" else 16384  # Larger chunks for more powerful models
        self.chunk_overlap = self.chunk_size // 10  # 10% overlap
        
        # Initialize safety settings
        self.safety_settings = {
            "HARM_CATEGORY_HARASSMENT": "BLOCK_MEDIUM_AND_ABOVE",
            "HARM_CATEGORY_HATE_SPEECH": "BLOCK_MEDIUM_AND_ABOVE",
            "HARM_CATEGORY_SEXUALLY_EXPLICIT": "BLOCK_MEDIUM_AND_ABOVE",
            "HARM_CATEGORY_DANGEROUS_CONTENT": "BLOCK_MEDIUM_AND_ABOVE",
        }

    def _process_with_progress(self, chunks, process_func, progress_callback=None):
        """Process chunks with progress updates."""
        results = []
        total_chunks = len(chunks)
        
        for i, chunk in enumerate(chunks):
            if progress_callback:
                progress = int((i + 1) / total_chunks * 100)
                progress_callback(progress)
            
            result = process_func(chunk, i, total_chunks)
            results.append(result)
            
        return results

    def analyze(self, text, progress_callback=None):
        """Analyze text using appropriate models based on mode."""
        if len(text) > self.chunk_size:
            return self._analyze_in_chunks(text, progress_callback)
        
        if self.mode == "quick":
            prompt = """Analyze this document concisely. Focus on:
            1. Key facts and findings
            2. Main insights
            3. Important implications
            Keep it brief and direct."""
            response = self.text_model.generate_content(prompt + "\n\n" + text)
            return response.text.strip() if response and response.text else "No analysis generated."
        
        # Standard mode: Use combination of models
        try:
            if progress_callback:
                progress_callback(20)
                
            # Get factual analysis from AQA
            aqa_response = self.aqa_model.generate_content(
                "Extract key verifiable facts from this document." + "\n\n" + text
            )
            
            if progress_callback:
                progress_callback(40)
                
            # Get deep insights using thinking model
            thinking_response = self.thinking_model.generate_content(
                """Analyze this document deeply, considering:
                1. Complex patterns and relationships
                2. Hidden implications
                3. Strategic recommendations
                4. Future impacts
                Show your reasoning process.""" + "\n\n" + text
            )
            
            if progress_callback:
                progress_callback(70)
                
            # Get final insights from Gemini 2.0
            insight_response = self.text_model.generate_content(
                """Synthesize the key insights from this document, focusing on:
                1. Critical findings
                2. Strategic implications
                3. Actionable recommendations""" + "\n\n" + text
            )
            
            if progress_callback:
                progress_callback(100)
            
            # Combine all analyses
            combined_analysis = ""
            if aqa_response and aqa_response.text:
                combined_analysis += "Key Facts:\n" + aqa_response.text.strip() + "\n\n"
            if thinking_response and thinking_response.text:
                combined_analysis += "Deep Analysis:\n" + thinking_response.text.strip() + "\n\n"
            if insight_response and insight_response.text:
                combined_analysis += "Key Insights:\n" + insight_response.text.strip()
                
            return combined_analysis if combined_analysis else "No analysis generated."
            
        except Exception as e:
            return f"Error during analysis: {str(e)}"

    def _analyze_in_chunks(self, text, progress_callback=None):
        # Create optimized chunks
        chunks = []
        start = 0
        while start < len(text):
            end = start + self.chunk_size
            
            # If not at the end, try to break at a sentence boundary
            if end < len(text):
                # Look for sentence boundaries (., !, ?)
                for i in range(min(end + 100, len(text)), end - 100, -1):
                    if text[i-1] in '.!?' and text[i:i+1].isspace():
                        end = i
                        break
            
            chunks.append(text[start:end])
            start = end - self.chunk_overlap
        
        def process_chunk(chunk, index, total):
            if self.mode == "quick":
                prompt = f"Analyze part {index + 1}/{total} of the document. Focus on key points and insights."
                model = self.text_model
            else:
                prompt = f"Provide detailed analysis for part {index + 1}/{total}, including facts, insights, and implications."
                model = self.thinking_model
            
            response = model.generate_content(prompt + "\n\n" + chunk)
            return response.text.strip() if response and response.text else f"No analysis for part {index + 1}"
        
        # Process chunks with progress tracking
        analyses = self._process_with_progress(chunks, process_chunk, progress_callback)
        
        # Combine results
        if self.mode == "quick":
            return "\n\n".join(analyses)
        
        # For standard mode, use Gemini 2.0 to synthesize
        final_prompt = "Synthesize these partial analyses into a coherent overall analysis:\n\n" + "\n\n".join(analyses)
        final_response = self.text_model.generate_content(final_prompt)
        return final_response.text.strip() if final_response and final_response.text else "Could not synthesize analyses"

    def summarize(self, text):
        # Use chunking if text is long
        if len(text) > 3000:
            return self._summarize_in_chunks(text)
        prompt = "Summarize the following document concisely:\n\n" + text
        response = self.text_model.generate_content(prompt)
        if response and response.text:
            return response.text.strip()
        else:
            return "No summary generated."

    def _summarize_in_chunks(self, text, chunk_size=3000, overlap=300):
        chunks = []
        start = 0
        while start < len(text):
            chunk = text[start:start+chunk_size]
            chunks.append(chunk)
            start += chunk_size - overlap
        summaries = []
        for i, chunk in enumerate(chunks):
            prompt = f"Summarize the following part ({i+1}/{len(chunks)}) of the document:\n\n" + chunk
            resp = self.text_model.generate_content(prompt)
            if resp and resp.text:
                summaries.append(resp.text.strip())
            else:
                summaries.append("No summary generated for this chunk.")
        combined_text = "\n".join(summaries)
        final_prompt = "Combine the following partial summaries into a coherent overall summary:\n\n" + combined_text
        final_resp = self.text_model.generate_content(final_prompt)
        if final_resp and final_resp.text:
            return final_resp.text.strip()
        else:
            return "No final summary generated."

    def analyze_image(self, image):
        """Analyzes an image using the vision model."""
        prompt = "Analyze this image from the PDF. Describe what you see, including any charts, diagrams, or visual elements. If there are any data visualizations, provide insights about them."
        response = self.vision_model.generate_content([prompt, image])
        if response and response.text:
            return response.text.strip()
        return "No image analysis generated."

    def extract_structured_data(self, text):
        """Extracts structured data from the text using advanced prompting."""
        prompt = """Extract key information from the document in JSON format. Include:
        {
            "title": "Document title if present",
            "key_points": ["List of main points"],
            "entities": {
                "people": ["Names of people mentioned"],
                "organizations": ["Organizations mentioned"],
                "locations": ["Locations mentioned"],
                "dates": ["Dates mentioned"]
            },
            "metrics": ["Any numerical data or statistics"],
            "action_items": ["Suggested next steps or actions"],
            "tables": ["Any tabular data detected"],
            "citations": ["Any references or citations"],
            "metadata": {
                "document_type": "Type of document",
                "language": "Primary language",
                "confidence_score": "Confidence in extraction (0-1)"
            }
        }"""
        
        response = self.text_model.generate_content(prompt + "\n\nDocument:\n" + text)
        if response and response.text:
            try:
                return json.loads(response.text)
            except json.JSONDecodeError:
                return {"error": "Could not parse structured data"}
        return {"error": "No structured data generated"}

    def extract_code_snippets(self, text):
        """Extracts and analyzes code snippets from the document."""
        prompt = """Find and analyze any code snippets in the document. For each snippet:
        1. Identify the programming language
        2. Extract the complete code block
        3. Provide a brief explanation of what the code does
        4. Note any potential issues or best practice violations
        5. Suggest improvements if applicable
        
        Format the response as JSON:
        {
            "code_blocks": [
                {
                    "language": "programming language",
                    "code": "actual code",
                    "explanation": "what it does",
                    "issues": ["list of issues"],
                    "improvements": ["suggested improvements"]
                }
            ]
        }"""
        
        response = self.text_model.generate_content(prompt + "\n\nDocument:\n" + text)
        if response and response.text:
            try:
                return json.loads(response.text)
            except json.JSONDecodeError:
                return {"error": "Could not parse code analysis"}
        return {"error": "No code analysis generated"}

    def extract_tables(self, text):
        """Extracts and structures tabular data from the document."""
        prompt = """Find and extract any tables from the document. For each table:
        1. Identify the table headers
        2. Extract all rows of data
        3. Provide a brief description of what the table represents
        
        Format the response as JSON:
        {
            "tables": [
                {
                    "headers": ["column names"],
                    "data": [["row values"]],
                    "description": "table description"
                }
            ]
        }"""
        
        response = self.text_model.generate_content(prompt + "\n\nDocument:\n" + text)
        if response and response.text:
            try:
                return json.loads(response.text)
            except json.JSONDecodeError:
                return {"error": "Could not parse table data"}
        return {"error": "No tables extracted"}

    def generate_embeddings(self, text):
        """Generates embeddings for semantic search and clustering."""
        try:
            response = self.embedding_model.generate_content(text)
            if response and hasattr(response, 'embedding'):
                return response.embedding
            return None
        except Exception as e:
            print(f"Error generating embeddings: {e}")
            return None

    def analyze_document_structure(self, text):
        """Analyzes the document's structure and organization."""
        prompt = """Analyze the document's structure and organization. Include:
        1. Document hierarchy (sections, subsections)
        2. Format and styling patterns
        3. Navigation elements (table of contents, references)
        4. Visual elements (figures, tables, diagrams)
        5. Consistency analysis
        
        Format the response as JSON:
        {
            "structure": {
                "hierarchy": ["list of sections"],
                "formatting": ["format patterns"],
                "navigation": ["navigation elements"],
                "visual_elements": ["list of visuals"],
                "consistency_score": "0-1 score",
                "improvement_suggestions": ["suggestions"]
            }
        }"""
        
        response = self.text_model.generate_content(prompt + "\n\nDocument:\n" + text)
        if response and response.text:
            try:
                return json.loads(response.text)
            except json.JSONDecodeError:
                return {"error": "Could not parse structure analysis"}
        return {"error": "No structure analysis generated"}

    def extract_key_topics(self, text):
        """Extracts and clusters key topics from the document."""
        prompt = """Identify and cluster the key topics discussed in this document.
        For each topic:
        1. Provide a representative label
        2. List related subtopics
        3. Include relevant keywords
        4. Note the approximate coverage (percentage of document)
        
        Format as JSON:
        {
            "topics": [
                {
                    "label": "topic name",
                    "subtopics": ["related subtopics"],
                    "keywords": ["relevant terms"],
                    "coverage_percentage": "0-100"
                }
            ]
        }"""
        
        response = self.text_model.generate_content(prompt + "\n\nDocument:\n" + text)
        if response and response.text:
            try:
                return json.loads(response.text)
            except json.JSONDecodeError:
                return {"error": "Could not parse topics"}
        return {"error": "No topics generated"}

    def analyze_audio_content(self, audio_file):
        """Analyze audio content from PDFs (e.g., embedded audio, voice notes)."""
        try:
            prompt = """Analyze this audio content. Include:
            1. Transcription
            2. Key points
            3. Speaker identification (if multiple)
            4. Tone and sentiment analysis"""
            response = self.text_model.generate_content([prompt, audio_file])
            return response.text.strip() if response and response.text else "No audio analysis generated."
        except Exception as e:
            return f"Error analyzing audio: {str(e)}"

    def execute_code_analysis(self, code_text):
        """Execute and analyze code snippets with safety checks."""
        try:
            tools = {
                'code_execution': {},
                'function_declarations': [
                    {'name': 'analyze_code_safety'},
                    {'name': 'run_code_snippet'}
                ]
            }
            
            prompt = """Analyze and safely execute this code:
            1. Check for security issues
            2. Run in isolated environment
            3. Provide output and explanation
            4. Suggest improvements"""
            
            response = self.text_model.generate_content(
                prompt + "\n\n" + code_text,
                tools=tools
            )
            return response.text.strip() if response and response.text else "No code analysis generated."
        except Exception as e:
            return f"Error executing code analysis: {str(e)}"

    def smart_analysis(self, content):
        """Use multiple functions together for comprehensive analysis."""
        try:
            tools = {
                'google_search': {},
                'code_execution': {},
                'function_declarations': [
                    {'name': 'extract_data'},
                    {'name': 'verify_facts'},
                    {'name': 'generate_citations'}
                ]
            }
            
            response = self.text_model.generate_content(content, tools=tools)
            return response.text.strip() if response and response.text else "No smart analysis generated."
        except Exception as e:
            return f"Error in smart analysis: {str(e)}"

    def process_long_document(self, text):
        """Process very long documents efficiently."""
        try:
            chunks = self._create_optimal_chunks(text, self.context_window // 2)
            
            def process_chunk(chunk):
                response = self.text_model.generate_content(
                    "Analyze this document section while maintaining context.",
                    chunk
                )
                return response.text.strip() if response and response.text else "No analysis for this section."
            
            analyses = self._process_with_progress(chunks, process_chunk)
            return "\n\n".join(analyses)
        except Exception as e:
            return f"Error processing long document: {str(e)}"

    def live_analysis(self, content):
        """Real-time analysis with voice and visual feedback."""
        try:
            tools = {
                'multimodal_live': {
                    'audio_input': True,
                    'video_input': True,
                    'text_output': True,
                    'audio_output': True
                }
            }
            
            response = self.text_model.generate_content(content, tools=tools)
            return response.text.strip() if response and response.text else "No live analysis generated."
        except Exception as e:
            return f"Error in live analysis: {str(e)}"


def display_menu():
    """Display the main menu with available commands."""
    table = Table(title="PDF Analyzer Menu", show_header=True, header_style="bold magenta")
    table.add_column("Command", style="cyan")
    table.add_column("Description", style="green")
    table.add_column("Status", style="yellow")
    
    # Get status indicators
    file_status = "[red]No file loaded[/red]" if not hasattr(display_menu, 'current_file') else f"[green]Loaded: {os.path.basename(display_menu.current_file)}[/green]"
    
    # Commands
    table.add_row("upload", "Upload a PDF/Document file", file_status)
    table.add_row("summary", "Generate document summary", "Requires file")
    table.add_row("analyze", "Generate detailed analysis", "Requires file")
    table.add_row("images", "Analyze images and diagrams", "Requires file")
    table.add_row("structure", "Analyze document structure", "Requires file")
    table.add_row("tables", "Extract and analyze tables", "Requires file")
    table.add_row("code", "Extract and analyze code snippets", "Requires file")
    table.add_row("topics", "Extract key topics and clusters", "Requires file")
    table.add_row("full", "Run full analysis", "Requires file")
    table.add_row("save", "Save results to file", "Requires analysis")
    table.add_row("clear", "Clear screen", "Always available")
    table.add_row("help", "Show this menu", "Always available")
    table.add_row("exit", "Exit the program", "Always available")
    
    console.print(table)

def upload_file():
    """Get PDF file path from user."""
    try:
        console.print("[yellow]Enter the path to your PDF file:[/yellow]")
        file_path = input("> ").strip()
        
        if os.path.exists(file_path):
            if file_path.lower().endswith('.pdf'):
                display_menu.current_file = file_path
                console.print(f"[green]Successfully loaded: {os.path.basename(file_path)}[/green]")
                return file_path
            else:
                console.print("[red]File must be a PDF document[/red]")
                return None
        else:
            console.print("[red]File not found[/red]")
            return None
            
    except Exception as e:
        console.print("[red]Error loading file[/red]")
        return None

def save_results(results, original_file):
    """Save analysis results to file."""
    if not results:
        console.print("[red]No results to save[/red]")
        return
        
    try:
        # Generate default filename based on original file
        base_name = os.path.splitext(os.path.basename(original_file))[0]
        default_filename = f"{base_name}_analysis.txt"
        
        console.print(f"[yellow]Save as (default: {default_filename}):[/yellow]")
        filename = input("> ").strip()
        
        if not filename:
            filename = default_filename
            
        # Add txt extension if not specified
        if not os.path.splitext(filename)[1]:
            filename += '.txt'
            
        with open(filename, 'w', encoding='utf-8') as f:
            # Write metadata
            f.write(f"=== Analysis Report for {os.path.basename(original_file)} ===\n")
            f.write(f"Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Write each result section
            for key, value in results.items():
                if key != 'metadata':
                    f.write(f"=== {key.replace('_', ' ').title()} ===\n\n")
                    if isinstance(value, dict):
                        f.write(json.dumps(value, indent=2))
                    else:
                        f.write(str(value))
                    f.write("\n\n")
            
        console.print(f"[green]Results saved to: {filename}[/green]")
        
    except Exception as e:
        console.print(f"[red]Error saving results: {str(e)}[/red]")

def clear_screen():
    """Clear the terminal screen."""
    os.system('cls' if os.name == 'nt' else 'clear')

def process_command(command, analyzer=None):
    """Process user commands."""
    command = command.lower().strip()
    
    if command == 'exit':
        return False
        
    elif command == 'help':
        display_menu()
        
    elif command == 'clear':
        clear_screen()
        display_menu()
        
    elif command == 'upload':
        upload_file()
        
    elif command in ['summary', 'analyze', 'images', 'structure', 'tables', 'code', 'topics', 'full']:
        if not hasattr(display_menu, 'current_file'):
            console.print("[red]Please upload a file first using the 'upload' command[/red]")
            return True
            
        # Ask for mode preference
        console.print("[yellow]Choose analysis mode:[/yellow]")
        console.print("1. Standard (More detailed, uses latest Gemini 2.0 models)")
        console.print("2. Quick (Faster, uses Gemini 1.5 Flash)")
        mode_choice = input("> ").strip()
        
        mode = "quick" if mode_choice == "2" else "standard"
        
        if not analyzer or analyzer.mode != mode:
            analyzer = PDFAnalyzer(mode=mode)
            
        try:
            with console.status("[bold green]Processing...") as status:
                def update_progress(progress):
                    status.update(f"[bold green]Running {command} analysis... {progress}%")
                
                text, images = extract_text_and_images_from_pdf(display_menu.current_file)
                results = {}
                
                if command in ['summary', 'full']:
                    status.update("[bold green]Generating summary...")
                    results['summary'] = analyzer.summarize(text)
                    
                if command in ['analyze', 'full']:
                    results['analysis'] = analyzer.analyze(text, progress_callback=update_progress)
                    
                if command in ['images', 'full'] and images:
                    status.update("[bold green]Analyzing images...")
                    results['image_analysis'] = [analyzer.analyze_image(img) for img in images]
                    
                if command in ['structure', 'full']:
                    status.update("[bold green]Analyzing document structure...")
                    results['structure'] = analyzer.analyze_document_structure(text)
                    
                if command in ['tables', 'full']:
                    status.update("[bold green]Extracting tables...")
                    results['tables'] = analyzer.extract_tables(text)
                    
                if command in ['code', 'full']:
                    status.update("[bold green]Analyzing code snippets...")
                    results['code_blocks'] = analyzer.extract_code_snippets(text)
                    
                if command in ['topics', 'full']:
                    status.update("[bold green]Extracting topics...")
                    results['topics'] = analyzer.extract_key_topics(text)
                
                # Store results for saving later
                process_command.last_results = results
                
                # Display results
                console.print("\n[bold green]Analysis complete![/bold green]")
                for key, value in results.items():
                    console.print(f"\n[bold cyan]=== {key.replace('_', ' ').title()} ===[/bold cyan]")
                    if isinstance(value, dict):
                        console.print(json.dumps(value, indent=2))
                    else:
                        console.print(value)
                        
        except Exception as e:
            console.print(f"[red]Error during analysis: {str(e)}[/red]")
            
    elif command == 'save':
        if not hasattr(process_command, 'last_results'):
            console.print("[red]No analysis results to save. Please run an analysis first.[/red]")
            return True
            
        if not hasattr(display_menu, 'current_file'):
            console.print("[red]No file information available.[/red]")
            return True
            
        save_results(process_command.last_results, display_menu.current_file)
        
    else:
        console.print("[red]Unknown command. Type 'help' to see available commands.[/red]")
        
    return True

@app.command()
def main():
    """Interactive PDF analysis tool powered by Google Gemini."""
    clear_screen()
    console.print("[bold cyan]Welcome to PDF Analyzer![/bold cyan]")
    console.print("[yellow]Type 'help' to see available commands or 'exit' to quit.[/yellow]\n")
    
    analyzer = PDFAnalyzer()
    running = True
    
    while running:
        command = input("\nEnter command > ").strip()
        running = process_command(command, analyzer)
        
    console.print("\n[bold cyan]Thank you for using PDF Analyzer![/bold cyan]")

if __name__ == "__main__":
    app() 