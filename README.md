# PDF Analyzer

⚠️ **Note: This is a Work in Progress (WIP).** Errors may occur, and features are subject to change. Use at your own discretion.

**PDF Analyzer** is an interactive command-line tool that leverages Google Gemini's generative AI to extract and analyze content from PDF documents. The tool supports extracting text and images, summarizing content, performing deep analysis (including code snippet extraction, table extraction, document structure analysis, and more), and even providing real-time progress updates—all in a single-file script.

## Features

*   **Text & Image Extraction:** Uses PyMuPDF and Pillow to extract text and images from PDFs.
*   **Multi-Mode Analysis:** Offers both "quick" and "standard" analysis modes:
    *   **Quick Mode:** Uses Gemini 1.5 Flash for fast, concise analysis.
    *   **Standard Mode:** Utilizes more advanced Gemini 2.0 models for deeper insights and detailed analysis.
*   **Advanced Capabilities:**
    *   Summarization and detailed analysis
    *   Structured data extraction (JSON output)
    *   Code snippet extraction and analysis
    *   Table extraction and analysis
    *   Document structure and key topics analysis
*   **Interactive CLI:** Easily navigate through commands such as `upload`, `summary`, `analyze`, `images`, `structure`, `tables`, `code`, `topics`, `full`, `save`, `clear`, `help`, and `exit`.
*   **Progress Tracking:** Provides real-time feedback during long-running analysis tasks.

## Prerequisites

*   Python 3.7+
*   Google Gemini API Key:
    *   Create a `.env` file in the project root with the following content:

    ```ini
    GEMINI_API_KEY=your_gemini_api_key_here
    ```

## Installation

1.  **Clone the Repository:**

    ```bash
    git clone [https://github.com/yourusername/pdf-analyzer.git](https://www.google.com/search?q=https://github.com/yourusername/pdf-analyzer.git)
    cd pdf-analyzer
    ```

2.  **Install Dependencies:**

    You can install the required packages using pip. If you have a `requirements.txt`, run:

    ```bash
    pip install -r requirements.txt
    ```

    Otherwise, install the dependencies manually:

    ```bash
    pip install python-dotenv PyPDF2 google-generativeai PyMuPDF Pillow typer rich
    ```

## Usage

1.  **Run the Script:**

    ```bash
    python script.py
    ```

    (Replace `script.py` with the actual filename if different.)

2.  **Interactive Commands:**

    Once running, you can use the following commands:

    *   `upload`: Upload a PDF file by providing its path.
    *   `summary`: Generate a concise summary of the document.
    *   `analyze`: Run a detailed analysis of the document.
    *   `images`: Analyze images and diagrams extracted from the PDF.
    *   `structure`: Analyze the document's structure and organization.
    *   `tables`: Extract and analyze tables.
    *   `code`: Extract and analyze code snippets.
    *   `topics`: Extract key topics and clusters.
    *   `full`: Run a full analysis pipeline (combining multiple features).
    *   `save`: Save the analysis results to a file.
    *   `clear`: Clear the terminal screen.
    *   `help`: Display the command menu.
    *   `exit`: Exit the program.

## Example Workflow

1.  **Upload a PDF:**

    Type `upload` and then provide the path to your PDF file.

2.  **Select Analysis Type:**

    Enter one of the analysis commands (e.g., `full` for a complete analysis). You will be prompted to choose between **Standard** (detailed) or **Quick** (fast) mode.

3.  **View Results:**

    The results are displayed in the terminal, organized by sections (e.g., summary, analysis, structure).

4.  **Save Results:**

    If you wish to save your analysis, type `save` and follow the prompt to specify the filename.

## Troubleshooting

*   **Missing `GEMINI_API_KEY`:**

    Ensure you have a valid API key in your `.env` file.

*   **File Loading Errors:**

    Double-check the file path you provide when uploading a PDF.

*   **API or Analysis Errors:**

    Verify your internet connection and the validity of your API key if you encounter issues during analysis.

## Future Roadmap

Planned features and improvements:
- **Enhanced OCR Support**: Integrate Tesseract OCR for scanned PDFs.
- **Improved Table Extraction**: Better handling of complex tables.
- **Multilingual Support**: Expand support for non-English documents.
- **Interactive Web Interface**: Develop a front-end for easier access.
- **More AI Models**: Experiment with alternative AI models for analysis.
- **Cloud Integration**: Allow processing PDFs from cloud storage (Google Drive, Dropbox, etc.).

## Acknowledgements

*   Google Generative AI
*   PyMuPDF (fitz)
*   Typer
*   Rich
*   Other open-source libraries and contributors.
