# JobMatchr

JobMatchr is a web application that leverages AI to analyze resumes against job descriptions. It provides detailed insights, including resume evaluation, missing keywords, and match percentage, to help job seekers optimize their resumes for better job matching.

![JobMatchr Screenshot](/templates/image1.png)

## Features

- **Resume Evaluation**: Get a professional evaluation of your resume against the job description.
- **Missing Keywords**: Identify essential keywords missing from your resume.
- **Match Percentage**: Calculate the dynamic match percentage based on skills, experience, and keywords.
- **Instant Analysis**: Receive immediate feedback with detailed insights.
- **Keyword Optimization**: Discover missing keywords to improve resume visibility.
- **Career Insights**: Personalized suggestions to enhance your resume for specific job roles.

![JobMatchr Analysis](/templates/image2.png)

## Technology Stack

- **Backend**: Flask
- **Frontend**: HTML, CSS, JavaScript
- **AI Models**: Gemini
- **Database**: Chroma
- **PDF Processing**: pdfplumber
- **Environment Management**: Python-dotenv

## Installation

1. **Clone the repository**:
    ```bash
    git clone https://github.com/Ashish-kharde1/JobMatchr.git
    cd JobMatchr
    ```

2. **Create a virtual environment**:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

4. **Set up environment variables**:
    Create a `.env` file in the project root and add your API keys:
    ```env
    GOOGLE_API_KEY=your_google_api_key
    ```

## Usage

1. **Run the application**:
    ```bash
    python main.py
    ```

2. **Open your browser** and navigate to `http://127.0.0.1:5000/`.

3. **Upload your resume** and paste the job description to get insights.

## Project Structure

- `main.py`: The main Flask application file.
- `templates/index.html`: The HTML template for the web interface.

## Contributing

Contributions are welcome! Please fork the repository and create a pull request with your changes.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
