AI-Resume-Builder

A. Project Title ResumeAI: Generative AI + Machine Learning Based Smart Resume Builder with ATS Scoring

B. Problem Statement: Students and job seekers often struggle to create professional, ATS-optimized resumes that can pass automated screening systems used by companies. Many resumes fail to highlight skills effectively or lack proper structure, resulting in fewer interview opportunities. This project solves that problem by building an AI-powered resume generator that automatically creates structured, professional resumes and evaluates them using machine learning models.

C. Features:

AI-generated professional resume using OpenAI GPT (GPT-4o-mini)
ATS Score prediction using sklearn MLP neural network
Resume strength classification using PyTorch neural network
Automatic PDF resume generation using FPDF
Clean and modern Streamlit web UI
Feature extraction from user input (skills, projects, education, etc.)
Instant resume generation (< 30 seconds)
One-click PDF download
Test mode support (no API required)
D. Tech Stack: Frontend / UI Streamlit Backend / AI Models OpenAI GPT (gpt-4o-mini) Scikit-learn (MLPRegressor neural network) PyTorch (Neural network classifier)

E. Libraries NumPy FPDF python-dotenv os, textwrap

F. How It Works User enters personal details (name, skills, education, projects) System extracts numerical features from input sklearn MLP model predicts ATS score PyTorch model classifies resume strength (Weak / Average / Strong) GPT generates a professional resume in structured format FPDF converts generated resume into downloadable PDF

G. How to Run the Project

Clone the repository git clone https://github.com/your-username/ResumeAI.git cd ResumeAI
Install dependencies pip install -r requirements.txt
Add API key (IMPORTANT)
Create a .env file:

OPENAI_API_KEY=your_api_key_here

OR enter API key in Streamlit sidebar.

Run the application streamlit run app.py
H. Output Preview ATS Score displayed after generation Resume Strength classification Downloadable professional PDF resume
