# Multi-Agent AI System for Job Screening - Project Summary

This project implements a complete end-to-end multi-agent AI system for enhancing job screening, built for a GenAI hackathon. The system automates the entire process from parsing job descriptions to sending interview invitations.

## System Architecture

The system consists of five specialized agents that work together in a pipeline:

1. **JD Summarizer Agent** (`agents/jd_summarizer.py`)
   - Parses job descriptions from CSV
   - Extracts structured data (skills, experience, education, etc.)
   - Uses Ollama embeddings with fallback to rule-based extraction

2. **CV Extractor Agent** (`agents/cv_extractor.py`)
   - Reads PDF resumes using PyMuPDF
   - Extracts key information like name, email, skills, etc.
   - Organizes resume data into structured format

3. **Matching Agent** (`agents/matcher.py`)
   - Uses `nomic-embed-text` model to create embeddings
   - Calculates cosine similarity between JDs and CVs
   - Generates match scores to rank candidates

4. **Shortlisting Agent** (`agents/shortlister.py`)
   - Filters candidates with scores ≥ 80%
   - Creates a shortlist for each job position
   - Prepares data for the email agent

5. **Interview Scheduler Agent** (`agents/emailer.py`)
   - Generates personalized invitation emails
   - Highlights matched skills from candidate's resume
   - Sends emails via SMTP (or simulates sending)

All data is persisted in a SQLite database (`memory.db`), allowing the system to maintain state across runs.

## Key Features

- **Semantic Matching**: Uses embeddings for intelligent candidate-job matching
- **Structured Data Extraction**: Parses unstructured text into usable data
- **Persistent Memory**: SQLite database stores all processing results
- **Data Visualization**: Generates an agent interaction diagram
- **Modular Architecture**: Agents communicate through well-defined interfaces

## Demo Version

A simplified demo version (`demo.py`) is available that doesn't require external dependencies. It simulates the full functionality of the system with mock data and processing, making it easy to showcase the concept.

## Technical Implementation

- **Language**: Python 3
- **Key Libraries**:
  - `pymupdf` for PDF parsing
  - `ollama` for embeddings
  - `scikit-learn` for vector similarity
  - `matplotlib` and `networkx` for visualization
  - `sqlite3` for database operations

## Next Steps

Future enhancements could include:
- Web interface for monitoring and management
- More sophisticated matching algorithms
- Integration with applicant tracking systems
- Dashboard for HR personnel
- Advanced NLP for deeper resume analysis

---

This multi-agent system demonstrates how AI can streamline the job screening process, improve candidate matching, and save time for HR professionals. The modular design allows for easy extensions and enhancements as needed. 