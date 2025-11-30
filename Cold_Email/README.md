# 📧 Cold Email Generator

An intelligent AI-powered application that scrapes job listings, extracts job details, matches them with your portfolio skills using semantic search, and generates personalized cold emails for business development outreach.

## 🎯 What It Does

This application automates the entire process of identifying relevant job opportunities and creating targeted outreach emails. It combines web scraping, natural language processing, and vector embeddings to connect job requirements with your portfolio expertise and automatically generate compelling cold emails.

## 📋 Key Features

- **Web Scraping**: Automatically scrapes job listings from career websites
- **Job Extraction**: Uses AI to parse job details into structured format (role, experience, skills, description)
- **Portfolio Matching**: Intelligently matches job requirements with your portfolio projects using semantic search
- **Email Generation**: Creates personalized cold emails highlighting relevant experience and portfolio links
- **Vector Database**: Leverages ChromaDB for fast semantic similarity search across portfolio items

## 🛠️ Technology Stack

- **Streamlit**: Web application framework for the user interface
- **LangChain**: LLM orchestration and prompt management
- **Groq API**: Fast language model inference for content generation
- **ChromaDB**: Vector database for embedding storage and semantic search
- **BeautifulSoup4**: Web scraping and HTML parsing
- **Pandas**: Data processing and portfolio management

## 📁 Project Files

**main.py**  
The main entry point for the Streamlit application. Orchestrates the entire workflow including user input handling, web scraping, job extraction, portfolio matching, and email generation display.

**chains.py**  
Contains the core LLM chain logic. Manages two critical operations: extracting structured job data from scraped content and generating personalized cold emails using the Groq API.

**portfolio.py**  
Manages portfolio data and vector database operations. Loads portfolio information from CSV files, stores embeddings in ChromaDB, and performs semantic similarity searches to find matching projects based on job skills.

**utils.py**  
Provides text cleaning and preprocessing utilities. Processes scraped HTML content by removing tags, URLs, special characters, and normalizing whitespace to prepare clean data for job extraction.

**email.py**  
An alternative Streamlit interface specifically designed for generating cold emails for direct job applications. Includes interactive forms for applicant details, company information, and email customization.

## 🔄 How It Works

**Step 1: Web Scraping**  
User provides a career page URL. The application loads and extracts all text content from the webpage.

**Step 2: Text Cleaning**  
Raw HTML is cleaned by removing tags, URLs, and special characters to create structured text suitable for analysis.

**Step 3: Job Extraction**  
The LLM analyzes the cleaned text and extracts job postings, converting them into structured JSON format with standardized fields.

**Step 4: Portfolio Loading**  
Portfolio data is loaded from CSV and converted into vector embeddings for semantic search capability.

**Step 5: Skill Matching**  
For each job found, required skills are extracted and matched against the portfolio database using semantic similarity.

**Step 6: Email Generation**  
The LLM generates a personalized cold email for each job, incorporating matched portfolio links and relevant experience.

**Step 7: Display & Export**  
Generated emails are displayed in the application and can be copied or downloaded for use.

## 📊 Data Files

**my_portfolio.csv**  
Your portfolio file should contain two columns:
- Techstack: Technologies and skills used
- Links: URL to the portfolio/project demonstration

This file is used to populate the vector database for semantic matching with job requirements.

## 🎓 Use Cases

**Business Development**: Identify companies posting jobs and reach out with relevant solutions and services

**Sales Outreach**: Generate targeted emails to hiring managers based on their specific job requirements

**Consulting Services**: Match your service offerings to company needs identified in their job postings

**Recruitment Marketing**: Create personalized emails for candidates based on job requirements

**Lead Generation**: Automatically identify and reach out to relevant prospects through job postings

## 🚀 Getting Started

1. Set up your Groq API key for LLM access
2. Prepare your portfolio CSV file with your projects and skills
3. Run the Streamlit application
4. Input a career page URL
5. View extracted jobs and automatically generated cold emails



## 🤝 Support

For questions or issues, please open a GitHub issue.

