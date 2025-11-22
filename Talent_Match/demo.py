#!/usr/bin/env python3
"""
Multi-Agent Job Screening System - Demo
=======================================

This is a simplified demo of the multi-agent job screening system.
It simulates the functionality without requiring all dependencies.

Run with: python demo.py
"""

import os
import sys
import time
import random
import json
import csv
from datetime import datetime
from typing import Dict, List, Any, Tuple

# Demo configuration
CSV_FILE = "job_description.csv"
RESUMES_DIR = "resumes"
NUM_RESUMES_TO_PROCESS = 5  # Limit to 5 to speed up demo

class DemoJDSummarizer:
    """Demo version of JD Summarizer Agent"""
    
    def process_all_jds(self) -> List[Dict[str, Any]]:
        """Process job descriptions from CSV"""
        print("🤖 JD Summarizer Agent processing job descriptions...")
        job_descriptions = []
        
        try:
            # Try different encodings
            encodings = ['utf-8', 'latin-1', 'cp1252']
            success = False
            for encoding in encodings:
                try:
                    with open(CSV_FILE, 'r', encoding=encoding) as file:
                        reader = csv.DictReader(file)
                        for i, row in enumerate(reader):
                            if i >= 5:  # Limit to first 5 JDs
                                break
                            
                            if 'Job Title' in row and 'Job Description' in row:
                                title = row['Job Title'].strip()
                                print(f"  ✓ Processing JD: {title}")
                                
                                # Simulate structured data extraction
                                summary = {
                                    'required_skills': ['Python', 'Database', 'Communication', 'Problem-solving'],
                                    'years_of_experience': '3-5 years',
                                    'education': "Bachelor's degree",
                                    'certifications': [],
                                    'responsibilities': [
                                        'Develop software applications',
                                        'Collaborate with team members',
                                        'Debug issues'
                                    ],
                                    'raw_jd': row['Job Description']
                                }
                                
                                job_descriptions.append({
                                    'title': title,
                                    'summary': summary
                                })
                    
                    # If we get here without error, break the loop
                    print(f"  ✓ Successfully read CSV with {encoding} encoding")
                    success = True
                    break
                except UnicodeDecodeError:
                    # Try next encoding
                    print(f"  ✗ Failed to read CSV with {encoding} encoding, trying next encoding")
                    continue
            
            # If no JDs were found, create some mock ones
            if not success or not job_descriptions:
                print("  ✓ Using mock job descriptions")
                job_titles = [
                    "Software Engineer",
                    "Data Scientist",
                    "Product Manager",
                    "Cloud Engineer",
                    "Machine Learning Engineer"
                ]
                
                for title in job_titles:
                    print(f"  ✓ Creating mock JD: {title}")
                    summary = {
                        'required_skills': ['Python', 'Database', 'Communication', 'Problem-solving'],
                        'years_of_experience': '3-5 years',
                        'education': "Bachelor's degree",
                        'certifications': [],
                        'responsibilities': [
                            'Develop software applications',
                            'Collaborate with team members',
                            'Debug issues'
                        ],
                        'raw_jd': f"This is a mock job description for {title}"
                    }
                    
                    job_descriptions.append({
                        'title': title,
                        'summary': summary
                    })
        except Exception as e:
            print(f"  ✗ Error loading job descriptions: {str(e)}")
            # Create mock data as fallback
            job_descriptions = [
                {
                    'title': 'Software Engineer',
                    'summary': {
                        'required_skills': ['Python', 'Database', 'Communication', 'Problem-solving'],
                        'years_of_experience': '3-5 years',
                        'education': "Bachelor's degree",
                        'certifications': [],
                        'responsibilities': [
                            'Develop software applications',
                            'Collaborate with team members',
                            'Debug issues'
                        ],
                        'raw_jd': 'This is a mock job description'
                    }
                }
            ]
        
        time.sleep(1)  # Simulate processing time
        return job_descriptions

class DemoCVExtractor:
    """Demo version of CV Extractor Agent"""
    
    def process_all_resumes(self) -> List[Dict[str, Any]]:
        """Process resumes from directory"""
        print("🤖 CV Extractor Agent processing resumes...")
        resume_data = []
        
        try:
            # Get list of PDF files
            resume_files = [f for f in os.listdir(RESUMES_DIR) if f.lower().endswith('.pdf')]
            
            # Limit to a few resumes for the demo
            resume_files = resume_files[:NUM_RESUMES_TO_PROCESS]
            
            for filename in resume_files:
                print(f"  ✓ Processing CV: {filename}")
                
                # Simulate CV data extraction
                cv_data = {
                    'filename': filename,
                    'name': f"Candidate {filename.split('.')[0]}",
                    'email': f"candidate_{filename.split('.')[0].lower()}@example.com",
                    'phone': f"555-{random.randint(100, 999)}-{random.randint(1000, 9999)}",
                    'education': "Bachelor's in Computer Science",
                    'work_experience': "Software Engineer, Data Analyst",
                    'skills': "Python, Java, SQL, Machine Learning, Communication",
                    'certifications': "AWS Certified Developer",
                    'tech_stack': "Python, Django, React, PostgreSQL",
                    'raw_text': "This is simulated resume content."
                }
                
                resume_data.append(cv_data)
                time.sleep(0.2)  # Simulate processing time
                
        except Exception as e:
            print(f"  ✗ Error processing resumes: {str(e)}")
        
        return resume_data

class DemoMatcher:
    """Demo version of Matcher Agent"""
    
    def match_all_jds_with_all_cvs(self, jd_data_list: List[Dict[str, Any]], cv_data_list: List[Dict[str, Any]]) -> Dict[str, List[Tuple[Dict[str, Any], float]]]:
        """Match job descriptions with resumes"""
        print("🤖 Matcher Agent calculating match scores...")
        all_matches = {}
        
        for jd in jd_data_list:
            job_title = jd['title']
            print(f"  ✓ Matching for job: {job_title}")
            
            matches = []
            for cv in cv_data_list:
                # Simulate semantic matching with random scores weighted towards higher ones
                # More realistic than completely random scores
                base_score = random.uniform(60, 95)
                score = base_score + (5 if "Python" in cv.get('skills', '') else 0)
                score = min(score, 100)  # Cap at 100
                
                matches.append((cv, score))
            
            # Sort by score
            matches.sort(key=lambda x: x[1], reverse=True)
            all_matches[job_title] = matches
            time.sleep(0.5)  # Simulate processing time
        
        return all_matches

class DemoShortlister:
    """Demo version of Shortlister Agent"""
    
    def shortlist_candidates(self, matches: Dict[str, List[Tuple[Dict[str, Any], float]]]) -> Dict[str, List[Tuple[Dict[str, Any], float]]]:
        """Shortlist candidates based on threshold"""
        print("🤖 Shortlister Agent selecting candidates...")
        threshold = 80.0
        shortlisted = {}
        
        for job_title, job_matches in matches.items():
            # Filter by threshold
            job_shortlisted = [(cv, score) for cv, score in job_matches if score >= threshold]
            
            if job_shortlisted:
                shortlisted[job_title] = job_shortlisted
        
        return shortlisted
    
    def print_shortlist_summary(self, shortlisted: Dict[str, List[Tuple[Dict[str, Any], float]]]) -> None:
        """Print summary of shortlisted candidates"""
        print("\n===== SHORTLISTED CANDIDATES =====")
        
        for job_title, candidates in shortlisted.items():
            print(f"\nJob: {job_title}")
            print(f"Number of shortlisted candidates: {len(candidates)}")
            
            for i, (cv, score) in enumerate(candidates):
                print(f"  {i+1}. {cv.get('name', 'Unknown')} - Score: {score:.2f}%")
    
    def get_shortlist_data(self, shortlisted: Dict[str, List[Tuple[Dict[str, Any], float]]]) -> List[Dict[str, Any]]:
        """Convert shortlisted data to format for emailer"""
        shortlist_data = []
        
        for job_title, candidates in shortlisted.items():
            for cv, score in candidates:
                entry = {
                    'job_title': job_title,
                    'cv_filename': cv.get('filename', ''),
                    'name': cv.get('name', 'Unknown'),
                    'email': cv.get('email', ''),
                    'phone': cv.get('phone', ''),
                    'score': score,
                    'skills': cv.get('skills', '')
                }
                shortlist_data.append(entry)
        
        return shortlist_data

class DemoEmailer:
    """Demo version of Interview Scheduler Agent"""
    
    def send_interview_invitations(self, shortlisted_data: List[Dict[str, Any]], jd_data_by_title: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Simulate sending interview invitations"""
        print("🤖 Interview Scheduler Agent generating emails...")
        email_results = []
        
        for candidate in shortlisted_data:
            job_title = candidate.get('job_title', '')
            name = candidate.get('name', 'Candidate')
            email = candidate.get('email', '')
            
            # Generate email content (simplified)
            print(f"  ✓ Sending email to: {name} <{email}> for {job_title} position")
            
            # Sample of the email that would be generated
            print(f"""
    --- Email Preview ---
    To: {email}
    Subject: Interview Invitation for {job_title} Position
    
    Dear {name},
    
    We're pleased to invite you for an interview for the {job_title} position.
    
    Best regards,
    The Recruitment Team
    --- End of Email ---
            """)
            
            # Track email sending result
            result = {
                'candidate': candidate,
                'job_title': job_title,
                'email': email,
                'success': True,
                'timestamp': datetime.now().isoformat()
            }
            
            email_results.append(result)
            time.sleep(0.3)  # Simulate email sending time
        
        return email_results

def generate_demo_diagram():
    """Generate a simple agent interaction diagram"""
    print("🤖 Generating Agent Interaction Diagram...")
    
    # Generate Mermaid diagram
    mermaid_code = """
    graph TD
        A[JD Summarizer Agent] -->|Passes JD summaries| C[Matching Agent]
        B[CV Extractor Agent] -->|Passes parsed CVs| C
        C -->|Passes match scores| D[Shortlisting Agent]
        D -->|Passes shortlisted candidates| E[Interview Scheduler Agent]
        
        classDef agent fill:#f9f,stroke:#333,stroke-width:2px;
        class A,B,C,D,E agent;
    """
    
    # Save diagram
    with open("agent_diagram_demo.md", 'w') as f:
        f.write(f"# Agent Interaction Diagram\n\n```mermaid\n{mermaid_code}\n```")
    
    print(f"  ✓ Generated agent diagram: agent_diagram_demo.md")

def run_demo():
    """Run the entire demo pipeline"""
    print("🤖 Starting Multi-Agent Job Screening Demo 🤖")
    print("============================================")
    
    # Step 1: Parse and summarize job descriptions
    jd_agent = DemoJDSummarizer()
    jd_summaries = jd_agent.process_all_jds()
    
    # Step 2: Extract data from resumes
    cv_agent = DemoCVExtractor()
    cv_data_list = cv_agent.process_all_resumes()
    
    # Step 3: Match JDs with CVs
    matcher = DemoMatcher()
    all_matches = matcher.match_all_jds_with_all_cvs(jd_summaries, cv_data_list)
    
    # Step 4: Shortlist candidates
    shortlister = DemoShortlister()
    shortlisted = shortlister.shortlist_candidates(all_matches)
    shortlister.print_shortlist_summary(shortlisted)
    
    # Step 5: Send interview invitations
    if shortlisted:
        shortlist_data = shortlister.get_shortlist_data(shortlisted)
        jd_data_by_title = {jd['title']: jd for jd in jd_summaries}
        
        emailer = DemoEmailer()
        email_results = emailer.send_interview_invitations(shortlist_data, jd_data_by_title)
    
    # Step 6: Generate interaction diagram
    generate_demo_diagram()
    
    print("\n✅ Job screening demo completed successfully! ✅")
    print("\nTo run the full system with all features:")
    print("  1. Install dependencies: pip install -r requirements.txt")
    print("  2. Run: python main.py")
    
    # Inform about the Streamlit UI
    print("\n🌐 Try the user-friendly Streamlit UI:")
    print("  1. Install Streamlit: pip install streamlit pandas")
    print("  2. Run: streamlit run app.py")
    print("  3. Open your browser to the URL shown")

if __name__ == "__main__":
    run_demo() 