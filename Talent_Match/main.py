import os
import sys
import time
from typing import Dict, List, Any, Tuple
import argparse
from datetime import datetime

# Import agents
from agents.jd_summarizer import JDSummarizerAgent
from agents.cv_extractor import CVExtractorAgent
from agents.matcher import MatcherAgent
from agents.shortlister import ShortlisterAgent
from agents.emailer import EmailerAgent

# Import utilities
from utils.diagram import DiagramGenerator
from db.memory import MemoryDB

def run_pipeline(args):
    """Run the entire job screening pipeline"""
    print("🤖 Starting Multi-Agent Job Screening System 🤖")
    print("==============================================")
    
    # Initialize database
    print("\n📊 Initializing database...")
    db = MemoryDB(args.db_file)
    
    # Step 1: Parse and summarize job descriptions
    print("\n📝 Running JD Summarizer Agent...")
    jd_agent = JDSummarizerAgent(args.jd_file)
    jd_summaries = jd_agent.process_all_jds()
    
    # Store JD summaries in database
    jd_ids = {}
    for jd in jd_summaries:
        job_title = jd['title']
        jd_id = db.insert_jd_summary(job_title, jd['summary'])
        jd_ids[job_title] = jd_id
        print(f"  ✓ Processed and stored: {job_title}")
    
    # Step 2: Extract data from resumes
    print("\n📄 Running CV Extractor Agent...")
    cv_agent = CVExtractorAgent(args.resumes_dir)
    cv_data_list = cv_agent.process_all_resumes()
    
    # Store CV data in database
    cv_ids = {}
    for cv_data in cv_data_list:
        filename = cv_data['filename']
        cv_id = db.insert_cv_data(filename, cv_data)
        cv_ids[filename] = cv_id
        print(f"  ✓ Processed and stored: {filename} ({cv_data['name']})")
    
    # Step 3: Match JDs with CVs
    print("\n🔍 Running Matcher Agent...")
    matcher = MatcherAgent()
    all_matches = matcher.match_all_jds_with_all_cvs(jd_summaries, cv_data_list)
    
    # Store match scores in database
    match_ids = {}
    for job_title, matches in all_matches.items():
        jd_id = jd_ids[job_title]
        print(f"  ✓ Generated {len(matches)} matches for: {job_title}")
        
        for cv_data, score in matches:
            filename = cv_data['filename']
            cv_id = cv_ids[filename]
            match_id = db.insert_match_score(jd_id, cv_id, score)
            match_key = f"{job_title}_{filename}"
            match_ids[match_key] = match_id
    
    # Step 4: Shortlist candidates
    print("\n👑 Running Shortlister Agent...")
    shortlister = ShortlisterAgent(threshold=args.threshold)
    shortlisted = shortlister.shortlist_candidates(all_matches)
    shortlister.print_shortlist_summary(shortlisted)
    
    # Store shortlisted candidates in database
    for job_title, candidates in shortlisted.items():
        jd_id = jd_ids[job_title]
        
        for cv_data, score in candidates:
            filename = cv_data['filename']
            cv_id = cv_ids[filename]
            match_key = f"{job_title}_{filename}"
            match_id = match_ids[match_key]
            
            db.insert_shortlisted(match_id, jd_id, cv_id, score)
            print(f"  ✓ Shortlisted: {cv_data['name']} for {job_title} (Score: {score:.2f}%)")
    
    # Step 5: Send interview invitations
    if args.send_emails:
        print("\n📧 Running Interview Scheduler Agent...")
        emailer = EmailerAgent(simulate=True)  # Use True for simulation, False to actually send emails
        
        # Get shortlisted data in format for emailer
        shortlist_data = shortlister.get_shortlist_data(shortlisted)
        
        # Create a mapping of job titles to JD data
        jd_data_by_title = {jd['title']: jd for jd in jd_summaries}
        
        # Send emails
        email_results = emailer.send_interview_invitations(shortlist_data, jd_data_by_title)
        emailer.print_email_summary(email_results)
        
        # Update database with email status
        current_time = datetime.now().isoformat()
        for result in email_results:
            if result['success']:
                # In a real system, we would look up the shortlist ID properly
                # For this demo, we're just noting that emails were sent
                print(f"  ✓ Recorded email sent to: {result['email']}")
    
    # Step 6: Generate interaction diagram
    print("\n📊 Generating Agent Interaction Diagram...")
    
    if args.diagram_type == 'mermaid':
        diagram_file = DiagramGenerator.save_mermaid_diagram("agent_diagram.md")
        print(f"  ✓ Generated Mermaid diagram: {diagram_file}")
    else:
        try:
            diagram_file = DiagramGenerator.generate_matplotlib_diagram("agent_diagram.png")
            print(f"  ✓ Generated Matplotlib diagram: {diagram_file}")
        except Exception as e:
            print(f"  ✗ Error generating Matplotlib diagram: {str(e)}")
            # Fallback to Mermaid
            diagram_file = DiagramGenerator.save_mermaid_diagram("agent_diagram.md")
            print(f"  ✓ Generated Mermaid diagram instead: {diagram_file}")
    
    print("\n✅ Job screening pipeline completed successfully! ✅")
    
    # Close database connection
    db.close()

def parse_arguments():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description='Multi-Agent Job Screening System')
    
    parser.add_argument('--jd-file', type=str, default='job_description.csv',
                        help='Path to job descriptions CSV file')
    
    parser.add_argument('--resumes-dir', type=str, default='resumes',
                        help='Directory containing resume PDFs')
    
    parser.add_argument('--db-file', type=str, default='memory.db',
                        help='SQLite database file path')
    
    parser.add_argument('--threshold', type=float, default=80.0,
                        help='Minimum score threshold for shortlisting (0-100)')
    
    parser.add_argument('--send-emails', action='store_true',
                        help='Send interview invitation emails')
    
    parser.add_argument('--diagram-type', type=str, choices=['mermaid', 'matplotlib'], 
                        default='mermaid',
                        help='Type of agent interaction diagram to generate')
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    run_pipeline(args) 