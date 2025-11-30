#cold eamil for job application usin streaamlit and chromadb
#model_name = "openai/gpt-oss-20b" 

import chromadb
import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
# import chromadb # Uncomment if you plan to use ChromaDB

# Initialize the language model
# Make sure to replace "YOUR_GROQ_API_KEY" with your actual Groq API key
# You can set this as an environment variable or use st.secrets for Streamlit Cloud
llm = ChatGroq(
    temperature=0.7,
    # groq_api_key=st.secrets.get("GROQ_API_KEY", "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"), # Placeholder for API key
    groq_api_key="your_groq_api_key_here",
    model_name="openai/gpt-oss-20b"
)

st.set_page_config(page_title="Cold Email Generator for Job Applications")
st.title("❄️ Cold Email Generator for Job Applications")
st.write("Generate a personalized cold email for your job application using AI.")

# Input fields for the user
with st.sidebar:
    st.header("Your Information")
    applicant_name = st.text_input("Your Name", "Vedant lonkar")
    applicant_email = st.text_input("Your Email", "vedantlonkar5555@gmail.com")
    applicant_skills = st.text_area("Your Key Skills/Experience (comma-separated)", "Data analysis, Machine Learning, Python, SQL, Cloud platforms (AWS/Azure)")

    st.header("Job and Company Details")
    recipient_name = st.text_input("Hiring Manager/Recipient Name (Optional)", "Hiring Team")
    company_name = st.text_input("Company Name", "Tech Innovations Inc.")
    job_title = st.text_input("Job Title You're Applying For", "Data Scientist")
    job_description_snippet = st.text_area("Key points from Job Description (Optional)", "Seeking a proactive data scientist to build predictive models and analyze large datasets. Experience with deep learning a plus.")
    
    st.header("Email Customization")
    call_to_action = st.text_input("Call to Action (e.g., 'schedule a brief call', 'connect on LinkedIn')", "schedule a brief virtual coffee chat")

# ChromaDB Integration Placeholder (Optional)
# If you wanted to store and retrieve job descriptions, past applications, or resume snippets,
# you could initialize and use ChromaDB here. For example:
client = chromadb.Client()
collection = client.get_or_create_collection(name="job_application_data")
if st.button("Store Job Description (Example)"):
    collection.add(documents=[job_description_snippet], metadatas=[{"company": company_name, "job_title": job_title}], ids=["job_desc_1"])
    st.success("Job description stored in ChromaDB!")


if st.button("Generate Cold Email"):
    if not applicant_name or not applicant_email or not company_name or not job_title:
        st.error("Please fill in at least your Name, Email, Company Name, and Job Title.")
    else:
        # Construct the prompt for the LLM
        prompt_template = ChatPromptTemplate.from_messages(
            [
                ("system", """You are an AI assistant specialized in writing professional and compelling cold emails for job applications.
                 Your goal is to craft a concise, impactful email that highlights the applicant's relevant skills and expresses genuine interest in the role and company.
                 Keep the tone professional and enthusiastic.
                 Include a clear call to action.
                 """),
                ("user", """
                 Write a cold email for a job application with the following details:
                 Applicant Name: {applicant_name}
                 Applicant Email: {applicant_email}
                 Applicant Skills/Experience: {applicant_skills}
                 Recipient Name: {recipient_name}
                 Company Name: {company_name}
                 Job Title: {job_title}
                 Key points from Job Description: {job_description_snippet}
                 Desired Call to Action: {call_to_action}

                 Structure the email as follows:
                 Subject: (Concise and attention-grabbing)
                 Body:
                 - Start with a polite greeting.
                 - Briefly introduce yourself and state the purpose (interest in {job_title} at {company_name}).
                 - Highlight 1-2 key skills/experiences from {applicant_skills} that directly relate to the {job_title} or {job_description_snippet}.
                 - Express your enthusiasm for {company_name} and briefly state why you are interested (e.g., company mission, innovation).
                 - Include the call to action: "I would appreciate the opportunity to {call_to_action} to discuss how my skills could benefit {company_name}."
                 - Conclude professionally.
                 - Signature.
                 """)
            ]
        )

        chain = prompt_template | llm

        with st.spinner("Generating your cold email..."):
            response = chain.invoke({
                "applicant_name": applicant_name,
                "applicant_email": applicant_email,
                "applicant_skills": applicant_skills,
                "recipient_name": recipient_name,
                "company_name": company_name,
                "job_title": job_title,
                "job_description_snippet": job_description_snippet,
                "call_to_action": call_to_action
            })
            
            st.subheader("Generated Cold Email:")
            st.markdown(f"```\n{response.content}\n```")
            st.success("Email generated successfully!")
            
            st.download_button(
                label="Download Email as .txt",
                data=response.content,
                file_name=f"cold_email_{company_name.replace(' ', '_')}_{job_title.replace(' ', '_')}.txt",
                mime="text/plain"
            )
