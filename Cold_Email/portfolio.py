import pandas as pd
import chromadb
import uuid
import os

class Portfolio:
    def __init__(self, path='my_portfolio.csv'):
        # Correct the path to the CSV file
        self.path = path
        # Ensure the vectorstore directory exists
        if not os.path.exists('vectorstore'):
            os.makedirs('vectorstore')
        self.client = chromadb.PersistentClient('vectorstore')
        self.collection = self.client.get_or_create_collection(name="portfolio")

    def load_portfolio(self):
        if self.collection.count() == 0:
            try:
                df = pd.read_csv(self.path)
                for _, row in df.iterrows():
                    self.collection.add(
                        documents=row["Techstack"],
                        metadatas={"links": row["Links"]},
                        ids=[str(uuid.uuid4())]
                    )
            except FileNotFoundError:
                raise FileNotFoundError(f"The file {self.path} was not found. Please ensure it is in the correct directory.")

    def query_links(self, skills):
        # Add a check to ensure 'skills' is a list. The LLM can sometimes return other types.
        if not isinstance(skills, list):
            skills = [] # Treat non-list inputs as if no skills were found.

        if not skills:
            return []
            
        # Ensure all items in the list are strings before joining
        query_text = " ".join(map(str, skills))
        results = self.collection.query(query_texts=[query_text], n_results=2)
        return results.get('metadatas', [[]])[0]