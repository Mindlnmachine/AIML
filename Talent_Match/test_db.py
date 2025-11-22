#!/usr/bin/env python3
"""
Test script for thread-safety of MemoryDB class
"""

import threading
import time
from db.memory import MemoryDB

def test_thread(n):
    """Test function to run in each thread"""
    db = MemoryDB('test_memory.db')
    data = {
        'required_skills': [f'Skill-{n}'],
        'certifications': [],
        'responsibilities': []
    }
    job_id = db.insert_jd_summary(f'Test Job {n}', data)
    print(f'Thread {n} inserted job with ID: {job_id}')
    time.sleep(0.5)
    db.close()

# Create and start 5 threads
threads = []
for i in range(5):
    t = threading.Thread(target=test_thread, args=(i,))
    threads.append(t)
    t.start()

# Wait for all threads to complete
for t in threads:
    t.join()

# Check all jobs
db = MemoryDB('test_memory.db')
all_jobs = db.get_all_jds()
print('All jobs:')
for job in all_jobs:
    print(f"  {job['id']}: {job['job_title']} - Skills: {job['required_skills']}")
db.close()

print("Test completed successfully!") 