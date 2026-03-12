import json
import os
from datetime import datetime

LOG_FILE = "logs.json"


def log_interaction(question, classification, response, response_time):
    log_entry = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "question": question,
        "classification": classification,
        "response": response,
        "response_time": round(response_time, 2)
    }

    if os.path.exists(LOG_FILE):

        with open(LOG_FILE, "r") as f:
            data = json.load(f)

    else:
        data = []

    data.append(log_entry)

    with open(LOG_FILE, "w") as f:
        json.dump(data, f, indent=4)