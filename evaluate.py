import json

LOG_FILE = "logs.json"


def evaluate_system():

    with open(LOG_FILE, "r") as f:
        logs = json.load(f)

    total = len(logs)

    correct = 0
    total_time = 0

    for entry in logs:

        total_time += entry["response_time"]

        if entry["classification"] != "error":
            correct += 1

    accuracy = (correct / total) * 100
    avg_time = total_time / total

    print("Total Questions:", total)
    print("Accuracy:", round(accuracy, 2), "%")
    print("Average Response Time:", round(avg_time, 2), "seconds")


if __name__ == "__main__":
    evaluate_system()