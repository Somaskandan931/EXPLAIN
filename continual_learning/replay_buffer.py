import json
from datetime import datetime

BUFFER_PATH = "continual_learning/replay_buffer.json"

def add_sample(text, label):
    sample = {
        "text": text,
        "label": label,
        "timestamp": str(datetime.now())
    }

    try:
        data = json.load(open(BUFFER_PATH))
    except:
        data = []

    data.append(sample)
    json.dump(data, open(BUFFER_PATH, "w"), indent=2)
