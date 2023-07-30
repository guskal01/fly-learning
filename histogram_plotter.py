import os
import json
import matplotlib.pyplot as plt

scores = []
for file in os.listdir("results"):
    path = "results/"+file+"/info.json"
    with open(path) as f:
        j = json.load(f)
        assert j['attacker'] == "HonestClient"
        assert j['defence'] == "FedAvg"
        scores.append(j['score'])

print(len(scores), "scores")

histfile = 'histogram.png'
plt.hist(scores, bins=10)
plt.savefig(histfile)
print("Saved to", histfile)
