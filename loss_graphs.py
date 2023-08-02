import os
import json
import matplotlib.pyplot as plt

from constants import *

folder = "matrix_results"

seen_attacks = []
seen_defenses = []
results = {}
for file in os.listdir(folder):
    if '.' in file: continue
    path = folder+"/"+file+"/info.json"
    with open(path) as f:
        j = json.load(f)
        assert 'Backdoor' not in j['attacker']
        attack = j['attacker'] + "(" + str(list(eval(j['attack_param']).values()))[1:-1].replace('.0,',',') + ")"
        defense = j['defence'] + "(" + str(list(eval(j['defence_param']).values()))[1:-1] + ")"
        
        attack = {"ExampleAttack()":"ExampleAttack", "HonestClient()":"HonestClient", "ShuffleAttacker(1)":"ShuffleAttack", "GAClient()":"GAAttack", "SimilarModel(1000000000, 1)":"SimilarModel"}[attack]
        defense = {"Krum(4, 1)":"Krum", "FedAvg()":"FedAvg", "FLTrust()":"FLTrust", "LFR(4)":"LFR", "Krum(4, 6)":"Multi-Krum", "LossDefense(4)":"LossDefense", "PCADefense(4)":"PCADefense", "FoolsGoldDefense()":"FoolsGold"}[defense]

        if attack == "GAAttack" and defense == "LFR":
            print(path)

        if attack not in seen_attacks:
            seen_attacks.append(attack)
        if defense not in seen_defenses:
            seen_defenses.append(defense)
        
        assert (attack, defense) not in results
        results[(attack, defense)] = j

seen_attacks.sort(key=lambda x:["HonestClient", "ExampleAttack", "SimilarModel", "ShuffleAttack", "GAAttack"].index(x))
seen_defenses.sort(key=lambda x:["FedAvg", "LFR", "LossDefense", "Krum", "Multi-Krum", "PCADefense", "FLTrust", "FoolsGold"].index(x))

print('Attacks:', seen_attacks)
print('Defenses:', seen_defenses)

print("Score matrix:")
for attack in seen_attacks:
    for defense in seen_defenses:
        if (attack, defense) in results:
            p = results[(attack, defense)]['score']
            print(f"{p:<10.4f}", end='\t')
        else:
            print(f"{'x':<10}", end='\t')
    print()

def plot(attack, defense, label_attack=True, label_defense=False, label_series=False, series='test_loss', alpha=0.5):
    if (attack, defense) not in results:
        print(f"Warning: {attack} vs. {defense} not in results, skipping.")
        return
    label = []
    if label_attack: label.append(attack)
    if label_defense: label.append(defense)
    if label_series: label.append({'train_loss': 'Train loss', 'test_loss': 'Test loss', 'backdoor_test_loss': 'Backdoor test loss'}[series])
    label = ' '.join(label)
    plt.plot(range(1, GLOBAL_ROUNDS+1), results[(attack,defense)][series], label=label, alpha=alpha)

def plot_all_test_losses():
    for attack in seen_attacks:
        for defense in seen_defenses:
            plot(attack, defense, label_defense=True)

def plot_all_attacks(defense):
    for attack in seen_attacks:
        plot(attack, defense, alpha=0.8)
    plt.title(defense)

def plot_all_defenses(attack):
    for defense in seen_defenses:
        plot(attack, defense, label_attack=False, label_defense=True, alpha=0.8)
    plt.title(attack)

# Plot test loss for each attack vs. FLTrust
#plot_all_attacks('FLTrust')

# Plot test loss for each defense without attack
plot_all_defenses('GAAttack')

#plot_all_test_losses()

plt.xlabel("Global round")
plt.ylabel("Test loss")

plt.legend()
plt.savefig(f"loss_graph.png")
plt.clf()

print()
