import os
import json
import matplotlib.pyplot as plt

from constants import *

folder = "results"

seen_attacks = []
seen_defenses = []
results = {}
for file in os.listdir("results"):
    path = folder+"/"+file+"/info.json"
    with open(path) as f:
        j = json.load(f)
        assert 'Backdoor' not in j['attacker']
        attack = j['attacker'] + "(" + str(list(eval(j['attack_param']).values()))[1:-1].replace('.0,',',') + ")"
        defense = j['defence'] + "(" + str(list(eval(j['defence_param']).values()))[1:-1] + ")"
        
        if attack not in seen_attacks:
            seen_attacks.append(attack)
        if defense not in seen_defenses:
            seen_defenses.append(defense)
        
        assert (attack, defense) not in results
        results[(attack, defense)] = j

print('Attacks:', seen_attacks)
print('Defenses:', seen_defenses)

print("Score matrix:")
for attack in seen_attacks:
    for defense in seen_defenses:
        if (attack, defense) in results:
            p = results[(attack, defense)]['score']
            print(f"{p:.4f}", end='\t')
        else:
            print("x", end='\t')
    print()

def plot(attack, defense, label_attack=True, label_defense=False, label_series=False, series='test_loss', alpha=0.5):
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
#plot_all_attacks('FLTrust()')

# Plot test loss for each defense without attack
plot_all_defenses('HonestClient()')

plt.legend()
plt.savefig(f"loss_graph.png")
plt.clf()

print()
