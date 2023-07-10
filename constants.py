import torch

GLOBAL_ROUNDS = 30
EPOCHS_PER_ROUND = 3
CLIENTS = 40
SELECT_CLIENTS = 10
LR = 0.001
TARGET_DISTANCES = [5, 10, 15, 20, 25, 30, 35, 40, 50, 60, 70, 80, 95, 110, 125, 145, 165]
N_ATTACKERS = 2

if torch.cuda.is_available():
    device = 'cuda'
else:
    print("USING CPU!!!")
    device = 'cpu'
