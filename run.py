import numpy as np
import logging
import sys
import torch
import pickle
from tqdm import tqdm
from matplotlib import pyplot as plt

sys.path.append("../")

from rl_6_nimmt import Tournament, GameSession
from rl_6_nimmt.agents import Human, DrunkHamster, BatchedACERAgent, Noisy_D3QN_PRB_NStep, MCSAgent, PolicyMCSAgent, PUCTAgent
from rl_6_nimmt.agents import PUCTCustomedAgent

logging.basicConfig(format="%(message)s",level=logging.INFO)
for name in logging.root.manager.loggerDict:
    if not "rl_6_nimmt" in name:
        logging.getLogger(name).setLevel(logging.WARNING)

agents = {}
    
#agents[f"Random"] = DrunkHamster()
#agents[f"D3QN"] = Noisy_D3QN_PRB_NStep(history_length=int(1e5), n_steps=10)
agents[f"ACER"] = BatchedACERAgent(minibatch=10)
agents[f"MCS"] = MCSAgent(mc_max=200)
agents[f"Alpha0.5"] = PUCTAgent(mc_max=200)
agents[f"Alpha0.5_customed"] = PUCTCustomedAgent(mc_max=200)

for agent in agents.values():
    try:
        agent.train()
    except:
        pass

merle = Human("Merle")

tournament = Tournament(min_players=2, max_players=4)

for name, agent in agents.items():
    tournament.add_player(name, agent)

print(tournament)

##
#nagents, tournament = pickle.load(open("./.tournament.pickle", "rb"))
#print(tournament)


##Let the games begin! Stage 1: 5 x 400 games, 6 players, 200 MC steps##
num_games = 4000
block_len = 400

try:
    tqdm._instances.clear()  # Important after cancelling any step
except:
    pass

while tournament.total_games < num_games:
    for _ in tqdm(range(block_len)):
        tournament.play_game()
    print(tournament)
        
    #if tournament.total_games < num_games:
    #tournament.evolve(max_players=6, max_per_descendant=2, copies=(2,))    