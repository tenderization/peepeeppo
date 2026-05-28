import argparse
import random
import util
import torch

def random_policy(marker, state):
    while True:
        idx = random.randint(0, 8)
        y = idx // 3
        x = idx % 3
        if not state[y][x]:
            state[y][x] = marker
            return state

def print_board(state):
    for y in range(len(state)):
        print(['X' if m == 1 else ('O' if m == -1 else ' ')  for m in state[y]])
    print()

def random_game():
    state = [[0, 0, 0],
             [0, 0, 0],
             [0, 0, 0]]
    while not util.full(state):
        state = random_policy( 1, state)
        print_board(state)
        if (util.longest_of(1, state) == 3):
            print("1 wins")
            return
        if util.full(state):
            print("cat's game")
            return
        state = random_policy(-1, state)
        print_board(state)
        if (util.longest_of(-1, state) == 3):
            print("-1 wins")
            return
    print("cat's game")

def mlp_model(d=256):
    class MLPModel(torch.nn.Module):
        def __init__(self):
            super(MLPModel, self).__init__()
            self.trunk = torch.nn.Sequential(
                torch.nn.Linear(9, d),
                torch.nn.ReLU(),
                torch.nn.Linear(d, d),
                torch.nn.ReLU(),)
            # redundant
            self.policy_head = torch.nn.Linear(d, 9)
            self.value_head = torch.nn.Linear(d, 1)

        def forward(self, x):
            x = self.trunk(x)
            return self.policy_head(x), self.value_head(x)
    m = MLPModel()
    return m

def rollout(m):
    states = list()
    with torch.no_grad():
        state = torch.zeros(1, 9, dtype=torch.float)
        while(not util.full(state) and util.longest_of(1, state.reshape(3, 3)) < 3):
            # switch player
            state = state * -1.0
            valid = state == 0.0
            logits, value_logit = m(state)
            masked = torch.full_like(logits, -torch.inf)
            masked = torch.where(valid, logits, masked)
            probs = torch.nn.functional.softmax(masked, -1)
            value = torch.nn.functional.tanh(value_logit)
            action = torch.searchsorted(probs.cumsum(-1).squeeze(), torch.rand(1))
            assert(not(state[0, action]))
            state[0, action] = 1.0
        if util.longest_of(1, state.reshape(3, 3)) == 3:
            print("won")
        else:
            print("cat's game")
        states.append(state.detatch.clone())
        print(state.reshape(3, 3))


def train():
    m = mlp_model()
    rollout(m)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', action='store_true')
    parser.add_argument('-t', action='store_true')
    args = parser.parse_args()
    if args.r:
        random_game()
    if args.t:
        train()
