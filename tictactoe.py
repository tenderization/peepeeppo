import time
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

def mlp_model(d=64):
    class MLPModel(torch.nn.Module):
        def __init__(self):
            super(MLPModel, self).__init__()
            self.trunk = torch.nn.Sequential(
                torch.nn.Linear(9, d),
                torch.nn.ReLU(),
                torch.nn.Linear(d, d),
                torch.nn.ReLU(),
                )
            # redundant
            self.policy_head = torch.nn.Linear(d, 9)
            self.value_head = torch.nn.Linear(d, 1)

        def forward(self, x):
            x = self.trunk(x)
            return self.policy_head(x), self.value_head(x)
    m = MLPModel()
    return m


def rollout(m):
    firststates = list()
    firstvalues = list()
    secondvalues = list()
    secondstates = list()
    firstactionprobs = list()
    secondactionprobs = list()
    firstactions = list()
    secondactions = list()
    first = True
    with torch.no_grad():
        state = torch.zeros(1, 9, dtype=torch.float)
        while(not util.full(state) and util.longest_of(1, state.reshape(3, 3)) < 3):
            # switch player
            state = state * -1.0
            valid = state == 0.0
            logits, value = m(state)
            masked = torch.full_like(logits, -torch.inf)
            masked = torch.where(valid, logits, masked)
            probs = torch.nn.functional.softmax(masked, -1)
            action = torch.searchsorted(probs.cumsum(-1).squeeze(), torch.rand(1))
            assert(not(state[0, action]))
            if first:
                firststates.append(state.detach().clone())
                firstvalues.append(value.detach().clone())
                firstactionprobs.append(probs[:, action])
                firstactions.append(torch.tensor([[action]], dtype=torch.int))
            else:
                secondstates.append(state.detach().clone())
                secondvalues.append(value.detach().clone())
                secondactionprobs.append(probs[:, action])
                secondactions.append(torch.tensor([[action]], dtype=torch.int))
            state[0, action] = 1.0
            first = not first
        if util.longest_of(1, state.reshape(3, 3)) == 3:
            targfirstvalue = 1.0 if not first else -1.0
        else:
            targfirstvalue = 0.0

    states = torch.cat(firststates + secondstates)
    values = torch.cat(firstvalues + secondvalues)
    actionprobs = torch.cat(firstactionprobs + secondactionprobs)
    actions = torch.cat(firstactions + secondactions)
    targvalues = torch.full_like(values, targfirstvalue)
    targvalues[len(firstvalues):] = -targfirstvalue
    assert(states.size(0) == values.size(0))
    assert(targvalues.size(0) == values.size(0))
    return states, actionprobs, actions, values, targvalues 


def lossfunc(probs, actions, oldactionprobs, advantages, values, targvalues, entropy_decay):
    epsilon = 0.2
    c1 = 0.5
    c2 = 0.02*entropy_decay

    r = torch.gather(probs, -1, actions)/oldactionprobs
    #clip_fraction = (torch.clip(r, 1 - epsilon, 1 + epsilon) != r).float().mean()
    #print("CLIP FRACTION", clip_fraction)

    lclip = torch.min(r * advantages,
                      torch.clip(r, 1 - epsilon, 1 + epsilon) * advantages).mean()
    lvf = torch.nn.functional.mse_loss(values, targvalues)
    dist = torch.distributions.categorical.Categorical(probs=probs)
    lentropy = dist.entropy().mean()
    return -lclip + c1*lvf - c2*lentropy
    #return -lclip
    #return c1 * lvf
    #return -c2*lentropy


def train():
    m = mlp_model()
    optim = torch.optim.Adam(m.parameters(), lr=5e-3, weight_decay=1e-4)
    target_batch_size = 16384
    iters = 1000
    epochs = 5
    for i in range(iters):
        statebatch = list()
        actionprobbatch = list()
        actionbatch = list()
        oldvaluebatch = list()
        targvaluebatch = list()
        total_batch_size = 0
        t0 = time.time() 
        while total_batch_size < target_batch_size:
            currstates, curractionprobs, curractions, currvalues, currtargvalues = rollout(m)
            statebatch.append(currstates)
            actionprobbatch.append(curractionprobs)
            actionbatch.append(curractions)
            oldvaluebatch.append(currvalues)
            targvaluebatch.append(currtargvalues)
            total_batch_size += len(currvalues)
        print("rollout took: ", time.time() - t0)
        states = torch.cat(statebatch)
        oldactionprobs = torch.cat(actionprobbatch)
        actions = torch.cat(actionbatch)
        oldvalues = torch.cat(oldvaluebatch)
        targvalues = torch.cat(targvaluebatch)
        advantages = targvalues - oldvalues
        print(targvalues.abs().mean())
        print("cat's game frac:", 1 - targvalues.abs().mean())
        for k in range(epochs):
            optim.zero_grad()
            newlogits, newvalues = m(states)
            masked = torch.full_like(newlogits, -torch.inf)
            valid = states == 0.0
            masked = torch.where(valid, newlogits, masked)
            newprobs = torch.nn.functional.softmax(masked, -1)
            if k == 0:
                torch.testing.assert_close(torch.gather(newprobs, -1, actions), oldactionprobs)
            decay = 1.0 if i == 0 else max(0.0, (1.0 -  ((i+1) * epochs + k) / (iters * epochs)))
            loss = lossfunc(newprobs, actions, oldactionprobs, advantages, newvalues, targvalues, decay)
            if k % 100 == 0:
                print(states[-1].reshape(3, 3))
                print(decay, loss)
            loss.backward()
            optim.step()
        #print(actionprobs, actions, targvalues, advantages)
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', action='store_true')
    parser.add_argument('-t', action='store_true')
    args = parser.parse_args()
    if args.r:
        random_game()
    if args.t:
        train()
