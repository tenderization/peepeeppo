import time
import argparse
import random
import util
import model
import torch
import multiprocessing

#torch.set_num_threads(1)

pool = None

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

class MLPModel(torch.nn.Module):
    def __init__(self, d):
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

def mlp_model(d=64):
    m = MLPModel(d)
    return m

def conv_model():
    m = model.ConvNet(3, 3)
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
    #m.eval()
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
            sample = torch.clamp(torch.rand(1), min=1e-5)
            action = torch.searchsorted(probs.cumsum(-1).squeeze(), sample)
            #action = torch.searchsorted(probs.cumsum(-1).squeeze(), torch.rand(1) + 1e-6)
            #action = torch.clamp(action, max=8)
            #dist = torch.distributions.categorical.Categorical(probs=probs)
            #action = dist.sample()
            if int(action) >= 9:
                print(probs)
                print(probs.cumsum(-1))
                print(valid)
                print(sample)
                raise Exception
            if state[0, int(action)]:
                print(probs)
                print(probs.cumsum(-1))
                print(valid)
                print(sample)
                raise Exception
            if first:
                firststates.append(state.detach().clone())
                firstvalues.append(value.detach().clone())
                firstactionprobs.append(probs[:, action])
                firstactions.append(action)
            else:
                secondstates.append(state.detach().clone())
                secondvalues.append(value.detach().clone())
                secondactionprobs.append(probs[:, action])
                secondactions.append(action)
            state[0, action] = 1.0
            first = not first
        if util.longest_of(1, state.reshape(3, 3)) == 3:
            targfirstvalue = 1.0 if not first else -1.0
        else:
            targfirstvalue = 0.0

    states = torch.cat(firststates + secondstates)
    values = torch.cat(firstvalues + secondvalues)
    actionprobs = torch.cat(firstactionprobs + secondactionprobs)
    actions = torch.tensor(firstactions + secondactions).reshape(states.size(0), 1)
    targvalues = torch.full_like(values, targfirstvalue)
    targvalues[len(firstvalues):] = -targfirstvalue
    assert(states.size(0) == values.size(0))
    assert(targvalues.size(0) == values.size(0))
    return states, actionprobs, actions, values, targvalues 

def rollout_multi(inp):
    m, n = inp
    tensorlists = list()
    for i in range(n):
        tensorlists.append(rollout(m))
    return tensorlists

def parallel_rollout(m, p=4, n=100):
    global pool
    if pool is None:
        pool = multiprocessing.Pool(n)
    parallel_rollouts = pool.map(rollout_multi, [(m, n)]*p)
    joined = list()
    for tensorlists in parallel_rollouts:
        joined += tensorlists
    return [torch.cat(tensorlist) for tensorlist in zip(*joined)]

def lossfunc(probs, actions, oldactionprobs, advantages, values, targvalues, entropy_decay):
    epsilon = 0.2
    c1 = 0.5
    c2 = 0.1*entropy_decay

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
    minibatch_size = 16384
    target_batch_size = 16384
    iters = 1000
    epochs = 5
    total_iters = iters*epochs*(target_batch_size//minibatch_size)
    #m = mlp_model()
    m = conv_model()
    optim = torch.optim.Adam(m.parameters(), lr=3e-3)
    scheduler = torch.optim.lr_scheduler.LinearLR(optim, start_factor=1.0, end_factor=0.001, total_iters=total_iters)
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
            #currstates, curractionprobs, curractions, currvalues, currtargvalues = parallel_rollout(m)
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
        print(states.shape, oldactionprobs.shape, actions.shape, oldvalues.shape, targvalues.shape, advantages.shape)
        if (targvalues.abs().mean() == 1.0):
            print("debug cat's games...")
            print(states[-32:].reshape(-1, 3, 3))
        dataset = torch.utils.data.dataset.TensorDataset(states, oldactionprobs, actions, oldvalues, targvalues, advantages)
        sampler = torch.utils.data.RandomSampler(dataset)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=minibatch_size, sampler=sampler, drop_last=True)
        firstsample = True
        #m.train()
        for k in range(epochs):
            for minibstates, miniboldactionprobs, minibactions, miniboldvalues, minibtargvalues, minibadvantages in dataloader:
                optim.zero_grad()
                newlogits, newvalues = m(minibstates)
                masked = torch.full_like(newlogits, -torch.inf)
                valid = minibstates == 0.0
                masked = torch.where(valid, newlogits, masked)
                newprobs = torch.nn.functional.softmax(masked, -1)
                if firstsample:
                    torch.testing.assert_close(torch.gather(newprobs, -1, minibactions), miniboldactionprobs, atol=5e-3, rtol=1e-3)
                    firstsample = False
                decay = 1.0 if i == 0 else max(0.0, 1 - (i + 100)/ iters)
                loss = lossfunc(newprobs, minibactions, miniboldactionprobs, minibadvantages, newvalues, minibtargvalues, decay)
                if k % 100 == 0:
                    print(minibstates[-1].reshape(3, 3))
                    print(decay, loss, scheduler.get_lr())
                loss.backward()
                optim.step()
                scheduler.step()
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
