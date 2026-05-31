import time
import argparse
import random
import util
import model
import torch

def random_policy(state):
    h = state.size(-2)
    while True:
        idx = random.randint(0, state.size(-1)-1)
        if not state[0,0,idx]:
            return idx

def greedy_policy(state):
    h = state.size(-2)
    w = state.size(-1)
    idx = 3
    while True:
        idx = idx % w
        if not state[0,0,idx]:
            return idx
        idx += 1

def greedy_policy2(state):
    h = state.size(-2)
    w = state.size(-1)
    idx = 3 + int(torch.sum(state.abs()) // 2)
    while True:
        idx = idx % w
        if not state[0,0,idx]:
            return idx
        idx += 1

def print_board(state):
    state = state.reshape(6, 7)
    for y in range(len(state)):
        print(['X' if m == 1 else ('O' if m == -1 else ' ')  for m in state[y]])
    print()

def random_game():
    h = 6
    w = 7
    state = torch.zeros(1, h, w, dtype=torch.float)
    turns = 0
    while torch.sum(torch.abs(state)) < h*w:
        state *= -1.0
        action = random_policy(state)
        y = h-(int(torch.sum(torch.abs(state), dim=-2)[0, action]) + 1)
        state[0, y, action] = 1.0
        if turns % 2 == 0:
            print_board(state)
        if (util.longest_of(1, state[0], 4) == 4):
            print("win")
            return
        if torch.sum(torch.abs(state)) >= h*w:
            print("cat's game")
            return
        turns += 1
    print("cat's game")

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
    device = next(m.parameters()).device
    h = 6
    w = 7
    size = h * w
    with torch.no_grad():
        state = torch.zeros(1, h, w, dtype=torch.float, device=device)
        while(torch.sum(torch.abs(state)) < size and util.longest_of_batch(1, state.cpu(), 4) < 4):
            # switch player
            state = state * -1.0
            valid = state[:, 0, :] == 0.0
            logits, value = m(state)
            masked = torch.full_like(logits, -torch.inf)
            masked = torch.where(valid, logits, masked)
            probs = torch.nn.functional.softmax(masked, -1)
            sample = torch.clamp(torch.rand(1, device=device), min=1e-5)
            action = torch.searchsorted(probs.cumsum(-1).squeeze(), sample)
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
            y = h-(int(torch.sum(torch.abs(state), dim=-2)[0, action]) + 1)
            state[0, y, action] = 1.0
            first = not first
        if util.longest_of_batch(1, state.cpu(), 4) >= 4:
            targfirstvalue = 1.0 if not first else -1.0
        else:
            targfirstvalue = 0.0
    states = torch.cat(firststates + secondstates)
    values = torch.cat(firstvalues + secondvalues)
    actionprobs = torch.cat(firstactionprobs + secondactionprobs)
    actions = torch.tensor(firstactions + secondactions).reshape(states.size(0), 1).to(device=device)
    targvalues = torch.full_like(values, targfirstvalue)
    targvalues[len(firstvalues):] = -targfirstvalue
    assert(states.size(0) == values.size(0))
    assert(targvalues.size(0) == values.size(0))
    return states, actionprobs, actions, values, targvalues 

def lossfunc(probs, actions, oldactionprobs, advantages, values, targvalues, entropy_decay):
    epsilon = 0.2
    c1 = 0.5
    c2 = 0.1*entropy_decay

    r = torch.gather(probs, -1, actions)/oldactionprobs

    lclip = torch.min(r * advantages,
                      torch.clip(r, 1 - epsilon, 1 + epsilon) * advantages).mean()
    lvf = torch.nn.functional.mse_loss(values, targvalues)
    dist = torch.distributions.categorical.Categorical(probs=probs)
    lentropy = dist.entropy().mean()
    return -lclip + c1*lvf - c2*lentropy

def sample_model_func(m):
    def sample_model(state):
        logits, value = m(state)
        masked = torch.full_like(logits, -torch.inf)
        valid = state[:, 0, :] == 0.0
        masked = torch.where(valid, logits, masked)
        probs = torch.nn.functional.softmax(masked, -1)
        action = torch.argmax(probs, dim=-1)
        return action
    return sample_model

def play(policy1, policy2):
    h = 6
    w = 7
    state = torch.zeros(1, h, w)
    polices = [policy1, policy2]
    player = 0
    while util.longest_of_batch(1, state, 4) < 4 and torch.sum(torch.abs(state)) < state.size(-1) * state.size(-2):
        state *= -1.0
        policy = polices[player % 2]
        action = policy(state)
        y = h-(int(torch.sum(torch.abs(state), dim=-2)[0, action]) + 1)
        assert(not state[0, y, action])
        state[0, y, action] = 1.0
        player += 1
    if util.longest_of_batch(1, state, 4) >= 4:
        if player % 2 == 1:
            return 1.0
        else:
            assert(torch.sum(state.abs()) % 2 == 0.0)
        return -1.0
    return 0.0

def eval_model(m, baselinepolicy):
    wins = 0
    ties = 0
    losses = 0
    evalgames = 100
    for i in range(100):
        modelfirst = torch.rand(1) > 0.5
        policies = [sample_model_func(m), baselinepolicy] if modelfirst else [baselinepolicy, sample_model_func(m)]
        value = play(*policies)
        if value == 1.0:
            if modelfirst:
                wins += 1.0
            else:
                losses += 1.0
        elif value == -1.0:
            if not modelfirst:
                wins += 1.0
            else:
                losses += 1.0
        else:
            ties += 1
    assert(wins + losses + ties == evalgames)
    return wins, losses, ties

def train(modelname, cuda):
    minibatch_size = 16384
    target_batch_size = 16384
    iters = 1000
    epochs = 5
    total_iters = iters*epochs*(target_batch_size//minibatch_size)
    device = 'cpu' if not cuda else 'cuda'
    m = model.ConvNet(6, 7, p=7).to(device=device)
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
        episodes = 0
        while total_batch_size < target_batch_size:
            episodes += 1
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
        print("cat's game frac:", 1 - targvalues.abs().mean(), "first player win frac:", targvalues[:total_batch_size//2].clamp(min=0).mean())
        print("average episode len:", total_batch_size/episodes)
        print(states.shape, oldactionprobs.shape, actions.shape, oldvalues.shape, targvalues.shape, advantages.shape)
        print(states[-1])
        dataset = torch.utils.data.dataset.TensorDataset(states, oldactionprobs, actions, oldvalues, targvalues, advantages)
        sampler = torch.utils.data.RandomSampler(dataset)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=minibatch_size, sampler=sampler, drop_last=True)
        firstsample = True
        #m.train()
        t0 = time.time()
        for k in range(epochs):
            for minibstates, miniboldactionprobs, minibactions, miniboldvalues, minibtargvalues, minibadvantages in dataloader:
                optim.zero_grad()
                newlogits, newvalues = m(minibstates)
                masked = torch.full_like(newlogits, -torch.inf)
                valid = minibstates[:, 0, :] == 0.0
                masked = torch.where(valid, newlogits, masked)
                newprobs = torch.nn.functional.softmax(masked, -1)
                if firstsample:
                    torch.testing.assert_close(torch.gather(newprobs, -1, minibactions), miniboldactionprobs, atol=7e-3, rtol=1e-3)
                    firstsample = False
                decay = 1.0 if i == 0 else max(0.0, 1 - (i + 100)/ iters)
                loss = lossfunc(newprobs, minibactions, miniboldactionprobs, minibadvantages, newvalues, minibtargvalues, decay)
                if k % 100 == 0:
                    print(decay, loss, scheduler.get_lr())
                loss.backward()
                optim.step()
                scheduler.step()
        print("epochs took:", time.time() - t0)
        wins, losses, ties = eval_model(m, random_policy)
        print(f"random eval wins: {wins} ties: {ties} losses: {losses}")
        wins, losses, ties = eval_model(m, greedy_policy)
        print(f"greedy eval wins: {wins} ties: {ties} losses: {losses}")
        wins, losses, ties = eval_model(m, greedy_policy2)
        print(f"greedy2 eval wins: {wins} ties: {ties} losses: {losses}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', action='store_true')
    parser.add_argument('-t')
    #parser.add_argument('-p')
    parser.add_argument('--cuda', action='store_true')
    args = parser.parse_args()
    if args.r:
        random_game()
    if args.t:
        train(args.t, args.cuda)
    #if args.p:
    #    play(args.p)

