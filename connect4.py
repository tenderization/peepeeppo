import os
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
        if (util.longest_of_batch(1, state, 4) == 4):
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

def play_policies(policy1, policy2):
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

def eval_policies(policy, baselinepolicy, evalgames=500):
    wins = 0
    ties = 0
    losses = 0
    for i in range(evalgames):
        modelfirst = torch.rand(1) > 0.5
        policies = [policy, baselinepolicy] if modelfirst else [baselinepolicy, policy]
        value = play_policies(*policies)
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
    minibatch_size = 32768
    target_batch_size = 32768
    iters = 10000
    epochs = 5
    total_iters = iters*epochs*(target_batch_size//minibatch_size)
    device = 'cpu' if not cuda else 'cuda'
    m = model.ConvNet(6, 7, channels=128, layers=4, p=7).to(device=device)
    optim = torch.optim.Adam(m.parameters(), lr=3e-4)
    scheduler = torch.optim.lr_scheduler.LinearLR(optim, start_factor=1.0, end_factor=0.001, total_iters=total_iters)
    if os.path.exists('test1000_c32_l2_d128.pth'):
        test_model = load_model('test1000_c32_l2_d128.pth')
    else:
        test_model = None
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
                    torch.testing.assert_close(torch.gather(newprobs, -1, minibactions), miniboldactionprobs, atol=5e-2, rtol=1e-3)
                    firstsample = False
                decay = 1.0 if i == 0 else max(0.0, 1 - (i + 100)/ iters)
                loss = lossfunc(newprobs, minibactions, miniboldactionprobs, minibadvantages, newvalues, minibtargvalues, decay)
                if k % 100 == 0:
                    print(decay, loss, scheduler.get_lr())
                loss.backward()
                optim.step()
                scheduler.step()
        print("epochs took:", time.time() - t0)
        wins, losses, ties = eval_policies(sample_model_func(m), random_policy, evalgames=500)
        print(f"random eval wins: {wins} ties: {ties} losses: {losses}")
        wins, losses, ties = eval_policies(sample_model_func(m), greedy_policy, evalgames=10)
        print(f"greedy eval wins: {wins} ties: {ties} losses: {losses}")
        wins, losses, ties = eval_policies(sample_model_func(m), greedy_policy2, evalgames=10)
        print(f"greedy2 eval wins: {wins} ties: {ties} losses: {losses}")
        if test_model is not None:
            wins, losses, ties = eval_policies(sample_model_func(m), sample_model_func(test_model), evalgames=100)
            print(f"test1000 model eval wins: {wins} ties: {ties} losses: {losses}")
        if i % 100 == 0:
            torch.save(m.state_dict(), f"{args.t}_iter{i}_c{m.channels}_l{m.layers}_d{m.d}.pth")
    torch.save(m.state_dict(), f"{args.t}_final_c{m.channels}_l{m.layers}_d{m.d}.pth")

def parse_model_config(model):
    _, name = os.path.split(model)
    vals = name.split('_')
    d = int("".join([char for char in vals[-1] if char.isdigit()]))
    l = int("".join([char for char in vals[-2] if char.isdigit()]))
    c = int("".join([char for char in vals[-3] if char.isdigit()]))
    return d, l, c

def load_model(model_name):
    h = 6
    w = 7
    state_dict = torch.load(model_name)
    dim, layers, channels = parse_model_config(model_name)
    m = model.ConvNet(h, w, channels=channels, layers=layers, d=dim, p=7)
    m.load_state_dict(state_dict)
    return m

def play(model_name):
    h = 6
    w = 7
    m = load_model(model_name)
    state = torch.zeros(1, h, w, dtype=torch.float)
    playerfirst = (torch.rand(1) > 0.5).all()
    print(playerfirst)
    while(True):
        state = state * -1.0
        if not playerfirst:
            valid = state[:, 0, :] == 0.0
            logits, value = m(state)
            masked = torch.full_like(logits, -torch.inf)
            masked = torch.where(valid, logits, masked)
            probs = torch.nn.functional.softmax(masked, -1)
            action = torch.argmax(probs, dim=-1)
            print("value:", value)
            playerfirst = False
            y = h-(int(torch.sum(torch.abs(state), dim=-2)[0, action]) + 1)
            state[0, y, action] = 1.0
            if util.longest_of_batch(1, state, 4) >= 4:
                print_board(state * -1.0)
                print("player lose")
                break
            if torch.sum(torch.abs(state)) >= h*w:
                print_board(state * -1.0)
                print("cat's game")
                break
        state = state * -1.0
        print_board(state)
        playeraction = int(input("player move:")) - 1
        assert(state[0, 0, playeraction] == 0.0)
        y = h-(int(torch.sum(torch.abs(state), dim=-2)[0, playeraction]) + 1)
        state[0, y, playeraction] = 1.0
        print_board(state)
        if util.longest_of_batch(1, state, 4) >= 4:
            print("player win")
            break
        if torch.sum(torch.abs(state)) >= h*w:
            print("cat's game")
            break
        playerfirst = False

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', action='store_true')
    parser.add_argument('-t')
    parser.add_argument('-p')
    parser.add_argument('--cuda', action='store_true')
    args = parser.parse_args()
    if args.r:
        random_game()
    if args.t:
        train(args.t, args.cuda)
    if args.p:
        play(args.p)

