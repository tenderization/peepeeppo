import torch
import time

import torch
from pathlib import Path

import peepeeppocpp

def longest_of_batch(marker, states, stop=3):
    return torch.ops.peepeeppocpp.longest_of(states, int(marker), int(stop))

def longest_of(marker, state, stop=3):
    # N^2
    h = len(state)
    w = len(state[0])
    longest = 0
    for transpose in (False, True):
        for idx0 in range(h if not transpose else w):
            curr_longest = 0
            for idx1 in range(w if not transpose else h):
                curr_state = state[idx0][idx1] if not transpose else state[idx1][idx0]
                if curr_state == marker:
                    curr_longest += 1
                    if curr_longest > longest:
                        longest = curr_longest
                        if longest >= stop:
                            return longest
                else:
                    curr_longest = 0
    # \
    #  \
    #   \
    # then
    #  /
    # /
    #/
    #for left in (False, True):
        left = transpose
        for x0 in range(w):
            curr_longest = 0
            for i in range(h):
                x = x0 + i if left else x0 - i
                y = h - 1 - i
                if x >= w or x < 0 or y < 0:
                    break
                curr_state = state[y][x]
                if curr_state == marker:
                    curr_longest += 1
                    if curr_longest > longest:
                        longest = curr_longest
                        if longest >= stop:
                            return longest
                else:
                    curr_longest = 0
            curr_longest = 0
            for i in range(h):
                x = x0 + i if left else x0 - i
                y = i
                if x >= w or x < 0 or y < 0:
                    break
                curr_state = state[y][x]
                if curr_state == marker:
                    curr_longest += 1
                    if curr_longest > longest:
                        longest = curr_longest
                        if longest >= stop:
                            return longest
                else:
                    curr_longest = 0
    return longest

def full(state):
    for y in range(len(state)):
        for x in range(len(state[0])):
            if not state[y][x]:
                return False
    return True

def test_longest_of():
    data0 = [[-1, 0, 1],
             [-1, 1, 1],
             [-1, 0, 1]]
    data1 = [[ 0, 0, 0],
             [-1, 0, 1],
             [-1, 1, 0]]
    data2 = [[1, -1, 1],
             [1, -1, 1],
             [-1, 1,-1]]

    assert(longest_of(-1, data0) == 3)
    assert(longest_of( 1, data0) == 3)
    assert(longest_of( 0, data0) == 1)
    assert(longest_of(-1, data1) == 2)
    assert(longest_of( 1, data1) == 2)
    assert(longest_of( 0, data1) == 3)
    assert(longest_of(-1, data2) == 2)
    assert(longest_of( 1, data2) == 2)
    assert(longest_of( 0, data2) == 0)

def test_longest_of2():
    data0 = [[0, 1, 0, 0],
             [1, 0, 0, 0],
             [0, 0, 0, 0],
             [0, 0, 0, 0]]
    data1 = [[0, 0, 0, 0],
             [0, 0, 0, 1],
             [0, 0, 1, 0]]
    assert(longest_of(1, data0, 4) == 2)
    assert(longest_of(1, data1, 4) == 2)
    assert(longest_of(0, data0, 4) == 4)
    assert(longest_of(0, data1, 4) == 4)

def test_extension():
    boards = torch.randint(-1, 1, (8192, 6, 7), dtype=torch.float)
    t0 = time.time()
    ref0 = torch.cat([torch.tensor([longest_of(-1, board, 4) for board in boards], dtype=torch.float)], dim=0)
    ref1 = torch.cat([torch.tensor([longest_of(0, board, 4) for board in boards], dtype=torch.float)], dim=0)
    ref2 = torch.cat([torch.tensor([longest_of(1, board, 4) for board in boards], dtype=torch.float)], dim=0)
    t1 = time.time()
    res0 = longest_of_batch(-1, boards, 4)
    res1 = longest_of_batch(0, boards, 4)
    res2 = longest_of_batch(1, boards, 4)
    t2 = time.time()
    print("nonbatched took: ", t1 - t0, " batch cpp extension took: ", t2 - t1)
    torch.testing.assert_close(ref0, res0, atol=0.0, rtol=0.0)
    torch.testing.assert_close(ref1, res1, atol=0.0, rtol=0.0)
    torch.testing.assert_close(ref2, res2, atol=0.0, rtol=0.0)


def test():
    test_longest_of()
    test_longest_of2()
    test_extension()

if __name__ == '__main__':
    test()
