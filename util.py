
def longest_of(marker, state):
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
                else:
                    curr_longest = 0
    # \
    #  \
    #   \
    # then
    #  /
    # /
    #/
    for left in (False, True):
        for x0 in range(w):
            curr_longest = 0
            for i in range(h):
                x = x0 + i if left else x0 - i
                y = i
                if x >= w or y >= h:
                    break
                curr_state = state[y][x]
                if curr_state == marker:
                    curr_longest += 1
                    if curr_longest > longest:
                        longest = curr_longest
                else:
                    curr_longest = 0
    return longest

def test_longest_of():
    data0 = [[-1, 0, 1],
             [-1, 1, 1],
             [-1, 0, 1]]
    data1 = [[ 0, 0, 0],
             [-1, 0, 1],
             [-1, 1, 0]]

    assert(longest_of(-1, data0) == 3)
    assert(longest_of( 1, data0) == 3)
    assert(longest_of( 0, data0) == 1)
    assert(longest_of(-1, data1) == 2)
    assert(longest_of( 1, data1) == 2)
    assert(longest_of( 0, data1) == 3)

def test():
    test_longest_of()

if __name__ == '__main__':
    test()
