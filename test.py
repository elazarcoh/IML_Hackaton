import numpy as np

d = np.loadtxt('submission/task1/src/human.txt', dtype=str)
def hamming2(s1, s2):
    """Calculate the Hamming distance between two bit strings"""
    assert len(s1) == len(s2)
    return sum(c1 != c2 for c1, c2 in zip(s1, s2))

# with open('r\s.txt', 'w') as inf, open('r\e.txt', 'w') as outf:
#     for s in d:
#         inf.write(s[:-20] + '\n')
#         outf.write(s[-20:] + '\n')

with open('r\\e.txt') as org, open('submission/task1/output/predictions.txt') as pre:
    s=[]
    for x,y in zip(org,pre):
       s.append(hamming2(x,y))
    print(np.mean(s))
