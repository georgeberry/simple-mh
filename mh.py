import numpy as np
from numpy.random import uniform
from math import e
import itertools
from matplotlib.pylab import *
import matplotlib.pyplot as plt

'''
metropolis-hastings implementation

should allow sampling from ising, poisson, normal

record rejection rate
'''

class Metropolis:
    '''
    just metropolis because that sounds cooler

    instantiate this once for every set of parameters
        N: samples to retain
        burn_in: discards this many steps at the beginning

    call Metropolis.DISTRIBUTION(parameters) to sample from a given distribution
    '''
    def __init__(self, N = 1000, burn_in = 500):
        self.N = N
        self.burn_in = burn_in            


    def poisson_hastings(self, pois_lambda, start_val):
        '''
        q(x, y) = q(y, x) for Hastings' implementation

        r(x, y) = pi(y)/pi(x)
        '''
        #record output
        points = []
        rejections = 0
        total = 0

        #
        i = start_val

        for k in range(self.N + self.burn_in):
            j = self.poisson_Q_hastings(i) #candidate j

            u = uniform(0, 1)

            if j > i:
                alpha = min(pois_lambda/(i + 1), 1)
            elif j < i:
                alpha = min(i/pois_lambda, 1)
            elif j == i:
                alpha = 1

            if u < alpha:
                points.append(j)
                i = j
                total += 1
            else:
                points.append(i)
                rejections += 1
                total += 1

        for each in range(self.burn_in):
            points.pop(0)

        points = np.array(points)

        avg = np.mean(points)
        std_dev = np.std(points)

        return avg, std_dev, rejections/total


    def poisson_Q_hastings(self, i):
        '''
        takes current state, X(t)
        returns candidate X(t+1)
        '''

        proposal_u = uniform(0,1)

        if i == 0:
            if proposal_u < .5:
                return 0
            else:
                return 1
        else:
            if proposal_u < .5:
                return i + 1
            else:
                return i - 1



    def std_norm_hastings(self, start_val, epsilon = 1, delta = 1):
        '''
        mean 0 variance 1 normal

        e: interval range on each side of X(t) to sample from
        d: pulls in value of uniform on both sides
        '''

        points = []
        rejections = 0
        total = 0

        i = start_val

        for k in range(self.N + self.burn_in):

            j = uniform(epsilon*i - delta, epsilon*i + delta) #candidate
            u = uniform(0, 1)

            alpha = min(e**(.5 * (i**2 - j**2)), 1)

            if u < alpha:
                points.append(j)
                i = j
                total += 1
            else:
                points.append(i)
                rejections += 1
                total += 1

        for each in range(self.burn_in):
            points.pop(0)

        points = np.array(points)

        avg = np.mean(points)
        std_dev = np.std(points)

        return avg, std_dev, rejections/total


    def ising(self, beta, L):
        '''
        beta: 1/temperature
        L: size of half of the lattice - 1 in one dimension.
            lattice goes from (-L to L)^2

        test ratio: pi(ksi^x)/pi(ksi)
        '''

        #random initial spins
        ising = np.random.choice(np.array([-1,1]), size = (L + 1, L + 1))

        energy = [] #contains energy
        magnetism = []
        rejections = 0
        total = 0

        #all pairs
        #treats out-of-scope as 0 (omits them)
        rows, cols = ising.shape
        pairs = {}
        prod = [np.array([x, y]) for x, y in itertools.product(range(rows), range(cols))]
        for prod1 in prod:
            for prod2 in prod:
                if abs(prod1[0] - prod2[0]) + abs(prod1[1] - prod2[1]) == 1:
                    if tuple(prod1) not in pairs:
                        pairs[tuple(prod1)] = []
                    pairs[tuple(prod1)].append(tuple(prod2))
                    #if sum(prod1) < sum(prod2):


        for k in range(self.N + self.burn_in):
            #candidate to change
            #pick at random
            candidate_point = (np.random.randint(0, L+1), np.random.randint(0, L+1))

            #test ratio
            #amazingly, depends only on current state
            exponent = 2 * beta * sum([ising[candidate_point]*ising[neighbor] for neighbor in pairs[candidate_point]])
            alpha = min(e**(-exponent), 1)
            u = uniform(0,1)
            if u < alpha:
                #flip sign
                ising[candidate_point] = -ising[candidate_point]
                energy.append(self.hamiltonian(ising, pairs))
                magnetism.append(np.sum(ising, axis=(0,1)))
                total += 1
            else:
                energy.append(self.hamiltonian(ising, pairs))
                magnetism.append(np.sum(ising, axis=(0,1)))
                rejections += 1
                total += 1

        for each in range(self.burn_in):
            energy.pop(0)

        energy = np.array(energy)
        magnetism = np.array(magnetism)
        energy_avg = np.mean(energy)
        energy_sd = np.std(energy)
        magnetism_avg = np.mean(magnetism)/(L + 1)^2
        magnetism_sd = np.std(magnetism)

        self.ising_matrix = ising

        return energy_avg, energy_sd, magnetism_avg, magnetism_sd, rejections/total


    def hamiltonian(self, matrix, pairs):
        h = 0
        for ego in pairs:
            ego_spin = matrix[ego]
            for neighbor in pairs[ego]:
                neighbor_spin = matrix[neighbor]
                h += ego_spin * neighbor_spin
        return -h

    def boardwalk(self, start_val, delta):
        '''
        we win if we hit number 587
        there are 1000 positions
        delta is the size of the window to sample from on either side
            min of 1, max 500
        '''
        winner = 587 #who knew
        expectation = 0

        i = start_val

        for k in range(self.N):
            if i == winner:
                expectation += 1000
            j = np.random.randint(max(i - delta, 0), min(i + delta + 1, 999))
            i = j

        return expectation/self.N




###show with
'''k = Metropolis(N = 10, burn_in = 5)
t = k.ising(1, 10)

matshow(k.ising_matrix)
savefig("/Users/georgeberry/Desktop/foo.pdf")'''


##loop through states
for N in [2000, 6000, 10000]:
    for burn_in in [0, 500, 2500, 5000]:
        if N > burn_in:
            m = Metropolis(N = N, burn_in = burn_in)
            for l in [1, 5, 50]:
                for pois_start in [1, 5, 50]:
                    t = m.poisson_hastings(l, pois_start)
                    print("type: poisson", "N =", N, "burn in =", burn_in, "lambda = ", l, "start val = ", pois_start, t)
            for norm_start in [0, 5, 50]:
                for epsilon in [1, 10]:
                    for delta in [1, .5, 2]:
                        t = m.std_norm_hastings(norm_start, epsilon, delta)
                        print("type: norm", "N =", N, "burn in =", burn_in, "start = ", norm_start, "epsilon = ", epsilon, "delta = ", delta, t)
            for beta in [.2, .4, .5, .6, .75]:
                for L in [5, 10]:
                    t = m.ising(beta, L)
                    print("type: ising", "N =", N, "burn in =", burn_in, "beta = ", beta, "L = ", L, t)
                    matshow(m.ising_matrix)
                    savefig("/Users/georgeberry/Desktop/ising/ising{}.pdf".format(''.join([str(N), str(burn_in), str(beta), str(L)])))

##ising plots
m = Metropolis(N = 10000, burn_in = 8000)

res = {5: {"energy": [], "magnetism": []}, 10: {"energy": [], "magnetism": []}}
betas = [.1, .2, .3, .4, .5, .6, .7, .8, .9]

for L in [5, 10]:
    for beta in betas:
        energy, a, magnetism, b, c = m.ising(beta, L)
        res[L]["energy"].append(energy)
        res[L]["magnetism"].append(magnetism)

betas.reverse()
res[5]["energy"].reverse()
res[5]["magnetism"].reverse()
res[10]["energy"].reverse()
res[10]["magnetism"].reverse()

with plt.xkcd():
    fig = plt.figure()
    ax = fig.add_axes((0.1, 0.2, 0.8, 0.7))

    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    plt.xticks(betas)
    plt.yticks([])
    plt.plot(betas, res[5]["energy"])
    plt.xlabel("inverse temperature")
    plt.ylabel("energy")

plt.savefig("/Users/georgeberry/Desktop/ising/ising5energy.pdf")

with plt.xkcd():
    fig = plt.figure()
    ax = fig.add_axes((0.1, 0.2, 0.8, 0.7))

    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    plt.xticks(betas)
    plt.yticks([])
    plt.plot(betas, res[5]["magnetism"])
    plt.xlabel("inverse temperature")
    plt.ylabel("magnetism")
plt.savefig("/Users/georgeberry/Desktop/ising/ising5magnetism.pdf")

with plt.xkcd():
    fig = plt.figure()
    ax = fig.add_axes((0.1, 0.2, 0.8, 0.7))

    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    plt.xticks(betas)
    plt.yticks([])
    plt.plot(betas, res[10]["energy"])
    plt.xlabel("inverse temperature")
    plt.ylabel("energy")
plt.savefig("/Users/georgeberry/Desktop/ising/ising10energy.pdf")

with plt.xkcd():
    fig = plt.figure()
    ax = fig.add_axes((0.1, 0.2, 0.8, 0.7))

    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    plt.xticks(betas)
    plt.yticks([])
    plt.plot(betas, res[10]["magnetism"])
    plt.xlabel("inverse temperature")
    plt.ylabel("magnetism")
plt.savefig("/Users/georgeberry/Desktop/ising/ising10magnetism.pdf")

with plt.xkcd():
    fig = plt.figure()
    ax = fig.add_axes((0.1, 0.2, 0.8, 0.7))

    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    plt.xticks(betas)
    plt.yticks([])
    plt.plot(betas, res[5]["energy"])
    plt.plot(betas, res[10]["energy"])
    plt.xlabel("inverse temperature")
    plt.ylabel("energy")

plt.savefig("/Users/georgeberry/Desktop/ising/energy.pdf")

###boardwalk game
m = Metropolis(N = 10000, burn_in = 0)

for start_val in [10, 200, 350, 587]:
    for delta in [10, 100, 300, 500]:
        avg = 0
        for r in range(1000):
            avg += m.boardwalk(start_val, delta)
        print("start:", start_val, "delta:", delta, "avg:", avg/1000)