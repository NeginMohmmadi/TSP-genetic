import random

def PMX_crossover(parent1, parent2):
    n = len(parent1)
    index_low = random.randint(0, n-2)
    index_high = random.randint(index_low+1, n-1)
    offspring = parent1[:index_low] + parent2[index_low: index_high+1] + parent1[index_high+1:]
    relations = {}
    for i in range(index_low, index_high+1):
        relations[parent2[i]] = parent1[i]
    for i in range(index_low):
        while offspring[i] in relations:
            offspring[i] = relations[offspring[i]]
    for i in range(index_high+1, n):
        while offspring[i] in relations:
            offspring[i] = relations[offspring[i]]
    return offspring

def CX_crossover(parent1: list, parent2: list):
    n = len(parent1)
    offspring = [-1 for _ in range(n)]
    parents = [parent1, parent2]
    for i in range(n):
        if offspring[i]==-1:
            random.shuffle(parents)
            offspring[i] = parents[0][i]
            j = parents[1].index(parents[0][i])
            while j!=i:
                offspring[j] = parents[0][j]
                j = parents[1].index(parents[0][j])
    return offspring

def OX2_crossover(parent1, parent2):
    offspring = parent1
    n = len(parent1)
    num_random = random.randint(1, n)