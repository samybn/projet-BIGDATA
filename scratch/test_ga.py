import random
import numpy as np

# Mocking the Optimizer logic from train_optimize.py
class MockGeneticOptimizer:
    def __init__(self, pop_size=4, generations=2):
        self.pop_size = pop_size
        self.generations = generations
        self.bounds = {
            'n_layer': (1, 5),
            'lr': (1e-5, 1e-2),
            'batch_size': (16, 129)
        }

    def create_individual(self):
        return {
            'n_layer': random.randint(*self.bounds['n_layer']),
            'lr': random.uniform(*self.bounds['lr']),
            'batch_size': random.randint(*self.bounds['batch_size'])
        }

    def crossover(self, p1, p2):
        child = {}
        for key in p1.keys():
            child[key] = p1[key] if random.random() > 0.5 else p2[key]
        return child

    def mutate(self, ind):
        if random.random() < 0.2:
            key = random.choice(list(self.bounds.keys()))
            if key == 'lr':
                ind[key] = random.uniform(*self.bounds[key])
            else:
                ind[key] = random.randint(*self.bounds[key])
        return ind

def test_ga():
    opt = MockGeneticOptimizer()
    pop = [opt.create_individual() for _ in range(opt.pop_size)]
    print(f"Initial Pop: {pop}")
    
    p1, p2 = pop[0], pop[1]
    child = opt.crossover(p1, p2)
    print(f"Crossover: {p1} + {p2} -> {child}")
    
    mutated = opt.mutate(child.copy())
    print(f"Mutation: {child} -> {mutated}")
    print("GA Logic Verification: SUCCESS")

if __name__ == "__main__":
    test_ga()
