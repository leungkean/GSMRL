import numpy as np
import matplotlib.pyplot as plt
import pickle

with open('./ppo_molecule_20_005/evaluate/test.pkl', 'rb') as f:
    transitions = pickle.load(f)['transitions']

transitions = np.sum(transitions, axis=0)
label = [234, 244, 322, 356, 393, 698, 725, 790, 792, 841, 80, 350, 465, 573, 583, 879, 901, 675, 147, 833]

plt.figure(figsize=(15,7.5))
plt.title('Molecule 20 Cost 0.05 Transitions')
plt.bar(list(range(20)), transitions, align='center')
plt.xticks(list(range(20)), label)
plt.show()