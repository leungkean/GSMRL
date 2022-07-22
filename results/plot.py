import numpy as np
import matplotlib.pyplot as plt
import pickle

with open('./ppo_molecule_20_005/evaluate/test.pkl', 'rb') as f:
    data = pickle.load(f)

transitions = data['transitions'].astype(np.float32) - 1.
transitions = np.clip(transitions, 0., None)
time_steps = data['num_acquisition']

for i in range(transitions.shape[0]):
    transitions[i] = transitions[i] / time_steps[i]

transitions = np.sum(transitions, axis=0)
label = [234, 244, 322, 356, 393, 698, 725, 790, 792, 841, 80, 350, 465, 573, 583, 879, 901, 675, 147, 833]

plt.figure(figsize=(15,7.5))
plt.title('Molecule 20 Transitions with Cost 0.05')
plt.bar(list(range(20)), transitions, align='center')
plt.xticks(list(range(20)), label)
plt.show()
