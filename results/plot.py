import numpy as np
import matplotlib.pyplot as plt
import pickle

with open('./ppo_molecule_20_001/evaluate/test.pkl', 'rb') as f:
    data = pickle.load(f)

transitions = data['transitions'].astype(np.float32) - 1.
transitions = np.clip(transitions, 0., None)

transitions = np.mean(transitions, axis=0)
label = [234, 244, 322, 356, 393, 698, 725, 790, 792, 841, 80, 350, 465, 573, 583, 879, 901, 675, 147, 833]

plt.figure(figsize=(15,7.5))
plt.title('Molecule 20 Transitions with Cost 0.01 \n Avg. Features Acquired: 16.2, Avg. Accuracy: 0.75')
plt.bar(list(range(20)), transitions, align='center')
plt.xticks(list(range(20)), label)
plt.xlabel('Feature')
plt.ylabel('Average Time Steps per Acquisition')
plt.show()
