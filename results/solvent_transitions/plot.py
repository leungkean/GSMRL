import numpy as np
import matplotlib.pyplot as plt
import pickle

with open('../ppo_solvent_exp/evaluate/test.pkl', 'rb') as f:
    data = pickle.load(f)

transitions = data['transitions'].astype(np.float32) - 1.
transitions = np.clip(transitions, 0., None)

transitions = np.mean(transitions, axis=0)
label = ['1313', '352', '1808', '1594', '1724', '650', '824', '1476', '1379', '439', '45', '204', '584', '222_solv', '2', '1971', '249', '1754', '1357', '1573']

plt.figure(figsize=(15,7.5))
plt.title('solvent_exp Transitions with Cost 5 \n Avg. Features Acquired: 2.24, Avg. Accuracy: 0.733')
plt.bar(list(range(20)), transitions, align='center')
plt.xticks(list(range(20)), label)
plt.xlabel('Feature')
plt.ylabel('Average Time Steps Acquired')
plt.show()
