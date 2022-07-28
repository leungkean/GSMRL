import numpy as np
import matplotlib.pyplot as plt
import pickle

with open('../ppo_solvent_20_exp_20/evaluate/test.pkl', 'rb') as f:
    data = pickle.load(f)

transitions = data['transitions'].astype(np.float32) - 1.
transitions = np.clip(transitions, 0., None)

transitions = np.mean(transitions, axis=0)
label_20 = ['1313', '352', '1808', '1594', '1724', '650', '824', '1476', '1379', '439', '45', '204', '584', '222_solv', '2', '1971', '249', '1754', '1357', '1573']
label_exp = ['e1', 't1', 'nn1', 'ne1', 'dipole1', 'ho1', 'lu1', 'holu_gap']

plt.figure(figsize=(18,9))
plt.title('solvent_20_exp_20 Transitions with Cheap Cost 0 and Exp Cost 20.0 \n RMSE: 37.22, Num. Acquisition: 20.1875')
plt.bar(list(range(28)), transitions, align='center')
plt.xticks(list(range(28)), label_20+label_exp)
plt.xlabel('Feature')
plt.ylabel('Average Time Steps Acquired')
plt.show()
