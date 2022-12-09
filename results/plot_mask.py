import pickle
import numpy as np
import matplotlib.pyplot as plt

window = 2

# DPDS questions
base_dpds = [str(i) for i in range(1,33) if i != 14 and i != 18]
# Time Labeled DPDS questions
time_dpds = []

for i in range(window):
    for dpds in base_dpds:
        time_dpds.append(dpds + '_' + str(i+1))

dataset_index = [1,2,3,4]

for ind in dataset_index: 
    plt.figure(figsize=(18.5,9.5))
    result_dir = f'./window{window}/psych{ind}/evaluate/'

    with open(result_dir + 'test.pkl','rb') as f:
        data = pickle.load(f)

    freq = data['transitions']
    freq[freq > 0] = 1
    freq = np.sum(freq, axis=0)/freq.shape[0]

    # Plot bar graph of freq of each question with labeled values to 2 decimal places
    plt_fig = plt.bar(time_dpds, freq, align='center', alpha=0.5)
    x = np.arange(1, len(time_dpds)+1)
    plt.xticks(x, time_dpds, rotation=-90)
    plt.ylabel('Frequency')
    plt.xlabel('DPDS Questions (Time Labeled)')
    plt.title('Frequency Each Question is Selected for Prediction')
    plt.bar_label(plt_fig, fmt='%.2f')
    plt.savefig(result_dir + 'freq.png')
    plt.show()
