import pickle
import numpy as np

window = 3

question_base = []

# DPDS1
question_base.append("In the past 24 hours… My mood was up and down.")
# DPDS2
question_base.append("I was not interested in doing much of anything.")
# DPDS3
question_base.append("I did something on impulse.")
# DPDS4
question_base.append("I insisted things be done my way.")
# DPDS5
question_base.append("I lost my temper.")
# DPDS6
question_base.append("I felt disconnected from my own body.")
# DPDS7 
question_base.append("I felt anxious.")
# DPDS8
question_base.append("I wanted people to notice my body.")
# DPDS9
question_base.append("I said something offensive to someone.")
# DPDS10
question_base.append("I heard things that weren't really there.")
# DPDS11
question_base.append("I felt depressed.")
# DPDS12
question_base.append("I did something dangerous just for the thrill.")
# DPDS13
question_base.append("I took advantage of someone.")
# DPDS15
question_base.append("My thoughts were confusing.")
# DPDS16
question_base.append("I wanted people to notice my talents.")
# DPDS17
question_base.append("I made sure everything I did was perfect.")
# DPDS19
question_base.append("I lied to someone.")
# DPDS20
question_base.append("I got lost in my fantasies.")
# DPDS21
question_base.append("I worried about being abandoned.")
# DPDS22
question_base.append("I didn't want to be around others.")
# DPDS23
question_base.append("I acted aggressively towards someone.")
# DPDS24
question_base.append("I did things others might find unusual.")
# DPDS25
question_base.append("I acted on impulse while feeling upset.")
# DPDS26
question_base.append("My relationships felt empty.")
# DPDS27
question_base.append("I behaved irresponsibly.")
# DPDS28
question_base.append("I acted on my emotions.")
# DPDS29
question_base.append("I thought about sex.")
# DPDS30
question_base.append("I put work above everything else.")
# DPDS31
question_base.append("I felt like I wanted to hurt someone.")
# DPDS32
question_base.append("I was suspicious of others.")

time_question = []

for i in range(window):
    for question in question_base:
        time_question.append(f'(Day {i+1}) ' + question)

time_question = tuple(time_question)
dataset_index = [1,2,3]

for ind in dataset_index:
    result_dir = f'./window{window}/psych{ind}/evaluate/'
    traj = {}

    with open(result_dir + 'test.pkl','rb') as f:
        data = pickle.load(f)

    transitions = data['transitions']

    for i in range(transitions.shape[0]):
        traj[i] = []
        prev_time = 0
        for j in np.flip(np.argsort(transitions[i])):
            if transitions[i,j] == 0:
                break
            # Check to see if agent chose invalid action
            if int(time_question[j][5]) < prev_time:
                print(f"ERROR at instance {j} for psych{ind}") 

            traj[i].append(time_question[j]) 
            prev_time = int(time_question[j][5])

        traj[i] = tuple(traj[i])

    with open(result_dir + 'question_trajectory.pkl','wb') as f:
        pickle.dump(traj,f)

