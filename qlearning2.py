import threading
import random
import numpy as np
import csv
import pandas as pd
import math
import os
import json

from merge_result import ClassMerge
from preprocess import LoadCategory

class QLearningUpdata:
    def __init__(self, in_record_path, out_table_path, where_pool, where_category_path, speak_focus_path,
                 care_number, decay_reward, base_reward, lower_limit, decay_qvalue, learning_rate):
        ## Init setting
        self.reward_table_size = [0, 0]
        self.reward_table = list(list())
        self.qtable_size = [0, 0]
        self.qtable = list(list())
        self.reward_table_header = list() ## to show on gui(table header).
        self.qtable_table_header = list() ## to show on gui(table header).
        
        self.record_number = 0 ##

        ## Define what POS(part-of-speech) care
        with open(speak_focus_path) as json_file: 
            self.speak_dict = json.load(json_file) 
        
        self.in_record_path = in_record_path ## Csv of record.
        self.out_table_path = out_table_path ## Csv of Q table.
        self.isstop = False

        self.care_number = care_number ## Number of data we use to calculate(only care about recent data).
        self.decay_reward = decay_reward ## Decay rate of reward giving.
        self.base_reward = base_reward ## Reward given when state happens.
        self.lower_limit = lower_limit ## Action need greater than limit (reward) to execute.
        self.decay_qvalue = decay_qvalue ## feature decay rate in Q learning formula.
        self.learning_rate = learning_rate ## flearning_rate in Q learning formula.

        self.where_category = LoadCategory(category_path=where_category_path)

        self.where_pool_a = where_pool[0]
        self.where_pool_b = where_pool[1]
        self.where_pool_c = where_pool[2]

    def start(self):
         ## Put the program in the sub thread, daemon=True means that the sub thread will be closed as the main thread closes.
        print("get q learning started!")
        threading.Thread(target=self.qLearning, daemon=True, args=()).start()

    def stop(self):
	    ## Switch to stop infinite loop.
        self.isstop = True
        print("get stopped!")
   
    def getdata(self):
	    ## When there is a need for reward table and reward table, the latest ones are returned.
        return(self.reward_table_size, self.reward_table_header, self.reward_table, self.qtable_size, self.qtable_table_header, self.qtable)

    def allPossibleNextAction(self, cur_pos, reward_np, state_size):
            action = list()
            ## Possible actions in current state
            step_matrix = [x >= self.lower_limit and x != None for x in reward_np[cur_pos]]

            for i in range(state_size):
                if(step_matrix[i]):
                    action.append(i)

            return(action)

    def getNextState(self, state_size):
            return random.randint(0, state_size - 1)
        
    def qLearning(self):
        while (not self.isstop):
            ## load past information
            file = pd.read_csv(self.in_record_path)
            
            ## If not have too much new record
            if (file.shape[0] - self.record_number) < self.care_number:
                continue
            ## Record the data points(index) are read
            self.record_number = file.shape[0]
                
            whens = list()
            whats = list()
            where_scores = list()
            
            speaking_n = list()
            speaking_nr = list()
            speaking_nrfg = list()
            speaking_nt = list()
            speaking_nz = list()
            speaking_ns = list()
            speaking_tg = list()

            speaking_v = list()
            speaking_df = list()
            speaking_vg = list()

            speaking_d = list()
            speaking_t = list()

            try:
                for info in range(self.care_number):
                    when = file.iloc[-(self.care_number)+info, :].when_state ## Get when sub state
                    what = file.iloc[-(self.care_number)+info, :].what_state ## Get what sub state
                    where_score = file.iloc[-(self.care_number)+info, :].where_state_score ## Get score of each where category to adjust
                    where_score = list(map(int, eval(where_score))) ## transform string to int
                    
                    ## 
                    n = eval(file.iloc[-(self.care_number)+info, :].n)
                    nr = eval(file.iloc[-(self.care_number)+info, :].nr)
                    nrfg = eval(file.iloc[-(self.care_number)+info, :].nrfg)
                    nt = eval(file.iloc[-(self.care_number)+info, :].nt)
                    nz = eval(file.iloc[-(self.care_number)+info, :].nz)
                    ns = eval(file.iloc[-(self.care_number)+info, :].ns)
                    tg = eval(file.iloc[-(self.care_number)+info, :].tg)
                    v = eval(file.iloc[-(self.care_number)+info, :].v)
                    df = eval(file.iloc[-(self.care_number)+info, :].df)
                    vg = eval(file.iloc[-(self.care_number)+info, :].vg)
                    d = eval(file.iloc[-(self.care_number)+info, :].d)
                    t = eval(file.iloc[-(self.care_number)+info, :].t)

                    whens.append(when)
                    whats.append(what)
                    where_scores.append(where_score)
                    speaking_n.append(n)
                    speaking_nr.append(nr)
                    speaking_nrfg.append(nrfg)
                    speaking_nt.append(nt)
                    speaking_nz.append(nz)
                    speaking_ns.append(ns)
                    speaking_tg.append(tg)
                    speaking_v.append(v)
                    speaking_df.append(df)
                    speaking_vg.append(vg)
                    speaking_d.append(d)
                    speaking_t.append(t)
            except IndexError as error:
                print("error")
                pass

            ## Adjust the result(where) by merge algorithm.--------------------------------------------------------------------------
            where_scores = np.array(where_scores)
            where_labels = ClassMerge(where_scores, self.where_pool_a, self.where_pool_b, self.where_pool_c)
            wheres = list()
            for label in where_labels:
                wheres.append(self.where_category[label]) ## Get where sub state
            ## ----------------------------------------------------------------------------------------------------------------------
            
            """[Create main state and action]
                [0, 1, 2, 3]
                     ↓       (Reverse)
                [3, 2, 1, 0] (For main state, first(latest) one is not need)
                   ↙  ↙  ↙   (Corresponding relationship)
                [3, 2, 1, 0] (For action, last one(lodest) is not need)
            """
            main_states = list()
            action = list()
            for i in range(len(whens)):
                self.speak_state = list()
                if speaking_n[i][0] != "":
                    self.focus(speaking_n[i], "n")
                if speaking_nr[i][0] != "":
                    self.focus(speaking_nr[i], "nr")
                if speaking_nrfg[i][2] != "":
                    self.focus(speaking_nrfg[i], "nrfg")
                if speaking_nt[i][0] != "":
                    self.focus(speaking_nt[i], "nt")
                if speaking_nz[i][0] != "":
                    self.focus(speaking_nz[i], "nz")
                if speaking_ns[i][0] != "":
                    self.focus(speaking_ns[i], "ns")
                if speaking_tg[i][0] != "":
                    self.focus(speaking_tg[i], "tg")
                if speaking_v[i][0] != "":
                    self.focus(speaking_v[i], "v")
                if speaking_df[i][0] != "":
                    self.focus(speaking_df[i], "df")
                if speaking_vg[i][0] != "":
                    self.focus(speaking_vg[i], "vg")
                if speaking_d[i][0] != "":
                    self.focus(speaking_d[i], "d")
                if speaking_t[i][0] != "":
                    self.focus(speaking_t[i], "t")
                main_state = whens[i] + "_" + wheres[i] + "_" + whats[i]

                for state in self.speak_state:
                    main_state += "_" + state
                main_states.append(main_state)

            ## Reverse, give reward from the latest state
            main_states.reverse()
            for i in range(len(main_states)-1): ## For action, last one(lodest) is not need
                action.append(main_states[i])

            main_states_rows = list(set(main_states))
            main_states_rows.sort(key=main_states.index)
            main_states = main_states[1:] ## For main state, first(latest) one is not need

            ## Define episode and step
            max_episodes = pow(len(main_states_rows), 2) ## Total episode need run
            max_episode_steps = (pow(len(main_states_rows), 2) + pow(len(main_states_rows), 2)) ** 0.5 ## Total step need run for each episode

            ## Create reward table
            reward_table = pd.DataFrame(index=main_states_rows, columns=main_states_rows)
            reward_table = reward_table.fillna(0)

            if not(reward_table.empty):
                ## Transform past information to reward table by reward formula
                for record_index, record_data in enumerate(main_states):
                    if record_index == 0:
                        reward_table.loc[record_data, action[record_index]] = float(reward_table.loc[record_data, action[record_index]]) + \
                                                                              self.base_reward
                    else:
                        decay_rate = pow(self.decay_reward, record_index)
                        reward_table.loc[record_data, action[record_index]] = float(reward_table.loc[record_data, action[record_index]]) + \
                                                                              self.base_reward * decay_rate
            
                ## Convert Reward table from dataframe to numpy
                reward_matrix = pd.DataFrame(reward_table).to_numpy()

                episode = 0 ## Initial episode
                state_size = len(main_states_rows)
                q_matrix = np.zeros((state_size, state_size))

                while episode < max_episodes:
                    ## Get starting place(current state)
                    cur_pos = random.randint(0, state_size - 1)
                    step = 0
                    while step < max_episode_steps:
                        ## Get all possible next states from current state(if the reward of action is lower than
                        ## lower_limit or this action can't happen(mark as none in reward table))
                        possible_actions = self.allPossibleNextAction(cur_pos, reward_matrix, state_size)
                        if len(possible_actions) != 0:
                            ## Select any one action randomly from possible_actions
                            action = random.choice(possible_actions)
                            ## Find the next state by random
                            next_state = self.getNextState(state_size)
                            ## Calculate q value by Q learning formula
                            ## Q[state, action] = Q[state, action] + lr * (reward + gamma * np.max(Q[new_state, :]) - Q[state, action])
                            q_matrix[cur_pos][action] = (1 - self.learning_rate) * q_matrix[cur_pos][action] + \
                                                        self.learning_rate * (reward_matrix[cur_pos][action] + \
                                                        self.decay_qvalue * max(q_matrix[next_state, :]) - q_matrix[cur_pos][action])

                            ## Go to next state(step)
                            cur_pos = next_state
                        step += 1
                    episode += 1

                ## Transform numpy array -> list -> dataframe
                qtable = q_matrix.tolist()
                qtable = pd.DataFrame(qtable)
                qtable.columns = main_states_rows
                qtable.index = main_states_rows

                if os.path.exists(self.out_table_path):
                    ## Get old q table
                    old_qtable = pd.read_csv(self.out_table_path, index_col=0)
                    old_main_states_rows = [ _ for _ in old_qtable.columns]

                    ## Find out states have not happened recently(new q table)
                    long_term_state = list()
                    for state in old_main_states_rows:
                        if state not in main_states_rows:
                            long_term_state.append(state)

                    ## Merge old Qvalue to new q table
                    for state in long_term_state:
                        qtable.loc[state] = 0
                        qtable[state] = float(0)
                        for old_state in old_main_states_rows:
                            qtable.loc[state, old_state] = old_qtable.loc[state, old_state]

                self.reward_table_size = [reward_matrix.shape[0], reward_matrix.shape[1]]
                self.reward_table_header = main_states_rows
                self.reward_table = reward_table.values.tolist()
                self.qtable_size = [qtable.shape[0], qtable.shape[1]]
                self.qtable_table_header = [ _ for _ in qtable.columns]
                self.qtable = qtable.values.tolist()
               
                # save to csv
                qtable.to_csv(self.out_table_path, index=True)
                print(qtable)
            else:
                pass
  
    def focus(self, speaks, pos):
        for speak in speaks:
            if speak in self.speak_dict[pos]:
                self.speak_state.append(speak)
                break
            

if __name__ == "__main__":
    x = QLearningUpdata(in_record_path="record/nobody_record_rtest.csv", 
                        out_table_path="q_table/Nobody_qtable.csv", 
                        where_pool=[[2], [1], []],
                        where_category_path="places_recognize/categories_places_uscc.txt",
                        speak_focus_path="nlp_recognize/speak_focus.json",
                        care_number=19, 
                        decay_reward=0.9, 
                        base_reward=100, 
                        lower_limit=10, 
                        decay_qvalue=0.9, 
                        learning_rate=0.1)
    
    x.start()
    
    while True:
        x.getdata()
        # print(np.array(reward_table))

        # if len(reward_table) != 0:
        #     break