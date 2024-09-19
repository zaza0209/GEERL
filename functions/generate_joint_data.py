# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 11:59:40 2023

joint data generating functions
"""
from memory_profiler import profile
from functions.generate_team_competition_data import match_teams, plot_state_trajectories
from collections import namedtuple, deque
import numpy as np
import pickle,os,sys
# from pympler import muppy, summary
import time
# import gc
#%%
class Team:
    def __init__(self, team_id, team_size, reward_function=None, transition_function=None,
                 transition_reward_function=None,
                 init_state=None, instt=False, record_history=True, horizon=7,
                 max_step=10000, weekly_summary="av_end",
                 is_team_training=False, return_trajectories=True,
                 is_training=True,is_indi=True, is_record_all=True,
                 transition_state_type= "daily",
                 gamma=0.9, transition_type="separate"):
        '''
        Team object for data generation

        Parameters
        ----------
        team_id : TYPE
            DESCRIPTION.
        team_size : TYPE
            DESCRIPTION.
        reward_function : TYPE
            DESCRIPTION.
        transition_function : TYPE
            DESCRIPTION.
        init_state : TYPE, optional
            DESCRIPTION. The default is None.
        instt : TYPE, optional
            DESCRIPTION. The default is False.
        record_history : TYPE, optional
            DESCRIPTION. The default is True.
        horizon : TYPE, optional
            DESCRIPTION. The default is 7.
        max_step : TYPE, optional
            DESCRIPTION. The default is 10000.
        weekly_summary : How to summarize the weekly variables for team-level learning.
            Choose from "av_end"(use the averge of the variables at the end of each week), "average" (average variables of a week), "concatenated" (concatenate all average variables during the week)
        is_team_eva: Is the data collected for team evaluation? If True, the daily recordings for previous weeks would be discarded to save memory.
        transition_type: if "separate": generate the reward and next state using separate functions;
            "joint": generate the reward and the next state using a single function.
        
        Returns
        -------
        None.

        '''
        self.team_id = team_id
        self.team_size = team_size
        # self.states = np.random.normal(0, 1, (team_size, 1)) if init_state is None else init_state(team_size)
        # if len(self.states.shape)<2:
        #     self.states = self.states.reshape(team_size, -1)
        self.reward_function = reward_function
        self.transition_function = transition_function
        self.transition_reward_function = transition_reward_function
        
        self.win_history = []  # Added win history list
        self.instt = instt
        self.matched_team=None
        self.transition_type=transition_type
        
        self.record_history = record_history
        # if not self.record_history:
        #     self.av_reward =0
        #     self.dis_reward = 0
        self.max_step=max_step
        self.horizon=horizon
        self.init_state = init_state
        self.is_training = is_training
        if not self.is_training and not is_record_all:
            self.av_reward =0
            self.dis_reward = 0
        # self.is_team = is_team
        self.is_indi=is_indi
        self.is_record_all=is_record_all
        self.gamma=gamma
        if is_team_training:
            self.is_team = True
        self.is_team_training = is_team_training
        self.return_trajectories = return_trajectories
        if weekly_summary not in ['average', 'av_end', 'concatenated',"concat_end", "separate_end"]:
            raise ValueError('Invalid weekly_summary.')
        self.weekly_summary = weekly_summary
        if transition_state_type not in ['daily', 'weekly']:
            raise ValueError('Invalid transition state type.')
        self.transition_state_type = transition_state_type
        
    def reset(self, state0=None, current_weekly_state=None, current_weekly_action=None,
              current_weekly_team_state=None, fixed_init_states=False):
        '''
        pesudo actions=0 for week -1
        '''
        # shape: (i, t)
        self.rewards_history = None
        # shape: (i, t)
        self.actions_history = None
        # shape: (t)
        self.matched_id_history = None
        self.trajectories = []
        self.current_weekly_state = None
        self.step_counts = 0
        self.day_counts=0
        self.week_counts=0
        
        if state0 is None:
            if self.init_state is None:
                self.states = np.random.normal(0, 1, (self.team_size, 1))  
            else:
                # print('init state')
                self.states = self.init_state(self.team_size)
                # print('self.states',self.states)
                
            if len(self.states.shape)<2:
                self.states = self.states.reshape(self.team_size, -1)
                
            tmp_history = self.states.copy().reshape((self.team_size, 1, -1))
            team_state_history = np.mean(self.states, axis=0).reshape(1, -1)
            Mit_history = []
            action_history = []
            for t in range(self.horizon):
                actions = np.random.randint(0, 2, self.team_size) #np.zeros(self.team_size)
                action_history.append(actions)
                if not fixed_init_states:
                    self.update_state(actions, other_team_current_states=np.mean(self.states, axis=0), Tit=np.mean(self.states, axis=0),
                                      competition_indicator=0)
                    # print('self.states',self.states)
                Mit_history.append(np.mean(self.states, axis=0))
                tmp_history = np.concatenate((tmp_history, self.states.reshape(self.team_size, 1, -1)), axis=1)
                team_state_history = np.concatenate((team_state_history, np.mean(self.states, axis=0).reshape(1, -1)), axis=0)
            Mit_history = np.array(Mit_history)
            action_history = np.array(action_history).reshape(self.team_size, -1)
            if self.weekly_summary == "average":
                weekly_state = np.mean(tmp_history[:, -(self.horizon+1):-1 ,:], axis=(0,1))
                weekly_team_state = np.mean(team_state_history[-(self.horizon+1):-1, :], axis=0)
                weekly_action = np.mean(action_history) if action_history.shape[0]>1 else np.random.randint(0, 2, 1) 
                weekly_Mit = np.mean(Mit_history, axis=0)
            elif self.weekly_summary == "av_end":
                weekly_state = np.mean(tmp_history[:, -1, :], axis=0)
                weekly_team_state = team_state_history[-1,  :]
                weekly_action = np.mean(action_history[:, -1]) if action_history.shape[0]>1 else np.mean(np.random.randint(0, 2, 1)) #np.mean(action_history[:, -1])
                # weekly_Mit = None #Mit_history[-1]
            elif self.weekly_summary == "concat_end":
                weekly_state = tmp_history[:, -1, :]
                weekly_action = action_history[:, -1] 
                # weekly_Mit = None
                weekly_team_state = team_state_history[-1,  :]
            elif self.weekly_summary == "concatenate":
                weekly_state = tmp_history[:, -(self.horizon+1):-1 ,:]
                weekly_team_state = team_state_history[-(self.horizon+1):-1, :]
                weekly_action = action_history if action_history.shape[0]>1 else np.random.randint(0, 2, 1) 
                # weekly_Mit = Mit_history
            elif self.weekly_summary == "separate_end":
                weekly_state = tmp_history[:, -1, :]
                weekly_action = action_history[:, -1] 
                weekly_team_state = team_state_history[-1,  :]
                
            self.current_weekly_state = weekly_state
            self.current_weekly_action = weekly_action
            self.current_weekly_team_state =  weekly_team_state
        else:
            self.states = state0
            self.current_weekly_state = current_weekly_state
            self.current_weekly_action = current_weekly_action
            self.current_weekly_team_state = current_weekly_team_state
        # print('finish self.states',self.states)
        return self.states
    
        
    # @profile
    def step(self, actions, other_team, other_team_current_states, other_team_current_weekly_state):
        current_states = self.states
        actions = actions.reshape((-1,1))
        if self.transition_state_type == "daily":
            Mit = other_team_current_states
            Tit = np.mean(self.states, axis=0) 
        elif self.transition_state_type == "weekly":  
            Mit = np.mean(other_team_current_weekly_state, axis=0)
            Tit = np.mean(self.current_weekly_state, axis=0)
        actions = np.atleast_1d(actions)
        if self.transition_type == "separate":
            rewards = self.get_reward(actions, Mit, Tit, competition_indicator=int(self.team_id == other_team.team_id))
            self.update_state(actions, Mit, Tit, competition_indicator=int(self.team_id == other_team.team_id))
        else:
            rewards = self.update_state_get_reward(actions, Mit, Tit, competition_indicator=int(self.team_id == other_team.team_id))
        
        
        self.day_counts+=1 
        self.step_counts+=1
        if self.rewards_history is not None:
            self.states_history = np.concatenate((self.states_history, current_states.reshape(self.team_size, 1, -1)), axis=1)
            self.next_states_history = np.concatenate((self.next_states_history, self.states.reshape(self.team_size, 1, -1)), axis=1)
            self.rewards_history = np.concatenate((self.rewards_history, rewards.reshape(self.team_size, 1)), axis = 1)
            self.actions_history = np.concatenate((self.actions_history, actions.reshape(self.team_size, 1)), axis = 1)
            self.matched_id_history = np.concatenate((self.matched_id_history, np.atleast_1d(other_team.team_id)))
            self.team_state_history = np.concatenate((self.team_state_history, Tit.reshape(1, -1)), axis=0)
            self.Mits_history = np.concatenate((self.Mits_history, Mit.reshape(1, -1)), axis=0)
        else:
            self.states_history = current_states.reshape((self.team_size, 1, -1))
            self.next_states_history = self.states.reshape((self.team_size, 1, -1))
            self.rewards_history = rewards.reshape(self.team_size, 1)
            self.actions_history = actions.reshape(self.team_size, 1)
            self.matched_id_history = np.array([other_team.team_id])
            self.team_state_history = Tit.reshape(1, -1)
            self.Mits_history = Mit.reshape(1, -1)
        '''    
        ## for individual level training, we should record all the historical data
        if (self.is_indi and self.is_training) or self.is_record_all:
            # self.states_history = np.concatenate((self.states_history, self.states.reshape(self.team_size, 1, -1)), axis=1)
            # self.team_state_history = np.concatenate((self.team_state_history, np.mean(self.states, axis=0).reshape(1, -1)), axis=0)
            
            # if self.day_counts != 1 or self.transition_state_type == "daily":
            if self.rewards_history is not None:
                self.states_history = np.concatenate((self.states_history, current_states.reshape(self.team_size, 1, -1)), axis=1)
                self.next_states_history = np.concatenate((self.next_states_history, self.states.reshape(self.team_size, 1, -1)), axis=1)
                self.rewards_history = np.concatenate((self.rewards_history, rewards.reshape(self.team_size, 1)), axis = 1)
                self.actions_history = np.concatenate((self.actions_history, actions.reshape(self.team_size, 1)), axis = 1)
                self.matched_id_history = np.concatenate((self.matched_id_history, np.atleast_1d(other_team.team_id)))
                self.team_state_history = np.concatenate((self.team_state_history, Tit.reshape(1, -1)), axis=0)
                self.Mits_history = np.concatenate((self.Mits_history, Mit.reshape(1, -1)), axis=0)
            else:
                self.states_history = current_states.reshape((self.team_size, 1, -1))
                self.next_states_history = self.states.reshape((self.team_size, 1, -1))
                self.rewards_history = rewards.reshape(self.team_size, 1)
                self.actions_history = actions.reshape(self.team_size, 1)
                self.matched_id_history = np.array([other_team.team_id])
                self.team_state_history = Tit.reshape(1, -1)
                self.Mits_history = Mit.reshape(1, -1)
        '''
        
        # ## individual level evaluation: only need to record the av r and dis r
        # elif self.is_indi and not self.is_training:
        #     self.av_reward = (self.av_reward*(self.step_counts-1) + np.mean(rewards))/self.step_counts
        #     self.dis_reward += self.gamma ** self.step_counts * np.mean(rewards)
        
        ## for team level training, only need to record one-week data
        if not ((self.is_indi and self.is_training) or self.is_record_all):
            # self.states_history = np.concatenate((self.states_history, self.states.reshape(self.team_size, 1, -1)), axis=1)
            # self.team_state_history = np.concatenate((self.team_state_history, np.mean(self.states, axis=0).reshape(1, -1)), axis=0)
            # if self.rewards_history is not None:
            #     self.states_history = np.concatenate((self.states_history, current_states.reshape(self.team_size, 1, -1)), axis=1)
            #     self.next_states_history = np.concatenate((self.next_states_history, self.states.reshape(self.team_size, 1, -1)), axis=1)
            #     self.rewards_history = np.concatenate((self.rewards_history, rewards.reshape(self.team_size, 1)), axis = 1)
            #     self.actions_history = np.concatenate((self.actions_history, actions.reshape(self.team_size, 1)), axis = 1)
            #     self.matched_id_history = np.concatenate((self.matched_id_history, np.atleast_1d(other_team.team_id)))
            #     self.team_state_history = np.concatenate((self.team_state_history, Tit.reshape(1, -1)), axis=0)
            #     self.Mits_history = np.concatenate((self.Mits_history, Mit.reshape(1, -1)), axis=0)
            # else:
            #     self.states_history = current_states.reshape((self.team_size, 1, -1))
            #     self.next_states_history = self.states.reshape((self.team_size, 1, -1))
            #     self.rewards_history = rewards.reshape(self.team_size, 1)
            #     self.actions_history = actions.reshape(self.team_size, 1)
            #     self.matched_id_history = np.array([other_team.team_id])
            #     self.team_state_history = Tit.reshape(1, -1)
            #     self.Mits_history = Mit.reshape(1, -1)
        
            if self.Mits_history.shape[1] > self.horizon:
                self.states_history = self.states_history[:, -(self.horizon):, :]
                self.next_states_history = self.next_states_history[:, -(self.horizon):, :]
                self.team_state_history = self.team_state_history[-(self.horizon):, :]
                self.Mits_history = self.Mits_history[-( self.horizon):, :]
                self.actions_history = self.actions_history.reshape[:, -(self.horizon):]
                self.matched_id_history = self.matched_id_history[-(self.horizon):]
                self.rewards_history = self.rewards_history[:, -(self.horizon):]
        if not self.is_training:
            # rewards
            self.av_reward = (self.av_reward*(self.step_counts-1) + np.mean(rewards))/self.step_counts
            self.dis_reward += self.gamma ** self.step_counts * np.mean(rewards)


        ## weekly summary        
        if self.day_counts == self.horizon:
            # print('week ending')
            self.week_counts+=1
            if self.is_record_all or (self.is_indi and self.transition_state_type=="weekly") or not self.is_indi:
                if self.weekly_summary == "average":
                    raise NotImplementedError('average weekly summary not implemented yet.')
                    # if using this type of summary, the action induced by the learnt policy would be based on the weekly state of the previous weeks
                    weekly_state = np.mean(self.next_states_history[:, -(self.horizon): ,:], axis=(0,1))
                    weekly_action = np.mean(self.actions_history[:, -self.horizon:])
                    weekly_reward = np.sum(self.rewards_history[:, -self.horizon:])
                    weekly_Mit = other_team_current_weekly_state #np.mean(self.Mits_history[-self.horizon:, :])
                    weekly_team_state = np.mean(self.team_state_history[-(self.horizon):, :], axis=0)
                elif self.weekly_summary == "av_end":
                    # weekly_Mit = np.mean(self.Mits_history[-1:, :])
                    weekly_state = np.mean(self.next_states_history[:, -1 , :], axis=0)
                    weekly_action = np.mean(self.actions_history[:, -1])
                    weekly_reward = np.sum(self.rewards_history[:, -self.horizon:])
                    weekly_Mit = self.Mits_history[-self.horizon, :]
                    weekly_team_state = self.team_state_history[-self.horizon, :]
                elif self.weekly_summary == "concat_end":
                    weekly_state = self.next_states_history[:, -1, :]
                    weekly_action = self.actions_history[:, -1] 
                    weekly_reward = np.sum(self.rewards_history[:, -self.horizon:])
                    weekly_Mit = self.Mits_history[-self.horizon, :]
                    weekly_team_state = self.team_state_history[-self.horizon, :]
                    # weekly_team_state = self.team_state_history[-1, :]
                elif self.weekly_summary == "concatenate":
                    weekly_state = self.next_states_history[:,-(self.horizon+1):-1, :] 
                    weekly_action = self.actions_history[:, -self.horizon:]
                    weekly_reward = np.sum(self.rewards_history[:, -self.horizon:])
                    weekly_Mit = self.Mits_history[-self.horizon:, :]
                    weekly_team_state = self.team_state_history[-self.horizon:, :]
                    # weekly_team_state = self.team_state_history[-(self.horizon+1):-1, :]
                elif self.weekly_summary == "separate_end":
                    weekly_state = self.next_states_history[:, -1 , :]
                    weekly_action = self.actions_history[:, -1] 
                    weekly_reward = np.mean(self.rewards_history[:, -self.horizon:], axis=-1)
                    weekly_Mit = self.Mits_history[-self.horizon, :]
                    weekly_team_state = self.team_state_history[-self.horizon, :]
                    # weekly_team_state = self.team_state_history[-1, :]
                
                if self.is_record_all or (self.is_training and not self.is_indi):
                    if self.weekly_summary != "separate_end":
                        self.trajectories.append({
                            'week': self.week_counts,
                            'team_id': self.team_id,
                            'state': self.current_weekly_state,
                            'reward': weekly_reward,
                            'next_state': weekly_state,
                            'matched_with': other_team.team_id,
                            'matching_state': weekly_Mit, #self.current_weekly_Mit , 
                            'weekly_actions': self.current_weekly_action,  #weekly_action, 
                            # 'weekly_team_state': weekly_team_state,
                            'weekly_team_state':self.current_weekly_team_state,
                            'next_weekly_team_state':weekly_team_state,
                            'next_weekly_actions': weekly_action
                        })
                    else:
                        self.trajectories.extend([{
                            'week': self.week_counts,
                            'team_id': self.team_id,
                            'state': self.current_weekly_state[i,:],
                            'reward': weekly_reward[i],
                            'next_state': weekly_state[i],
                            'matched_with': other_team.team_id,
                            'matching_state': weekly_Mit, #self.current_weekly_Mit , 
                            'weekly_actions': self.current_weekly_action[i],  #weekly_action,
                            # 'weekly_team_state': weekly_team_state,
                            'weekly_team_state':self.current_weekly_team_state,
                            'next_weekly_team_state':weekly_team_state,
                            'next_weekly_actions': weekly_action[i],
                            'subject_id':i
                        } for i in range(self.team_size)])
                self.current_weekly_state = weekly_state.copy()
                self.current_weekly_team_state = weekly_team_state.copy()
                self.current_weekly_action = weekly_action.copy()
                # self.current_weekly_Mit = weekly_Mit.copy()
                self.day_counts = 0
                
        return self.states, rewards, bool(self.max_step<self.step_counts)
    
    def update_state_get_reward(self, actions, other_team_current_states, Tit, competition_indicator=None):
        next_week = self.week_counts if self.day_counts+1 <self.horizon else self.week_counts+1
        self.states, rewards = self.transition_reward_function(self.states, actions, other_team_current_states, Tit,
                                               t=self.step_counts,
                                               week=self.week_counts,
                                               competition_indicator = competition_indicator,
                                               next_week = next_week,
                                               next_dow=self.day_counts+1 if self.day_counts+1 < self.horizon else 1)
        return rewards
    
    def get_reward(self, actions, other_team_current_states, Tit, competition_indicator=None):
        # bug: rewards = self.reward_function(self.states, actions, matched_team, self.step_counts)
        rewards = []
        
        for i, (subject_state, action) in enumerate(zip(self.states, actions)):
            reward = self.reward_function(Sijt=subject_state, Aijt=action, t= self.step_counts, i=i, 
                                          matched_team_current_states=other_team_current_states, 
                                          team_current_states=Tit,
                                          week=self.week_counts, 
                                          competition_indicator=competition_indicator,
                                          )
            rewards.append(reward)
        rewards = np.array(rewards)
        return rewards
    
    
    def update_state(self, actions, other_team_current_states, Tit, competition_indicator=None):
        next_week = self.week_counts if self.day_counts+1 <self.horizon else self.week_counts+1
        self.states = self.transition_function(self.states, actions, other_team_current_states, Tit,
                                               t=self.step_counts,
                                               week=self.week_counts,
                                               competition_indicator = competition_indicator,
                                               next_week = next_week,
                                               next_dow=self.day_counts+1 if self.day_counts+1 < self.horizon else 1)
        
            
    def state_combine_friends_fun(self):
        states_tmp=np.empty((0, self.team_size))
        for i in range(self.team_size):
            s = np.concatenate((self.states[i].flatten(), self.states[:i].flatten(), self.states[i+1:].flatten())).reshape(1,-1)
            states_tmp = np.concatenate([states_tmp, s], axis=0)
            
        self.states = states_tmp
            
    def register_match_outcome(self, my_rewards, opponent_rewards):
        self.win_history.extend(my_rewards > opponent_rewards)
        
        
    def process_weekly_summary(self, horizon):
        raise NotImplementedError("process_weekly_summary not implemented")
        # Initialize the list of weekly trajectories
        trajectories = []
        num_weeks = int(len(self.actions_history)/horizon)

        for week in range(num_weeks):
            # Calculate the average state over the week for the team
            start_idx = 1 + week * horizon
            end_idx = min(start_idx + horizon + 1, self.states_history.shape[1])
            weekly_state = np.mean(self.states_history[:, start_idx:end_idx], axis=1)

            # Sum the rewards over the week for the team
            start_idx = 1 + week * horizon
            end_idx = min(start_idx + horizon, self.rewards_history.shape[1])
            weekly_reward = np.mean(self.rewards_history[:, start_idx:end_idx], axis=1)

            # Get the next state, assuming it's the state at the end of the week
            start_idx = 1 + (1 + week) * horizon
            end_idx = min(start_idx + horizon + 1, self.states_history.shape[1])
            next_state = np.mean(self.states_history[:, start_idx:end_idx], axis=1)

            # The following line is just an example and may need modification
            matched_team_id = self.matched_id_history[1:week * horizon]

            start_idx = 1 + week * horizon
            end_idx = min(start_idx + horizon + 1, len(self.Mits_history))
            matching_state = np.mean(self.Mits_history[start_idx:end_idx])

            week_summary = {
                'week': week,
                'team_id': self.team_id,
                'state': weekly_state,
                'reward': weekly_reward,
                'next_state': next_state,
                'matched_with': matched_team_id,
                'matching_state': matching_state
            }
            
            # Append the week summary to the trajectories list
            trajectories.append(week_summary)
        return trajectories

#%%
def centralized_execution(team_a, team_b, individual_policy,
                         ):
    # state_concat = np.concatenate((team_a.states,team_b.states), axis=0).reshape(1,-1)
    action = individual_policy(team_a.states, team_b.states, h = team_a.day_counts)
    actions_a = action[..., :team_a.team_size]
    actions_b = action[..., -team_b.team_size:]
    return actions_a, actions_b

def state_processing_for_actions(team_a, team_b,action_space, individual_policy,
                                current_states_a, team_b_current_weekly_state,
                                state_combine_friends=False,
                                only_states=0, include_Mit=1, include_team_effect=0,
                                instt=0,round_Mit_function=None, epsilon=0, history_length=1,
                                transition_state_type = "weekly", 
                                horizon=None,include_weekend_indicator=0):
    
    # if individual_policy:
    # Initialize or update the history of states_a
    if not hasattr(team_a, 'states_a_history'):
        team_a.states_a_history = deque(maxlen=history_length)

    
    if only_states:
        states_a = team_a.states[...,np.newaxis,:]
    elif state_combine_friends:
        states=np.empty((0, team_a.team_size))
        for i in range(team_a.team_size):
            s = np.concatenate((team_a.states[i].flatten(), team_a.states[:i].flatten(), team_a.states[i+1:].flatten())).reshape(1,-1)
            states = np.concatenate([states, s], axis=0)
        if transition_state_type == "daily":
            Mit = np.mean(team_b.states, axis=0)
        elif transition_state_type == "weekly":
            Mit = np.mean(team_b_current_weekly_state, axis=0)
        if include_Mit:
            states_a = np.concatenate((np.tile(Mit.reshape(1, 1, -2), (team_a.team_size, 1, 1)), states[...,np.newaxis,:]), axis=-1) 
        else:
            states_a = states[...,np.newaxis,:]

    elif include_team_effect:
        if instt:
            states =  team_a.states[...,np.newaxis,:-1]
            instt_a =  team_a.states[...,-1]
            instt_b = np.atleast_1d(team_b.states[0,-1])
            matching_mask = np.equal(instt_a, instt_b[np.newaxis, :])
            matching_indices = matching_mask.astype(int).reshape(-1,1)
            if include_Mit:
                if round_Mit_function:
                    Mits = round_Mit_function(np.mean(team_b.states[...,:-1], axis=0))
                else:
                    Mits = np.mean(team_b.states[...,:-1], axis=0)
                states_a = np.concatenate((np.tile(Mits, (team_a.team_size, 1, 1)), np.tile(current_states_a[...,:-1], (team_a.team_size, 1, 1)), states, matching_indices[...,np.newaxis]), axis=-1) 
            else:
                states_a = np.concatenate((np.tile(current_states_a[...,:-1], (team_a.team_size, 1, 1)), states, matching_indices[...,np.newaxis]), axis=-1) 

        else:
            if transition_state_type == "daily":
                Mit = np.mean(team_b.states, axis=0)
                Tit = current_states_a
            elif transition_state_type == "weekly":
                Mit = np.mean(team_b_current_weekly_state, axis=0)
                Tit = np.mean(team_a.current_weekly_state, axis=0)
            if include_Mit:
                states_a = np.concatenate((np.tile(Mit.reshape(1, 1, -2), (team_a.team_size, 1, 1)),  np.tile(Tit, (team_a.team_size, 1, 1)), team_a.states[...,np.newaxis,:]), axis=-1)
            else:
                states_a = np.concatenate((np.tile(Tit, (team_a.team_size, 1, 1)), team_a.states[...,np.newaxis,:]), axis=-1)
  
    else:
        if instt:
            states =  team_a.states[...,np.newaxis,:-1]
            instt_a =  team_a.states[...,-1]
            instt_b = np.atleast_1d(team_b.states[0,-1])
            matching_mask = np.equal(instt_a, instt_b[np.newaxis, :])
            matching_indices = matching_mask.astype(int).reshape(-1,1)
            if include_Mit:
                if round_Mit_function:
                    Mits = round_Mit_function(np.mean(team_b.states[...,:-1], axis=0))
                else:
                    Mits = np.mean(team_b.states[...,:-1], axis=0)
                states_a = np.concatenate((np.tile(Mits, (team_a.team_size, 1, 1)), states, matching_indices[...,np.newaxis]), axis=-1) 
            else:
                states_a = np.concatenate((states, matching_indices[...,np.newaxis]), axis=-1) 
        else:
            if transition_state_type == "daily":
                Mit = np.mean(team_b.states, axis=0)
            elif transition_state_type == "weekly":
                Mit = np.mean(team_b_current_weekly_state, axis=0)
            if include_Mit:
                states_a = np.concatenate((np.tile(Mit.reshape(1, 1, -2), (team_a.team_size, 1, 1)), team_a.states[...,np.newaxis,:]), axis=-1)
            else:
                states_a = team_a.states[...,np.newaxis,:]
    if transition_state_type == "weekly" and include_weekend_indicator and not only_states:
        states_a = np.concatenate((states_a, np.tile(int(team_a.day_counts == horizon-1), (states_a.shape[0], 1, 1))), axis=2)
    # print('states.shape', states_a.shape)
    # Append the current states_a to the history
    team_a.states_a_history.append(states_a)

    # Concatenate states_a from the history for each subject
    concatenated_states = []
    for i in range(team_a.team_size):
        # Extract the history for the current subject
        subject_history = [team_a.states_a_history[j][i] for j in range(len(team_a.states_a_history))]
    
        # Check if we have fewer observations than history_length
        if len(subject_history) < history_length:
            # Calculate the padding required
            padding_shape = (1,(history_length - len(subject_history)) * states_a.shape[-1])
            # Pad with zeros
            padding = np.zeros(padding_shape)
            subject_history_padded = np.concatenate([padding] + subject_history, axis=1)
        else:
            subject_history_padded = np.concatenate(subject_history, axis=1)
    
        concatenated_states.append(subject_history_padded)
        
    states_a_history_concatenated = np.array(concatenated_states)

    # Reshape for action selection
    states_a_history_concatenated = states_a_history_concatenated.reshape(team_a.team_size, 1, -1)
    # Determine actions
    # if individual_policy:
    # if np.random.uniform() < epsilon:
    #     actions_a = np.random.choice(action_space, team_a.team_size)
    # else:
    actions_a = individual_policy(states_a_history_concatenated)  # Dimension team_size

    # else:
    #     actions_a = np.random.choice(action_space, team_a.team_size) # np.random.binomial(1, 0.5, team_a.team_size)
    
    return actions_a

def individaul_action_execution(team_a, team_b, action_space, 
                                individual_policy, centralized_actions,
                                current_states_a, current_states_b, 
                                team_a_current_weekly_state,
                                team_b_current_weekly_state,
                                state_combine_friends=False,
                                only_states=0, include_Mit=1, include_team_effect=0,
                                instt=0,round_Mit_function=None, epsilon=0, history_length=1,
                                transition_state_type = "weekly", 
                                horizon=None,include_weekend_indicator=0, selfmatched=None):
        
    
    if centralized_actions:
        if np.random.uniform() < epsilon:
            actions_a = np.random.choice(action_space, team_a.team_size) # np.random.binomial(1, 0.5, team_a.team_size)
            actions_b = np.random.choice(action_space, team_b.team_size) # np.random.binomial(1, 0.5, team_a.team_size)
        else:
            actions_a, actions_b = centralized_execution(team_a, team_b, individual_policy)
    elif individual_policy:
        if selfmatched is None:
            raise ValueError('selfmatched information not provided')
        
        if np.random.uniform() < epsilon:
            actions_a = np.random.choice(action_space, team_a.team_size) # np.random.binomial(1, 0.5, team_a.team_size)
        else:
            actions_a = state_processing_for_actions(team_a=team_a, team_b=team_b, 
                                                     action_space=action_space, 
                                                     individual_policy=individual_policy,
                                                     state_combine_friends=state_combine_friends,
                                            current_states_a=current_states_a, team_b_current_weekly_state = team_b_current_weekly_state,
                                            only_states=only_states, include_Mit=include_Mit,
                                            include_team_effect=include_team_effect,
                                            instt=instt,round_Mit_function=round_Mit_function,
                                            epsilon=epsilon,history_length=history_length,
                                            horizon=horizon,
                                            transition_state_type=transition_state_type, include_weekend_indicator=include_weekend_indicator)
        if not selfmatched:
            if np.random.uniform() < epsilon:
                actions_b = np.random.choice(action_space, team_b.team_size) # np.random.binomial(1, 0.5, team_a.team_size)
            else:
                actions_b = state_processing_for_actions(team_a=team_b, team_b=team_a, 
                                                         action_space=action_space, 
                                                         individual_policy=individual_policy,
                                                         state_combine_friends=state_combine_friends,
                                                current_states_a=current_states_b, team_b_current_weekly_state = team_a_current_weekly_state,
                                                only_states=only_states, include_Mit=include_Mit,
                                                include_team_effect=include_team_effect,
                                                instt=instt,round_Mit_function=round_Mit_function,
                                                epsilon=epsilon,history_length=history_length,
                                                horizon=horizon,
                                            transition_state_type=transition_state_type, include_weekend_indicator=include_weekend_indicator)
        else:
            actions_b = None
    else:
        # print('action_space',action_space)
        actions_a = np.random.choice(action_space, team_a.team_size) # np.random.binomial(1, 0.5, team_a.team_size)
        actions_b = np.random.choice(action_space, team_b.team_size) # np.random.binomial(1, 0.5, team_a.team_size)

    return actions_a, actions_b
    
#%%
def run_horizon(teams, horizon, num_weeks, individual_policy=None, 
                epsilon=0,
                instt=False, only_states=0, include_Mit=1, include_team_effect=0,
                state_combine_friends=0,
                action_space=[0,1], no_selfmatch=False,
                team_policy=None, round_Mit_function=None,
                verbose=0, history_length=1, 
                transition_state_type = "weekly", delete_week_end=0,
                include_weekend_indicator=0,  centralized_actions=0):
    if team_policy == None:
        team_policy="complete"
    if team_policy in ["complete", "same_instt", "diff_instt", "self_match"]:
        team_list = match_teams(teams, no_selfmatch=no_selfmatch, random_type=team_policy)
    else:
        if verbose: start=time.time()
        team_list = team_policy(teams)
        if verbose: 
            print('matching elapse', time.time() - start)
            sys.stdout.flush()
    for t in range(horizon):
        #%%
        for i in range(0, len(team_list), 2):
            team_a, team_b = team_list[i], team_list[i+1]
            current_states_a = np.mean(team_a.states, axis=0)
            current_states_b = np.mean(team_b.states, axis=0)
            team_a_current_weekly_state = team_a.current_weekly_state
            team_b_current_weekly_state = team_b.current_weekly_state
            
            selfmatched = bool(team_a == team_b)
            actions_a, actions_b = individaul_action_execution(team_a, team_b,
                                        action_space,  individual_policy, 
                                        centralized_actions, 
                                        current_states_a=current_states_a, 
                                        current_states_b=current_states_b, 
                                        team_a_current_weekly_state=team_a_current_weekly_state, 
                                        team_b_current_weekly_state= team_b_current_weekly_state,
                                        only_states=only_states, include_Mit=include_Mit,
                                        include_team_effect=include_team_effect,
                                        instt=instt,round_Mit_function=round_Mit_function,
                                        epsilon=epsilon,history_length=history_length,
                                        horizon=horizon,
                                        transition_state_type=transition_state_type, 
                                        include_weekend_indicator=include_weekend_indicator,
                                        selfmatched=selfmatched)
            team_a.step(actions_a, team_b, current_states_b, team_b_current_weekly_state)      
            if not selfmatched:
                # print("==== team_b")
                team_b.step(actions_b, team_a, current_states_a, team_a_current_weekly_state)

#%%
# Simulation function to generate the new data
# @profile
def simulate_teams(teams, num_weeks, individual_policy=None, 
                   epsilon=0,
                   instt=False, only_states=0, include_Mit=1,include_team_effect=0,
                   state_combine_friends=0,
                   action_space=[0,1], no_selfmatch=False,
                   team_policy=None, save_path=None, round_Mit_function=None,
                   is_online=False, 
                   verbose=0, history_length=1, horizon_eva=None,
                   episode = 1, transition_state_type = "weekly", delete_week_end=0,
                   include_weekend_indicator=0, burnin=0, reset_teams=True,
                   centralized_actions=0, burnin_individual_policy=None, burnin_team_policy=None):
    '''
    generate data for individual-level and team-level learning or evaluation

    Parameters
    ----------
    teams : TYPE
        DESCRIPTION.
    num_weeks : TYPE
        DESCRIPTION.
    individual_policy : TYPE, optional
        DESCRIPTION. The default is None.
    epsilon : TYPE, optional
        DESCRIPTION. The default is 0.
    instt : TYPE, optional
        DESCRIPTION. The default is False.
    only_states : TYPE, optional
        DESCRIPTION. The default is 0.
    include_team_effect : TYPE, optional
        DESCRIPTION. The default is 0.
    action_space : TYPE, optional
        DESCRIPTION. The default is [0,1].
    team_policy : TYPE, optional
        DESCRIPTION. The default is None.
    save_path : TYPE, optional
        DESCRIPTION. The default is None.
    round_Mit_function : TYPE, optional
        DESCRIPTION. The default is None.
    is_online : Whether the data is to be used for online training. 
        The default is False. If Fasle, the frist element in trajectories for each team will not be returned (generated by random matching policy).
    episode: how many episode of data to collect
    Returns
    -------
    None.

    '''
    trajectories=[]
    rollouts = []
    if horizon_eva:
        horizon=horizon_eva
    else:
        horizon = teams[0].horizon
    if team_policy == None:
        team_policy="complete"
        
    # play for several episodes
    for _ in range(episode):
        
        if reset_teams:
            for team in teams:
                _ = team.reset()
        if burnin > 0:
            if burnin > horizon: # and centralized_actions:
                for week in range(int(burnin/horizon)):
                    run_horizon(teams, horizon=horizon, num_weeks=num_weeks, 
                                individual_policy=burnin_individual_policy, epsilon=epsilon, instt=instt, 
                                only_states=only_states, include_Mit=include_Mit, include_team_effect=include_team_effect,
                                state_combine_friends=state_combine_friends, action_space=action_space,
                                no_selfmatch=no_selfmatch, team_policy=burnin_team_policy,
                                round_Mit_function=round_Mit_function,
                                verbose=verbose, history_length=history_length, 
                                transition_state_type = transition_state_type, delete_week_end = delete_week_end,
                                include_weekend_indicator = include_weekend_indicator,
                                centralized_actions=centralized_actions)
       
            else:
                run_horizon(teams, horizon=burnin, num_weeks=num_weeks, 
                            individual_policy=burnin_individual_policy, epsilon=epsilon, instt=instt, 
                            only_states=only_states, include_Mit=include_Mit, include_team_effect=include_team_effect,
                            state_combine_friends=state_combine_friends, action_space=action_space,
                            no_selfmatch=no_selfmatch, team_policy=burnin_team_policy,
                            round_Mit_function=round_Mit_function,
                            verbose=verbose, history_length=history_length, 
                            transition_state_type = transition_state_type, delete_week_end = delete_week_end,
                            include_weekend_indicator = include_weekend_indicator,
                            centralized_actions=centralized_actions)
            for team in teams:
                _ = team.reset(state0 = team.states, current_weekly_state = team.current_weekly_state, 
                               current_weekly_action = team.current_weekly_action, current_weekly_team_state=team.current_weekly_team_state)
            #%%
        for week in range(num_weeks):
            if verbose: 
                print('week', week)
                sys.stdout.flush()
            run_horizon(teams, horizon=horizon, num_weeks=num_weeks, 
                        individual_policy=individual_policy, epsilon=epsilon, instt=instt, 
                        only_states=only_states, include_Mit=include_Mit, include_team_effect=include_team_effect,
                        state_combine_friends=state_combine_friends, action_space=action_space,
                        no_selfmatch=no_selfmatch, team_policy=team_policy,
                        round_Mit_function=round_Mit_function,
                        verbose=verbose, history_length=history_length, 
                        transition_state_type = transition_state_type, delete_week_end = delete_week_end,
                        include_weekend_indicator = include_weekend_indicator,
                        centralized_actions=centralized_actions)
        #%%
        ## states shape: (i, t, p)
        # rewards shape: (i, t)
        # Mit shape: (t, p)
        # actions shape: (i, t)
        # matched id shape: (t)
        for team in teams:
            # if collect_for_training:
            trajectories.extend(team.trajectories)
            # if not is_team_eva:
            if hasattr(team, "av_reward"):
                res={
                  "av_rewards": team.av_reward,
                  "dis_rewards":team.dis_reward
                 }
            else:
                # res={
                #     "team_id":team.team_id,
                #  "states": team.states_history[:, :-1, :],
                #  "actions": team.actions_history[:, :-1],
                #  "rewards": team.rewards_history[:, :-1],
                #  "next_states": team.next_states_history[:, :-1, :],
                #  "Mits":team.Mits_history[:-1,:],
                #  "next_Mits":team.Mits_history[1:,:],
                #  'team_states':team.team_state_history[:-1,:],
                #  'next_team_states':team.team_state_history[1:,:],
                #  "matched_id":team.matched_id_history[:-1]
                #  }
                res={
                    "team_id":team.team_id,
                 "states": team.states_history,
                 "actions": team.actions_history,
                 "rewards": team.rewards_history,
                 "next_states": team.next_states_history,
                 "Mits":team.Mits_history,
                 "next_Mits":team.Mits_history,
                 'team_states':team.team_state_history,
                 'next_team_states':team.team_state_history,
                 "matched_id":team.matched_id_history
                 }
            rollouts.append(res)
    #%%
    if save_path is not None:
        max_team_size = 0
        for team in teams:
            if max_team_size < team.team_size:
                max_team_size = team.team_size
        # Save rollouts and trajectories using pickle
        if not os.path.exists(save_path):
            os.makedirs(save_path, exist_ok=True)
        if len(rollouts)>1:
            with open(save_path+'/rollouts.pkl', 'wb') as f:
                pickle.dump({"rollouts":rollouts,
                             "max_team_size":max_team_size}, f)
        if len(trajectories)>1:
            with open(save_path+'/trajectories.pkl', 'wb') as f:
                pickle.dump({"trajectories": trajectories,
                             "max_team_size":max_team_size
                             }, f)
        
    return rollouts, trajectories
