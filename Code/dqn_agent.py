"""
    AI Project by Florian Kleinicke
    Q-learning for Snake in an OpenAI/PLE environment

    The agent. Here the models and the enviroment get also initilized.
"""

import dqn_replayMemory as rpl
import dqn_model as dqnmodel
import dqn_enviroment as dqnenv

#get the in dqn_snake defined variables
#should have passed them over as args
from dqn_snake import BATCH_SIZE,EPSILON_START,EPSILON_END,EPSILON_DECAY,TARGET_UPDATE_INTERVAL,CHECKPOINT_INTERVAL
from dqn_snake import PLAY_INTERVAL,PLAY_REPEAT,LEARNING_RATE,GLOBALFOLDERNAME,GLOBALACTIVATEADDITIONAL
from dqn_snake import use_cuda,logger

import argparse
import copy
import glob
import logging
import math
import os
import re
from collections import deque

from random import random
import random as rnd
import time

import numpy as np
import pylab
import torch

from torch import optim
from torch.autograd import Variable
from torch.nn import functional as F


class Agent(object):
    def __init__(self, args: argparse.Namespace, cuda=use_cuda, action_repeat: int = 4):
        print("add is",GLOBALACTIVATEADDITIONAL)
        # Init
        #todo: move these to the argument list
        self.folderName=GLOBALFOLDERNAME#"dqn_checkpoints6_normal"
        print("Startig with experiment ",self.folderName)
        self.activateAdditional=GLOBALACTIVATEADDITIONAL#True
        self.clip: bool = args.clip
        self.seed: int = args.seed
        self.action_repeat: int = action_repeat
        self.frame_skipping: int = args.skip_action
        self._state_buffer = deque(maxlen=self.action_repeat)
        self._additional_state_buffer = deque(maxlen=self.action_repeat)
        self.step = 0
        self.best_score = args.best or -10000
        self.best_count = 0
        self.starttime=time.time()
        self.lasttime=self.starttime
        self.maxIter=args.maxIter

        self._play_steps = deque(maxlen=5)

        # Environment
        self.env = dqnenv.Environment(args.game, record=args.record, seed=self.seed,activateAdditional=self.activateAdditional,videoFolder=self.folderName)
        self.env.n_action=4
        #self.env.action_space.n_
        # DQN Model
        self.dqn_hidden_state = self.dqn_cell_state = None
        self.target_hidden_state = self.target_cell_state = None

        self.mode: str = args.model.lower()
        if self.mode == 'dqn':
            self.dqn: DQN = dqnmodel.DQN(self.env.action_space,self.activateAdditional)
        elif self.mode == 'smaller':
            self.dqn: DQN = dqnmodel.DQN_smaller(self.env.action_space,self.activateAdditional)
        elif self.mode == 'verysmall':
            self.dqn: DQN = dqnmodel.DQN_verysmall(self.env.action_space,self.activateAdditional)
        elif self.mode == 'tiny':
            self.dqn: DQN = dqnmodel.DQN_tiny(self.env.action_space,self.activateAdditional)
        elif self.mode == 'other':
            self.dqn: DQN = dqnmodel.DQN_other(self.env.action_space,self.activateAdditional)
        elif self.mode == 'big':
            self.dqn: DQN = dqnmodel.DQN_big(self.env.action_space,self.activateAdditional)
        elif self.mode == 'big2':
            self.dqn: DQN = dqnmodel.DQN_big2(self.env.action_space,self.activateAdditional)
        elif self.mode == 'connected':
            self.dqn: DQN = dqnmodel.DQN_connected(self.env.action_space,self.activateAdditional)
        elif self.mode == 'test1':
            self.dqn: DQN = dqnmodel.DQN_test1(self.env.action_space,self.activateAdditional)
        elif self.mode == 'test2':
            self.dqn: DQN = dqnmodel.DQN_test2(self.env.action_space,self.activateAdditional)
        elif self.mode == 'insp':
            self.dqn: DQN = dqnmodel.DQN_insp(self.env.action_space,self.activateAdditional)
        elif self.mode == 'fully':
            self.dqn: DQN = dqnmodel.DQN_fully(self.env.action_space,self.activateAdditional)
        elif self.mode == 'frameskip':
            self.dqn: DQN = dqnmodel.DQN_frameskip(self.env.action_space,self.activateAdditional)

        else:
            print("Error: didn't understood modelname")
        #    self.dqn: DQN = DQN(self.env.action_space,self.activateAdditional)

        if cuda:
            self.dqn.cuda()

        # DQN Target Model
        self.target: DQN = copy.deepcopy(self.dqn)

        # Optimizer
        self.optimizer = optim.Adam(self.dqn.parameters(), lr=LEARNING_RATE)

        # Replay Memory
        self.replay = rpl.ReplayMemory()

        # Epsilon
        self.epsilon = EPSILON_START

    def select_action(self, states, additional_states):
        #(self, states: np.array, additional_states: np.array) -> tuple:

        # Decrease epsilon value
        self.epsilon = EPSILON_END + (EPSILON_START - EPSILON_END) * \
                                     math.exp(-1. * self.step / EPSILON_DECAY)

        #print("select action")
        #print(additional_states)
        if self.epsilon > random()*1e12: #todo: 1e12 entfernen
            # Random Action

            sample_action = rnd.randint(0, 3)#np.random.choice(range(4)) random.randrange(0, 3.5, 1)#
            #sample_action = self.env.game.action_space.sample()
            action = torch.LongTensor([[sample_action]])
            return action


        states = states.reshape(1, self.action_repeat, self.env.width, self.env.height)
        if use_cuda:
            states_variable: Variable = Variable(torch.FloatTensor(states).cuda())
            additional_states_variable: Variable = Variable(torch.FloatTensor(additional_states).cuda())
        else:
            states_variable: Variable = Variable(torch.FloatTensor(states))
            additional_states_variable: Variable = Variable(torch.FloatTensor(additional_states))

        #if self.mode == 'dqn':
        states_variable.volatile = True
            #print(additional_states_variable)
            #print(additional_states_variable.data.cpu().numpy())
        action = self.dqn(states_variable,additional_states_variable).data.cpu().max(1)[1]
            ###print("action",action)

        #print("action",action)
        return action


    def get_initial_states(self):
        state = self.env.reset()
        state,additional_state = self.env.get_screen()
        states = np.stack([state for _ in range(self.action_repeat)], axis=0)
        additional_states = np.stack([additional_state for _ in range(self.action_repeat)], axis=0)
        #additional_states = additional_state#np.stack([additional_state for _ in range(self.action_repeat)], axis=0)

        self._state_buffer = deque(maxlen=self.action_repeat)
        self._additional_state_buffer = deque(maxlen=self.action_repeat)
        for _ in range(self.action_repeat):
            self._state_buffer.append(state)
            self._additional_state_buffer.append(additional_state)
        return states,additional_states

    def add_state(self, state,additional_state):
        self._state_buffer.append(state)
        self._additional_state_buffer.append(additional_state)

    def recent_states(self):
        return np.array(self._state_buffer),np.array(self._additional_state_buffer)

    def print_states(self, filename,score,count):
        dirpath = os.path.dirname(filename)

        if not os.path.exists(dirpath):
            os.mkdir(dirpath)

        fh = open(filename, "a")
        fh.write(f'{self.step}, {score}, {count}, {time.time()-self.starttime}\n')
        fh.close

    def train(self, gamma: float = 0.99, mode: str = 'rgb_array'):
        # Initial States
        reward_sum = 0.
        q_mean = 0.
        target_mean = 0.
        print("maximum number of iterations",self.maxIter)
        #stops after approx maxIter steps, you could also stop after certain time.
        #time.time()-self.starttime<1e5
        while self.step<self.maxIter+2:
            states,additional_states = self.get_initial_states()
            losses = []
            checkpoint_flag = False
            target_update_flag = False
            play_flag = False
            play_steps = 0
            real_play_count = 0
            real_score = 0

            reward = 0
            done = False

            #performs one game
            while True:
                self.step
                # Get Action
                #print("additional_states",additional_states)
                action: torch.LongTensor = self.select_action(states,additional_states).view(1,1)

                for _ in range(self.frame_skipping):
                    observation, reward, done, info = self.env.step(action[0, 0])
                    #gets the new state and append three older
                    next_state,next_additional_state = self.env.get_screen()
                    self.add_state(next_state,next_additional_state)

                    if done:
                        break

                # Store the infomation in Replay Memory
                next_states,next_additional_states = self.recent_states()
                if done:
                    self.replay.put(states,additional_states, action, reward, None,None)
                else:
                    self.replay.put(states,additional_states, action, reward, next_states,next_additional_states)

                # Change States
                states = next_states
                additional_states=next_additional_states
                # Optimize
                ##print("memory",len(self.replay.memory))
                if self.replay.is_available():
                    #print(self.replay.memory)
                    #print(("before",self.dqn.affine1.weight[:,1024:]))
                    loss, reward_sum, q_mean, target_mean = self.optimize(gamma)
                    #print(("after",self.dqn.affine1.weight[:,1024:]))
                    #time.sleep(5)
                    #if math.isnan(loss):
                    #print("nan loss?",loss)
                    losses.append(loss[0])

                if done:
                    break

                # Increase step
                self.step += 1
                play_steps += 1

                # Target Update
                if self.step % TARGET_UPDATE_INTERVAL == 0:
                    self._target_update()
                    target_update_flag = True
                # Checkpoint
                if self.step % CHECKPOINT_INTERVAL == 0:
                    self.save_checkpoint(filename=f'{self.folderName}/chkpoint_{self.mode}_{self.step}.pth.tar')
                    checkpoint_flag = True


                # Play
                if self.step % PLAY_INTERVAL == 0:
                    play_flag = True

                    scores = []
                    counts = []
                    for _ in range(PLAY_REPEAT):
                        score, real_play_count = self.play(logging=False, human=False)
                        scores.append(score)
                        counts.append(real_play_count)
                        logger.debug(f'[{self.step}] [Validation] play_score: {score}, play_count: {real_play_count}')
                    real_score = int(np.mean(scores))
                    real_play_count = int(np.mean(counts))

                    self.print_states(
                        filename=f'{self.folderName}/scores.txt',score=real_score,count=real_play_count)

                    if self.best_score <= real_score:
                        self.print_states(
                            filename=f'{self.folderName}/bestscores.txt',score=real_score,count=real_play_count)
                        self.best_score = real_score
                        self.best_count = real_play_count
                        logger.debug(f'[{self.step}] [CheckPoint] Play: {self.best_score} [Best Play] [checkpoint]')
                        self.save_checkpoint(
                            filename=f'{self.folderName}/chkpoint_{self.mode}_{self.best_score}.pth.tar')


            self._play_steps.append(play_steps)

            # Return Play results
            if play_flag:
                play_flag = False
                logger.info(f'[{self.step}] [Validation] mean_score: {real_score}, mean_play_count: {real_play_count}')

            # Logging
            mean_loss = np.mean(losses)
            if math.isnan(mean_loss):
                mean_loss=-1.

            target_update_msg = '  [target updated]' if target_update_flag else ''
            save_msg = '  [checkpoint!]' if checkpoint_flag else ''
            spendTime=time.time()-self.starttime
            stepTime=time.time()-self.lasttime
            self.lasttime=time.time()

            logger.info(f'[{self.step}] totalTime: {spendTime:<8.4} stepTime: {stepTime:<8.4}'
                        f'Loss:{mean_loss:<8.4} Play:{play_steps:<3}  AvgPlay:{self.play_step:<4.3} ' #
                        f'RewardSum:{reward_sum:<3} Q:[{q_mean:<6.4}] '
                        f'T:[{target_mean:<6.4}] '
                        f'Epsilon:{self.epsilon:<6.4}{target_update_msg}')

    def optimize(self, gamma: float):

        # Get Sample
        transitions = self.replay.sample(BATCH_SIZE)

        # Mask
        if use_cuda:
            non_final_mask = torch.ByteTensor(list(map(lambda ns: ns is not None, transitions.next_state))).cuda()
            final_mask = 1 - non_final_mask

            state_batch: Variable = Variable(torch.cat(transitions.state).cuda())
            additional_state_batch: Variable = Variable(torch.cat(transitions.additional_state).cuda())
            action_batch: Variable = Variable(torch.cat(transitions.action).cuda())
            reward_batch: Variable = Variable(torch.cat(transitions.reward).cuda())
            non_final_next_state_batch = Variable(torch.cat([ns for ns in transitions.next_state if ns is not None]).cuda())
            non_final_next_state_batch.volatile = True
            non_final_next_additional_state_batch = Variable(torch.cat([ns for ns in transitions.next_additional_state if ns is not None]).cuda())
            non_final_next_additional_state_batch.volatile = True

        else:
            non_final_mask = torch.ByteTensor(list(map(lambda ns: ns is not None, transitions.next_state)))
            #print(non_final_mask)
            #print(np.shape(non_final_mask))
            final_mask = 1 - non_final_mask

            state_batch: Variable = Variable(torch.cat(transitions.state))
            additional_state_batch: Variable = Variable(torch.cat(transitions.additional_state))
            action_batch: Variable = Variable(torch.cat(transitions.action))
            reward_batch: Variable = Variable(torch.cat(transitions.reward))
            non_final_next_state_batch = Variable(torch.cat([ns for ns in transitions.next_state if ns is not None]))
            non_final_next_state_batch.volatile = True
            non_final_next_additional_state_batch = Variable(torch.cat([ns for ns in transitions.next_additional_state if ns is not None]))
            non_final_next_additional_state_batch.volatile = True

        # Reshape States and Next States
        state_batch = state_batch.view([BATCH_SIZE, self.action_repeat, self.env.width, self.env.height])
        non_final_next_state_batch = non_final_next_state_batch.view(
            [-1, self.action_repeat, self.env.width, self.env.height])
        non_final_next_state_batch.volatile = True
        #print(np.shape(state_batch))
        #print(np.shape(additional_state_batch))
        additional_state_batch = additional_state_batch.view([BATCH_SIZE, self.action_repeat,self.env.additional])
        non_final_next_additional_state_batch = non_final_next_additional_state_batch.view(
            [-1, self.env.additional])
        non_final_next_additional_state_batch.volatile = True

        # Clipping Reward between -2 and 2
        reward_batch.data.clamp_(-1, 1)

        # Predict by DQN Model
        #if self.mode == 'dqn':
        q_pred = self.dqn(state_batch,additional_state_batch)
        #print("gather action_batch", np.shape(action_batch)) 32x1?
        q_values = q_pred.gather(1, action_batch)

        # Predict by Target Model
        if use_cuda:
            target_values = Variable(torch.zeros(BATCH_SIZE, 1).cuda())
        else:
            target_values = Variable(torch.zeros(BATCH_SIZE, 1))
        #if self.mode == 'dqn':
        target_pred = self.target(non_final_next_state_batch,non_final_next_additional_state_batch)


        #todo: lerne weitere parameter von qlearning -> already done?
        target_values[non_final_mask] = reward_batch[non_final_mask] + target_pred.max(1)[0] * gamma
        target_values[final_mask] = reward_batch[final_mask].detach()

        loss = F.smooth_l1_loss(q_values, target_values)

        # loss = torch.mean((target_values - q_values) ** 2)
        self.optimizer.zero_grad()
        loss.backward(retain_variables=True)

        if self.clip:
            for param in self.dqn.parameters():
                param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

        reward_score = int(torch.sum(reward_batch).data.cpu().numpy()[0])
        q_mean = torch.sum(q_pred, 0).data.cpu().numpy()[0]
        target_mean = torch.sum(target_pred, 0).data.cpu().numpy()[0]

        return loss.data.cpu().numpy(), reward_score, q_mean, target_mean

    def _target_update(self):
        self.target = copy.deepcopy(self.dqn)

    def save_checkpoint(self, filename='{self.folderName}/checkpoint.pth.tar'):
        dirpath = os.path.dirname(filename)

        if not os.path.exists(dirpath):
            os.mkdir(dirpath)

        #fh = open(“dirpath/scores.txt”, “a”)
        #fh.write(self.step,self.best_score,self.best_count)
        #fh.close

        checkpoint = {
            'dqn': self.dqn.state_dict(),
            'target': self.target.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'step': self.step,
            'best': self.best_score,
            'best_count': self.best_count
        }
        torch.save(checkpoint, filename)

    def load_checkpoint(self, filename='{self.folderName}/checkpoint.pth.tar', epsilon=None):
        if use_cuda:
            checkpoint = torch.load(filename)
        else:
            checkpoint = torch.load(filename, map_location=lambda storage, loc: storage)
        self.dqn.load_state_dict(checkpoint['dqn'])
        self.target.load_state_dict(checkpoint['target'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.step = checkpoint['step']
        self.best_score = self.best_score or checkpoint['best']
        self.best_count = checkpoint['best_count']

    def load_latest_checkpoint(self, epsilon=None):
        r = re.compile('chkpoint_(.*)_(?P<number>-?\d+)\.pth\.tar$')
        #r = re.compile('chkpoint_(dqn)_(?P<number>-?\d+)\.pth\.tar$')

        files = glob.glob(f'{self.folderName}/chkpoint_{self.mode}_*.pth.tar')
        #files = glob.glob(f'chkpoint_{self.mode}_*.pth.tar')
        #print(r)
        print(files)
        if files:
            files = list(map(lambda x: [int(r.search(x).group('number')), x], files))
            files = sorted(files, key=lambda x: x[0])
            latest_file = files[-1][1]
            self.load_checkpoint(latest_file, epsilon=epsilon)
            print(f'latest checkpoint has been loaded - {latest_file}')
        else:
            print('no latest checkpoint for {self.folderName}')

    def play(self, logging=True, human=True):
        observation = self.env.game.reset()
        states,additional_states = self.get_initial_states()
        count = 0
        total_score = 0

        self.env.game.seed(self.seed)

        while True:
            # screen = self.env.game.render(mode='human')

            states = states.reshape(1, self.action_repeat, self.env.width, self.env.height)
            if use_cuda:
                states_variable: Variable = Variable(torch.FloatTensor(states).cuda())
                additional_states_variable: Variable = Variable(torch.FloatTensor(additional_states).cuda())
            else:
                states_variable: Variable = Variable(torch.FloatTensor(states))
                additional_states_variable: Variable = Variable(torch.FloatTensor(additional_states))


            #if self.mode == 'dqn':
            dqn_pred = self.dqn(states_variable,additional_states_variable)
            #print(states_variable,additional_states_variable)
            action = dqn_pred.data.cpu().max(1)[1].view(1, 1)[0, 0]

            for i in range(self.frame_skipping):
                if human:
                    screen = self.env.game.render(mode='human')
                observation, reward, done, info = self.env.step(action)
                # States <- Next States
                next_state,additional_state = self.env.get_screen()
                self.add_state(next_state,additional_state)
                states,additional_states = self.recent_states()
                ###print(i,additional_states)

                total_score += reward

                if done:
                    break

            # Logging
            count += 1
            if logging:
                action_dist = torch.sum(dqn_pred, 0).data.cpu().numpy()[0]
                print(f'[{count}] action:{action} {action_dist}, reward:{reward}')

            if done:
                break
        self.env.game.close()
        return total_score, count

    def inspect(self):


        print(dir(self.dqn.conv1))

        for param in list(self.dqn.parameters()):
            print(param.size())

        print("conv2",self.dqn.conv2.kernel_size)
        print("conv3",self.dqn.conv3.kernel_size)
        print("lin1",self.dqn.affine1.in_features,self.dqn.affine1.out_features)
        print("lin2",self.dqn.affine2.in_features,self.dqn.affine2.out_features)
        weights=self.dqn.affine1.weight
        print(self.dqn.affine1.weight)
        print((self.dqn.affine1.weight[:,1024:]))
        import visualize
        inputs = torch.randn(1,4,64,64)
        inputs2=additional_states=torch.zeros(4,12)
        print("state",self.dqn.state_dict)
        y=self.dqn(Variable(inputs),Variable(inputs2))
        g=visualize.make_dot(y,self.dqn.state_dict)
        g.view()
        #for k, v in self.dqn.state_dict().iteritems():
        #    print("Layer {}".format(k))
        #    print(v)

    #todo: delete those?
    @property
    def play_step(self):
        return np.nan_to_num(np.mean(self._play_steps))
"""
    def _sum_params(self, model):
        return np.sum([torch.sum(p).data[0] for p in model.parameters()])

    def imshow(self, sample_image: np.array, transpose=False):
        if transpose:
            sample_image = sample_image.transpose((1, 2, 0))
        pylab.imshow(sample_image, cmap='gray')
        pylab.show()

    def toimage(self, image: np.array, name: str):
        toimage(image, cmin=0, cmax=255).save(name)
"""
