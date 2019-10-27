"""
    AI Project by Florian Kleinicke
    Q-learning for Snake in an OpenAI/PLE environment
    For the lecture Artificial Intelligence by Prof. Dr. Björn Ommer, SS 2017/18, Heidelberg

    Code basis source: https://github.com/AndersonJo/dqn-pytorch/blob/master/dqn.py
    I took this q learning basis, applied it on snake and modified it in multiple areas. F.e. I added additional information (about distance to food) into the neural network)
"""

"""
    In this file all the global parameters are set and the agent is started.
"""

import dqn_agent

import argparse
import logging
from random import random

import numpy as np
import torch
from gym.wrappers import Monitor
from torch import nn, optim


#Cuda
use_cuda=False
use_cuda = torch.cuda.is_available()
print("Cuda:",use_cuda)

# Training
BATCH_SIZE = 32

# Replay Memory
REPLAY_MEMORY = 50000

# Epsilon
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 100000

# ETC Options
TARGET_UPDATE_INTERVAL = 1000
CHECKPOINT_INTERVAL = 50000
PLAY_INTERVAL = 900
PLAY_REPEAT = 1
#LEARNING_RATE = 0.0001

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1','True'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0','False'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser(description='DQN Configuration')
parser.add_argument('--model', default='dqn', type=str, help='forcefully set step')
parser.add_argument('--step', default=None, type=int, help='forcefully set step')
parser.add_argument('--best', default=None, type=int, help='forcefully set best')
parser.add_argument('--load_latest', dest='load_latest', action='store_true', help='load latest checkpoint')
parser.add_argument('--no_load_latest', dest='load_latest', action='store_false', help='train from the scrach')
parser.add_argument('--checkpoint', default=None, type=str, help='specify the checkpoint file name')
parser.add_argument('--mode', dest='mode', default='train', type=str, help='play, train or inspect')
parser.add_argument('--game', default='Snake-v0', type=str, help='only Pygames are supported')
parser.add_argument('--clip', dest='clip', action='store_true', help='clipping the delta between -1 and 1')
parser.add_argument('--noclip', dest='clip', action='store_false', help='not clipping the delta')
parser.add_argument('--skip_action', default=4, type=int, help='Skipping actions') #4
parser.add_argument('--record', dest='record', action='store_true', help='Record playing a game')
parser.add_argument('--inspect', dest='inspect', action='store_true', help='Inspect CNN')
parser.add_argument('--seed', default=111, type=int, help='random seed')
parser.add_argument('--lr', dest='lr',default=0.0001, type=float, help='learning_rate')
parser.add_argument('--folder',dest='folder', default='7', type=str, help='change foldername addtion')
parser.add_argument('--add', dest='add',default=False, type=str2bool, help='should additonal features be used')
parser.add_argument('--maxIter', dest='maxIter',default=3e6, type=float, help='max number of iterations')


parser.set_defaults(clip=True, load_latest=True, record=False, inspect=False)
parser: argparse.Namespace = parser.parse_args()

LEARNING_RATE = parser.lr

GLOBALFOLDERNAME=f'dqn_checkpoints_{parser.folder}_{parser.add}_{parser.model}_{parser.lr}'


GLOBALACTIVATEADDITIONAL=parser.add

# Random Seed
torch.manual_seed(parser.seed)
if use_cuda:
    torch.cuda.manual_seed(parser.seed)
np.random.seed(parser.seed)

# Logging
logger = logging.getLogger('DQN')
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(message)s')

file_handler = logging.FileHandler(f'dqn_{parser.folder}_{parser.add}_{parser.model}_{parser.lr}.log')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)


def main(parser):
    #usage: python dqn_snake.py --add False --folder 7 --lr 0.0001 --model dqn

    print("Settings",GLOBALFOLDERNAME,GLOBALACTIVATEADDITIONAL,LEARNING_RATE)
    agent = dqn_agent.Agent(parser)

    if parser.load_latest and not parser.checkpoint:
        agent.load_latest_checkpoint()
    elif parser.checkpoint:
        agent.load_checkpoint(parser.checkpoint)
    print(parser.mode.lower())
    if  parser.mode.lower()== 'play':
        agent.play()
    elif parser.mode.lower() == 'train':
        agent.train()
    elif parser.mode.lower() == 'inspect':
        agent.inspect()


if __name__ == '__main__':
    main(parser)
