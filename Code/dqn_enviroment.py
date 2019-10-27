"""
    AI Project by Florian Kleinicke
    Q-learning for Snake in an OpenAI/PLE environment
"""

import gym
import gym_ple
from random import sample
import numpy as np
from gym.wrappers import Monitor
from torchvision import transforms as T



"""
    The enviroment of the snake game is here initilized
"""
class Environment(object):
    def __init__(self, game, record=False, width=64, height=64, seed=0,additional=12,activateAdditional=True,videoFolder="0/"):
        self.activateAdditional=activateAdditional
        self.game = gym.make(game)
        self.game.seed(seed)
        print("record",record)
        if record:
            print("record")
            self.game = Monitor(self.game, f'./videos/{videoFolder}', force=True)

        self.width = width
        self.height = height
        self.additional=additional
        self._toTensor = T.Compose([T.ToPILImage(), T.ToTensor()])
        gym_ple

    def play_sample(self, mode: str = 'human'):
        observation = self.game.reset()

        while True:
            screen = self.game.render(mode=mode)
            if mode == 'rgb_array':
                screen,_ = self.preprocess(screen)
            action = self.game.action_space.sample()
            observation, reward, done, info = self.game.step(action)
            if done:
                break
        self.game.close()

    """
        looks if the snake is able to go into the desired direction. If not, another guess is calculated.
    """
    def desiredmove(self,head, nextstep):
        if head==0:
            if nextstep is not 3:
                return nextstep
            else:
                return 1
        elif head==1:
            if nextstep is not 2:
                return nextstep
            else:
                return 0
        elif head==2:
            if nextstep is not 1:
                return nextstep
            else:
                return 3
        elif head==3:
            if nextstep is not 0:
                return nextstep
            else:
                return 2
        else:
            #time.sleep(5)

            return -1

    #returns position of food for snake
    def getfood(self,observation):
        for i in range(observation.shape[0]):
            #i is north->south
            for j in range(observation.shape[1]):
                #j is west->east
                if observation[i,j]==100:
                    #if not food_found:
                        #food_found=True
                        #print("position of food is to the south from {} to {} and to the east from {} to {}".format(i,i+5,j,j+5))
                        foodlocation=(i,i+5,j,j+5)
                        return foodlocation
        print("Error, couldn't find food!!!")
        print(np.array(observation))
        return (-4,-4,-4,-4)

    """
        gets position of the snakes head and the direction its facing
    """
    def getsnake(self,observation):
        direction_counter=0
        headlocation=(-20,-20)#-1
        direction=-1
        dirx=observation.shape[0]-1
        diry=observation.shape[1]-1
        for i in range(0,dirx+1):
            #i is north->south
            for j in range(0,diry+1):
                #j is west->east

                #0 means direction snake is looking at
                if observation[i,j]==0:
                        direction_counter+=1
                        #only takes the second of the three direction pixel. If only one is visible that one is taken.
                        if observation[max(0,i-2),j]==255:
                            headlocation=(max(0,i-2),j)
                            direction=3
                            if direction_counter > 1 :
                                return (headlocation,direction)
                        elif observation[min(i+2,dirx),j]==255:
                            headlocation=(min(i+2,dirx),j)
                            direction=0
                            if direction_counter > 1 :
                                return (headlocation,direction)
                        elif observation[i,max(0,j-2)]==255:
                            headlocation=(i,max(0,j-2))
                            direction=2
                            if direction_counter > 1 :
                                return (headlocation,direction)
                        elif observation[i,min(j+2,diry)]==255:
                            headlocation=(i,min(j+2,diry))
                            direction=1
                            if direction_counter > 1 :
                                return (headlocation,direction)
                        #else:
                        #There was a problem, that sometimes the direction is displayed 3 pixel ahead of the snake instead of 2 pixel.
                        elif observation[max(0,i-3),j]==255:
                            headlocation=(max(0,i-3),j)
                            direction=3
                            if direction_counter > 1 :
                                return (headlocation,direction)
                        elif observation[min(i+3,dirx),j]==255:
                            headlocation=(min(i+3,dirx),j)
                            direction=0
                            if direction_counter > 1 :
                                return (headlocation,direction)
                        elif observation[i,max(0,j-3)]==255:
                            headlocation=(i,max(0,j-3))
                            direction=2
                            if direction_counter > 1 :
                                return (headlocation,direction)
                        elif observation[i,min(j+3,diry)]==255:
                            headlocation=(i,min(j+3,diry))
                            direction=1
                            if direction_counter > 1 :
                                return (headlocation,direction)
                        else:
                            pass
                            #print("no direction could be found!!!!")
        return headlocation,direction

    """
        returns the positions of food, snake head, distance and guesses for the next action
    """
    def printpositions(self,observation):
        #init the output values for case they are not found in image
        food_found=False
        direction_counter=0
        foodlocation=(-4,-4,-4,-4)
        headlocation=(-3,-3)
        direction=-1
        foodlocation=self.getfood(observation)

        headlocation,direction=self.getsnake(observation)

        #get direction:
        newdirection=-1
        #generates biggest distance to food
        distsouth1=foodlocation[0]-headlocation[0]
        distsouth2=foodlocation[1]-headlocation[0]
        disteast1=foodlocation[2]-headlocation[1]
        disteast2=foodlocation[3]-headlocation[1]
        distances=np.array([distsouth1,distsouth2,disteast1,disteast2])

        bigindex=np.argmax(np.absolute(distances))

        #goes into direction of biggest distance
        pos=distances[bigindex]
        if bigindex <=1:
            if pos>=0:
                newdirection=self.desiredmove(direction,3)
            else:
                newdirection=self.desiredmove(direction,0)
        else:
            if pos>=0:
                newdirection=self.desiredmove(direction,2)
            else:
                newdirection=self.desiredmove(direction,1)

        #didn't manage to add all inputs in an array in some nice way
        additional_states=np.append(np.append(np.append(foodlocation,headlocation),[direction,newdirection]),distances)

        #if no food is found, something is wrong -> set everything to -2
        if (foodlocation[0]<0):
            additional_states.fill(-2)
            print("didn't found anything")

        return additional_states

    """
        used to put background at 0 and have equally distant values
    """
    def revaluescreen(self,element):
        if element<12:
            return 85
        if element<50:
            return 0
        if element<150:
            return 170
        else:
            return 255

    """
        shapes the image in a way, we want (1 channel) and calculates additional features if required
    """
    def preprocess(self, screen):
        #print(np.shape(screen))
        preprocessed = screen[:, :,1]
        #print(preprocessed)
        #print(preprocessed)
        #print([[self.revaluescreen(e) for e in row] for row in screen[:, :,1]])
        #for i in preprocessed
        #print(np.shape(preprocessed))
        if self.activateAdditional:
            additional_states=self.printpositions(preprocessed)#np.zeros(12)#
        else:
            additional_states=np.zeros(12)

        preprocessed=np.array([[self.revaluescreen(e) for e in row] for row in screen[:, :,1]])

        #print (preprocessed)
        preprocessed: np.array = preprocessed.astype('float32') / 255.

        return preprocessed,additional_states

    def init(self):
        """
        @return observation
        """
        return self.game.reset()

    def get_screen(self):
        screen = self.game.render('rgb_array')
        screen,additional_states = self.preprocess(screen)
        #print("additional_states",additional_states)
        return screen,additional_states

    def step(self, action: int):
        observation, reward, done, info = self.game.step(action)
        return observation, reward, done, info

    def reset(self):
        """
        :return: observation array
        """
        observation = self.game.reset()
        observation = self.preprocess(observation)
        return observation

    @property
    def action_space(self):
        return self.game.action_space.n
