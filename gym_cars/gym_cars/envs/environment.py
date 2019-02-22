import os
import cv2
import time
import numpy as np

from resizeimage import resizeimage

THERE_IS_NOT_SCREEN_FLAG=True
if THERE_IS_NOT_SCREEN_FLAG==True:
    os.environ["SDL_VIDEODRIVER"] = "dummy"

import random
import math
import numpy as np
import matplotlib.pyplot as plt
import pygame
from pygame.color import THECOLORS

import pymunk
from pymunk.vec2d import Vec2d
from pymunk.pygame_util import draw


import sys
import gym
from gym import error, spaces, utils
from gym.utils import seeding



# PyGame init
WIDTH  = 1000
HEIGHT = 700

pygame.init()
clock = pygame.time.Clock()

# EDIT: add constants
# Showing sensors and redrawing slows things down.
draw_screen  = True

#shapes
MIN_BALL_RADIUS         = 50
PRIZE_RELATED_TO_CAR    = 100
OBSTABLE_RELATED_TO_CAR = 300
CAT_RELATED_TO_CAR      = 5
CAR_RADIUS              = 25
EDGE                    = CAR_RADIUS+10
WIDTH_OF_FRAME          = 3
PIXEL_DELTA 		= 1
BARRIER_FACTOR          = 1.5
FRAME_FOR_PRIZE         = 5

#colors
FOE    = THECOLORS["blue"]
FRIEND = THECOLORS["pink"]
SCREAN = THECOLORS["black"]
OUT    = -1

#prizes
NUM_OF_LIVES                  = 5
PRIZESCORE                    = 50
DETHSCORE                     = -100
MAX_NUM_OF_PRIZES_AT_A_TIME   = 1
MAX_NUM_OF_STEPS_FOR_THE_GAME = 1000000
MAX_NUM_OF_COLLECTED_PRIZES   = 100000000000000

#velocity
CAR_VELOCITY          = 100
OBSTACLE_VELOCITY_MIN = int( CAR_VELOCITY * 0.6 )
OBSTACLE_VELOCITY_MAX = int( CAR_VELOCITY * 0.8 )
CAT_VELOCITY_MIN      = int( CAR_VELOCITY * 0.7 )
CAT_VELOCITY_MAX      = int( CAR_VELOCITY * 0.9 )


#mass
MASS_OF_OBSTACLE = 10000
MASS_OF_CAR      = MASS_OF_OBSTACLE / 1000
MASS_OF_CAT      = MASS_OF_OBSTACLE / 1000
MASS_OF_PRIZE    = MASS_OF_OBSTACLE / 100

#screen
TIME_OF_STEP = 1.0/10
screen = pygame.display.set_mode((WIDTH, HEIGHT))
# Turn off alpha since we don't use it.
screen.set_alpha(None)

#collisions 
COLLTYPE = { "car":0,
	     "obstacle":1,
             "prize":2,
	     "wallLeft":3,
	     "wallUp":4,
	     "wallRight":5,
	     "wallDown":6,
	     "wallVerticalL":7,
	     "wallHorizantalU":8,
	     "wallVerticalR":9,
	     "wallHorizantalD":10
	     
           }


class carsEnv(gym.Env):

    metadata = {"render.modes": ["human", "ansi"]}

    def __init__(self):
        
        ############################################## dopamin initilization ##############################################
        self.screen_size       = 210
        self.observation_space = spaces.Box(low=0, high=255, dtype=np.uint8, shape=(HEIGHT,WIDTH))
        self.action_space      = spaces.Discrete(len([0,1,2,3]))
        self.reward_range      = 150
        self.lives             = NUM_OF_LIVES
        ############################################## dopamin initilization ##############################################
        
	self.recover_from_reset      = False
        self.num_of_collected_prizes = 0

        # Physics stuff.
        self.space         = pymunk.Space()
        self.space.gravity = pymunk.Vec2d(0., 0.)

        # Create the car.
        self.create_car(100, 100, 0.5)

        # Record steps.
        self.num_steps = 0

        # Create walls.
        static = [

	    #left wall
            pymunk.Segment(                           
                self.space.static_body,
                (0, 1), (0, HEIGHT), WIDTH_OF_FRAME),

	    #up wall
            pymunk.Segment(
                self.space.static_body,
                (1, HEIGHT), (WIDTH, HEIGHT),WIDTH_OF_FRAME),

	    #right wall
            pymunk.Segment(
                self.space.static_body,
                (WIDTH-1, HEIGHT), (WIDTH-1, 1), WIDTH_OF_FRAME),

	    #down wall
            pymunk.Segment(
                self.space.static_body,
                (1, 1), (WIDTH, 1), WIDTH_OF_FRAME),]

	'''
	    #vertical L wall barrier
            pymunk.Segment(
                self.space.static_body,
                ( WIDTH / 2 - WIDTH_OF_FRAME - PIXEL_DELTA, BARRIER_FACTOR * MIN_BALL_RADIUS ), ( WIDTH / 2 - WIDTH_OF_FRAME - PIXEL_DELTA, HEIGHT - BARRIER_FACTOR * MIN_BALL_RADIUS ), WIDTH_OF_FRAME ),
 
	    #horizontal U wall barrier 
            pymunk.Segment(
                self.space.static_body,
                 ( BARRIER_FACTOR * MIN_BALL_RADIUS, HEIGHT / 2 + WIDTH_OF_FRAME + PIXEL_DELTA ), ( WIDTH - BARRIER_FACTOR * MIN_BALL_RADIUS, HEIGHT / 2 + WIDTH_OF_FRAME + PIXEL_DELTA ), WIDTH_OF_FRAME ),

            #vertical R wall barrier 
            pymunk.Segment(
                self.space.static_body,
                ( WIDTH / 2 + WIDTH_OF_FRAME + PIXEL_DELTA, BARRIER_FACTOR * MIN_BALL_RADIUS ), ( WIDTH / 2 + WIDTH_OF_FRAME + PIXEL_DELTA, HEIGHT - BARRIER_FACTOR * MIN_BALL_RADIUS ), WIDTH_OF_FRAME ),
 
	    #horizontal D wall barrier
            pymunk.Segment(
                self.space.static_body,
                 ( BARRIER_FACTOR * MIN_BALL_RADIUS, HEIGHT / 2 - WIDTH_OF_FRAME - PIXEL_DELTA), ( WIDTH - BARRIER_FACTOR * MIN_BALL_RADIUS, HEIGHT / 2 - WIDTH_OF_FRAME - PIXEL_DELTA ), WIDTH_OF_FRAME ),
	'''
    		
        
        for i,s in enumerate(static):
            s.friction = 1.
            s.group = 1
            #s.collision_type = 1
            s.color = THECOLORS['red']
	    s.collision_type = COLLTYPE[ "wallLeft" ] + i
        self.space.add(static)

        # Create some obstacles, semi-randomly.
        # We'll create three and they'll move around to prevent over-fitting.
        self.obstacles = []
	'''
        self.obstacles.append( self.create_obstacle( WIDTH * 0.2 , HEIGHT * 0.7 , 120) )
        self.obstacles.append( self.create_obstacle( WIDTH * 0.7 , HEIGHT * 0.7 , 70 ) )
        self.obstacles.append( self.create_obstacle( WIDTH * 0.7 , HEIGHT * 0.2 , 50 ) )
        self.obstacles.append( self.create_obstacle( WIDTH * 0.35, HEIGHT * 0.35, 40 ) )
        '''
        #prizes + Create first prize
        self.prizes = []  
        self.put_prize()
        
	#collisions
	#self.colision_from_up   = False
	#self.colision_from_left = False
	self.dont_move           = False
	self.got_prize           = False
	self.wall_collision      = False
	self.need_to_die         = False

	self.space.add_collision_handler(COLLTYPE["car"], COLLTYPE["obstacle"], begin =\
					 self.carAndObstacleCollision, pre_solve = self.carAndObstacleCollision)

    	self.space.add_collision_handler(COLLTYPE["car"], COLLTYPE["prize"], begin =\
					 self.carAndPrizeCollision, pre_solve = self.carAndPrizeCollision)

	self.space.add_collision_handler(COLLTYPE["car"], COLLTYPE["wallLeft"], begin =\
					 self.carAndWallLeftcollision, pre_solve = self.carAndWallLeftcollision, separate = self.carAndWallseparator )

	self.space.add_collision_handler(COLLTYPE["car"], COLLTYPE["wallUp"], begin =\
					 self.carAndWallUpcollision, pre_solve = self.carAndWallUpcollision,  separate = self.carAndWallseparator) 

	self.space.add_collision_handler(COLLTYPE["car"], COLLTYPE["wallRight"], begin =\
					 self.carAndWallRightcollision, pre_solve = self.carAndWallRightcollision, separate = self.carAndWallseparator)

	self.space.add_collision_handler(COLLTYPE["car"], COLLTYPE["wallDown"], begin =\
					 self.carAndWallDowncollision, pre_solve = self.carAndWallDowncollision, separate = self.carAndWallseparator)

	self.space.add_collision_handler(COLLTYPE["car"], COLLTYPE["wallHorizantalU"], begin =\
					 self.carAndWallDowncollision,  pre_solve = self.carAndWallDowncollision, separate = self.carAndWallseparator)

	self.space.add_collision_handler(COLLTYPE["car"], COLLTYPE["wallHorizantalD"], begin =\
					 self.carAndWallUpcollision,  pre_solve = self.carAndWallUpcollision, separate = self.carAndWallseparator)

        self.space.add_collision_handler(COLLTYPE["car"], COLLTYPE["wallVerticalL"], begin =\
					 self.carAndWallRightcollision,  pre_solve = self.carAndWallRightcollision, separate = self.carAndWallseparator)

	self.space.add_collision_handler(COLLTYPE["car"], COLLTYPE["wallVerticalR"], begin =\
					 self.carAndWallLeftcollision,  pre_solve = self.carAndWallLeftcollision, separate = self.carAndWallseparator)

	  
  

    def carAndObstacleCollision(self, space, arbiter):
        self.need_to_die = True
        return True


    def carAndPrizeCollision(self, space, arbiter):
        self.got_prize = True
        return True

    #wall collisions
    def carAndWallLeftcollision(self, space, arbiter):
        driving_direction = Vec2d(1, 0).rotated(self.car_body.angle)
	if driving_direction.x<=0:
            self.dont_move = True
	else:
            self.dont_move = False

        return True

    def carAndWallUpcollision(self, space, arbiter):
	driving_direction = Vec2d(1, 0).rotated(self.car_body.angle)
	if driving_direction.y>=0:
            self.dont_move = True
	else:
            self.dont_move = False

        return True

    def carAndWallRightcollision(self, space, arbiter):
	driving_direction = Vec2d(1, 0).rotated(self.car_body.angle)
	if driving_direction.x>=0:
            self.dont_move = True
	else:
            self.dont_move = False

        return True


    def carAndWallDowncollision(self, space, arbiter):
	driving_direction = Vec2d(1, 0).rotated(self.car_body.angle)
	if driving_direction.y<=0:
            self.dont_move = True
	else:
            self.dont_move = False

        return True


    def carAndWallVerticalcollision(self, space, arbiter):
	'''
        driving_direction = Vec2d(1, 0).rotated(self.car_body.angle)
	if driving_direction.y>=0 && colision_from_down == False:
            colision_from_up = True
        elif driving_direction.y<=0 && colision_from_up == False:
            colision_from_down = True
	'''
        return True
	

    def carAndWallHorizantalcollision(self, space, arbiter):
	
        return True
	
    # separators
    def carAndWallseparator(self, space, arbiter):
        self.dont_move = False
        return

    def _seed(self, seed=None):
        self.np_random, seed1 = seeding.np_random(seed)
        # Derive a random seed. This gets passed as a uint, but gets
        # checked as an int elsewhere, so we need to keep it below
        # 2**31.
        seed2 = seeding.hash_seed(seed1 + 1) % 2**31

        # Empirically, we need to seed before loading the ROM.
        seed1=0
        return [seed1, seed2]    
        



    def create_obstacle(self, x, y, r):
        
        c_body                 = pymunk.Body(MASS_OF_OBSTACLE, pymunk.inf)
        c_shape                = pymunk.Circle(c_body, r)
        c_shape.elasticity     = 1
        c_body.position        = x, y
        c_shape.color          = THECOLORS["blue"]
	c_shape.collision_type = COLLTYPE["obstacle"]
        self.space.add(c_body, c_shape)
        return c_body
    
    
    
    def create_prize(self, x, y, r):
        
        inertia                = pymunk.moment_for_circle(1, 0, 14, (0, 0))
        c_body                 = pymunk.Body(MASS_OF_PRIZE, inertia)
        c_shape                = pymunk.Circle(c_body, r)
        c_shape.elasticity     = 1.0
	c_shape.collision_type = COLLTYPE["prize"]
        c_body.position        = x, y
        c_shape.color          = THECOLORS["pink"]
        self.space.add(c_body, c_shape)
        return c_body, c_shape

    
    
    def create_cat(self):
        
        inertia = pymunk.moment_for_circle(1, 0, 14, (0, 0))
        self.cat_body = pymunk.Body(MASS_OF_CAT, inertia)
        self.cat_body.position = 50, HEIGHT - 100
        self.cat_shape = pymunk.Circle(self.cat_body, 30)
        self.cat_shape.color = THECOLORS["orange"]
        self.cat_shape.elasticity = 1.0
        self.cat_shape.angle = 0.5
        direction = Vec2d(1, 0).rotated(self.cat_body.angle)
        self.space.add(self.cat_body, self.cat_shape)

        
        
    def create_car(self, x, y, r):
        
        inertia = pymunk.moment_for_circle(1, 0, 14, (0, 0))
        self.car_body = pymunk.Body(MASS_OF_CAR, inertia)
        self.car_body.position = x, y
        self.car_shape = pymunk.Circle(self.car_body, CAR_RADIUS)
        self.car_shape.color = THECOLORS["green"]
        self.car_shape.elasticity = 1.0
        self.car_body.angle = r
        driving_direction = Vec2d(1, 0).rotated(self.car_body.angle)
        self.car_body.apply_impulse(driving_direction)
        self.space.add(self.car_body, self.car_shape)
    
    
    
    def put_prize(self):
        
        if len(self.prizes)>=MAX_NUM_OF_PRIZES_AT_A_TIME:
            return
        
        x=random.randint(0+FRAME_FOR_PRIZE, WIDTH-1-FRAME_FOR_PRIZE)
        y=random.randint(0+FRAME_FOR_PRIZE, HEIGHT-1-FRAME_FOR_PRIZE)
        tmp_c,tmp_s=self.create_prize(x, y, 30)
        self.prizes.append((tmp_c,tmp_s))
     
    
    
    def remove_prize(self):
        
        while len(self.prizes)>0:
            tmp_prize=self.prizes.pop()
            self.space.remove(tmp_prize)
       
    
    
    def getReward(self,driving_direction):

        reward=0   # Set the reward.

	if self.need_to_die==True:
     	    reward     = DETHSCORE
	    self.lives = self.lives - 1
	    self.oneDeathReset()
	    
	
	# EDIT: if we hit a prize we get points
	if self.got_prize == True:
		
	    reward = PRIZESCORE
	    #set prize to false
	    self.got_prize = False

	    #if we touch it we need to make it disapear
	    self.remove_prize()

	    #if we touch we increase counter		
	    self.num_of_collected_prizes=self.num_of_collected_prizes + 1
        reward = reward - 1           
        return reward
    
    
    
    def frame_step(self, action):
	#print("#######################################",action)
	#if self.recover_from_reset == True:
    	#    self.recover_from_reset=False
	#    self.lives = NUM_OF_LIVES

	self.num_steps += 1	

        #did we finish the game?
        done = False

	#what is the action
        if action == 0:  # Turn left.
	    car_velocity         = 1
            self.car_body.angle -= .2
        elif action == 1:  # Turn right.
 	    car_velocity         = 1
            self.car_body.angle += .2
	elif action == 2:
 	    car_velocity         = 1
	    self.car_body.angle += .0
	elif action == 3:
	    car_velocity         = 0
	    self.car_body.angle += .0
                   
        # add prize.
        if self.num_steps % PRIZE_RELATED_TO_CAR == 0:
            self.put_prize()    
            
        # Move obstacles.
        if self.num_steps % OBSTABLE_RELATED_TO_CAR == 0:
            self.move_obstacles()
        
	#move car
	driving_direction = self.moveCar(car_velocity)

	#print("car vel",self.car_body.velocity)

	# Update the screen and stuff.
        screen.fill(THECOLORS["black"])  
        draw(screen, self.space)
        self.space.step(TIME_OF_STEP)
        
        if draw_screen:
            pygame.display.flip()
            
        clock.tick()
        
        
        # Get the current location and the readings there.
        x, y = self.car_body.position

        reward = self.getReward(driving_direction)
        
	
	if self.wall_collision:
	    self.wall_collision = False

	if self.num_of_collected_prizes >= MAX_NUM_OF_COLLECTED_PRIZES or self.lives<1:
            done = True
            return 0, done

        return reward, done
    
    

    def moveCar(self,car_velocity):
	driving_direction = Vec2d(1, 0).rotated(self.car_body.angle)
        self.car_body.velocity = CAR_VELOCITY * driving_direction * car_velocity
	new_car_pos_x = self.car_body.position[0]+self.car_body.velocity[0] * TIME_OF_STEP
	new_car_pos_y = self.car_body.position[1]+self.car_body.velocity[1] * TIME_OF_STEP

	if self.dont_move == True:
	    #print("we sropped the car")		
	    #time.sleep(5)
	    self.car_body.velocity.x = 0
	    self.car_body.velocity.y = 0


	'''
	if new_car_pos_x < 0 - CAR_RADIUS or new_car_pos_x > HEIGHT + CAR_RADIUS or new_car_pos_y < 0 - CAR_RADIUS or new_car_pos_y > WIDTH + CAR_RADIUS:
	    self.car_body.velocity = (0,0)
	    driving_direction=(0,0)
	'''
	return driving_direction



    def move_obstacles(self):
        
        # Randomly move obstacles around.
        for obstacle in self.obstacles:
            speed = random.randint(OBSTACLE_VELOCITY_MIN, OBSTACLE_VELOCITY_MAX)
	    angle = np.random.rand() * 2 * np.pi - np.pi 
            direction = Vec2d(1, 0).rotated( angle )
            obstacle.velocity = speed * direction

            
            

    def car_is_crashed(self, readings):  
        '''
        EDIT: adaptations because readings in now a list of tuples
        '''
        if (readings[0][0] == 1 and readings[0][1]==FOE) or\
           (readings[1][0] == 1 and readings[1][1]==FOE) or\
           (readings[2][0] == 1 and readings[2][1]==FOE):
            return True
        else:
            return False


    
    def get_rotated_point(self, x_1, y_1, x_2, y_2, radians):
        # Rotate x_2, y_2 around x_1, y_1 by angle.
        x_change = (x_2 - x_1) * math.cos(radians) + \
            (y_2 - y_1) * math.sin(radians)
        y_change = (y_1 - y_2) * math.cos(radians) - \
            (x_1 - x_2) * math.sin(radians)
        new_x = x_change + x_1
        new_y = HEIGHT - (y_change + y_1)
        return int(new_x), int(new_y)



    def oneDeathReset(self):

        #reset obstacle position
	'''
        self.obstacles[0].position = ( WIDTH * 0.2 , HEIGHT * 0.7  )
        self.obstacles[1].position = ( WIDTH * 0.7 , HEIGHT * 0.7  )
        self.obstacles[2].position = ( WIDTH * 0.7 , HEIGHT * 0.2  )
        self.obstacles[3].position = ( WIDTH * 0.35, HEIGHT * 0.35 )

	#reset obstacle velocity 
        for obstacle in self.obstacles:
            speed = random.randint(OBSTACLE_VELOCITY_MIN, OBSTACLE_VELOCITY_MAX)
            angle = np.random.rand() * 2 * np.pi - np.pi 
            direction = Vec2d(1, 0).rotated( angle )
            obstacle.velocity = speed * direction
	'''
	#reset car position
	self.car_body.position  = (100, 100)

	#reset all collision flags
	self.dont_move      = False
	self.got_prize      = False
	self.wall_collision = False
	self.need_to_die    = False
    	
	#put new prize
	self.remove_prize()
	self.put_prize()

    def get_observation(self):
	obs = pygame.surfarray.array3d(screen)
	return obs

	
    def getScreenGrayscale(self,screen_data = None):

	obs = self.get_observation()
	obs = cv2.cvtColor( obs, cv2.COLOR_BGR2GRAY )

	return obs

    def _reset(self):
	
	self.num_of_collected_prizes = 0
	self.lives                   = NUM_OF_LIVES
        self.num_steps               = 0
	self.oneDeathReset()
	obs = self.get_observation()
        return obs	
        
        
        
    def _step(self, action):
        
        (reward, done) = self.frame_step(action)
        obs            = self.get_observation()
        info           = {}

        return obs, reward, done, info
    


    def _render(self, mode="human", close=False):
        
        pass


#game = carsEnv()
#while True:
#    game._step(1)
#    time.sleep(1)
	
    


