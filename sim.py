import pygame
import numpy as np
import math
import cv2 as cv
import random
import action_set
import time
import matplotlib.pyplot as plt
from lidar import Lidar

#OPENCV GOES Y then X

TAU = 2 * math.pi

class Environment:
    def __init__(self, total_iter):
        self.total_iter = total_iter
        self.clock = pygame.time.Clock()


        self.width = 64
        self.height = 64

        self.black_floor = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        self.floor = self.black_floor.copy()
        self.lidar_occupancy = np.zeros((self.height, self.width, 1), dtype=np.uint8)

        self.input = np.zeros((20, 20, 2), dtype=np.uint8)

        self.black = (0,0,0)
        self.grey = (70,70,70)
        self.blue = (255,0,0)
        self.green = (0,255,0)
        self.red = (0,0,255)
        self.white = (255,255,255)

        self.mode = "wall"

        self.spawn = [7, 7]
        self.spawnfl = [10, 9]
        self.spawnfr = [10, 5]
        self.spawnbr = [4, 5]
        self.spawnbl = [4, 9]

        self.init_pos = np.array([self.spawn,
                        self.spawnfl,
                        self.spawnfr,
                        self.spawnbr,
                        self.spawnbl
                        ], dtype=np.float32)

        self.bot_pos = np.array([self.spawn,
                        self.spawnfl,
                        self.spawnfr,
                        self.spawnbr,
                        self.spawnbl
                        ], dtype=np.float32)

        self.bot_angle = math.pi/2
        self.prev_dist = 100000000
        self.actions = action_set.get_action_set()

        pygame.init()
        self.display = pygame.display.set_mode((self.width, self.height))
        
        #cv.namedWindow("Main Game", cv.WINDOW_NORMAL)
        # cv.namedWindow("Model Input", cv.WINDOW_NORMAL)

        
        self.clock = pygame.time.Clock()

        self.start_time = time.time()

        self.circles, self.walls = self.generate_random_obstacles(random.randint(1, 3), random.randint(1, 6))

        self.draw_obstacles()

        self.goal_pos = self.generate_goal_pos(0)

        self.draw_goal_color(self.floor)
            
        self.lidar = Lidar(self.bot_pos[0], self.bot_angle, self.height, self.width)

        self.state_sequence = []
        

    def seed(self, seed):
        random.seed(seed)

    def reset(self, it, simple):
        # print("Oops")
        self.prev_dist = 100000000
        self.distance_to_goal = 0    
        self.man_dist_to_goal = 0

        self.start_t = it


        if not simple:
        # if iter < 5000 or iter > 20000:
        #     print("UGHGHG")
            self.circles, self.walls = self.generate_random_obstacles(random.randint(1, 2), random.randint(2, 4))

            self.floor = self.black_floor.copy()
           # self.circles.clear()
            #self.walls.clear()
            self.draw_obstacles()
        # self.lidar_occupancy = self.black_floor.copy()

        # if iter < 5000 or iter > 20000:
            #cv.imshow("obs", self.floor)
            self.spawn_bot()
        # else:
        #     self.bot_pos = self.init_pos.copy()
        #     self.bot_angle = math.pi/2
            self.goal_pos = self.generate_goal_pos(it)
        else:
            self.simple_reset()
            self.floor = self.black_floor.copy()
            self.circles.clear()
            self.walls.clear()
            self.draw_obstacles()

        self.lidar = Lidar(self.bot_pos[0], self.bot_angle, self.height, self.width)

        # if iter > 2000:
        
        self.draw_goal_color(self.floor)

        self.lidar.update_pose(self.bot_pos[0], self.bot_angle)
        detections_cart, detections_cart_global, detections_polar = self.lidar.sense_obstacles(self.floor)
        
        self.prior_dets = detections_cart

        # print(detections)
        self.lidar_occupancy = np.zeros((self.height, self.width, 1), dtype=np.uint8)
        self.lidar_occupancy = self.draw_detections(self.lidar_occupancy, detections_cart)
        
        self.start_time = time.time()

       

        state = self.convert_det_to_input(detections_cart, self.bot_pos)
        #state = state[18:19, 0:1, :]
       
        self.state_sequence.clear()
        self.state_sequence.append(state)
        self.state_sequence.append(state)
        self.state_sequence.append(state)

        
        seq = np.array(self.state_sequence)
        stacked_seq_scan = np.concatenate((seq[0, :, :, :], seq[1, :, :, :], seq[2, :, :, :]), axis = 2)


        return state
                    
    

    def convert_det_to_input(self, detections, bot_pos):#input is 360 x 2 -> need to pad with 40 copies of goal position

        front_of_car = (bot_pos[1] + bot_pos[2])/2.0
        rounded_bot_pos = np.rint(front_of_car)
        relative_goal = self.goal_pos-rounded_bot_pos
        # print(self.bot_pos[0], rounded_bot_pos, self.goal_pos)
        # print(relative_goal)
        copied_goal = np.tile(np.array(relative_goal).reshape((1, 2)), (2, 20, 1))
        copied_bot_pos = np.tile(rounded_bot_pos.reshape((1, 2)), (1, 20, 1))

        # print(copied_bot_pos[0], copied_bot_pos.shape)
        # print(copied_goal[0],  copied_goal.shape)

        out = np.zeros((21, 20, 2), dtype=np.int32)

        # print("asdgadsg", copied_goal[-1, :, :])

        rect_det = np.array(detections).reshape(18, 20, 2)

        # print("sfd", rect_det[3, :, :])

        out[:18, :, :] = rect_det
        
        out[18:20, :, :] = copied_goal
        out[20:, :, :] = copied_bot_pos

        # print("sasdfasdf", out[3, :, :])
        # print()
        # print("asdf", out.shape)
        # print("goal", out[18, 0, :], "pos", out[20, 0, :])

        return out

    def overwrite_state_hers(self, state, new_state, final_state):
        # copied_bot_pos = np.tile(self.bot_pos[0].reshape((1, 2)), (1, 20, 1))
        # self.input[19:, :, :] = copied_bot_pos
        # print("before: ", state[:, 19:, :, :])
        # print("target goal: ", goal_state[18:19, :, :])
        
            # print("before:")
            # print("s", state[:, 18:, 0, :])
            # print("n", new_state[18:, 0, :])
            # print("f", final_state[18:, 0, :])
        rel_goal_bot_pos = final_state[20:, :, :] - state[:, 20:, :, :].squeeze(0)
        next_rel_goal_bot_pos = final_state[20:, :, :] - new_state[20:, :, :]

        # print("goal: ", goal_state[20:, :, :])
        # print("state: ", state[:, 20:, :, :].squeeze(0))

        # print("shhnn", next_rel_goal_bot_pos, next_rel_goal_bot_pos.shape)

        rel_goal_bot_pos = np.tile(rel_goal_bot_pos, (2, 1, 1))
        next_rel_goal_bot_pos = np.tile(next_rel_goal_bot_pos, (2, 1, 1))


        state[:, 18:20, :, :] = np.expand_dims(rel_goal_bot_pos, axis=0)
        new_state[18:20, :, :] = next_rel_goal_bot_pos

        # print("after:")
        # print("s", state[:, 18:, 0, :])
        # print("n", new_state[18:, 0, :])
        # print("f", final_state[18:, 0, :])

        # print(state[:, 18:20, :, :])
        # print(goal_state[18:20, :, :])

        return state, new_state
    
    def her_reward(self, state, t, orig_distance):
        if state[0, 18, 0, 0] <= 4:# and (abs(self.bot_angle - angle) < 0.5 or abs(self.bot_angle + angle - TAU) < 0.5) :
            # print("Found Goal Her")
            goal_reward = max(1.0, orig_distance)#self.distance_to_goal# + abs(math.pi - min(abs(self.bot_angle - angle), abs(self.bot_angle + angle - TAU))) * 5
            return goal_reward
        
        progress_reward = -0.01 * t

        return progress_reward


    def reward(self, did_collide, t):

        # front_of_car = (self.bot_pos[1] + self.bot_pos[2])/2.0
        # # print(self.bot_pos[1], self.bot_pos[2], (self.bot_pos[1] + self.bot_pos[2])/2.0)
        # dist = np.linalg.norm(front_of_car-self.goal_pos)

        diff = self.bot_pos[0] - self.goal_pos
        dist = np.linalg.norm(diff) #8d
        man_dist = abs(diff[0]) + abs(diff[1])

        # print(front_of_car, self.goal_pos, dist)

        angle = math.atan2(self.goal_pos[1]-self.bot_pos[0][1], self.goal_pos[0]-self.bot_pos[0][0])

        angle = self.limit_angle(angle)

        # print(angle)

        done = False
        
        time_punishment = 0.0

        print(t, self.start_t, t-self.start_t)
        #if t - self.start_t > 300:
        #    print("Ran out of Time")
        #    time_punishment = -3#min(-1, -dist/32)
        #    done = True
        #    return time_punishment, done
            
        if did_collide:
            print("Collided")
            # collision_punishment = -1.0 * (25-(time.time()-self.start_time)) - 25.0 * (dist/180.0)
            collision_punishment = min(-1, self.man_dist_to_goal)
            done = True
            return collision_punishment, done
        
        goal_reward = 0.0
        if dist < 4:# and (abs(self.bot_angle - angle) < 0.5 or abs(self.bot_angle + angle - TAU) < 0.5) :
            goal_reward = max(1.0, self.man_dist_to_goal)#self.distance_to_goal# + abs(math.pi - min(abs(self.bot_angle - angle), abs(self.bot_angle + angle - TAU))) * 5
            print("Found Goal w/ Reward: ", goal_reward)
            done = True
            return goal_reward, done
        
        #progress_reward = -0.01

        if man_dist < self.prev_dist - 0.15:
            # print("Closer")
            #print(dist, self.man_dist_to_goal)
            progress_reward = min(1, self.man_dist_to_goal/(dist + 1))*0.01
            # pass
        else:
            # print("Farther")
            progress_reward = -0.01

        # total_reward = goal_reward + progress_reward + time_punishment + collision_punishment

        # print(self.bot_pos[0], self.goal_pos, dist, angle, self.bot_angle, progress_reward )

        self.prev_dist = man_dist

        return progress_reward, done
        
    def perform_action_8d(self, action_id=0):
        action = self.actions[action_id]

        y_change = action[0]
        x_change = action[1]

        self.bot_pos[:, 0] = self.bot_pos[:, 0] + y_change
        self.bot_pos[:, 1] = self.bot_pos[:, 1] + x_change


    def perform_action(self, action_id=0):
        action = self.actions[action_id]

        self.bot_angle += action[1] * math.pi/180.0

        self.bot_angle = self.limit_angle(self.bot_angle)

        x_change = math.cos(self.bot_angle) * action[0]
        y_change = math.sin(self.bot_angle) * action[0]

        self.bot_pos[:, 0] = self.bot_pos[:, 0] + y_change
        self.bot_pos[:, 1] = self.bot_pos[:, 1] + x_change
        
        self.bot_pos[1:] = self.rotate(action[1])
       

        # print(self.bot_pos[0], self.bot_angle, action, x_change, y_change)

    def user_input_step(self, t, save_dir):
        pygame.event.get()
        # print(self.bot_pos[0],self.bot_angle)
        self.floor = self.black_floor.copy()
        # self.lidar_occupancy = self.black_floor.copy()
        self.draw_obstacles()
        self.draw_goal_color(self.floor)

        action_id = self.user_input()

        done = False
        if action_id >= 0:

            self.perform_action_8d(action_id)

            self.lidar.update_pose(self.bot_pos[0], self.bot_angle)
            detections_cart, detections_cart_global, detections_polar = self.lidar.sense_obstacles(self.floor)
            
            collided_with_det = self.check_collision(detections_cart_global)
            if collided_with_det:
                state = self.convert_det_to_input(self.prior_dets, self.prior_bot_pos)
                self.bot_pos = self.prior_bot_pos.copy()
                self.lidar.update_pose(self.bot_pos[0], self.bot_angle)
            else:
                self.prior_dets = detections_cart
                self.prior_bot_pos = self.bot_pos.copy()
                state = self.convert_det_to_input(detections_cart, self.bot_pos)
            #print(detections_cart)
            self.state_sequence.append(state)
            _, done = self.reward(collided_with_det, t)

            #if t %  == 0:
        #print(len(self.state_sequence))
        if len(self.state_sequence) == 3:
            triple_state = np.array(self.state_sequence)
            #print(triple_state.shape)
            reshaped_triple_state = triple_state.reshape(triple_state.shape[0], -1)
            #print(reshaped_triple_state)
            np.savetxt(save_dir + "/scans/" + str(t), reshaped_triple_state)
            with open(save_dir + "/labels/" + str(t), 'w') as output_file:
                output_file.write(str(action_id))

            
            self.state_sequence.clear()
    
        cv.fillPoly(self.floor, [np.array([[round(pt[1]), round(pt[0])] for pt in self.bot_pos[1:]], dtype=np.int32)], self.red, cv.LINE_8)
        # cv.fillPoly(self.lidar_display, [np.array([[round(pt[1]), round(pt[0])] for pt in self.bot_pos[1:]], dtype=np.int32)], self.red, cv.LINE_8)
        
        # concat = np.hstack((self.floor, with_goal_bot))


        #cv.imshow("Main Game", self.floor)
        #plt.imshow(self.floor)
        #plt.show()
        # cv.imshow("Model Input", cv.normalize(with_goal_bot, None, 0, 255, cv.NORM_MINMAX))
        self.floor_disp = np.flip(self.floor, axis=0)
        img_surface = pygame.surfarray.make_surface(self.floor.swapaxes(0, 1))
        self.display.blit(img_surface, (0, 0))
        pygame.display.flip()
        pygame.display.update()
        # self.clock.tick_busy_loop(60)


        # cv.imshow("hi", cv.normalize(self.translated_occupancy, None, 0, 255, cv.NORM_MINMAX))
        
        #state = state[18:19, 0:1, :]
        #print(state)

        return done

    def step(self, action_id, t):
        pygame.event.get()
        # print(self.bot_pos[0],self.bot_angle)
        self.floor = self.black_floor.copy()
        # self.lidar_occupancy = self.black_floor.copy()
        self.draw_obstacles()
        self.draw_goal_color(self.floor)

        # self.user_input()
        # self.perform_action(action_id)
        
        self.perform_action_8d(action_id)
        # print("after", self.bot_pos[0],self.bot_angle)

        self.lidar.update_pose(self.bot_pos[0], self.bot_angle)
        detections_cart, detections_cart_global, detections_polar = self.lidar.sense_obstacles(self.floor)
        
        collided_with_det = self.check_collision(detections_cart)
      

        #print(collided_with_det, len(detections_cart), len(self.prior_dets), self.bot_pos, self.prior_bot_pos)
        if collided_with_det:
            state = self.convert_det_to_input(self.prior_dets, self.prior_bot_pos)
            self.bot_pos = self.prior_bot_pos.copy()
            self.lidar.update_pose(self.bot_pos[0], self.bot_angle)
        else:
            self.prior_dets = detections_cart
            self.prior_bot_pos = self.bot_pos.copy()
            state = self.convert_det_to_input(detections_cart, self.bot_pos)

          
        if len(self.state_sequence) > 2:
            self.state_sequence.pop(0)
        self.state_sequence.append(state)

        seq = np.array(self.state_sequence)
        stacked_seq_state = np.concatenate((seq[0, :, :, :], seq[1, :, :, :], seq[2, :, :, :]), axis = 2)



        # print(detections)
        #self.lidar_occupancy = self.draw_detections(self.lidar_occupancy, detections_cart)

        # with_goal_bot = self.draw_goal_gray(self.lidar_occupancy)
        # with_goal_bot[round(self.bot_pos[0][0]), round(self.bot_pos[0][1])] = 3

        
        
        #self.lidar_display = self.lidar_occupancy.copy()
        cv.fillPoly(self.floor, [np.array([[round(pt[1]), round(pt[0])] for pt in self.bot_pos[1:]], dtype=np.int32)], self.red, cv.LINE_8)
        # cv.fillPoly(self.lidar_display, [np.array([[round(pt[1]), round(pt[0])] for pt in self.bot_pos[1:]], dtype=np.int32)], self.red, cv.LINE_8)
        
        # concat = np.hstack((self.floor, with_goal_bot))

        #cv.imshow("Main Game", self.floor)
        #plt.imshow(self.floor)
        #plt.show()
        # cv.imshow("Model Input", cv.normalize(with_goal_bot, None, 0, 255, cv.NORM_MINMAX))
        self.floor_disp = np.flip(self.floor, axis=0)
        img_surface = pygame.surfarray.make_surface(self.floor.swapaxes(0, 1))
        self.display.blit(img_surface, (0, 0))
        pygame.display.flip()
        pygame.display.update()
        reward, done = self.reward(collided_with_det, t)
        
        if t % 3 == 0:
            print("goal", self.goal_pos, "rel_goal", state[18, 0, :], "pos", state[20, 0, :], "ang", self.bot_angle, self.actions[action_id], "rew:", reward)

        # cv.imshow("hi", cv.normalize(self.translated_occupancy, None, 0, 255, cv.NORM_MINMAX))
        
        #state = state[18:19, 0:1, :]
        #print(state)

        return state, reward, done, None

    def user_input(self):
        keys = pygame.key.get_pressed() 

        # if keys[pygame.K_UP] or keys[pygame.K_w]:
        #     # print(self.bot_pos[:,1] - 1)
        #     # if self.bot_pos[0][1] > 2:
        #         # self.bot_pos[:, 1] = np.clip(self.bot_pos[:, 1] - 1, 0, 127) 
        #         self.bot_pos[:, 1] = self.bot_pos[:, 1] - 1


        # if keys[pygame.K_DOWN] or keys[pygame.K_s]:
        #     # if self.bot_pos[0][1] < 125:
        #         self.bot_pos[:, 1] = np.clip(self.bot_pos[:, 1] + 1, 0, 127)
        #         self.bot_pos[:, 1] = self.bot_pos[:, 1] + 1

        # if keys[pygame.K_LEFT] or keys[pygame.K_a]:
        #     # if self.bot_pos[0][0] > 2:
        #         self.bot_pos[:, 0] = np.clip(self.bot_pos[:, 0] - 1, 0, 127)
        #         self.bot_pos[:, 0] = self.bot_pos[:, 0] - 1

        # if keys[pygame.K_RIGHT] or keys[pygame.K_d]:
        #     # self.bot_pos[0] = min(127, self.bot_pos[0] + 1)
        #     # if self.bot_pos[0][1] < 125:
        #         self.bot_pos[:, 0] = np.clip(self.bot_pos[:, 0] + 1, 0, 127)
        #         self.bot_pos[:, 0] = self.bot_pos[:, 0] + 1
            

        # if keys[pygame.K_q]:
        #     self.bot_pos[1:] = self.rotate(-10)
        #     self.bot_angle -= 10 * math.pi/180

        #     if (self.bot_angle < 0.0):
        #         self.bot_angle += TAU
        
        # if keys[pygame.K_e]:
        #     self.bot_pos[1:] = self.rotate(10)
        #     self.bot_angle += 10 * math.pi/180

        #     if (self.bot_angle >= TAU):
        #         self.bot_angle -= TAU

        # if keys[pygame.K_b]:
        #     self.mode = "wall"
        #     print("Wall Mode")
        
        # if keys[pygame.K_c]:
        #     self.mode = "circle"
        #     print("Circle Mode")

        action_id = -1

        if keys[pygame.K_UP]:
            # if keys[pygame.K_LEFT]:
            #     action_id = 4
            # elif keys[pygame.K_RIGHT]:
            #     action_id = 3
            # else:
            action_id = 0
        # if keys[pygame.K_DOWN]:
        #     if keys[pygame.K_LEFT]:
        #         action_id = 6
        #     elif keys[pygame.K_RIGHT]:
        #         action_id = 5
        #     else:
        #         action_id = 2

        if keys[pygame.K_UP] and not keys[pygame.K_RIGHT] and not keys[pygame.K_LEFT] and not keys[pygame.K_DOWN]:
            action_id = 2

        if keys[pygame.K_DOWN] and not keys[pygame.K_LEFT] and not keys[pygame.K_UP] and not keys[pygame.K_RIGHT]:
            action_id = 0

        if keys[pygame.K_LEFT] and not keys[pygame.K_RIGHT] and not keys[pygame.K_UP] and not keys[pygame.K_DOWN]:
            action_id = 1

        if keys[pygame.K_RIGHT] and not keys[pygame.K_LEFT] and not keys[pygame.K_UP] and not keys[pygame.K_DOWN]:
            action_id = 3

        

        return action_id
        # # self.perform_action(action_id)
        
        # events = pygame.event.get()

        # for event in events:
        #     if self.mode == "circle":
        #         if event.type == pygame.MOUSEBUTTONUP:
        #             print("UP")
        #             pos = pygame.mouse.get_pos()
        #             print("@POS: ", pos)
        #             cv.circle(self.floor, pos, 1, self.white -1, lineType=cv.LINE_8)
        #         # if event.type == pygame.MOUSEBUTTONDOWN:
        #         #     print("DOWN")
        #         # if event.type == pygame.MOUSEMOTION:
        #         #     print("MOVE")
                
    def draw_obstacles(self):
        for circ in self.circles:
            #print(circ)
            self.floor[circ[0], circ[1], :] = self.white
        for wall in self.walls:
            #print(wall)
            cv.line(self.floor, (wall[0], wall[1]), (wall[2], wall[3]), self.white, 2, cv.LINE_8)
        self.floor = cv.copyMakeBorder(self.floor[1:self.height-1, 1:self.width-1, :], 1, 1, 1, 1, cv.BORDER_CONSTANT, None, self.white)


        imgray = cv.cvtColor(self.floor, cv.COLOR_BGR2GRAY)
        ret, thresh = cv.threshold(imgray, 127, 255, 0)
        contours, hierarchy = cv.findContours(thresh, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)

        self.obstacle_contours = contours
    
    def draw_detections(self, image, detections):
        for detection in detections:
            image[detection[0], detection[1], :] = 1
        return image
    
    def draw_goal_color(self, image):
        image[self.goal_pos[0], self.goal_pos[1], :] = self.green
    
    def draw_goal_gray(self, image):
        retImage = image.copy()
        retImage[self.goal_pos[0], self.goal_pos[1], :] = 2

        return retImage


    def generate_random_obstacles(self, num_walls, num_circles):
        circles = []
        walls = []

        post_low_bound = 5
        for i in range(num_circles):
            x, y = random.randint(post_low_bound, self.width-post_low_bound-1), random.randint(post_low_bound, self.height-post_low_bound-1)
            circles.append([x, y])

        distance = random.uniform(5, 15)

        wall_low_bound = 15#int(self.width/5)


        while len(walls) < num_walls:
            spawn_angle = random.uniform(0, TAU-0.01)

            x = random.randint(wall_low_bound, self.width-wall_low_bound-1)
            x1 = round(x + math.cos(spawn_angle) * distance)
            y = random.randint(wall_low_bound, self.height-wall_low_bound-1)
            y1 = round(y + math.sin(spawn_angle) * distance)

			# print(self.bot_pos[0], y, x, self.floor[y, x, :])

            if 0 < x1 < self.width-1 and 0 < y1 < self.height-1:
                walls.append([x, y, x1, y1])
                
        return circles, walls
    
    
    def spawn_bot(self):
        ylow = 10
        yhigh = self.height - 10
        xlow = 10
        xhigh = self.width - 10

        done = False
        while not done:
            x = random.randint(xlow, xhigh)
            y = random.randint(ylow, yhigh)
            raw_angle = random.randint(0, 359)
            # self.bot_angle = np.deg2rad(raw_angle)

            self.bot_angle = np.deg2rad(90) #8d

            # print(raw_angle)

            self.bot_angle = self.limit_angle(self.bot_angle)

            # print(self.bot_angle)

            
            spawn = [y, x]
            spawnfl = [y+3, x+2]
            spawnfr = [y+3, x-2]
            spawnbr = [y-3, x-2]
            spawnbl = [y-3, x+2]

            self.bot_pos = np.array([spawn,
                            spawnfl,
                            spawnfr,
                            spawnbr,
                            spawnbl
                            ], dtype=np.float32)
            self.prior_bot_pos = self.bot_pos.copy()
            # print("sd", self.bot_pos[1:], raw_angle-90)
            # self.bot_pos[1:] = self.rotate(raw_angle-90)
            # print("afs", self.bot_pos[1:])

            bot_contour = np.array([[[round(point[1]), round(point[0])]] for point in self.bot_pos[1:]], dtype=np.int32)
        
            done = True
            forbidden_pixels = np.argwhere(np.sum(self.floor - self.white, axis=2) == 0)
            #print(np.sum(self.floor - self.white, axis=2))
            #print(forbidden_pixels)
            for det in forbidden_pixels:
                if abs(cv.pointPolygonTest(bot_contour, (round(det[1]), round(det[0])), True)) < 8:
                    done = False
                    break
            self.spawn = spawn
        # print(self.bot_pos[0], self.bot_angle)
        return
    
    def generate_goal_pos(self, it):
    
        done = False

        try_counter = 0
        while not done:

            front_of_car = (self.bot_pos[1] + self.bot_pos[2])/2.0

            # spawn_angle = random.uniform(0, TAU-0.01)
            distance = random.uniform(10, 10 + (min(0.5 * self.total_iter, it)/(0.5 * self.total_iter)) * self.width/3)

            spawn_angle = self.bot_angle + random.uniform(-math.pi, math.pi)

            spawn_angle = self.limit_angle(spawn_angle)

            x = round(front_of_car[1] + math.cos(spawn_angle) * distance)
            y = round(front_of_car[0] + math.sin(spawn_angle) * distance)
            # print(self.bot_pos[0], y, x, self.floor[y, x, :])


            if 3 < x < self.width-4 and 3 < y < self.height-4:
            #sum(self.floor[y, x, :]) == 0:
                disp = np.zeros_like(self.floor)
                cv.drawContours(disp, self.obstacle_contours, -1, self.white, 1, cv.LINE_8)
                disp[y, x, :] = self.green
                #cv.imshow("skdjfn", disp)
                done = True
                for contour in self.obstacle_contours:
                    obs_d = abs(cv.pointPolygonTest(contour, (x, y), True))
                    # print(obs_d)
                    if obs_d < 5:
                        done = False
            try_counter += 1

            if try_counter > 10:
                self.spawn_bot()
                try_counter = 0
        self.distance_to_goal = distance
        self.man_dist_to_goal = abs(x) + abs(y)
        print("Goal Dist: ", self.distance_to_goal)
        return [y, x]

    def simple_reset(self):
        self.spawn = [7, 15]
        self.spawnfl = [10, 17]
        self.spawnfr = [10, 13]
        self.spawnbr = [4, 13]
        self.spawnbl = [4, 17]

        self.bot_pos = np.array([self.spawn,
                        self.spawnfl,
                        self.spawnfr,
                        self.spawnbr,
                        self.spawnbl
                        ], dtype=np.float32)
                        
        self.goal_pos = [17, 15]
        self.distance_to_goal = 10

    def rotate(self, angle):
        rads = np.deg2rad(angle)
        c_y, c_x = self.bot_pos[0]
        return np.array(
            [
                [
                    c_y + np.sin(rads) * (px - c_x) + np.cos(rads) * (py - c_y),
                    c_x + np.cos(rads) * (px - c_x) - np.sin(rads) * (py - c_y)
                ]
                for py, px in self.bot_pos[1:]
            ]
        , dtype=np.float32)
    
    def check_collision(self, obstacles):
        bot_contour = np.array([[[round(point[1]), round(point[0])]] for point in self.bot_pos[1:]], dtype=np.int32)
       
        for det in obstacles:
            if cv.pointPolygonTest(bot_contour, (det[1], det[0]), False) >= 0:
                return True
            
        if len(obstacles) < 360:
            return True
        return False
    
    def limit_angle(self, angle):
        if (angle < 0.0):
                angle += TAU
            
        if (angle >= TAU):
               angle -= TAU

        return angle
