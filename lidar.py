import numpy as np

import cv2 as cv
import math

class Lidar:
    def __init__(self, pos, angle, height, width):
        self.pos = [pos[0], pos[1]] #(y, x)
        self.width = width
        self.height = height
        self.angle = angle
        self.beam_number = 360
        self.radial_noise = 0.0
        self.axial_noise = 0.4 #cells
        self.sigma = np.array([self.axial_noise, self.radial_noise])
        self.range = 400

    def update_pose(self, pos, angle):
        self.pos = [pos[0], pos[1]]
        self.angle = angle

    def get_points_on_line(self, x1, y1, x2, y2):
        points = []
        issteep = abs(y2-y1) > abs(x2-x1)
        if issteep:
            x1, y1 = y1, x1
            x2, y2 = y2, x2
        rev = False
        if x1 > x2:
            x1, x2 = x2, x1
            y1, y2 = y2, y1
            rev = True
        deltax = x2 - x1
        deltay = abs(y2-y1)
        error = int(deltax / 2)
        y = y1
        ystep = None
        if y1 < y2:
            ystep = 1
        else:
            ystep = -1
        for x in range(x1, x2 + 1):
            if issteep:
                points.append((y, x))
            else:
                points.append((x, y))
            error -= deltay
            if error < 0:
                y += ystep
                error += deltax
        # Reverse the list if the coordinates were reversed
        if rev:
            points.reverse()
        return points
    
    def distance(self, x, y):
        py = (y - self.pos[0])**2
        px = (x - self.pos[1])**2
        return math.sqrt(px + py)
    
    
    def noisify(self, distance, angle):
        mean = np.array([distance,angle])
        covariance = np.diag(self.sigma ** 2) #noise of distance and angle measurements are not correlated, hence only diagonal is non-zero. https://www.cuemath.com/algebra/covariance-matrix/
        distance, angle = np.random.multivariate_normal(mean,covariance) #gets the actual noisy values from Gaussian distribution  
        distance = max(distance, 0)
        angle = max(angle, 0)
        return [distance, angle]

    def sense_obstacles(self, map):
        datacart = [] #stores distance and angle of ONLY WALLS from bot's position  
        datacart_global = []
        datapolar = []
        y1, x1 = round(self.pos[0]), round(self.pos[1])  
        # print(x1, y1, self.angle)
        for angle in np.linspace(0, 2*np.pi, self.beam_number, False): #scan from 0 to 2pi, 1 degree intervals (Resolution). For every angle, check if there is a wall
            beam_angle = self.angle + angle
            y2, x2 = (round(y1 + self.range * np.sin(beam_angle)), round(x1 + self.range * np.cos(beam_angle))) #coordinate of end of line segment
            
            line_pnts = self.get_points_on_line(x1, y1, x2, y2) #returns in x, y format
            
            for x, y in line_pnts: #If wall is within laser's path ...

                if 0 <= x < self.width and 0 <= y < self.height: #if within the window/map. Reference point is pixel (0,0).
                    color = map[y, x, :] #extract RGB value on map at every point of iteration within laser's path and do the quick check below ... 
                    if (color[0], color[1], color[2]) == (255, 255, 255): #if color is white, aka the walls, calculate this distance from the robot
                        distance = self.distance(x, y) #References method above 
                        output = self.noisify(distance, beam_angle) #add uncertainty to measurements
                        noisy_dets = self.polar_to_cartesian(x1, y1, output)
                        datacart.append(noisy_dets[0]) #[rel_y, rel_x, glo_y, glo_x] --> This is the return 
                        datacart_global.append(noisy_dets[1])
                        datapolar.append([output[1], output[1]]) #a, d
                        break
        
        #when sensor completes a full turn, return the data to be drawn in the map ...which is the responsiblity of the buildEnvironment class in the env.py file
        # if len(data)>0:
        return datacart, datacart_global, datapolar #[[y, x]]
        # else:
        #     return False

    def polar_to_cartesian(self, x0, y0, polar):
        rel_x2, rel_y2, glo_x2, glo_y2 = (round(polar[0] * np.cos(polar[1])), round(polar[0] * np.sin(polar[1])), round(x0 + polar[0] * np.cos(polar[1])), round(y0 + polar[0] * np.sin(polar[1]))) #coordinate of end of line segment
        return ([rel_y2, rel_x2], [min(max(glo_y2, 0), self.height-1), min(max(glo_x2, 0), self.width-1)])
