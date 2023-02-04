import glob 
import os 
import sys 
import random
import weakref
import numpy as np 
import cv2  
import tensorflow_probability as tfp 
import tensorflow as tf
import time
import math
try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    print("no file")

import carla 
from carla import ColorConverter as cc 

global current_frame
current_frame = None
car_speed = 0 
steering_strength = 0.35
gas_strength = 1.0
brake_strength = 0.6
actions = {
            0: [gas_strength -.75, -steering_strength],
            1: [gas_strength -.75, steering_strength],
            2: [gas_strength, 0.],
            3: [-brake_strength, 0]
            }



def get_img( image):
        global current_frame
        # image.convert(cc.LogarithmicDepth)
        image = np.array(image.raw_data)
        img = image.reshape((1,220,220,4))
        img = img[:,110:220,:,:3]
        
        # print("img" ,img)
        current_frame = img


load_model_action = tf.keras.models.load_model("./steer_model")




client = carla.Client('localhost',2000)
client.set_timeout(10.0)
print("Client")
world = client.load_world('Town02')
map_ = world.get_map()
blueprint_vehicle = world.get_blueprint_library().filter('model3')[0]
spawn_points = map_.get_spawn_points()
try:
    spawn_point = random.choice(spawn_points)
except:
    raise RuntimeError("no spwan location there")

vehicle = world.try_spawn_actor(blueprint_vehicle , spawn_point)
print("vehicle spwaned")
blueprint = world.get_blueprint_library().find('sensor.camera.rgb')
blueprint.set_attribute('image_size_x', '220')
blueprint.set_attribute('image_size_y', '220')
blueprint.set_attribute('fov', '120')
position = carla.Transform(carla.Location(x=1.5 ,z=3))
try:
    camera = world.try_spawn_actor(blueprint,position , attach_to = vehicle)
    print("Spawning camera")
except:
    raise RuntimeError("Could not spawn camera sensor")

camera.listen(lambda data : get_img(data))

while True:
    # print("current_frame"  , current_frame)
    try:
        if current_frame== None:
            pass 

    except:
        if current_frame.all() :
            print("controlling vehicle")
        
            cv2.imshow("img" , current_frame[0])
            cv2.waitKey(100)
            action = load_model_action(current_frame.astype(np.float))
            prob = action.numpy()
            control = carla.VehicleControl()
            
            vel = vehicle.get_velocity()
            vel = 3.6 * math.sqrt(vel.x ** 2 + vel.y ** 2 + vel.z ** 2)
            print("velocty" ,vel)
            if vel <12:
                control.throttle = 0.2
            else:
                control.brake= 1
            
            print("prob" , prob[0][0]/100)
            control.steer = float(prob[0][0]/100)
            vehicle.apply_control(control)
            time.sleep(0.1)



