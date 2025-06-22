"""camera_pid controller."""

from controller import Display, Keyboard, Robot, Camera
from vehicle import Car, Driver
import numpy as np
import cv2
from datetime import datetime
import os
import math
import time
from PIL import Image
import joblib
import pandas as pd
import matplotlib.pyplot as plt

DATASET_PATH = r"C:\Users\eclip\OneDrive\Documentos\MNA\6to Trimestre\Navegacion Autonoma\Navigation Dataset"
MODEL_PATH = os.path.join(os.getcwd(), "second_model.joblib")
WARNING_DISTANCE = 20
STOP_DISTANCE = 5
SPEED = 20


#Getting image from camera
def get_image(camera):
    """
    Get the image from the Webots camera from a camera instance
    """
    raw_image = camera.getImage()  
    
    image = np.frombuffer(raw_image, np.uint8).reshape(
        (camera.getHeight(), camera.getWidth(), 4)
    )

    image = image[:,:,:3]
    pil_image = Image.fromarray(image).convert("L")

    box = (0, 70, 290, 160)
    crop_image = pil_image.crop(box)

    return crop_image


#Display image 
def display_image(display, image):
    # Image to display
    
    image_rgb = np.dstack((image, image,image,))
    
    # Display image
    image_ref = display.imageNew(
        image_rgb.tobytes(),
        Display.RGB,
        width=image_rgb.shape[1],
        height=image_rgb.shape[0],
    )
    display.imagePaste(image_ref, 0, 0, False)


#initial angle and speed 
manual_steering = 0
steering_angle = 0
angle = 0.0
speed = 20

# set target speed
def set_speed(kmh):
    global speed            #robot.step(50)

#update steering angle
def set_steering_angle(wheel_angle):
    global angle, steering_angle
    # Check limits of steering
    if (wheel_angle - steering_angle) > 0.1:
        wheel_angle = steering_angle + 0.1
    if (wheel_angle - steering_angle) < -0.1:
        wheel_angle = steering_angle - 0.1
    steering_angle = wheel_angle

    if wheel_angle == 0 : 
        global manual_steering
        global new_manual_steering 
        manual_steering = 0
        new_manual_steering = 0

    # limit range of the steering angle
    if wheel_angle > 0.8:
        wheel_angle = 0.8
    elif wheel_angle < -0.8:
        wheel_angle = -0.8
    # update steering angle
    angle = wheel_angle

#validate increment of steering angle
def change_steer_angle(inc):
    global manual_steering
    global new_manual_steering
    # Apply increment
    new_manual_steering = manual_steering + inc
    # Validate interval 
    if new_manual_steering <= 25.0 and new_manual_steering >= -25.0: 
        manual_steering = new_manual_steering
        set_steering_angle(manual_steering * 0.02)
    # Debugging
    if manual_steering == 0:
        print("going straight")
    else:
        turn = "left" if steering_angle < 0 else "right"
        print("turning {} rad {}".format(str(steering_angle),turn))


# main
def main():
    # Create the Robot instance.
    robot = Car()
    driver = Driver()

    # Get the time step of the current world.
    timestep = int(robot.getBasicTimeStep())

    # Initialize distance sensor 
    ds = robot.getDevice('ds_front')
    ds.enable(timestep)

    # Initialize radar 
    radar = robot.getDevice("radar")
    radar.enable(timestep)

    # Load CNN model
    model = joblib.load(MODEL_PATH)

    # Create camera instance
    camera = robot.getDevice("camera")
    camera.enable(timestep)  # timestep

    # processing display
    display_img = Display("display_image")

    #create keyboard instance
    keyboard=Keyboard()
    keyboard.enable(timestep)

    while robot.step() != -1:
        # Get image from camera
        image = get_image(camera)
        # print(image.size)

        arr = np.array(image, dtype="float32") / 255.0

        input_tensor = np.expand_dims(np.array(arr), axis=0)

        st_angle = model.predict(input_tensor)
        
        # # Sense distance 
        # distance = ds.getValue()
        # print(ds.getValue())

        num_targets = radar.getNumberOfTargets()
        targets = radar.getTargets()
        print("targets: {}".format(num_targets))
        # print(f"Distance: {targets[1]}")
        
        if num_targets > 0:
            print("OBSTACLE DETECTED")
            for i in targets:
                if i.distance < WARNING_DISTANCE and i.distance > STOP_DISTANCE:
                    print("DECREASING SPEED")
                    speed = 8
                elif i.distance < STOP_DISTANCE:
                    print("STOPPING VEHICLE")
                    speed = 0
                else: 
                    speed = 20
        else:
            speed = SPEED
        
        
        print(f"Speed: {speed}\n Steering angle: {st_angle}")


        set_steering_angle(st_angle.item())
            
    
        #update angle and speed
        driver.setSteeringAngle(angle)
        driver.setCruisingSpeed(speed)

    

if __name__ == "__main__":
    main()
