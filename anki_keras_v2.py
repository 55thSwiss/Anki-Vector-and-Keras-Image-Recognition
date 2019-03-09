#!/usr/bin/env python3

"""

"""

import os
import sys
import time
import random

try:
    from PIL import Image
except ImportError:
    sys.exit('Cannot import from PIL: Do `pip3 install '
             '--user Pillow` to install')

import anki_vector

from keras.applications import resnet50
from keras.preprocessing import image

import numpy as np

# Load the ResNet50 model
resnet_model = resnet50.ResNet50(weights='imagenet')

robot = anki_vector.Robot(anki_vector.util.parse_command_args().serial, enable_camera_feed=True)
screen_dimensions = anki_vector.screen.SCREEN_WIDTH, anki_vector.screen.SCREEN_HEIGHT
current_directory = os.path.dirname(os.path.realpath(__file__))
image_file = os.path.join(current_directory, 'resources', "latest.jpg")


def detect_labels(path):
    # resnet_model = resnet50.ResNet50(weights='imagenet')
    print('Detect labels, image = {}'.format(path))

    # Load Keras' ResNet50 model that was pre-trained
    # against the ImageNet database
    model = resnet50.ResNet50()

    # Load the image file, resizing it to 224x224
    # pixels (required by this model)
    img = image.load_img(path, target_size=(224, 224))

    # Convert the image to a numpy array
    x = image.img_to_array(img)

    # Add a forth dimension since Keras expects a list of images
    x = np.expand_dims(x, axis=0)

    # Scale the input image to the range used in the trained network
    x = resnet50.preprocess_input(x)

    # Run the image through the deep neural network to make a prediction
    predictions = model.predict(x)

    # Look up the names of the predicted classes. Index zero
    # is the results for the first image.
    predicted_classes = resnet50.decode_predictions(predictions, top=1)

    print("This is an image of:")

    for imagenet_id, name, likelihood in predicted_classes[0]:
        print(" - {}: {:2f} likelihood".format(name, likelihood))
        name = name.format(name)
        print('this is the ' + name)
        return ("{}".format(name))


def connect_robot():
    print('Connect to Vector...')
    robot.connect()


def disconnect_robot():
    robot.disconnect()
    print('Vector disconnected')


def stand_by():
    # If necessary, move Vector's Head and Lift to make it easy to see his face
    robot.behavior.set_lift_height(0.0)


def show_camera():
    print('Show camera')
    robot.camera.init_camera_feed()
    robot.vision.enable_display_camera_feed_on_face(True)


def close_camera():
    print('Close camera')
    robot.vision.enable_display_camera_feed_on_face(False)
    robot.camera.close_camera_feed()


def save_image(file_name):
    print('Save image')
    robot.camera.latest_image.save(file_name, 'JPEG')


def show_image(file_name):
    print('Show image = {}'.format(file_name))

    # Load an image
    image = Image.open(file_name)

    # Convert the image to the format used by the Screen
    print("Display image on Vector's face...")
    screen_data = anki_vector.screen.convert_image_to_screen_data(
                                image.resize(screen_dimensions))
    robot.screen.set_screen_with_image_data(screen_data, 5.0, True)


def robot_say(text):
    print('Say {}'.format(text))
    robot.say_text(text)


def analyze():
    stand_by()
    show_camera()
    robot_say('My lord, I found something interesting. Give me 5 seconds.')
    time.sleep(5)

    robot_say('Prepare to take a photo')
    robot_say('3')
    time.sleep(1)
    robot_say('2')
    time.sleep(1)
    robot_say('1')
    robot_say('Cheers')

    save_image(image_file)
    show_image(image_file)
    time.sleep(1)

    robot_say('Start to analyze the object')
    text = detect_labels(image_file)
    show_image(image_file)
    robot_say('Might be {}'.format(text))

    close_camera()
    robot_say('Over, goodbye!')


def main():
    while True:
        connect_robot()
        try:
            analyze()
        except Exception as e:
            print('Analyze Exception: {}', e)

        disconnect_robot()
        time.sleep(random.randint(30, 60 * 5))


if __name__ == "__main__":
    main()
