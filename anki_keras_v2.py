#!/usr/bin/env python3

"""
This script uses Anki Vector's camera to take a photo and perform
image recognition with Keras built in ResNet50 neural network and
the Image-Net database.
"""

import os
import random
import sys
import time

import anki_vector

from keras.applications import resnet50
from keras.preprocessing import image

import numpy as np

try:
    from PIL import Image
except ImportError:
    sys.exit('Cannot import from PIL: Do `pip3 install '
             '--user Pillow` to install')


# Load the ResNet50 model from Keras
resnet_model = resnet50.ResNet50(weights='imagenet')

# setup robot object
robot = anki_vector.Robot(anki_vector.util.parse_command_args().
                          serial, enable_camera_feed=True)
screen_dimensions = anki_vector.screen.SCREEN_WIDTH, anki_vector.screen.SCREEN_HEIGHT
# define the path where this script it
current_directory = os.path.dirname(os.path.realpath(__file__))
# check for resources folder and creat it if necessary
image_path = os.path.join(current_directory, 'resources')
if not os.path.exists(image_path):
    os.makedirs(image_path)
# path to find the image later
image_file = os.path.join(current_directory, 'resources', "latest.jpg")


def format_picture(image_file):
    '''
    ResNet50 takes images of 224x224 pixels. By using parameters
    'target_size=(224, 224)' the wide angle image on Vector is
    squished making the image recognition more difficult, this
    changes the image into the appropriate size ahead of time
    by cropping to the center of the image.
    '''
    print('formatting picture')
    im = Image.open(image_file)
    width, height = im.size   # Get dimensions

    left = (width - 224)/2
    top = (height - 224)/2
    right = (width + 224)/2
    bottom = (height + 224)/2

    im = im.crop((left, top, right, bottom))
    im.save(image_file)


def detect_labels(path):
    '''
    The image recogition function using Keras and Image-Net.
    '''
    # resnet_model = resnet50.ResNet50(weights='imagenet')
    print('Detect labels, image = {}'.format(path))

    # Load Keras' ResNet50 model that was pre-trained
    # against the ImageNet database
    model = resnet50.ResNet50()

    # Load the image file, resizing it to 224x224
    # pixels (required by this model)
    img = image.load_img(path)

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
    predicted_classes = resnet50.decode_predictions(predictions, top=3)

    robot_say("My top three guesses are")

    for imagenet_id, name, likelihood in predicted_classes[0]:
        # robot_say("{}: {:2f} likelihood".format(name, likelihood))
        robot_say("{}".format(name))
        time.sleep(1)


def connect_robot():
    print('Connect to Vector...')
    robot.connect()


def disconnect_robot():
    robot.disconnect()
    print('Vector disconnected')


def stand_by():
    # If necessary, move Vector's Head and Lift to make it easy to see his face
    robot.behavior.set_lift_height(0.0)
    robot.behavior.set_head_angle(anki_vector.util.degrees(6.0))


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
    robot_say('What is that...?')
    time.sleep(1)

    show_image(image_file)
    time.sleep(1)

    save_image(image_file)
    time.sleep(1)

    format_picture(image_file)

    robot_say('Hang on, let me think about this for a minute.')
    detect_labels(image_file)
    time.sleep(1)

    show_image(image_file)
    time.sleep(1)

    robot_say('Then again, I\'m not too smart.')

    close_camera()

    robot_say('Goodbye!')


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
