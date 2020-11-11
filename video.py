""" AUTHOR : AARUSHI KOHLI """

""" COMPUTER VISION AND INTERNET OF THINGS """

"""INTERNSHIP AT THE SPARKS FOUNDATION."""

"""TASK 1: IMPLEMENT AN OBJECT DETECTOR WHICH IDENTIFIES THE CLASSES OF THE OBJECTS IN AN IMAGE OR VIDEO."""

   
""" DETECTION IN VIDEO """



"""Object detection is a technology related to computer vision that deals with detecting instances of semantic objects of a certain class (such as humans
,buildings, or vehicles) in digital videos and images. Object detection has proved to be a prominent module for numerous important applications like
video surveillance, autonomous driving, face detection, etc."""


""" Various specialized algorithms have been developed to detect and recognize objects in vidoes and images like RCNNs,SSD,RetinaNet and YOLO.Here we 
are gonna use a trained YOLOv3 model to perform detection."""

""" This model can recognize 80 different objects given below:
person, bicycle, car, motorcycle, airplane,
bus, train, truck, boat, traffic light, fire hydrant, stop_sign,
parking meter, bench, bird, cat, dog, horse, sheep, cow, elephant, bear, zebra, 
giraffe, backpack, umbrella, handbag, tie, suitcase, frisbee, skis, snowboard,
sports ball, kite, baseball bat, baseball glove, skateboard, surfboard, tennis racket,
bottle, wine glass, cup, fork, knife, spoon, bowl, banana, apple, sandwich, orange,
broccoli, carrot, hot dog, pizza, donot, cake, chair, couch, potted plant, bed,
dining table, toilet, tv, laptop, mouse, remote, keyboard, cell phone, microwave,
oven, toaster, sink, refrigerator, book, clock, vase, scissors, teddy bear, hair dryer,
toothbrush. """
 
""" For the task we will download any random video from net and copy it in a folder with YOLOv3 Model and a python file named as test.py ."""


from imageai.Detection import VideoObjectDetection
import os

execution_path = os.getcwd()

""" we create an instance of the VideoObjectDetection class."""

detector = VideoObjectDetection()

""" we set the model type to YOLOv3, which corresponds to the YOLO model we downloaded and copied to the folder."""

detector.setModelTypeAsYOLOv3()

"""we set the model path to the file path of the model file we copied into the folder."""

detector.setModelPath( os.path.join(execution_path , "yolo.h5"))

""" we load the model into the instance of the VideoObjectDetection class that we created."""

detector.loadModel()

""" Now,we call the detectObjectsFromVideo function and parsed the following values into it:

i. input_file_path: This refers to the file path of the video we copied into the folder.

ii. output_file_path: This refers to the file path to which the detected video will be saved.

iii. frames_per_second: This refers to the number of image frames that we want the detected video to have within a second.

iv. log_progress: This is used to state that the detection instance should report the progress of the detection in the command line interface."""

video_path = detector.detectObjectsFromVideo(input_file_path=os.path.join( execution_path, "Pexels Videos 2099536.mp4"),
                                output_file_path=os.path.join(execution_path, "Pexels Videos 2099536_1.mp4")
                                , frames_per_second=29, log_progress=True)
print(video_path)
""" THIS DETECTION PROCESS MAY TAKE A FEW MINUTES.  ONCE DONE , THE DETECTED VIDEO CAN BE FOUND IN THE FOLDER . """