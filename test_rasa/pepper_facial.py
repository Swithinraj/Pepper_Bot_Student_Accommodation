from imutils.video import VideoStream
import face_recognition
import cv2
import os
import pickle
import time
from qibullet import SimulationManager
from gtts import gTTS
from playsound import playsound
import numpy as np
import random
import requests
from profanity import profanity
import threading
from inference_engine import inference_engine
import speech_recognition as sr

#file path for the bayesian network model

file_path_b= "/home/swithin/Desktop/hci/RASA/test_rasa/hcir_project.dsl"

class BehaviorRealizer():
    def __init__(self):
        simulation_manager = SimulationManager()
        client = simulation_manager.launchSimulation(gui=True)
        self.pepper = simulation_manager.spawnPepper(client, spawn_ground_plane=True)

def preference_collection(file_path):
    preferences = []  # List to store accommodation preferences

    # Dialogue flow to gather accommodation preferences
    print("Let's find you the perfect accommodation. I will ask you questions regarding your preferences for the accommodation. LET'S START!")
    speak("Let's find you the perfect accommodation. I will ask you questions regarding your preferences for the accommodation. LET'S START!")

    print("Could you tell me about your preferred location for accommodation? Bonn, Rheinbach, or Sankt Augustin")
    speak("Could you tell me about your preferred location for accommodation? Bonn, Rheinbach, or Sankt Augustin")
    preferences.append(input("Preferred location: ").lower())

    print("What is your budget? expensive, medium or affordable?")
    speak("What is your budget? expensive, medium or affordable?")
    preferences.append(input("Your budget: ").lower())

    print("What is your preferred space of living? small, medium or large?")
    speak("What is your preferred space of living? small, medium or large?")
    preferences.append(input("Preferred space: ").lower())

    print("Which mode of transport is your highest priority? s-bahn, bus or fahrrad?")
    speak("Which mode of transport is your highest priority? s-bahn, bus or fahrrad?")
    transport= input("Preferred mode of transport: ").lower()
    if transport =="s-bahn":
        transport = "sbahn"
    else:
        pass
    preferences.append(transport)

    print("Do you want to live near or far from the central station?")
    speak("Do you want to live near or far from the central station?")
    preferences.append(input("Preferred distance from HBF: ").lower())

    print("What clubs are you interested in? literature, sports, dance, or gaming?")
    speak("What clubs are you interested in? literature, sports, dance, or gaming?")
    preferences.append(input("Preferred club: ").lower())

    print("Thank you for giving me your preferences! Please wait while I find the best suitable accommodation for you...")
    speak("Thank you for giving me your preferences! Please wait while I find the best suitable accommodation for you.")

    time.sleep(2)

    accommodation = inference_engine(file_path, preferences)

    text= f"The best accoommodation for you is, {accommodation}!"

    print(text)

    timings_explain(text)
    
    return accommodation

def rasa(name):
    try:
        message = "start"
        while message is not None:
            bot_message = ""
            prev_message = message
            message = input("Enter your question: ")

            # Check if user wants to exit the conversation
            if message.lower() in ['bye', 'thanks']:
                if prev_message =="start":
                    speak ("already?, maybe did you say bye by mistake?")
                    bye= input("Yes or No?")
                    if bye.lower() == "yes":
                        speak("Then, continue with your request")
                        message= input("Continue with your request: ")
                        message= message.lower()
                        pass
                    else:
                        speak("okay then, Until next time!")
                        break
                else:
                    speak("Until next time!")
                    break


            # Lowercase user input for consistency
            message = message.lower()

            # Apply profanity filter and remove censored words
            filtered_message = profanity.censor(message)

            print("Your message: ",filtered_message)

            # Send user message to Rasa server
            r = requests.post('http://localhost:5002/webhooks/rest/webhook', json={"message": filtered_message})

            # Process bot's response
            for response in r.json():
                bot_message = response.get('text', '')

            # Speak the bot's response
            timings_explain(bot_message)

        return (bot_message)

    except Exception as e:
        print(f"An error occurred: {e}")

        return None

# Capturing new faces
def embedding():
    speak("Please enter your name.")
    name = input("Enter your name: ")
    ref_id = random.randint(100, 1000)

    try:
        with open("ref_name.pkl", "rb") as f:
            ref_dict = pickle.load(f)
    except FileNotFoundError:
        ref_dict = {}

    ref_dict[ref_id] = name

    with open("ref_name.pkl", "wb") as f:
        pickle.dump(ref_dict, f)

    try:
        with open("ref_embed.pkl", "rb") as f:
            embed_dict = pickle.load(f)
    except FileNotFoundError:
        embed_dict = {}

    webcam = VideoStream(src=0).start()
    speak("please be still, while i capture you!")

    for i in range(5):
        print("Capturing")
        while True:
            frame = webcam.read()
            if frame is None or frame.size == 0:
                print("Frame is empty or invalid.")
                continue

            cv2.imshow('Video', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            rgb_small_frame = small_frame[:, :, ::-1]
            face_locations = face_recognition.face_locations(rgb_small_frame)
            if face_locations:
                face_encoding = face_recognition.face_encodings(frame)[0]
                if ref_id in embed_dict:
                    embed_dict[ref_id].append(face_encoding)
                else:
                    embed_dict[ref_id] = [face_encoding]
                break

    with open("ref_embed.pkl", "wb") as f:
        pickle.dump(embed_dict, f)

    webcam.stop()
    speak("Thank you for your patience!")
#Recognising the faces and if faces does not exist in data, embedding function is called
def stream():
    # Load reference data from pickle files
    try:
        with open("ref_name.pkl", "rb") as f:
            ref_dict = pickle.load(f)
    except FileNotFoundError:
        ref_dict = {}

    try:
        with open("ref_embed.pkl", "rb") as f:
            embed_dict = pickle.load(f)
    except FileNotFoundError:
        embed_dict = {}

    known_face_encodings = []
    known_face_names = []

    for ref_id, embed_list in embed_dict.items():
        for embed in embed_list:
            known_face_encodings.append(embed)
            known_face_names.append(ref_id)

    recognized = False
    names= None
    while True:
        video_capture = cv2.VideoCapture(0)
        ret, frame = video_capture.read()
        video_capture.release()

        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        if face_encodings:
            for face_encoding in face_encodings:
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)

                if len(face_distances) > 0:
                    best_match_index = np.argmin(face_distances)
                    if matches[best_match_index]:
                        name = known_face_names[best_match_index]
                        if name in ref_dict:
                            if not recognized:
                                # print(f"Hello, {ref_dict[name]}!")
                                names= ref_dict[name]
                                recognized = True
                                break  # Break out of the loop once recognized
                        else:
                            print("Name not found in ref_dict:", name)
            else:
                if recognized:  # If recognized, break out of the main loop
                    break
                else:
                    speak("Could not recognize you!, please wait.")
                    embedding()  # Start embedding if face is detected but not recognized
                    break
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    if names== None:
        names=stream()
        main_name= names
        return main_name
    else:
        return names

def speak(text):
    to_speak = gTTS(text=text, lang='en')
    filename = 'voice.mp3'
    to_speak.save(filename)
    playsound(filename)

def wave_to_human(robot, no_wave):
    robot.setAngles("LShoulderPitch", 1.0, 1.0)
    robot.setAngles("RShoulderPitch", -1.0, 1.0)
    robot.setAngles("RShoulderRoll", 0.0, 0.0)
    robot.setAngles("RElbowYaw", 0.0, 0.0)
    robot.setAngles("RWristYaw", 0.0, 0.0)
    for i in range(no_wave):
        robot.setAngles("RElbowRoll", -0.5, 1.0)
        time.sleep(1.0)
        robot.setAngles("RElbowRoll", 0.5, 0.0)
        time.sleep(1.0)
    robot.setAngles("RShoulderPitch", 1.0, 1.0)
    robot.setAngles("RElbowYaw", 0.0, 1.0)
    robot.setAngles("RElbowRoll", 0.5, 1.0)

def nod_to_explain(robot):
    nod_no = 5
    for i in range(nod_no):
        robot.setAngles("HeadPitch", -0.6, 0.08)
        time.sleep(0.3)
        robot.setAngles("HeadPitch", 0.8, 0.15)
        time.sleep(1)
        robot.setAngles("HeadPitch", 0.0, 0.1)

def timings_greets(text, no_wave):

    #Thread initialization
    speak_thread = threading.Thread(target=speak, args=(text,))
    wave_thread = threading.Thread(target=wave_to_human, args=(behavior_realizer_class.pepper, no_wave))

    wave_thread.start()
    speak_thread.start()
    wave_thread.join()
    speak_thread.join()

def timings_explain(text):

    #Thread initialization
    speak_thread = threading.Thread(target=speak, args=(text,))
    nod_thread = threading.Thread(target=nod_to_explain, args=(behavior_realizer_class.pepper,))

    nod_thread.start()
    speak_thread.start()

    while speak_thread.is_alive():
        time.sleep(1)

    nod_thread.join()
    speak_thread.join()

def reset_all_joints(robot):
    robot.setAngles("LShoulderPitch", 1.0, 1.0)
    time.sleep(0.3)
    robot.setAngles("RShoulderPitch", 1.0, 1.0)
    time.sleep(0.3)


behavior_realizer_class = BehaviorRealizer()
recognized = False

reset_all_joints(behavior_realizer_class.pepper)
name= stream()

if name!= "Unknown" and not recognized:
    print(f"Hello, {name}!")
    recognized = True
    timings_greets(f"Hello, {name}!" + "do you want help with your accommodation selection?", 2)
    selection= input("Yes or No? ").lower()
    if selection == "yes" or selection == "y":
        speak ("Cool!")
        best_accommodation = preference_collection(file_path_b)
        bot_message=rasa(name)

    else:
        speak("ah it is sad that i cant help with other matters with my current capability. I am sorry.")

    timings_greets(f"Bye, {name}!", 4)

#rasa run -m models --endpoints endpoints.yml --port 5002 --credentials credentials.yml