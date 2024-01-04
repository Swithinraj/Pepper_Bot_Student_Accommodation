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
import speech_recognition as sr
from datetime import datetime


class ChatBot:
    def __init__(self, name):
        print("----- Starting up", name, "-----")
        self.name = name

    def speech_to_text(self):
        recognizer = sr.Recognizer()
        while True:  # Keep listening until valid input is received
            with sr.Microphone() as mic:
                print("Listening...")
                recognizer.adjust_for_ambient_noise(mic)
                audio = recognizer.listen(mic)
            try:
                text = recognizer.recognize_google(audio)
                print("Your input  --> ", text)
                return text.lower()  # Return the recognized text
            except sr.UnknownValueError:
                print("ERROR: Could not understand you. Please say again.")
                self.text_to_speech("Could not understand you. Please say again.")

    def text_to_speech(self, text):
        to_speak = gTTS(text=text, lang="en")
        filename = 'voice.mp3'
        to_speak.save(filename)
        playsound(filename)
        os.remove(filename)
        return text

chat_bot= ChatBot("Pepper")

class BehaviorRealizer():
    def __init__(self):
        simulation_manager = SimulationManager()
        client = simulation_manager.launchSimulation(gui=True)
        self.pepper = simulation_manager.spawnPepper(client, spawn_ground_plane=True)


def rasa(name):

    chat_bot.text_to_speech(f"Ask me something you want to know about your accommodation {name}.")
    try:
        message = "start"
        while message is not None:
            bot_message = ""
            prev_message = message
            message = chat_bot.speech_to_text()
            message= message.lower()

            # Check if user wants to exit the conversation
            if message in ['bye','goodbye', 'thank you','thanks', 'see you']:
                if prev_message =="start":
                    chat_bot.text_to_speech ("already?, maybe did you say bye by mistake? Yes or NO")
                    bye= chat_bot.speech_to_text()
                    if bye.lower() == "yes":
                        chat_bot.text_to_speech("Then, continue with your request")
                        message= chat_bot.speech_to_text()
                        pass
                    else:
                        chat_bot.text_to_speech("okay then, Until next time!")
                        break
                else:
                    chat_bot.text_to_speech("Until next time!")
                    break

            # Apply profanity filter and remove censored words
            filtered_message = profanity.censor(message)

            print("Your request: ",filtered_message)

            # Send user message to Rasa server
            r = requests.post('http://localhost:5002/webhooks/rest/webhook', json={"message": filtered_message})

            # Process bot's response
            for response in r.json():
                bot_message = response.get('text', '')

            # chat_bot.text_to_speech the bot's response
            timings_explain(bot_message)

        return (bot_message)

    except Exception as e:
        print(f"An error occurred: {e}")

        return None

# Capturing new faces
def embedding():
    chat_bot.text_to_speech("what is your name?")
    name = chat_bot.speech_to_text()
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
    chat_bot.text_to_speech("please be still, while i capture you!")

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
    chat_bot.text_to_speech("Thank you for your patience!")
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
                    chat_bot.text_to_speech("Could not recognize you!, please wait.")
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
    speech_thread = threading.Thread(target=chat_bot.text_to_speech, args=(text,))
    wave_thread = threading.Thread(target=wave_to_human, args=(behavior_realizer_class.pepper, no_wave))

    wave_thread.start()
    speech_thread.start()
    wave_thread.join()
    speech_thread.join()

def timings_explain(text):

    #Thread initialization
    speech_thread = threading.Thread(target=chat_bot.text_to_speech, args=(text,))
    nod_thread = threading.Thread(target=nod_to_explain, args=(behavior_realizer_class.pepper,))

    nod_thread.start()
    speech_thread.start()

    while speech_thread.is_alive():
        time.sleep(1)

    nod_thread.join()
    speech_thread.join()

def reset_all_joints(robot):
    robot.setAngles("LShoulderPitch", 1.0, 1.0)
    time.sleep(0.3)
    robot.setAngles("RShoulderPitch", 1.0, 1.0)
    time.sleep(0.3)

def greeting_based_on_time():
    current_time = datetime.now()
    current_hour = current_time.hour

    if 5 <= current_hour < 12:
        return "Good morning!"
    elif 12 <= current_hour < 18:
        return "Good afternoon!"
    else:
        return "Good evening!"
    

behavior_realizer_class = BehaviorRealizer()
recognized = False

reset_all_joints(behavior_realizer_class.pepper)
name= stream()

if name!= "Unknown" and not recognized:
    recognized = True
    timings_greets(f"Hello, {name}!" + greeting_based_on_time(), 2)
    bot_message=rasa(name)
    time.sleep(3)
    timings_greets(f"Bye, {name}!", 4)

#rasa run -m models --endpoints endpoints.yml --port 5002 --credentials credentials.yml