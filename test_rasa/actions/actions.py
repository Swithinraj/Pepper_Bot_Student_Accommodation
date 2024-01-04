from typing import Any, Text, Dict, List
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
import pyAgrum as gum
import numpy as np
import random
import time



class ActionInferAccommodation(Action):
    def name(self) -> Text:
        return "action_infer_accommodation"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        
        file_path= "/home/swithin/Desktop/hci/RASA/test_rasa/hcir_project.dsl"

        # Retrieve slot values
        location = tracker.get_slot("location")
        budget = tracker.get_slot("budget")
        size = tracker.get_slot("size")
        transport = tracker.get_slot("transport")
        dist = tracker.get_slot("dist")
        clubs = tracker.get_slot("clubs")

        location = location if location is not None else 0
        budget = budget if budget is not None else 0
        size = size if size is not None else 0
        transport = transport if transport is not None else 0
        dist = dist if dist is not None else 0
        clubs = clubs if clubs is not None else 0

        # Store slot values in a list
        preferences = [location, budget, size, transport, dist, clubs]

        preferences = word_check(preferences)

        accommodation = inference_engine(file_path, preferences)
        
        dispatcher.utter_message(text= f"The best accommodation for you is, {accommodation}!")

        time.sleep(3)

        return []
    

def levenshtein_distance(s1, s2):
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]

def word_check(preferences):
    predefined_words = [
        'bonn', 'sankt_augustin', 'rheinbach', 'expensive', 'affordable', 'medium',
        'small', 'large', 'near', 'far', 'sbahn', 'bus', 'fahrrad', 'literature',
        'sports', 'dance', 'gaming'
    ]

    correct_preferences = []
    for pref in preferences:
        if pref is not None and pref != 0:  # Check for None and not equal to 0
            similarity_scores = {word: levenshtein_distance(pref, word) for word in predefined_words}
            best_match = min(similarity_scores, key=similarity_scores.get)
            correct_preferences.append(best_match)
        else:
            correct_preferences.append(0)

    return correct_preferences


def inference_engine(filepath, preferences):
    # Unpack preferences
    city, rent, space, mobility, dist, clubs = preferences

    try:
        # Load the Bayesian Network from the given file
        bn = gum.loadBN(filepath)
        ie = gum.LazyPropagation(bn)

        # Set the evidence variables and their values
        evidence_dict = {
            'City': city,
            'Rent': rent,
            'Living_space': space,
            'Mobility': mobility,
            'Distance_station': dist,
            'Clubs': clubs
        }
        ie.setEvidence(evidence_dict)

        # Set the target variable
        target_variable = bn.idFromName('Accommodation')

        # Perform inference
        ie.makeInference()
        marginal_accommodation = ie.posterior(target_variable)

        # Find the index of the maximum value in the marginal distribution
        max_accommodation_index = np.argmax(marginal_accommodation.toarray())

        # Get the label of the accommodation with the highest value
        accommodation_labels = bn.variable(target_variable).labels()
        max_accommodation_label = accommodation_labels[max_accommodation_index]

        # Return the accommodation with the highest value
        return max_accommodation_label

    except Exception as e:
        # If an error occurs, return a random accommodation from the list
        accommodations = [
            'EndenicherAllee', 'Europaring', 'FerdinandStr',
            'HirschbergerStr', 'KeramikerStr', 'Kirschallee', 'Jesuitenhof'
        ]
        return random.choice(accommodations)




