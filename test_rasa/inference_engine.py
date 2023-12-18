import pyAgrum as gum
import numpy as np
import random

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
        print(f"An error occurred: {e}")
        # If an error occurs, return a random accommodation from the list
        accommodations = [
            'EndenicherAllee', 'Europaring', 'FerdinandStr',
            'HirschbergerStr', 'KeramikerStr', 'Kirschallee', 'Jesuitenhof'
        ]
        return random.choice(accommodations)
