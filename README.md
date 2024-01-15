## Pepper Bot Student Accommodation
1. Create a virtual environment using `pipenv`
2. Setup rasa by following: https://rasa.com/docs/rasa/installation/environment-set-up/
3. Then clone this repo: `git clone https://github.com/Swithinraj/Pepper_Bot_Student_Accommodation/`
4. cd `test_rasa/`
5. Run this command: `rasa train` and then it run this command in the same terminal: `rasa run -m models --endpoints endpoints.yml --port 5002 --credentials credentials.yml`
6. Open a new terminal: source your virtual environment, inside `test_rasa/`, run : `rasa action`
7. In another terminal, inside `test_rasa/`, run: `python3 pepper_rasa.py`
