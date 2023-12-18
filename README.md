## Install Rasa

## The first step is to install Rasa. You can do this by running the following command in your terminal:

  ```bash
  $ sudo apt update
  $ sudo apt install python3-dev python3-pip

## Setting up a Virtual Environment

  $ python3 -m venv ./venv
  $ source ./venv/bin/activate

## Install Rasa in the Virtual Environment

  $ pip3 install rasa

## To create a new Rasa project. we can do this by running the following command:

  $ rasa init --no-prompt

## This will create a new Rasa project with some default files and folders.
