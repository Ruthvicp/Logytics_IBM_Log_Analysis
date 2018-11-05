# Logytics - IBM Log Analysis Hackathon Use case

In this project, we pursued the use case to build an unsupervised model to identify and correlate warnings and error messages.

To accomplish this we used Python, Spark, and Flask.

---
### Youtube Demo - https://youtu.be/5pVbRnonJ7c
### Presentation - https://app.box.com/s/hnsn92no1hz6ujyntyn69yg3ruyspiiy

---
## Architecture

## Data Processing

Data processing was performed using Python and Spark in Scala.

1. First the data was brought into Python, split, stripped of all special characters, and the numbers were replaced.
2. Results from Python were fed into Spark for their word counts and uniqe log messages.

### To run the code:

1. The Python file [here](https://github.com/Ruthvicp/Logytics_IBM_Log_Analysis/blob/master/LOGytics/cleanData.py)  will run as is.
2. The Spark in Scala will need the winutils file and a JDK to run.

---

## Deep Learning

1. Built a character generation model - 
2. Built a word generation model -
3. Built a sentence generation model -
4. Built a word generation model on Bidirectional LSTM - 

out of all the models, sentence model (3) has given better and faster results

| Model        | Training time         | Epochs  |
| ------------- |:-------------:| -----:|
| Character generation   | 4.5 hours | 1 |
| word generation      | 2 hours      |   5 |
| Sentence generation | 2 hours      |    10 |
| word generation on bidirectional LSTM | 4 hours      |    5 |

---

## Flask Web Page
1. run logytics_run.py to start Flask server
2. It will load index.html which is home page
It has 4 sections which describe - Logytics, Prediction mechanism, Analytics & model visualization
3. In the demo -  we will load the trained model and send the log file to model in order to predict the future log

---

## Machine Learning
convert the given documents into vectors and perform clustering on the data.
This given us clusters of tags present in the log file. 
The last section in the index.html displays the clustering and tabulates the results


---
