# scp_classes
This is an NLP model that helps [SCP](https://scp-wiki.wikidot.com/) scientists with objects classification.
## Overview
I created a simple BERT model for object classification in SCP setting. This repository includes a code for model fine-tune and usage. I also attached some training report.
## Training data
The training data was collected from scp-wiki.wikidot.com, but I don't remember what code I used for scraping because it was long time ago.
The dataset includes titles, descriptions and containment procedures for 4207 objects with class labels (2948 in train and 1259 in test).
## Model
I used pre-trained XLMRoberta model and fine-tuned it to the SCP objects data for a three-class classification task. There is one fully-connected layer in a classification head.
## Experiments
There are some experiments with different document zones: titles, descriptions and containment procedures.
![alt text](https://github.com/8Michelle/scp_classes/blob/master/assets/wandb_plots.png | width=100)
## Results
