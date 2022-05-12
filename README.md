# Project: United States Board of Veterans' Appeals Sentence Annotator

The data & directive for this project were outlined by the <b> <a href= "https://www.in.tum.de/legaltech/home/"> Professorship for Legal Tech</a> </b> for the course titled "Legal Data Science & Informatics (IN2395)" taught during Winter 2021-22 at the <b> Technical University of Munich </b>. 

In the project we are tasked with developing a sentence level annotator for case text originating from the US Board of Veterans' Appeals. 

As a part of this project the course participants were tasked with manually labelling legal cases. As a pre-requisite they were provided with some legal background via lectures and workshops then subsequently instructed to annotate sentences from a total of 141 BVA cases using the <a href="https://gloss2.savelka.net/"> Gloss Legal Annotator Tool </a>. The task of annotating was divided amongst the 50+ participants, hence the resulting annotated documents are the shared Intellectual Property of all course participants. For this reason I have not included any reference to the data in this repo, and I have also removed .json & .txt files used throughout the notebook <b>LDSI-Project-SHM</b> from the project directory.

Some functions to tokenize and parse the case text were taken from the <b>LDSI_W21_Classifier_Workshop_clear.ipynb</b> notebook provided by the <a href= "https://www.in.tum.de/legaltech/home/"> Professorship for Legal Tech</a> </b>.

In the <u>LDSI-Project-SHM</u> notebook I have featurized the sentences as TF-IDF vectors & sentence embeddings then applied them to 28 machine learning models.

The top performing TF-IDF based model and Word Embedding based model(F1-Score: 86% & 85% respectively) have been saved as "best_model.joblib" and "best2_model.joblib" and can be tested on a sample case text provided in "Check.txt" via "analyze.py" and "analyze_second_best.py".

```
$ python analyze.py Check.txt
$ python analyze_second_best.py Check.txt
```

To download the necessary dependencies please run the following comman
```
pip install -r requirements.txt
```


