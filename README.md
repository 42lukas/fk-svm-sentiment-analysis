# FK-SVM Twitter Sentiment Analysis

This project implements and explains the method from the paper  
**“Application of Support Vector Machine (SVM) in the Sentiment Analysis of Twitter Dataset”**,  
focusing on the hybrid **Fisher Kernel Support Vector Machine (FK-SVM)** approach.

## Dataset
Uses the **Sentiment140** Twitter dataset (1.6M labeled tweets).

## Installation

To use this repo just clone the repository:

```
git clone https://github.com/42lukas/fk-svm-sentiment-analysis.git
```
Afterwards download the Sentiment140 dataset. We used this Website:
```https://colab.research.google.com/github/littlecolumns/ds4j-notebooks/blob/master/investigating-sentiment-analysis/notebooks/Cleaning%20the%20Sentiment140%20data.ipynb#scrollTo=TX8WqwSLyaR-```

If you donwloaded this dataset, import it and use the python scripts in the ```format_dataset``` folder to clean and noramlize it. Finally the whole dataset will be split up into 3 parts, train (80%), val (10%) and test (10%).

Then you should be able to just run the ```main.py``` and train the model. You will be asked to select between ```PLSA``` and ```none```. We recommand to use ```none``` if your computer lack performance. 

Afterwards use the ```test_model.py``` to finally test you trained model. 

## License
MIT

## Author
**Lukas** \
**Mihoshi**