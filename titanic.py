import pandas as pd
from ObjectOrientedTitanic import ObjectOrientedTitanic

import warnings
warnings.filterwarnings('ignore')
print("Warnings were ignored")

train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")

objectOrientedTitanic=ObjectOrientedTitanic(train, test)

objectOrientedTitanic.information()

objectOrientedTitanic.preprocessing(strategy_type='strategy1')
objectOrientedTitanic.information()
objectOrientedTitanic.visualize(visualizer_type="RadViz", number_of_features=None)
objectOrientedTitanic.machine_learning()