import pandas as pd
import warnings
import numpy as np
import renders as rs
from IPython.display import display # Allows the use of display() for DataFrames
from sklearn.cross_validation import train_test_split
from sklearn.tree import DecisionTreeRegressor

warnings.filterwarnings('ignore')
df = pd.read_excel("Task_Data_Scientist_Dataset.xlsx")
df0 = pd.read_excel(open("Task_Data_Scientist_Dataset.xlsx",'rb'),sheetname = 0)
df1 = pd.read_excel(open("Task_Data_Scientist_Dataset.xlsx",'rb'),sheetname = 1)
df2 = pd.read_excel(open("Task_Data_Scientist_Dataset.xlsx",'rb'),sheetname = 2)
df3 = pd.read_excel(open("Task_Data_Scientist_Dataset.xlsx",'rb'),sheetname = 3)
df4 = pd.read_excel(open("Task_Data_Scientist_Dataset.xlsx",'rb'),sheetname = 4)

df.head()
df1 = df
df1.
sex=df2[['Sex','Client']].drop_duplicates()
sex.groupby(['Sex'])['Client#'].aggregate('count').reset_index().sort_values('Client', ascending=False)
# Import libraries necessary for this project


# Show matplotlib plots inline (nicely formatted in the notebook)
%matplotlib inline

# Load the wholesale customers dataset
try:
    data = pd.read_csv("customers.csv")
    data.drop(['Region', 'Channel'], axis = 1, inplace = True)
    print "Wholesale customers dataset has {} samples with {} features each.".format(*data.shape)
except:
    print "Dataset could not be loaded. Is the dataset missing?"
