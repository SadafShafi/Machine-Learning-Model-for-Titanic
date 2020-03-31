from sklearn import datasets
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate	
from sklearn.model_selection import cross_val_score	
from sklearn.naive_bayes import GaussianNB,MultinomialNB,BernoulliNB,CategoricalNB
import csv
from xgboost import XGBClassifier	
from sklearn.cluster import KMeans
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegression

global dataset
global TestDataset

dataset = pd.read_csv("train.csv")
TestDataset = pd.read_csv("test.csv")

def uniquify_me(*lister):

	auxiliaryList = []
	
	for word in lister:
	
		if word not in auxiliaryList:
			
			auxiliaryList.append(word)

	return auxiliaryList

def DataPreProcessing():                    
	
	global dataset
	global TestDataset
	fillNa_value = np.float(0.0)
	TestDataset.fillna(fillNa_value,inplace=True)
	TestDataset = TestDataset.loc[:, TestDataset.columns != 'PassengerId']
	TestDataset = TestDataset.loc[:, TestDataset.columns != 'Cabin']



	# for abc in TestDataset["Age"]:
	# print(type(fillNa_value))
	# for x in TestDataset["Age"]:
	# 	print(type(x))
	# 	exit()
	##################################### Dealing with Tickets#########################################
	index = 0

	num = 1
	a = 0
	# tiketo = pd.DataFrame(columns = TestDataset["Ticket"])
	
	while(a < len(TestDataset["Ticket"])):
		b = a+1
		while(b < len(TestDataset["Ticket"])-1):
			if(TestDataset["Ticket"][a] == TestDataset["Ticket"][b]):
				TestDataset["Ticket"][b] = num
			b+=1
		a+=1
		num+=1

	index = 0

	for a in TestDataset["Ticket"]:
		if(type(a) == type("asdkjb")):
			TestDataset["Ticket"][index] = num
			num += 1
		index += 1
	

	# ................................................................................................
	a = 0

	while(a < len(dataset["Ticket"])):
		b = a+1
		while(b < len(dataset["Ticket"])-1):
			if(dataset["Ticket"][a] == dataset["Ticket"][b]):
				dataset["Ticket"][b] = num
			b+=1
		a+=1
		num+=1

	index = 0

	for a in dataset["Ticket"]:
		if(type(a) == type("asdkjb")):
			dataset["Ticket"][index] = num
			num += 1
		index += 1

		############################# Adressing Missing Ages ########################################

	index = 0
	for age,sex,name,spo in zip(TestDataset["Age"],TestDataset["Sex"],TestDataset["Name"],TestDataset["SibSp"]) :
		if( sex == 'female'):
			if "Mrs." in name:
				if age == fillNa_value:
					# print("*******************************")
					# age.replace(33.5,inplace = True)
					TestDataset["Age"][index] = 33.5
					# TestDataset.at('Age',index,33.5)
			elif "Miss" in name:
				if age == fillNa_value:
					# print("*******************************")
					# age.replace(17.8,inplace = True)
					# TestDataset.at('Age',index,17.8)
					TestDataset["Age"][index]  = 17.8

		else:
			if "Mr." in name:
				if age == fillNa_value:
					# print("*******************************")
					# TestDataset.replace(23.3,inplace = True)
					TestDataset["Age"][index]  = 23.3
					# TestDataset.at('Age',index,23.3)
					


		index += 1


	index = 0
	for age,sex,name,spo in zip(dataset["Age"],dataset["Sex"],dataset["Name"],dataset["SibSp"]) :
		if( sex == 'female'):
			# print("entering the echelon of female -----------------------------------------------------------------------------------------------")
			if "Mrs." in name:
				if age == fillNa_value:
					# print("*******************************")
					# age.replace(33.5,inplace = True)
					# dataset.at('Age',index, 33.5)
					dataset["Age"][index] = 33.5
			elif "Miss" in name:

				if age == fillNa_value:
					# print("*******************************")
					# age.replace(17.8,inplace = True)
					# dataset.at('Age',index, 17.8)
					dataset["Age"][index]  = 17.8

		else:
			# print("entering the male world -----------------------------------------------------------------------------------------------")
			if "Mr." in name:
				
				if age == fillNa_value:
					# print("*******************************")
					
					# TestDataset.replace(23.3,inplace = True)
					# dataset.at('Age',index, 23.3)
					dataset["Age"][index]  = 23.3
					# print("here commeth the mister  ----------------------------\n",dataset["Age"][index])

	for xyz,sex in zip(TestDataset["Name"],TestDataset["Sex"]):
		if "Mr." in xyz:
			TestDataset.replace(xyz,1,inplace = True)
		elif("Mrs." in xyz):
			TestDataset.replace(xyz,2,inplace = True)
		elif("Miss." in xyz):
			TestDataset.replace(xyz,3,inplace = True)	
		elif("Don." in xyz):
			TestDataset.replace(xyz,4,inplace = True)	
		elif("Dr." in xyz):
			if sex == "male" :
				TestDataset.replace(xyz,5,inplace = True)
			else:
				TestDataset.replace(xyz,6,inplace = True)
		elif("Ms." in xyz):
			TestDataset.replace(xyz,7,inplace = True)
		elif("Lady." in xyz):
			TestDataset.replace(xyz,8,inplace = True)
		elif("Capt." in xyz):
			TestDataset.replace(xyz,9,inplace = True)
		elif("Master." in xyz):
			TestDataset.replace(xyz,10,inplace = True)
		elif("Dona." in xyz):
			TestDataset.replace(xyz,11,inplace = True)
		elif("Col." in xyz):
			TestDataset.replace(xyz,12,inplace = True)		
		elif("Rev." in xyz):
			TestDataset.replace(xyz,13,inplace = True)
		elif("Mme." in xyz):
			TestDataset.replace(xyz,14,inplace = True)
		elif("Major." in xyz):
			TestDataset.replace(xyz,15,inplace = True)
		elif("Sir." in xyz):
			TestDataset.replace(xyz,16,inplace = True)
		elif("Mlle." in xyz):
			TestDataset.replace(xyz,17,inplace = True)
		elif("Countess." in xyz):
			TestDataset.replace(xyz,18,inplace = True)
		elif("Jonkheer." in xyz):
			TestDataset.replace(xyz,19,inplace = True)				
		else:
			TestDataset.replace(xyz,0,inplace = True)

	
	TestDataset.replace('male',1,inplace=True)
	TestDataset.replace('female',0,inplace=True)
	TestDataset.replace('Q',1,inplace=True)
	TestDataset.replace('S',2,inplace=True)
	TestDataset.replace('C',3,inplace=True)

	
	dataset.fillna(fillNa_value,inplace=True)
	dataset = dataset.loc[:, dataset.columns != 'PassengerId']
	dataset = dataset.loc[:, dataset.columns != 'Cabin']


    # Dealing with Cabin column
	# lister = []	
	# # TestDataset["Cabin"].fillna(0,inplace = True)
	# for x in TestDataset["Cabin"] :
	# 	if x == fillNa_value:
	# 		pass
	# 	else:
	# 		lister.append(x)
	
	# for x in dataset["Cabin"] :
	# 	if x == fillNa_value:
	# 		pass
	# 	else:
	# 		lister.append(x)

	# lister = uniquify_me(*lister)
	# number = 1
	# for cabin in lister:
	# 	TestDataset.replace(cabin,number,inplace  = True)
	# 	dataset.replace(cabin,number,inplace  = True)
	# 	number += 1


	for xyz,sex in zip(dataset["Name"],dataset["Sex"]):

		if("Mr." in xyz):
			dataset.replace(xyz,1,inplace = True)
		elif("Mrs." in xyz):
			dataset.replace(xyz,2,inplace = True)
		elif("Miss." in xyz):
			dataset.replace(xyz,3,inplace = True)	
		elif("Don." in xyz):
			dataset.replace(xyz,4,inplace = True)	
		elif("Dr." in xyz):
			if sex == "male" :
				dataset.replace(xyz,5,inplace = True)
			else:
				dataset.replace(xyz,6,inplace = True)
		elif("Ms." in xyz):
			dataset.replace(xyz,7,inplace = True)
		elif("Lady." in xyz):
			dataset.replace(xyz,8,inplace = True)
		elif("Capt." in xyz):
			dataset.replace(xyz,9,inplace = True)
		elif("Master." in xyz):
			dataset.replace(xyz,10,inplace = True)
		elif("Dona." in xyz):
			dataset.replace(xyz,11,inplace = True)
		elif("Col." in xyz):
			dataset.replace(xyz,12,inplace = True)		
		elif("Rev." in xyz):
			dataset.replace(xyz,13,inplace = True)	
		elif("Mme." in xyz):
			dataset.replace(xyz,14,inplace = True)
		elif("Major." in xyz):
			dataset.replace(xyz,15,inplace = True)
		elif("Sir." in xyz):
			dataset.replace(xyz,16,inplace = True)
		elif("Mlle." in xyz):
			dataset.replace(xyz,17,inplace = True)
		elif("Countess." in xyz):
			dataset.replace(xyz,18,inplace = True)
		elif("Jonkheer." in xyz):
			dataset.replace(xyz,19,inplace = True)			
		else:
			dataset.replace(xyz,0,inplace = True)


	dataset.replace('male',1,inplace=True)
	dataset.replace('female',0,inplace=True)
	dataset.replace('Q',1,inplace=True)
	dataset.replace('S',2,inplace=True)
	dataset.replace('C',3,inplace=True)
	
	X = dataset.loc[:, dataset.columns != 'Survived']
	Y = dataset['Survived']	

	X_Test = TestDataset.loc[:, TestDataset.columns != 'Survived']
	

	
	X_train,X_test,y_train,y_test = train_test_split(
													 X,Y,
													 test_size=0.3,
													 random_state=12
													 )
	

	return X_train,X_test,y_train,y_test,X,Y,X_Test
	

	
def TestingAlgorithms():	

	X_train,X_test,y_train,y_test,X,Y,X_Test = DataPreProcessing()


	clc = LogisticRegression()
	clf = GaussianNB()
	ber = BernoulliNB()
	mul = MultinomialNB()
	sss = svm.SVC()

	
	model = XGBClassifier()

	ladyADA = AdaBoostClassifier(n_estimators=500, random_state=9)
	ForestGump = RandomForestClassifier(n_estimators=500,max_depth=10, random_state=3)
	from sklearn.ensemble import BaggingClassifier
	bagging = BaggingClassifier(GaussianNB(),
                            max_samples=0.5, max_features=0.5)


	
	cv = 12
	# print("Logistic regression : ",cross_val_score(clc, X, Y, cv=cv).mean())
	# print("Gaussin NB : ",cross_val_score(clf, X, Y, cv=cv).mean())
	# print("BernoulliNB : ",cross_val_score(ber, X, Y, cv=cv).mean())
	# print("MultinomialNB : ",cross_val_score(mul, X, Y, cv=cv).mean())
	# print("SVM : ",cross_val_score(sss, X, Y, cv=cv).mean())
	# print("Random Forest : ",cross_val_score(ForestGump, X, Y, cv=cv).mean())
	# print("AdaBoost : ",cross_val_score(ladyADA, X, Y, cv=cv).mean())
	print("Bagging : ",cross_val_score(bagging, X, Y, cv=cv).mean())
	print("xgboost  : ",cross_val_score(model, X, Y, cv=cv).mean())
	

def TrainMeFinally():

	X_train,X_test,y_train,y_test,X,Y,X_Test = DataPreProcessing()
	
	ADA = RandomForestClassifier(max_depth=10, random_state=2)
	# ADA = XGBClassifier()
	ADA.fit(X,Y)

	return ADA.predict(X_Test)




'''
Next try ensamble method

'''



# X_train,X_test,y_train,y_test,X,Y,X_Test = DataPreProcessing()
# TestingAlgorithms()
# exit()

PId = TestDataset['PassengerId']	
pred = TrainMeFinally()
# Start writing it to a csv file
with open('innovators.csv', 'w', newline='') as file:
	writer = csv.writer(file)
	# print(map(lambda x:[x], pred))
	zipped_lists=zip(PId,pred)
	writer.writerow(('PassengerId','Survived'))
	for row in zipped_lists:
		writer.writerow(row)
print("done")
	

