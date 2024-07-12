#classification based
#identify mimp features
#identify the best classifier
# it is non parametric supervised learning method used for cassiification and regression
    #it is used for both classification and regression


from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

#data
iris=load_iris()
x=iris.data
y=iris.target


#split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

#model
model=DecisionTreeClassifier()
model.fit(x_train,y_train)

#prediction
y_pred=model.predict(x_test)

#evaluation
accuracy=accuracy_score(y_test,y_pred)
print(accuracy)
