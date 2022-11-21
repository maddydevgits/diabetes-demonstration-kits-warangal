import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from flask import Flask,request,render_template # deployments

app=Flask(__name__) # flask-app

dataset=pd.read_csv('diabetes.csv')

# Input Cols - X
X=dataset.iloc[:,:-1] # exclude the last col
# We have to convert df into array
X=X.values

# Output Cols - Y
Y=dataset.iloc[:,-1]
Y=Y.values

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2)
# 80% of data - train, 20% of data - test

# Models Creation
logistic_classifier=LogisticRegression()
logistic_classifier.fit(X_train,Y_train)

knn_classifier=KNeighborsClassifier()
knn_classifier.fit(X_train,Y_train)

dt_classifier=DecisionTreeClassifier()
dt_classifier.fit(X_train,Y_train)

rf_classifier=RandomForestClassifier()
rf_classifier.fit(X_train,Y_train)

# Evaluate the Models - Test it first

logistic_pred=logistic_classifier.predict(X_test)
knn_pred=knn_classifier.predict(X_test)
dt_pred=dt_classifier.predict(X_test)
rf_pred=rf_classifier.predict(X_test)

print(accuracy_score(logistic_pred,Y_test))
print(accuracy_score(knn_pred,Y_test))
print(accuracy_score(dt_pred,Y_test))
print(accuracy_score(rf_pred,Y_test)) # best classifier - RF Classifier


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict',methods=['post','get'])
def predict():
    pregnancies=request.form['pregnancies']
    glucose=request.form['glucose']
    bp=request.form['bp']
    st=request.form['st']
    insulin=request.form['insulin']
    bmi=request.form['bmi']
    dpf=request.form['dpf']
    age=request.form['age']
    data=[[pregnancies,glucose,bp,st,insulin,bmi,dpf,age]]
    outcome=rf_classifier.predict(data)
    if(outcome[0]==0):
        outcome='No Diabetes'
    else:
        outcome='Diabetes Found'
    return render_template('index.html',result=outcome)


if __name__=="__main__":
    app.run(port=5000,debug=True)
