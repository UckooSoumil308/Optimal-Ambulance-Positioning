from django.db.models import Count
from django.db.models import Q
from django.shortcuts import render, redirect, get_object_or_404

from sklearn.ensemble import VotingClassifier
import warnings
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
# Create your views here.
from Remote_User.models import ClientRegister_Model,ambulance_positioning_prediction,detection_ratio,detection_accuracy

def login(request):


    if request.method == "POST" and 'submit1' in request.POST:

        username = request.POST.get('username')
        password = request.POST.get('password')
        try:
            enter = ClientRegister_Model.objects.get(username=username,password=password)
            request.session["userid"] = enter.id

            return redirect('ViewYourProfile')
        except:
            pass

    return render(request,'RUser/login.html')

def Register1(request):

    if request.method == "POST":
        username = request.POST.get('username')
        email = request.POST.get('email')
        password = request.POST.get('password')
        phoneno = request.POST.get('phoneno')
        country = request.POST.get('country')
        state = request.POST.get('state')
        city = request.POST.get('city')
        ClientRegister_Model.objects.create(username=username, email=email, password=password, phoneno=phoneno,
                                            country=country, state=state, city=city)

        return render(request, 'RUser/Register1.html')
    else:
        return render(request,'RUser/Register1.html')

def ViewYourProfile(request):
    userid = request.session['userid']
    obj = ClientRegister_Model.objects.get(id= userid)
    return render(request,'RUser/ViewYourProfile.html',{'object':obj})


def Predict_Ambulance_Positioning_Type(request):

        if request.method == "POST":

            Fid= request.POST.get('Fid')
            Temp= request.POST.get('Temp')
            Wind= request.POST.get('Wind')
            Date_Time= request.POST.get('Date_Time')
            Age_band_of_driver= request.POST.get('Age_band_of_driver')
            Sex_of_driver= request.POST.get('Sex_of_driver')
            Vehicle_driver_relation= request.POST.get('Vehicle_driver_relation')
            Driving_experience= request.POST.get('Driving_experience')
            Type_of_vehicle= request.POST.get('Type_of_vehicle')
            Area_accident_occured= request.POST.get('Area_accident_occured')
            Types_of_Junction= request.POST.get('Types_of_Junction')
            Type_of_collision= request.POST.get('Type_of_collision')
            Number_of_vehicles_involved= request.POST.get('Number_of_vehicles_involved')
            Number_of_casualties= request.POST.get('Number_of_casualties')
            Vehicle_movement= request.POST.get('Vehicle_movement')
            Cause_of_accident= request.POST.get('Cause_of_accident')
            Accident_severity= request.POST.get('Accident_severity')
            Ambulace_Allocated= request.POST.get('Ambulace_Allocated')
            Ambulace_Pickedup= request.POST.get('Ambulace_Pickedup')
            Latitude= request.POST.get('Latitude')
            Longitude= request.POST.get('Longitude')


            df = pd.read_csv('Datasets.csv')


            def apply_results(label):
                if (label == 0):
                    return 0  # In Position
                elif (label == 1):
                    return 1  # Not In Position

            df['results'] = df['Label'].apply(apply_results)

            cv = CountVectorizer(lowercase=False)

            X = df["Fid"]
            y = df['results']


            print("X Values")
            print(X)
            print("Labels")
            print(y)

            X = cv.fit_transform(X)

            models = []
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
            X_train.shape, X_test.shape, y_train.shape

            print("Deep Neural Network (DNN)")

            from sklearn.neural_network import MLPClassifier
            mlpc = MLPClassifier().fit(X_train, y_train)
            y_pred = mlpc.predict(X_test)
            print("ACCURACY")
            print(accuracy_score(y_test, y_pred) * 100)
            print("CLASSIFICATION REPORT")
            print(classification_report(y_test, y_pred))
            print("CONFUSION MATRIX")
            print(confusion_matrix(y_test, y_pred))
            models.append(('MLPClassifier', mlpc))


            print("KNeighborsClassifier")
            from sklearn.neighbors import KNeighborsClassifier
            kn = KNeighborsClassifier()
            kn.fit(X_train, y_train)
            knpredict = kn.predict(X_test)
            print("ACCURACY")
            print(accuracy_score(y_test, knpredict) * 100)
            print("CLASSIFICATION REPORT")
            print(classification_report(y_test, knpredict))
            print("CONFUSION MATRIX")
            print(confusion_matrix(y_test, knpredict))
            models.append(('KNeighborsClassifier', kn))

            classifier = VotingClassifier(models)
            classifier.fit(X_train, y_train)
            y_pred = classifier.predict(X_test)

            Fid1 = [Fid]
            vector1 = cv.transform(Fid1).toarray()
            predict_text = classifier.predict(vector1)

            pred = str(predict_text).replace("[", "")
            pred1 = pred.replace("]", "")

            prediction = int(pred1)

            if prediction == 0:
                val = 'In Position'
            elif prediction == 1:
                val = 'Not In Position'

            print(val)
            print(pred1)

            ambulance_positioning_prediction.objects.create(
            Fid=Fid,
            Temp=Temp,
            Wind=Wind,
            Date_Time=Date_Time,
            Age_band_of_driver=Age_band_of_driver,
            Sex_of_driver=Sex_of_driver,
            Vehicle_driver_relation=Vehicle_driver_relation,
            Driving_experience=Driving_experience,
            Type_of_vehicle=Type_of_vehicle,
            Area_accident_occured=Area_accident_occured,
            Types_of_Junction=Types_of_Junction,
            Type_of_collision=Type_of_collision,
            Number_of_vehicles_involved=Number_of_vehicles_involved,
            Number_of_casualties=Number_of_casualties,
            Vehicle_movement=Vehicle_movement,
            Cause_of_accident=Cause_of_accident,
            Accident_severity=Accident_severity,
            Ambulace_Allocated=Ambulace_Allocated,
            Ambulace_Pickedup=Ambulace_Pickedup,
            Latitude=Latitude,
            Longitude=Longitude,
            Prediction=val)

            return render(request, 'RUser/Predict_Ambulance_Positioning_Type.html',{'objs':val})
        return render(request, 'RUser/Predict_Ambulance_Positioning_Type.html')

