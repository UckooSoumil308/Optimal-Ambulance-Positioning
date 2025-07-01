from django.db.models import  Count, Avg
from django.shortcuts import render, redirect
from django.db.models import Count
from django.db.models import Q
import datetime
import xlwt
from django.http import HttpResponse

import string
import re

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.metrics import accuracy_score

import pandas as pd


# Create your views here.
from Remote_User.models import ClientRegister_Model,ambulance_positioning_prediction,detection_ratio,detection_accuracy


def serviceproviderlogin(request):
    if request.method  == "POST":
        admin = request.POST.get('username')
        password = request.POST.get('password')
        if admin == "Admin" and password =="Admin":
            return redirect('View_Remote_Users')

    return render(request,'SProvider/serviceproviderlogin.html')



def View_Predicted_Ambulance_Positioning_Type(request):

    obj = ambulance_positioning_prediction.objects.all()
    return render(request, 'SProvider/View_Predicted_Ambulance_Positioning_Type.html', {'objs': obj})

def View_Predicted_Ambulance_Positioning_Type_Ratio(request):
    detection_ratio.objects.all().delete()
    ratio = ""
    kword = 'In Position'
    print(kword)
    obj = ambulance_positioning_prediction.objects.all().filter(Prediction=kword)
    obj1 = ambulance_positioning_prediction.objects.all()
    count = obj.count();
    count1 = obj1.count();
    ratio = (count / count1) * 100
    if ratio != 0:
        detection_ratio.objects.create(names=kword, ratio=ratio)

    ratio1 = ""
    kword1 = 'Not In Position'
    print(kword1)
    obj1 = ambulance_positioning_prediction.objects.all().filter(Prediction=kword1)
    obj11 = ambulance_positioning_prediction.objects.all()
    count1 = obj1.count();
    count11 = obj11.count();
    ratio1 = (count1 / count11) * 100
    if ratio1 != 0:
        detection_ratio.objects.create(names=kword1, ratio=ratio1)


    obj = detection_ratio.objects.all()
    return render(request, 'SProvider/View_Predicted_Ambulance_Positioning_Type_Ratio.html', {'objs': obj})


def View_Remote_Users(request):
    obj=ClientRegister_Model.objects.all()
    return render(request,'SProvider/View_Remote_Users.html',{'objects':obj})


def charts(request,chart_type):
    chart1 = detection_ratio.objects.values('names').annotate(dcount=Avg('ratio'))
    return render(request,"SProvider/charts.html", {'form':chart1, 'chart_type':chart_type})

def charts1(request,chart_type):
    chart1 = detection_accuracy.objects.values('names').annotate(dcount=Avg('ratio'))
    return render(request,"SProvider/charts1.html", {'form':chart1, 'chart_type':chart_type})

def likeschart(request,like_chart):
    charts =detection_accuracy.objects.values('names').annotate(dcount=Avg('ratio'))
    return render(request,"SProvider/likeschart.html", {'form':charts, 'like_chart':like_chart})

def likeschart1(request,like_chart):
    charts =detection_ratio.objects.values('names').annotate(dcount=Avg('ratio'))
    return render(request,"SProvider/likeschart1.html", {'form':charts, 'like_chart':like_chart})

def Download_Trained_DataSets(request):

    response = HttpResponse(content_type='application/ms-excel')
    # decide file name
    response['Content-Disposition'] = 'attachment; filename="Predicted_Datasets.xls"'
    # creating workbook
    wb = xlwt.Workbook(encoding='utf-8')
    # adding sheet
    ws = wb.add_sheet("sheet1")
    # Sheet header, first row
    row_num = 0
    font_style = xlwt.XFStyle()
    # headers are bold
    font_style.font.bold = True
    # writer = csv.writer(response)
    obj = ambulance_positioning_prediction.objects.all()
    data = obj  # dummy method to fetch data.
    for my_row in data:
        row_num = row_num + 1

        ws.write(row_num, 0, my_row.Fid, font_style)
        ws.write(row_num, 1, my_row.Temp, font_style)
        ws.write(row_num, 2, my_row.Wind, font_style)
        ws.write(row_num, 3, my_row.Date_Time, font_style)
        ws.write(row_num, 4, my_row.Age_band_of_driver, font_style)
        ws.write(row_num, 5, my_row.Sex_of_driver, font_style)
        ws.write(row_num, 6, my_row.Vehicle_driver_relation, font_style)
        ws.write(row_num, 7, my_row.Driving_experience, font_style)
        ws.write(row_num, 8, my_row.Type_of_vehicle, font_style)
        ws.write(row_num, 9, my_row.Area_accident_occured, font_style)
        ws.write(row_num, 10, my_row.Types_of_Junction, font_style)
        ws.write(row_num, 11, my_row.Type_of_collision, font_style)
        ws.write(row_num, 12, my_row.Number_of_vehicles_involved, font_style)
        ws.write(row_num, 13, my_row.Number_of_casualties, font_style)
        ws.write(row_num, 14, my_row.Vehicle_movement, font_style)
        ws.write(row_num, 15, my_row.Cause_of_accident, font_style)
        ws.write(row_num, 16, my_row.Accident_severity, font_style)
        ws.write(row_num, 17, my_row.Ambulace_Allocated, font_style)
        ws.write(row_num, 18, my_row.Ambulace_Pickedup, font_style)
        ws.write(row_num, 19, my_row.Latitude, font_style)
        ws.write(row_num, 20, my_row.Longitude, font_style)
        ws.write(row_num, 21, my_row.Prediction, font_style)


    wb.save(response)
    return response

def Train_Test_DataSets(request):

    detection_accuracy.objects.all().delete()

    df = pd.read_csv('Datasets.csv')

    def apply_results(label):
        if (label == 0):
            return 0 # In Positioning
        elif (label == 1):
            return 1 # Not In Positioning

    df['results'] = df['Label'].apply(apply_results)

    cv = CountVectorizer(lowercase=False)

    X = df["Fid"].apply(str)
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
    print("X_test")
    print(X_test)
    print(X_train)

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
    detection_accuracy.objects.create(names="Deep Neural Network (DNN)",
                                      ratio=accuracy_score(y_test, y_pred) * 100)


    # SVM Model
    print("SVM")
    from sklearn import svm
    lin_clf = svm.LinearSVC()
    lin_clf.fit(X_train, y_train)
    predict_svm = lin_clf.predict(X_test)
    svm_acc = accuracy_score(y_test, predict_svm) * 100
    print(svm_acc)
    print("CLASSIFICATION REPORT")
    print(classification_report(y_test, predict_svm))
    print("CONFUSION MATRIX")
    print(confusion_matrix(y_test, predict_svm))
    models.append(('svm', lin_clf))
    detection_accuracy.objects.create(names="SVM", ratio=svm_acc)


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
    detection_accuracy.objects.create(names="KNeighborsClassifier", ratio=accuracy_score(y_test, knpredict) * 100)

    obj = detection_accuracy.objects.all()

    return render(request,'SProvider/Train_Test_DataSets.html', {'objs': obj})














