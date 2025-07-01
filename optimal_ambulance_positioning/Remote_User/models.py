from django.db import models

# Create your models here.
from django.db.models import CASCADE


class ClientRegister_Model(models.Model):

    username = models.CharField(max_length=30)
    email = models.EmailField(max_length=30)
    password = models.CharField(max_length=10)
    phoneno = models.CharField(max_length=10)
    country = models.CharField(max_length=30)
    state = models.CharField(max_length=30)
    city = models.CharField(max_length=30)


class ambulance_positioning_prediction(models.Model):

    Fid= models.CharField(max_length=3000)
    Temp= models.CharField(max_length=3000)
    Wind= models.CharField(max_length=3000)
    Date_Time= models.CharField(max_length=3000)
    Age_band_of_driver= models.CharField(max_length=3000)
    Sex_of_driver= models.CharField(max_length=3000)
    Vehicle_driver_relation= models.CharField(max_length=3000)
    Driving_experience= models.CharField(max_length=3000)
    Type_of_vehicle= models.CharField(max_length=3000)
    Area_accident_occured= models.CharField(max_length=3000)
    Types_of_Junction= models.CharField(max_length=3000)
    Type_of_collision= models.CharField(max_length=3000)
    Number_of_vehicles_involved= models.CharField(max_length=3000)
    Number_of_casualties= models.CharField(max_length=3000)
    Vehicle_movement= models.CharField(max_length=3000)
    Cause_of_accident= models.CharField(max_length=3000)
    Accident_severity= models.CharField(max_length=3000)
    Ambulace_Allocated= models.CharField(max_length=3000)
    Ambulace_Pickedup= models.CharField(max_length=3000)
    Latitude= models.CharField(max_length=3000)
    Longitude= models.CharField(max_length=3000)
    Prediction= models.CharField(max_length=3000)

class detection_ratio(models.Model):

    names = models.CharField(max_length=300)
    ratio = models.CharField(max_length=300)

class detection_accuracy(models.Model):

    names = models.CharField(max_length=300)
    ratio = models.CharField(max_length=300)


