from django.contrib import admin
from django.db import models
from apis.models import *

# Register your models here.

admin.site.register(CustomUser)
admin.site.register(Scenario)
admin.site.register(ScenarioSolution)
