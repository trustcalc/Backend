from rest_framework.views import APIView
from ...models import CustomUser, ScenarioSolution
from rest_framework.response import Response


class registerUser(APIView):
    def get(self, request, email):
        uploaddic = {}

        SolutionName = []

        userexist = CustomUser.objects.filter(email=email)
        if userexist:
            userobj = CustomUser.objects.get(email=email)
            scenarioobj = ScenarioSolution.objects.filter(
                user_id=userobj.id).values()

            if scenarioobj:
                for i in scenarioobj:
                    SolutionName.append(i.SolutionName)
            uploaddic['SolutionName'] = SolutionName
            return Response(uploaddic)

        else:
            print("User not exist.... Created new")
            return Response("User not exist.... Created new")

    def post(self, request):
        if request.data is not None:
            userexist = CustomUser.objects.filter(username=request.data['fullname'],
                                                  email=request.data['email'])
            if userexist:
                print("User Already Exist!")
                return Response("User Already Exist!")
            else:
                userform = CustomUser.objects.create(
                    name=request.data['fullname'],
                    email=request.data['email'],
                    password=request.data['password'],
                )
                userform.save()
                print("Successfully Created User!")
                return Response("Successfully Created User!")

        return Response("Successfully add!")
