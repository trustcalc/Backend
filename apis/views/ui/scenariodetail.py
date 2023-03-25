from rest_framework.views import APIView
from rest_framework.response import Response
from ...models import Scenario, CustomUser


class scenariodetail(APIView):
    def get(self, request, scenarioId):

        scenario = Scenario.objects.get(id=scenarioId)

        print('id:', scenario.description, scenario.scenario_name)
        if (scenario is not None):
            return Response({
                'scenarioName': scenario.scenario_name,
                'description': scenario.description,
            }, status=200)
        else:
            return Response("Not Exist", status=201)

    def delete(self, request, scenarioId):
        print('id:', scenarioId)
        Scenario.objects.get(id=scenarioId).delete()
        return Response('Delete Success', status=200)

    def put(self, request):
        scenario = Scenario.objects.get(
            id=request.data['id'])

        scenario.scenario_name = request.data['name']
        scenario.description = request.data['description']
        scenario.save()

        return Response("successfully changed")