from rest_framework.decorators import authentication_classes, api_view, parser_classes
from rest_framework.parsers import MultiPartParser, FormParser
from ...authentication import CustomUserAuthentication
from  apis.views.helpers_views import handle_score_request

# 1)TrustScore
@ parser_classes([MultiPartParser, FormParser])
@ api_view(['POST'])
@ authentication_classes([CustomUserAuthentication])
def get_trust_score(request):
    return handle_score_request('trust', 'trustscore', request.data, request.user.id)

# 2)TrustingAIScoreSupervised
@ parser_classes([MultiPartParser, FormParser])
@ api_view(['POST'])
@ authentication_classes([CustomUserAuthentication])
def get_trusting_AI_scores_supervised(request):
    return handle_score_request('trust', 'trusting_AI_scores_supervised', request.data, request.user.id)

# 3)TrustingAIScoreUnSupervised
@ parser_classes([MultiPartParser, FormParser])
@ api_view(['POST'])
@ authentication_classes([CustomUserAuthentication])
def get_trusting_AI_scores_unsupervised(request):
    return handle_score_request('trust', 'trusting_AI_scores_unsupervised', request.data, request.user.id)
