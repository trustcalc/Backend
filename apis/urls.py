import sys

from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static

from apis.views.ui import scenario,scenariodetail,solution,solutiondetail,analyze,compare,dashboard,userpage
from apis.views.user import auth, registeruser, user, userset
from apis.views.score import account,explain,fair,robust,pillar,trust
from apis.views.helpers_views import download


sys.path.extend([r"Backend", r"Backend/apis"])

urlpatterns = [

     #0 Scenario, Solution Endpoints
    path('scenario', scenario.scenario().as_view()),
    path('scenario/<str:email>', solution.solution.as_view()),
    path('scenario/<int:scenarioId>',scenario.scenario.as_view(), name="scenario"),
    path('scenario:detail', scenariodetail.scenariodetail.as_view(),name="scenariodetail"),

    path('solution', solution.solution().as_view()),
    path('solution/<str:email>', solution.solution.as_view()),
    path('solution_detail', solutiondetail.solutiondetail.as_view(),name="solutiondetail"),
    path('solution_detail/<int:id>',solutiondetail.solutiondetail.as_view(), name="solutiondetail"),

    # 1)Accountability Scores
    path('factsheet_completness_score',account.get_factsheet_completness_score),
    path('missing_data_score/', account.get_missing_data_score),
    path('normalization_score/', account.get_normalization_score),
    path('regularization_score/', account.get_regularization_score),
    path('train_test_split_score/', account.get_train_test_split_score),

    # 2)Robustness Scores
    path('clever_score/', robust.get_clever_score),
    path('clique_method_score/', robust.get_clique_method_score),
    path('confidence_score/', robust.get_confidence_score),
    path('carliwagnerwttack_score/', robust.get_carliwagnerwttack_score),
    path('deepfoolattack_score/', robust.get_deepfoolattack_score),
    path('fast_gradient_attack_score/',robust.get_fast_gradient_attack_score),
    path('loss_sensitivity_score/', robust.get_loss_sensitivity_score),

    # 3)Explainability Scores
    path('modelsize_score/', explain.get_modelsize_score),
    path('correlated_features_score/',explain.get_correlated_features_score),
    path('algorithm_class_score/', explain.get_algorithm_class_score),
    path('feature_relevance_score/', explain.get_feature_relevance_score),
    path('permutation_feature_importance_score/',
         explain.get_permutation_feature_importance_score),

    # 4)Fairness Scores
    path('disparate_impact_score/', fair.get_disparate_impact_score),
    path('class_balance_score/', fair.get_class_balance_score),
    path('overfitting_score/', fair.get_overfitting_score),
    path('underfitting_score/', fair.get_underfitting_score),
    path('statistical_parity_difference_score/',fair.get_statistical_parity_difference_score),
    path('equal_opportunity_difference_score/',fair.get_equal_opportunity_difference_score),
    path('average_odds_difference_score/',fair.get_average_odds_difference_score),

    # 5)Pillar Scores
    path('accountability_score/', pillar.get_accountability_score),
    path('robustnesss_score/', pillar.get_robustness_score),
    path('explainability_score/', pillar.get_explainability_score),
    path('fairness_score/', pillar.get_fairness_score),

    # 6)Trust Scores
    path('trustscore/', trust.get_trust_score),
    path('trusting_AI_scores_supervised/', trust.get_trusting_AI_scores_supervised),
    path('trusting_AI_scores_unsupervised/',trust.get_trusting_AI_scores_unsupervised),

     #7) Usermanagement
    path('registerUser', registeruser.registerUser.as_view(), name="registerUser"),
    path('auth', auth.auth.as_view(), name="auth"),

    path('setuser/<str:email>', userset.userset.as_view(), name="userset"),
    path('setuser', userset.userset.as_view(), name="userset"),

    path('user', user.user.as_view(), name="user"),
    path('user/<str:email>/', user.user.as_view(), name="user"),
    
    #8)UI Pages
    path('userpage', userpage.userpage.as_view()),
    path('userpage/<str:email>', userpage.userpage.as_view()),
    
    path('dashboard',dashboard.dashboard.as_view()),
    path('dashboard/<str:email>',dashboard.dashboard.as_view(), name="dashboard"),

    path('analyze', analyze.analyze.as_view()),
    path('analyze/<str:email>',analyze.analyze.as_view(), name="analyze"),

    path('compare', compare.compare.as_view()),
    path('compare/<str:email>',compare.compare.as_view(), name="compare"),


    path('factsheet_download', download.as_view(), name="download")

] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
