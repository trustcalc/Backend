from os.path import isfile, join
from django.test import TestCase

# Create your tests here.
from .models import CustomUser, Scenario, ScenarioSolution
from .apiback.ui.dashboard import unsupervised_FinalScore, finalScore_supervised
from .apiback.public import save_files_return_paths
from algorithms.unsupervised.Functions.Accountability.NormalizationScore import normalization_score as get_normalization_score_unsupervised
from algorithms.supervised.Functions.Robustness.CleverScore_supervised import get_clever_score_supervised
from algorithms.unsupervised.Functions.Robustness.CleverScore import clever_score as get_clever_score_unsupervised
from algorithms.unsupervised.Functions.Fairness.Fairness import analyse as get_fairness_score_unsupervised
from algorithms.supervised.Functions.Fairness.FarinessScore_supervised import get_fairness_score_supervised
from algorithms.unsupervised.Functions.Explainability.Explainability import analyse as get_explainability_score_unsupervised
from algorithms.supervised.Functions.Explainability.ExplainabilityScore_supervised import get_explainability_score_supervised
from algorithms.unsupervised.Functions.Accountability.Accountability import analyse as get_accountability_score_unsupervised
from algorithms.supervised.Functions.Accountability.AccountabilityScore_supervised import get_accountability_score_supervised
from algorithms.supervised.Functions.Fairness.AverageOddsDifferenceScore_supervised import get_average_odds_difference_score_supervised
from algorithms.supervised.Functions.Fairness.EqualOpportunityDifferenceScore_supervised import get_equal_opportunity_difference_score_supervised
from algorithms.unsupervised.Functions.Fairness.StatisticalParityDifferenceScore import get_statistical_parity_difference_score_unsupervised
from algorithms.supervised.Functions.Fairness.StatisticalParityDifferenceScore import get_statistical_parity_difference_score_supervised
from algorithms.unsupervised.Functions.Fairness.UnderfittingScore import underfitting_score as get_underfitting_score_unsupervised
from algorithms.supervised.Functions.Fairness.UnderfittingScore_supervised import get_underfitting_score_supervised
from algorithms.unsupervised.Functions.Fairness.OverfittingScore import overfitting_score as get_overfitting_score_unsupervised
from algorithms.supervised.Functions.Fairness.OverfittingScore_supervised import get_overfitting_score_supervised
from algorithms.supervised.Functions.Fairness.ClassBalanceScore_supervised import get_class_balance_score_supervised
from algorithms.unsupervised.Functions.Fairness.DisparateImpactScore import disparate_impact_score as get_disparate_impact_score_unsupervised
from algorithms.supervised.Functions.Fairness.DisparateImpactScore_supervised import get_disparate_impact_score_supervised
from algorithms.unsupervised.Functions.Explainability.PermutationFeatureScore import permutation_feature_importance_score as get_permutation_feature_importance_score_unsupervised
from algorithms.supervised.Functions.Explainability.FeatureRelevanceScore_supervised import get_feature_relevance_score_supervised
from algorithms.supervised.Functions.Explainability.AlgorithmClassScore_supervised import get_algorithm_class_score_supervised
from algorithms.unsupervised.Functions.Explainability.CorrelatedFeaturesScore import correlated_features_score as get_correlated_features_score_unsupervised
from algorithms.supervised.Functions.Explainability.CorrelatedFeaturesScore_supervised import get_correlated_features_score_supervised
from algorithms.unsupervised.Functions.Explainability.ModelSizeScore import model_size_score as get_modelsize_score_unsupervised
from algorithms.supervised.Functions.Explainability.ModelSizeScore_supervised import get_model_size_score_supervised as get_modelsize_score_supervised
from algorithms.supervised.Functions.Robustness.LossSensitivityScore_supervised import get_loss_sensitivity_score_supervised
from algorithms.supervised.Functions.Robustness.ERFastGradientAttackScore_supervised import get_er_fast_gradient_attack_score_supervised as get_fast_gradient_attack_score_supervised
from algorithms.supervised.Functions.Robustness.ERDeepFoolAttackScore_supervised import get_deepfool_attack_score_supervised as get_deepfoolattack_score_supervised
from algorithms.supervised.Functions.Robustness.ERCarliniWagnerScore_supervised import get_er_carlini_wagner_score_supervised as get_carliwagnerwttack_score_supervised
from algorithms.supervised.Functions.Robustness.ConfidenceScore_supervised import get_confidence_score_supervised
from algorithms.supervised.Functions.Robustness.CliqueMethodScore_supervised import get_clique_method_supervised as get_clique_method_score_supervised
from algorithms.unsupervised.Functions.Robustness.CleverScore import clever_score as get_clique_method_score_unsupervised
from algorithms.unsupervised.Functions.Accountability.TrainTestSplitScore import train_test_split_score as get_train_test_split_score_unsupervised
from algorithms.unsupervised.Functions.Accountability.RegularizationScore import regularization_score as get_regularization_score_unsupervised
from algorithms.supervised.Functions.Accountability.MissingDataScore_supervised import get_missing_data_score_supervised
from algorithms.unsupervised.Functions.Accountability.MissingDataScore import missing_data_score as get_missing_data_score_unsupervised
from algorithms.supervised.Functions.Accountability.NormalizationScore_supervised import get_normalization_score_supervised
from algorithms.supervised.Functions.Accountability.RegularizationScore_supervised import get_regularization_score_supervised
from algorithms.supervised.Functions.Accountability.TrainTestSplitScore_supervised import get_train_test_split_score_supervised
from algorithms.supervised.Functions.Accountability.FactSheetCompletnessScore_supervised import get_factsheet_completeness_score_supervised
from algorithms.unsupervised.Functions.Accountability.FactSheetCompletnessScore import get_factsheet_completeness_score_unsupervised
from algorithms.supervised.Functions.Robustness.Robustness_supervised import get_robustness_score_supervised
from algorithms.unsupervised.Functions.Robustness.Robustness import analyse as get_robustness_score_unsupervised
from algorithms.TrustScore.TrustScore import trusting_AI_scores_unsupervised
from algorithms.TrustScore.TrustScore import trusting_AI_scores_supervised

# class CustomUserTestCase(TestCase):
#     def setUp(self):
#         CustomUser.objects.create(
#             email="user1", password="user1-password", username="user1", is_admin=False)
#         CustomUser.objects.create(
#             email="user2", password="user2-password", username="user2", is_admin=True)

#     def test_customuser(self):
#         """Animals that can speak are correctly identified"""
#         user1 = CustomUser.objects.get(email="user1")
#         user2 = CustomUser.objects.get(email="user2")
#         self.assertEqual(user1.get_user_name(), 'user1')
#         self.assertEqual(user1.get_user_password(), 'user1-password')
#         self.assertEqual(user2.get_user_name(), 'user2')
#         self.assertEqual(user2.get_user_password(), 'user2-password')


# class ScenarioTestCase(TestCase):
#     def setUp(self):
#         tempUser = CustomUser.objects.create(
#             email="temp", password="temp", username="temp")
#         Scenario.objects.create(
#             scenario_name="testscenario", description="testscenario", user_id=tempUser.get_user_id())

#     def test_scenario(self):
#         tempUser = CustomUser.objects.get(email="temp")
#         temp = Scenario.objects.get(scenario_name="testscenario")
#         self.assertEqual(temp.get_description(), 'testscenario')
#         self.assertEqual(temp.get_user_id(), temp.get_user_id())


# class SoltionTestCase(TestCase):
#     def setUp(self):
#         tempUser = CustomUser.objects.create(
#             email="temp", password="temp", username="temp")
#         print('id:', tempUser.get_user_id())
#         tempScenario = Scenario.objects.create(
#             user_id=tempUser.id,
#             scenario_name="temp",
#             description="temp"
#         )
#         ScenarioSolution.objects.create(
#             user_id=tempUser.id,
#             scenario_id=tempScenario.id,
#             solution_name="temp",
#             description="temp",
#             solution_type="supervised",
#             training_file=None,
#             test_file=None,
#             protected_features="temp",
#             protected_values="temp",
#             target_column="temp",
#             outlier_data_file=None,
#             favourable_outcome="temp",
#             factsheet_file=None,
#             model_file=None,
#             metrics_mappings_file=None,
#             weights_metrics=None,
#             weights_pillars=None
#         )

#     def test_solution(self):
#         tempUser = CustomUser.objects.get(email="temp")
#         tempScenario = Scenario.objects.get(user_id=tempUser.id)
#         tempSolution = ScenarioSolution.objects.get(
#             scenario_id=tempScenario.id)
#         self.assertEqual(tempSolution.get_description(), 'temp')
#         self.assertEqual(tempSolution.get_solution_name(), 'temp')
#         self.assertEqual(tempSolution.get_solution_type(), 'supervised')
#         print('file:', tempSolution.get_traing_file(), 'asdb')
#         self.assertEqual(tempSolution.get_traing_file(), '')
#         self.assertEqual(tempSolution.get_test_file(), '')
#         self.assertEqual(tempSolution.get_protected_feature(), 'temp')
#         self.assertEqual(tempSolution.get_protected_value(), 'temp')
#         self.assertEqual(tempSolution.get_target_column(), 'temp')
#         self.assertEqual(tempSolution.get_outlier_file(), '')
#         self.assertEqual(tempSolution.get_favourable_outcome(), 'temp')
#         self.assertEqual(tempSolution.get_factsheet_file(), '')
#         self.assertEqual(tempSolution.get_model_file(), '')
#         self.assertEqual(tempSolution.get_metrics_mapping_file(), '')
#         self.assertEqual(tempSolution.get_weights_metrics(), '')
#         self.assertEqual(tempSolution.get_weights_pillars(), '')


class SolutionTestCase(TestCase):
    def setUp(self):
        import os
        from pathlib import Path
        from os import listdir
        BASE_DIR = Path(__file__).resolve().parent.parent

        tempUser = CustomUser.objects.create(
            username='temp', email='temp', password='temp')
        tempScenario = Scenario.objects.create(
            user_id=tempUser.id, scenario_name='temp', description='temp')
        # for sup
        commonPath = 'files/TestValues_2003/supervised/working'
        mapFile = 'files/TestValues_2003/supervised/weights&mapping/mapping_metrics_default_sup.json'
        weightsMetrics = 'files/TestValues_2003/supervised/weights&mapping/weights_metrics_default_sup.json'
        weightsPillars = 'files/TestValues_2003/supervised/weights&mapping/weights_pillars_default_sup.json'
        folder_path = os.path.join(
            BASE_DIR, 'media/files/TestValues_2003/supervised/working')
        # for folder in listdir(folder_path):
        #     trainFile = join(commonPath, folder, 'train.csv')
        #     modelFile = join(commonPath, folder, 'model.pkl')
        #     testFile = join(commonPath, folder, 'test.csv')
        #     factFile = join(commonPath, folder, 'factsheet.json')
        #     path_module, path_traindata, path_testdata, path_factsheet, mappings_config, path_outliersdata, weights_metrics, weights_pillars = save_files_return_paths(
        #         modelFile,
        #         trainFile,
        #         testFile,
        #         factFile,
        #         mapFile,
        #         None,
        #         weightsMetrics,
        #         weightsPillars)

        #     model_path, training_dataset_path, test_dataset_path, factsheet_path, mappings_path, target_column = save_files_return_paths(
        #         modelFile, trainFile, testFile, factFile, mapFile, '')

        #     # account:
        #     print('factsheet score:' + folder, get_factsheet_completeness_score_supervised(
        #         model_path, training_dataset_path, test_dataset_path, factsheet_path, mappings_path))

        #     print('missingdata score:' + folder, get_missing_data_score_supervised(
        #         model_path, training_dataset_path, test_dataset_path, factsheet_path, mappings_path))

        #     print('normalization score:' + folder, get_normalization_score_supervised(
        #         model_path, training_dataset_path, test_dataset_path, factsheet_path, mappings_path))

        #     # print('regularization score:' + folder, get_regularization_score_supervised(
        #     #     model_path, training_dataset_path, test_dataset_path, factsheet_path, mappings_path))

        #     print('train_test score:' + folder, get_train_test_split_score_supervised(
        #         model_path, training_dataset_path, test_dataset_path, factsheet_path, mappings_path))

        #     # robust:
        #     print('clever_score score:' + folder, get_clever_score_supervised(
        #         model_path, training_dataset_path, test_dataset_path, factsheet_path, mappings_path))

        #     print('clique_method_score score:' + folder, get_clique_method_score_supervised(
        #         model_path, training_dataset_path, test_dataset_path, factsheet_path, mappings_path))

        #     print('confidence_score score:' + folder, get_confidence_score_supervised(
        #         model_path, training_dataset_path, test_dataset_path, factsheet_path, mappings_path))

        #     print('carliwagnerwttack_score score:' + folder, get_carliwagnerwttack_score_supervised(
        #         model_path, training_dataset_path, test_dataset_path, factsheet_path, mappings_path))

        #     print('loss_sensitivity_score score:' + folder, get_loss_sensitivity_score_supervised(
        #         model_path, training_dataset_path, test_dataset_path, factsheet_path, mappings_path))

        #     print('deepfoolattack_score score:' + folder, get_deepfoolattack_score_supervised(
        #         model_path, training_dataset_path, test_dataset_path, factsheet_path, mappings_path))

        #     print('fast_gradient_attack_score score:' + folder, get_fast_gradient_attack_score_supervised(
        #         model_path, training_dataset_path, test_dataset_path, factsheet_path, mappings_path))
        #     # explain
        #     print('modelsize_score score:' + folder, get_modelsize_score_supervised(
        #         model_path, training_dataset_path, test_dataset_path, factsheet_path, mappings_path))

        #     print('correlated_features_score score:' + folder, get_correlated_features_score_supervised(
        #         model_path, training_dataset_path, test_dataset_path, factsheet_path, mappings_path))

        #     print('algorithm_class_score score:' + folder, get_algorithm_class_score_supervised(
        #         model_path, training_dataset_path, test_dataset_path, factsheet_path, mappings_path))

        #     print('feature_relevance_score score:' + folder, get_feature_relevance_score_supervised(
        #         model_path, training_dataset_path, test_dataset_path, factsheet_path, mappings_path))

        #     print('permutation_feature_importance_score score:' +
        #           "The metric function isn't applicable for supervised ML/DL solutions")

        #     # fairness:
        #     print('disparate_impact_score score:' + folder, get_disparate_impact_score_supervised(
        #         model_path, training_dataset_path, test_dataset_path, factsheet_path, mappings_path, target_column))

        #     print('class_balance_score score:' + folder, get_class_balance_score_supervised(
        #         model_path, training_dataset_path, test_dataset_path, factsheet_path, mappings_path, target_column))

        #     # print('overfitting_score score:' + folder, get_overfitting_score_supervised(
        #     #     model_path, training_dataset_path, test_dataset_path, factsheet_path, mappings_path, target_column))

        #     print('underfitting_score score:' + folder, get_underfitting_score_supervised(
        #         model_path, training_dataset_path, test_dataset_path, factsheet_path, mappings_path, target_column))

        #     print('statistical_parity_difference_score score:' + folder, get_statistical_parity_difference_score_supervised(
        #         model_path, training_dataset_path, test_dataset_path, factsheet_path, mappings_path, target_column))

        #     print('equal_opportunity_difference_score score:' + folder, get_equal_opportunity_difference_score_supervised(
        #         model_path, training_dataset_path, test_dataset_path, factsheet_path, mappings_path, target_column))

        #     print('average_odds_difference_score score:' + folder, get_average_odds_difference_score_supervised(
        #         model_path, training_dataset_path, test_dataset_path, factsheet_path, mappings_path, target_column))
        #     # pillar:
        #     # print('accountability_score score:' + folder, get_accountability_score_supervised(
        #     # model_path, training_dataset_path, test_dataset_path, factsheet_path, mappings_path, target_column))

        #     print('robustnesss_score score:' + folder, get_robustness_score_supervised(
        #         model_path, training_dataset_path, test_dataset_path, factsheet_path, mappings_path, target_column))

        #     # print('explainability_score score:' + folder, get_explainability_score_supervised(
        #     #     model_path, training_dataset_path, test_dataset_path, factsheet_path, mappings_path, target_column))

        #     # print('fairness_score score:' + folder, get_fairness_score_supervised(
        #     #     model_path, training_dataset_path, test_dataset_path, factsheet_path, mappings_path, target_column))
        #     # trust:
        #     # print('trustscore score:' + folder, trusting_AI_scores_supervised(
        #     #     model_path, training_dataset_path, test_dataset_path, factsheet_path, mappings_path, target_column).score)

        #     # print('trusting_AI_scores_supervised score:' + folder, get_factsheet_completeness_score_supervised(
        #     #     model_path, training_dataset_path, test_dataset_path, factsheet_path, mappings_path))

        #     # print('account score:' + folder, trusting_AI_scores_supervised(
        #     #     model_path, training_dataset_path, test_dataset_path, factsheet_path, mappings_path, target_column))

        #     # print('finalScoreSupervised:' + folder + ':::', finalScore_supervised(path_module, path_traindata,
        #     #       path_testdata, path_factsheet, path_outliersdata, weights_metrics, weights_pillars))

        # for unsup
        commonPath = 'files/TestValues_2003/unsupervised/working'
        mapFile = 'files/TestValues_2003/unsupervised/weights&mapping/mapping_metrics_default_unsup.json'
        weightsMetrics = 'files/TestValues_2003/unsupervised/weights&mapping/weights_metrics_default_unsup.json'
        weightsPillars = 'files/TestValues_2003/unsupervised/weights&mapping/weights_pillars_default_unsup.json'
        folder_path = os.path.join(
            BASE_DIR, 'media/files/TestValues_2003/unsupervised/working')
        for folder in listdir(folder_path):
            trainFile = join(commonPath, folder, 'train.csv')
            modelFile = join(commonPath, folder, 'model.joblib')
            testFile = join(commonPath, folder, 'test.csv')
            factFile = join(commonPath, folder, 'factsheet.json')
            outFile = join(commonPath, folder, 'outliers.csv')

            path_module, path_traindata, path_testdata, path_factsheet, mappings_config, path_outliersdata, weights_metrics, weights_pillars = save_files_return_paths(
                modelFile,
                trainFile,
                testFile,
                factFile,
                mapFile,
                None,
                weightsMetrics,
                weightsPillars)

            model_path, training_dataset_path, test_dataset_path, factsheet_path, mappings_path, outliers_data, target_column = save_files_return_paths(
                modelFile, trainFile, testFile, factFile, mapFile, outFile, '')

            # account
            print('factsheet score' + folder, get_factsheet_completeness_score_unsupervised(
                model_path, training_dataset_path, test_dataset_path, factsheet_path, mappings_path))

            print('missingdata' + folder, get_missing_data_score_unsupervised(
                model_path, training_dataset_path, test_dataset_path, factsheet_path, mappings_path))

            print('normalization' + folder, get_normalization_score_unsupervised(
                model_path, training_dataset_path, test_dataset_path, factsheet_path, mappings_path))

            print('regularization' + folder, get_regularization_score_unsupervised(
                model_path, training_dataset_path, test_dataset_path, factsheet_path, mappings_path))

            print('train_test' + folder, get_train_test_split_score_unsupervised(
                model_path, training_dataset_path, test_dataset_path, factsheet_path, mappings_path))
            # robust
            print('clever_score' + folder, get_clever_score_unsupervised(
                model_path, training_dataset_path, test_dataset_path, factsheet_path, mappings_path))

            print('clique_method_score' + folder, get_clique_method_score_unsupervised(
                model_path, training_dataset_path, test_dataset_path, factsheet_path, mappings_path))

            print('confidence_score' + folder,
                  "The metric function isn't applicable for unsupervised ML/DL solutions")

            print('carliwagnerwttack_score' + folder,
                  "The metric function isn't applicable for unsupervised ML/DL solutions")

            print('loss_sensitivity_score' + folder,
                  "The metric function isn't applicable for unsupervised ML/DL solutions")

            print('deepfoolattack_score' + folder,
                  "The metric function isn't applicable for unsupervised ML/DL solutions")

            print('fast_gradient_attack_score' + folder,
                  "The metric function isn't applicable for unsupervised ML/DL solutions")
            # explain
            print('modelsize_score' + folder, get_modelsize_score_unsupervised(
                model_path, training_dataset_path, test_dataset_path, factsheet_path, mappings_path))

            print('correlated_features_score' + folder, get_correlated_features_score_unsupervised(
                model_path, training_dataset_path, test_dataset_path, factsheet_path, mappings_path))

            print('algorithm_class_score' + folder,
                  "The metric function isn't applicable for unsupervised ML/DL solutions")

            print('feature_relevance_score' + folder,
                  "The metric function isn't applicable for unsupervised ML/DL solutions")

            print('permutation_feature_importance_score' + folder, get_permutation_feature_importance_score_unsupervised(
                model_path, training_dataset_path, test_dataset_path, factsheet_path, mappings_path, outliers_data))
            # fairness
            print('disparate_impact_score' + folder, get_disparate_impact_score_unsupervised(
                model_path, training_dataset_path, test_dataset_path, factsheet_path, mappings_path, target_column, outliers_data))

            print('class_balance_score' + folder,
                  "The metric function isn't applicable for unsupervised ML/DL solutions")

            print('overfitting_score' + folder, get_overfitting_score_unsupervised(
                model_path, training_dataset_path, test_dataset_path, factsheet_path, mappings_path, target_column, outliers_data))

            print('underfitting_score' + folder, get_underfitting_score_unsupervised(
                model_path, training_dataset_path, test_dataset_path, factsheet_path, mappings_path, target_column, outliers_data))

            print('statistical_parity_difference_score' + folder, get_statistical_parity_difference_score_unsupervised(
                model_path, training_dataset_path, test_dataset_path, factsheet_path, mappings_path, target_column, outliers_data))

            print('equal_opportunity_difference_score' + folder,
                  "The metric function isn't applicable for unsupervised ML/DL solutions")

            print('average_odds_difference_score' + folder,
                  "The metric function isn't applicable for unsupervised ML/DL solutions")
            # pillar
            print('accountability_score' + folder, get_accountability_score_unsupervised(
                model_path, training_dataset_path, test_dataset_path, factsheet_path, mappings_path, target_column, outliers_data))

            print('robustnesss_score' + folder, get_robustness_score_unsupervised(
                model_path, training_dataset_path, test_dataset_path, factsheet_path, mappings_path, target_column, outliers_data))

            print('explainability_score' + folder,
                  get_explainability_score_unsupervised(
                      model_path, training_dataset_path, test_dataset_path, factsheet_path, mappings_path, target_column, outliers_data))

            print('fairness_score' + folder, get_fairness_score_unsupervised(
                model_path, training_dataset_path, test_dataset_path, factsheet_path, mappings_path, target_column, outliers_data))
            # trustscore
            print('trustscore' + folder, trusting_AI_scores_unsupervised(
                model_path, training_dataset_path, test_dataset_path, factsheet_path, mappings_path, target_column, outliers_data).score)

            # print('average_odds_difference_score:' + folder + ':::', unsupervised_FinalScore(path_module, path_traindata,
            #       path_testdata, path_factsheet, path_outliersdata, weights_metrics, weights_pillars))

    def test(self):
        return 'ok'


class SolutionTestCase(TestCase):
    def setUp(self):
        import os
        import pandas as pd
        from pathlib import Path
        from os import listdir
        BASE_DIR = Path(__file__).resolve().parent.parent

        folder_path = os.path.join(
            BASE_DIR, 'media/files/TestValues_2003/supervised/working')
        for folder in listdir(folder_path):
            factFile = join(folder_path, folder, 'factsheet.json')
            factData = pd.read_json(factFile)

            if ("scores" in factData):
                print('supervised scores:' + folder, factData['scores'])
            else:
                print('supervised test' + folder +
                      'has no scores in factsheet.json')

        folder_path = os.path.join(
            BASE_DIR, 'media/files/TestValues_2003/unsupervised/working')
        for folder in listdir(folder_path):
            factFile = join(folder_path, folder, 'factsheet.json')
            factData = pd.read_json(factFile)

            if ("scores" in factData):
                print('unsupervised scores:' + folder, factData['scores'])
            else:
                print('unsupervised test' + folder +
                      'has no scores in factsheet.json')

    def test(self):
        return 'ok'
