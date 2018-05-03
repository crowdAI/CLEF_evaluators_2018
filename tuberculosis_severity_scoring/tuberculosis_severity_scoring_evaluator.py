import csv
import numpy as np
from sklearn import metrics
"""
Evaluator class
Evaluates one single runfile
_evaluate method is called by the CrowdAI framework and returns an object holding up to 2 different scores
"""
class TuberculosisSeverityScoringEvaluator:

    """
    Constructor
    Parameter 'answer_file_path': Path of file containing ground truth
    """
    def __init__(self, answer_file_path, debug_mode=False):
        #Ground Truth file
        self.answer_file_path = answer_file_path
        self.gt = self.load_gt()

    """
    This is the only method that will be called by the framework
    Parameter 'submission_file_path': Path of the submitted runfile
    returns a _result_object that can contain up to 2 different scores
    """
    def _evaluate(self, client_payload, context={}):
        submission_file_path = client_payload['submission_file_path']

        predictions = self.load_predictions(submission_file_path)

        predictedScores = np.asarray([], float)
        predictedProbs = np.asarray([], float)
        trueScores = np.asarray([], float)
        trueClasses = np.asarray([], float)
        for key in predictions:
            prediction = predictions[key]
            gtruth = self.gt[key]

            predictedScores = np.append(predictedScores, prediction[0])
            predictedProbs = np.append(predictedProbs, prediction[1])
            trueScores = np.append(trueScores, gtruth[0])
            trueClasses = np.append(trueClasses, gtruth[1])

        squaredErrors = np.power(predictedScores - trueScores, 2)
        rmse = np.power(np.mean(squaredErrors), 0.5)

        fpr, tpr, thr = metrics.roc_curve(trueClasses, predictedProbs, pos_label=1)
        auc = metrics.auc(fpr, tpr)

        #_result_object = {
        #  "RMSE": rmse,
        #  "AUC" : auc
        #}

        _result_object = {
          "score": rmse,
          "score_secondary" : auc
        }

        return _result_object

    """
    Loads and returns a predictions object (dictionary) that contains the submitted data that will be used in the _evaluate method
    Parameter 'submission_file_path': Path of the submitted runfile
    Validation of the runfile format will also be handled here
    THE VALIDATION PART CAN BE IMPLEMENTED BY IVAN IF YOU WISH (ivan.eggel@hevs.ch)
    """
    # Format: <Patient-ID>,<Severity score>,<Probability of "HIGH" severity>
    # Validation already implemented by Ivan
    # Return a dict where the value of dict[patient_id] is a tuple (severity_score, probability)
    def load_predictions(self,submission_file_path):
        predictions = {}
        patient_ids_gt = list(self.gt.keys())
        allowed_severity_scores = set([self.gt[i][0] for i in self.gt])
        occured_patient_ids = []
        with open(submission_file_path) as csvfile:
            reader = csv.reader(csvfile, delimiter=',', quoting=csv.QUOTE_NONE)
            lineCnt = 0

            for row in reader:
                lineCnt += 1

                #Not 3 comma separated tokens on line => Error
                if len(row) != 3:
                    raise Exception("Wrong format: Each line must consist of an image ID followed by a comma, a TB severity score and a probability ({}). {}"
                        .format("<patient_id>,<tb_severity_score>,<probability>", self.line_nbr_string(lineCnt)))

                patient_id = row[0]
                # Patient ID does not exist in testset => Error
                if patient_id not in patient_ids_gt:
                    raise Exception("Patient ID '{}' in submission file does not exist in testset {}"
                        .format(patient_id, self.line_nbr_string(lineCnt)))

                # Patient ID occured at least twice in file => Error
                if patient_id in occured_patient_ids:
                    raise Exception("Patient ID '{}' was specified more than once in submission file {}"
                        .format(patient_id, self.line_nbr_string(lineCnt)))

                occured_patient_ids.append(patient_id)

                # 2nd row of line is not an int or contained in possible_tb_types => Error
                try:
                    svr_score = int(row[1])
                    if svr_score not in allowed_severity_scores:
                        raise ValueError
                except ValueError:
                    raise Exception("Invalid TB severity score '{}'. Possible values are: {}. {}"
                        .format(row[1], allowed_severity_scores, self.line_nbr_string(lineCnt)))

                # Probability not a number or not between 0 and 1 => Error
                try:
                    probability = float(row[2])
                    if probability < 0 or probability > 1:
                        raise ValueError
                except ValueError:
                    raise Exception("Probability must be a number between 0 and 1 {}"
                        .format(self.line_nbr_string(lineCnt)))

                predictions[patient_id] = (svr_score, probability)

            # In case not all images from the testset are contained in the file => Error
            if(len(occured_patient_ids) != len (patient_ids_gt)):
                raise Exception("Number of patient IDs in submission file not equal to number of patient IDs in testset")

        return predictions


    """
    Load and return groundtruth data
    """
    def load_gt(self):
        gt = {}
        with open(self.answer_file_path) as csvfile:
            reader = csv.reader(csvfile, delimiter=',', quoting=csv.QUOTE_NONE)
            lineCnt = 0
            for row in reader:
                patient_id = row[0]
                svr_score = int(row[1])
                probability = float(row[2])
                gt[patient_id] = (svr_score, probability)
        return gt



    def line_nbr_string(self, line_nbr):
        return "(Line nbr {})".format(line_nbr)



"""
Test evaluation a runfile
provide path to ground truth file in constructor
call _evaluate method with path of submitted file as argument
"""
if __name__ == "__main__":

  #Ground truth file
  gt_file_path = "gt_file.csv"

  #Submission file
  submission_file_path = gt_file_path
  #submission_file_path = "runs/00_run_ok.csv"
  #submission_file_path = "runs/01_wrong_patient_id.csv"
  #submission_file_path = "runs/02_patient_id_more_than_once.csv"
  #submission_file_path = "runs/03_wrong_svr_score.csv"
  #submission_file_path = "runs/04_probability_not_number.csv"
  #submission_file_path = "runs/05_probability_not_btw_0_and_1.csv"
  #submission_file_path = "runs/06_not_all_patient_ids_included.csv"
  #submission_file_path = "runs/07_not_3_tokens.csv"


  _client_payload = {}
  _client_payload["submission_file_path"] = submission_file_path
  #Create instance of Evaluator
  evaluator = TuberculosisSeverityScoringEvaluator(gt_file_path)
  #Call _evaluate method
  result = evaluator._evaluate(_client_payload)
  print(result)
