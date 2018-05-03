import csv
import pandas as pd
import numpy as np
from sklearn import metrics

class TuberculosisMdrDetectionEvaluator:

    def __init__(self, answer_file_path, debug_mode=False):
        #Ground Truth file
        self.answer_file_path = answer_file_path
        self.gt = self.load_gt()

    def _evaluate(self, client_payload, context={}):
        submission_file_path = client_payload['submission_file_path']

        predictions = self.load_predictions(submission_file_path)

        predictedProbs = np.asarray([], float)
        trueClasses = np.asarray([], float)
        for key in predictions:
            prediction = float(predictions[key])
            gtruth = self.gt.loc[self.gt[0] == key]

            predictedProbs = np.append(predictedProbs, prediction)
            trueClasses = np.append(trueClasses, gtruth.values[0][1])

        predictedClasses = (predictedProbs > 0.5).astype(float)
        matched = (predictedClasses == trueClasses).astype(int)
        acc = matched.sum() / len(matched)

        fpr, tpr, thr = metrics.roc_curve(trueClasses, predictedProbs, pos_label=1)
        auc = metrics.auc(fpr, tpr)

        #_result_object = {
        #    "ACC": acc,
        #    "AUC": auc
        #}

        _result_object = {
            "score": acc,
            "score_secondary": auc
        }
        return _result_object

    def load_predictions(self,submission_file_path):
        pairs = {}
        patient_ids_gt = self.gt[0].tolist()

        with open(submission_file_path) as csvfile:
            reader = csv.reader(csvfile, delimiter=',', quoting=csv.QUOTE_NONE)
            lineCnt = 0
            occured_patients = []
            for row in reader:
                lineCnt += 1
                #Not 2 comma separated tokens on line => Error
                if len(row) != 2:
                    raise Exception("Wrong format: Each line must consist of a patient ID followed by a comma and a score ({}) {}"
                        .format("<patient_id>,<score>", self.line_nbr_string(lineCnt)))

                patient_id = row[0]

                # Patient ID does not exist in testset => Error
                if patient_id not in patient_ids_gt:
                    raise Exception("Patient ID '{}' in submission file does not exist in testset {}"
                        .format(patient_id, self.line_nbr_string(lineCnt)))

                # Patient ID occured at least twice in file => Error
                if patient_id in occured_patients:
                    raise Exception("Patient ID '{}' was specified more than once in submission file {}"
                        .format(patient_id, self.line_nbr_string(lineCnt)))

                occured_patients.append(patient_id)

                # 2nd value on row not a number or not between 0 and 1 => Error
                try:
                    probability = float(row[1])
                    if probability < 0 or probability > 1:
                        raise ValueError
                except ValueError:
                    raise Exception("Score must be a number between 0 and 1 {}"
                        .format(self.line_nbr_string(lineCnt)))

                pairs[row[0]] = row[1]

            # In case not all images from the testset are contained in the file => Error
            if(len(occured_patients) != len (patient_ids_gt)):
                raise Exception("Number of patient IDs in submission file not equal to number of patient IDs in testset")

        return pairs


    def load_gt(self):
        return pd.read_csv(self.answer_file_path, sep=",", header=None)
        #print(self.gt[0].tolist())


    def line_nbr_string(self, line_nbr):
        return "(Line nbr {})".format(line_nbr)




if __name__ == "__main__":

  #Ground truth file
  gt_file_path = "gt_file.csv"
  #Submission file

  submission_file_path = gt_file_path # => perfect run
  #submission_file_path = "runs/00_run_ok.csv"
  #submission_file_path = "runs/01_not_2_tokens.csv"
  #submission_file_path = "runs/02_wrong_patient_id.csv"
  #submission_file_path = "runs/03_same_patient_id_more_than_once.csv"
  #submission_file_path = "runs/04_wrong_score_no_number.csv"
  #submission_file_path = "runs/05_wrong_score_greater_1.csv"
  #submission_file_path = "runs/06_not_all_patient_ids.csv"

  _client_payload = {}
  _client_payload["submission_file_path"] = submission_file_path
  #Create instance of Evaluator
  evaluator = TuberculosisMdrDetectionEvaluator(gt_file_path)
  #Call _evaluate method
  result = evaluator._evaluate(_client_payload)
  print(result)
