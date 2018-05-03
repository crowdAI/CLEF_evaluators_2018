import csv
import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.metrics.classification import confusion_matrix


class TuberculosisTbTypeEvaluator:

    def __init__(self, answer_file_path,debug_mode=False):
        #Ground Truth file
        self.answer_file_path = answer_file_path
        self.gt = self.load_gt()

    def cohensKappa(self, trueClasses, predictedClasses):
        confusionMat = metrics.confusion_matrix(trueClasses, predictedClasses)

        m = confusionMat.shape[0]

        f = np.diag(np.ones(m))             # unweighted

        n = np.sum(confusionMat)            # sum of all matrix elements
        confusionMat = confusionMat / n     # proportion
        r = np.sum(confusionMat, axis=1)    # rows sum
        s = np.sum(confusionMat, axis=0)    # columns sum
        r = np.reshape(r, (len(r), 1))
        s = np.reshape(s, (1, len(s)))
        Ex = np.matmul(r, s)                # expected proportion for random agree
        po = np.sum(np.multiply(confusionMat, f))
        pe = np.sum(np.multiply(Ex, f))
        kappa = (po - pe) / (1 - pe)

        return kappa

    def _evaluate(self, client_payload, context={}):
        submission_file_path = client_payload['submission_file_path']

        predictions = self.load_predictions(submission_file_path)

        predictedClasses = np.asarray([], float)
        trueClasses = np.asarray([], float)
        for key in predictions:
            prediction = float(predictions[key])
            gtruth = self.gt.loc[self.gt[0] == key]

            predictedClasses = np.append(predictedClasses, prediction)
            trueClasses = np.append(trueClasses, gtruth.values[0][1])

        matched = (predictedClasses == trueClasses).astype(int)
        acc = matched.sum() / len(matched)

        kappa = self.cohensKappa(trueClasses, predictedClasses)

        #_result_object = {
        #  "ACC": acc,
        #  "Kappa": kappa
        #}

        _result_object = {
          "score": acc,
          "score_secondary": kappa
        }


        return _result_object

    def load_predictions(self,submission_file_path):
        pairs = {}
        patient_ids_gt = self.gt[0].tolist()
        possible_tb_types = set(self.gt[1].tolist())
        with open(submission_file_path) as csvfile:
            reader = csv.reader(csvfile, delimiter=',', quoting=csv.QUOTE_NONE)
            lineCnt = 0
            occured_patients = []
            for row in reader:
                lineCnt += 1
                #Not 2 comma separated tokens on line => Error
                if len(row) != 2:
                    raise Exception("Wrong format: Each line must consist of an patient ID followed by a comma and the TB type ({}) {}"
                        .format("<patient_id>,<tb_type>", self.line_nbr_string(lineCnt)))

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

                # 2nd row of line is not an int or contained in possible_tb_types => Error
                try:
                    tb_type = int(row[1])
                    if tb_type not in possible_tb_types:
                        raise ValueError
                except ValueError:
                    raise Exception("TB type '{}' does not exist {}. Possible values are: {}"
                        .format(row[1], self.line_nbr_string(lineCnt), possible_tb_types))

                pairs[row[0]] = row[1]

            # In case not all patients from the testset are contained in the file => Error
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
  #submission_file_path = gt_file_path # => Perfect run
  submission_file_path = "runs/00_run_ok.csv"
  #submission_file_path = "runs/01_not_2_tokens.csv"
  #submission_file_path = "runs/02_wrong_patient_id.csv"
  #submission_file_path = "runs/03_same_patient_id_more_than_once.csv"
  #submission_file_path = "runs/04_wrong_tb_type_no_int.csv"
  #submission_file_path = "runs/05_wrong_tb_type_greater_5.csv"
  #submission_file_path = "runs/06_not_all_patient_ids.csv"

  _client_payload = {}
  _client_payload["submission_file_path"] = submission_file_path

  #Create instance of Evaluator
  evaluator = TuberculosisTbTypeEvaluator(gt_file_path)
  #Call _evaluate method
  result = evaluator._evaluate(_client_payload)
  print(result)
