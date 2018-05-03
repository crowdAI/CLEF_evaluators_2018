"""
Evaluator class
Evaluates one single runfile
_evaluate method is called by the CrowdAI framework and returns an object holding up to 2 different scores
"""

import csv
import datetime


class BirdMonophoneEvaluator:

    """
    Constructor
    Parameter 'answer_file_path': Path of file containing ground truth
    """
    def __init__(self,  answer_file_path,
                        allowed_classes_file_path,
                        debug_mode=False):
        #Ground truth file
        self.answer_file_path = answer_file_path

        #allowed ids in the predictions files
        self.allowed_classes_file_path = allowed_classes_file_path

        #Ground truth data
        self.gt = self.load_gt()




    """
    This is the only method that will be called by the framework
    Parameter 'submission_file_path': Path of the submitted runfile
    returns a _result_object that can contain up to 2 different scores
    """
    def _evaluate(self, client_payload, context={}):
        submission_file_path = client_payload['submission_file_path']
        #Load predictions
        predictions = self.load_predictions(submission_file_path)

        if predictions != None:
            #Compute first score
            rmap_foreground = self.retrieval_mean_average_precision_foreground(predictions)
            #Compute second score
            rmap_background = self.retrieval_mean_average_precision_background(predictions)

            #Create object that is returned to the CrowdAI framework
            # _result_object = {
            #     "mrr_only_foreground_species": rmap_foreground,
            #     "map_including_background_species" : rmap_background
            # }
            _result_object = {
                "score": rmap_foreground,
                "score_secondary" : rmap_background
            }

            return _result_object


    """
    Load and return groundtruth data
    """
    def load_gt(self):
        gt = {}
        gt['foreground'] = {}
        gt['with_background'] = {}

        allowed_classes = self.load_allowed_classes()

        with open(self.answer_file_path) as csvfile:
            reader = csv.reader(csvfile, delimiter=';', quoting=csv.QUOTE_NONE)
            for row in reader:
                query = row[0]

                classid_foreground = row[1]
                gt['foreground'][query] = set() #more convenient for having one score function
                gt['foreground'][query].add(classid_foreground)

                classids_all = set()
                classids_all.add(classid_foreground)

                for classid_background in row[2].split(','):
                    if classid_background in allowed_classes:
                        classids_all.add(classid_background)
                gt['with_background'][query] = classids_all

        return gt


    """
    Load and return allowed class ids in the predictions files
    """
    def load_allowed_classes(self):
        #...
        #return gt
        allowed_classes = set()
        with open(self.allowed_classes_file_path) as f:
            for classid in f.readlines():
                allowed_classes.add(classid.rstrip("\n"))
        return allowed_classes


    """
    Loads and returns a predictions object (dictionary) that contains the submitted data that will be used in the _evaluate method
    Parameter 'submission_file_path': Path of the submitted runfile
    Validation of the runfile format will also be handled here
    THE VALIDATION PART CAN BE IMPLEMENTED BY IVAN IF YOU WISH (ivan.eggel@hevs.ch)
    """

    def load_predictions(self, submission_file_path):
        #...
        #returns predictions
        query_to_correct_classid_ranks = {}
        query_to_correct_classid_ranks['foreground'] = {}
        query_to_correct_classid_ranks['with_background'] = {}

        allowed_query_ids = self.gt['foreground'].keys()
        allowed_classes = self.load_allowed_classes()

        max_rank = 100 #max nbr of classes for query_tc

        with open(submission_file_path) as csvfile:
            reader = csv.reader(csvfile, delimiter=';', quoting=csv.QUOTE_NONE)
            lineCnt = 0
            occured_observations = {}

            for row in reader:
                lineCnt += 1
                if lineCnt % 100000 == 0:
                    print(lineCnt)

                if len(row) != 4:
                    raise Exception("Wrong format: Each line must consist of a Media ID, Class ID, score and rank separated by semicolons ({}) {}"
                        .format("<MediaId>;<ClassId>;<Score>;<Rank>", self.line_nbr_string(lineCnt)))

                query_id = row[0]

                if query_id not in allowed_query_ids:
                    raise Exception("MediaID '{}' in submission file does not exist in testset {}"
                        .format(query_id, self.line_nbr_string(lineCnt)))

                class_id = row[1]
                # Class ID not in testset => Error
                if class_id not in allowed_classes:
                    raise Exception("'{}' is not a valid class ID {}"
                        .format(class_id, self.line_nbr_string(lineCnt)))


                # 3rd value in line is not a number  => Error
                try:
                    probability = float(row[2])
                    if probability < 0 or probability > 1:
                        raise ValueError
                except ValueError:
                    raise Exception("Score must be a probability between 0 and 1 {}"
                        .format(self.line_nbr_string(lineCnt)))

                # Rank not an int between 1 and 100 => Error
                try:
                    rank = int(row[3])
                    if rank < 1 or rank > max_rank:
                        raise ValueError
                except ValueError:
                    raise Exception("Rank 'must be an integer between 1 and {}. {}"
                        .format(max_rank, self.line_nbr_string(lineCnt)))


                values_for_observation = occured_observations.get(query_id,list())
                class_ids_for_observation = [tup[0] for tup in values_for_observation]

                # Same query_id combined with class_id present more than once => Error
                if class_id in class_ids_for_observation:
                    raise Exception("Same prediction (query_id;class_id) present more than once ({};{}) {}"
                        .format(query_id, class_id, self.line_nbr_string(lineCnt)))

                #add tuple to observations
                values_for_observation.append((class_id, probability, rank, lineCnt))
                occured_observations[query_id] = values_for_observation

                #add to dict. this dict will be returned by the function later
                for focus in ['foreground','with_background']:
                    if class_id in self.gt[focus][query_id]:
                        if not query_id in query_to_correct_classid_ranks[focus]:
                            query_to_correct_classid_ranks[focus][query_id] = set()
                        query_to_correct_classid_ranks[focus][query_id].add(rank)

            for q_id in occured_observations:
                # Sort by rank (tup[2])
                values_sorted = sorted(occured_observations[q_id], key=lambda tup: (tup[2]))
                last_rank = 0
                for curr_class_id, curr_probability, curr_rank, curr_line in values_sorted:
                    #Ranking for media_id not consecutive => Error
                    if curr_rank != (last_rank+1):
                        raise Exception("Ranking must be consecutive {}"
                            .format(self.line_nbr_string(curr_line)))
                    last_rank = curr_rank

        return query_to_correct_classid_ranks



    def line_nbr_string(self, line_nbr):
        return "(Line nbr {})".format(line_nbr)



    """
    Compute and return the primary score
    Parameter 'predictions' : predictions object generated by the load_predictions method
    NO VALIDATION OF THE RUNFILE SHOULD BE IMPLEMENTED HERE
    We assume that the predictions in the parameter are valid
    Valiation should be handled in the load_predictions method
    """
    def retrieval_mean_average_precision_foreground(self, query_to_correct_classid_ranks):
        return self.compute_map_score('foreground', query_to_correct_classid_ranks)


    """
    Compute and return the secondary score
    Parameter 'predictions' : predictions object generated by the load_predictions method
    NO VALIDATION OF THE RUNFILE SHOULD BE IMPLEMENTED HERE
    We assume that the predictions in the parameter are valid
    Valiation should be handled in the load_predictions method
    """
    def retrieval_mean_average_precision_background(self, query_to_correct_classid_ranks):
        return self.compute_map_score('with_background', query_to_correct_classid_ranks)


    def compute_map_score(self, by_type, query_to_correct_classid_ranks):
        
        excluded = set(["8794","28335"])
        map = 0.0

        for query in self.gt[by_type]:
            if query not in excluded:
                ap = 0.0
                count_relevant = 0

                if query in query_to_correct_classid_ranks[by_type]:
                    correct_ranks = query_to_correct_classid_ranks[by_type][query]
                    for rank in correct_ranks:
                        count_relevant += 1
                        ap +=  float(count_relevant) / float(rank)

                ap = ap / float(len(self.gt[by_type][query]))
                map += ap

        map =  map / float (len(self.gt[by_type]) - len(excluded))

        return map

"""
Test evaluation a runfile
provide path to ground truth file in constructor
call _evaluate method with path of submitted file as argument
"""
if __name__ == "__main__":

  #Ground truth file
  gt_file_path = "gt_file.csv"

  #allowed classids
  allowed_classes_file_path = 'allowed_classes.txt'



  #Submission file
  submission_file_path = "runs/00_perfect_only_foreground.csv"
  #submission_file_path = "runs/01_perfect_with_background_equiproba.csv"
  #submission_file_path = "runs/02_perfect_foreground_with_background.csv"
  #submission_file_path = "runs/03_univ-tln-run1.csv"



  #submission_file_path = "runs/04_not_4_tokens.csv"
  #submission_file_path = "runs/05_wrong_media_id.csv"
  #submission_file_path = "runs/06_wrong_class_id.csv"
  #submission_file_path = "runs/07_invalid_score.csv"
  #submission_file_path = "runs/08_invalid_rank.csv"
  #submission_file_path = "runs/09_same_prediction_more_than_once.csv"
  #submission_file_path = "runs/10_ranking_not_consecutive.csv"

  _client_payload = {}
  _client_payload["submission_file_path"] = submission_file_path


  #Create instance of Evaluator
  evaluator = BirdMonophoneEvaluator(gt_file_path, allowed_classes_file_path)
  #Call _evaluate method
  result = evaluator._evaluate(_client_payload)
  print(result)
