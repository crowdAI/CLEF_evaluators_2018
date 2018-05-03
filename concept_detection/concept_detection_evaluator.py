import csv
from sklearn.metrics import f1_score
"""
Evaluator class
Evaluates one single runfile
_evaluate method is called by the CrowdAI framework and returns an object holding up to 2 different scores
"""
class ConceptDetectionEvaluator:

    """
    Constructor
    Parameter 'answer_file_path': Path of file containing ground truth
    """
    def __init__(self, answer_file_path, debug_mode=False):
      self.answer_file_path = answer_file_path
      #Ground truth pairs {image_id:concepts}
      self.gt_pairs = self.load_gt()


    """
    This is the only method that will be called by the framework
    Parameter 'submission_file_path': Path of the submitted runfile
    returns a _result_object that can contain up to 2 different scores
    """
    def _evaluate(self, client_payload, context={}):
      submission_file_path = client_payload['submission_file_path']

      candidate_pairs = self.load_predictions(submission_file_path)
      f1_score = self.compute_f1(candidate_pairs)

    #  _result_object = {
    #      "f1": f1_score,
    #      "score_secondary" : 0 # no secondary score
    # }

      _result_object = {
          "score": f1_score,
          "score_secondary" : 0
      }
      return _result_object



    """
    Loads and returns a predictions object (dictionary) that contains the submitted data that will be used in the _evaluate method
    Parameter 'submission_file_path': Path of the submitted runfile
    Validation of the runfile format will also be handled here
    THE VALIDATION PART CAN BE IMPLEMENTED BY IVAN IF YOU WISH (ivan.eggel@hevs.ch)
    """
    def load_predictions(self, submission_file_path):
        pairs = {}
        image_ids_gt = set(self.gt_pairs.keys())
        max_num_concepts = 1286 # max num concepts for an image in gt file
        with open(submission_file_path) as csvfile:
            reader = csv.reader(csvfile, delimiter='\t', quoting=csv.QUOTE_NONE)
            lineCnt = 0
            occured_images = []
            for row in reader:
                lineCnt += 1

                # empty line => Error
                if(len(row) < 1):
                    raise Exception("Wrong format: Each line must at least consist of an image ID {}"
                        .format(self.line_nbr_string(lineCnt)))
                #in case more than 2 tab separated tokens => Error
                elif(len(row) > 2):
                    raise Exception("Wrong format: Line consist of more than 2 tokens separated by a tab {}"
                        .format(self.line_nbr_string(lineCnt)))

                image_id = row[0]
                # Image ID does not exist in testset => Error
                if image_id not in image_ids_gt:
                    raise Exception("Image ID '{}' in submission file does not exist in testset {}"
                        .format(image_id,self.line_nbr_string(lineCnt)))

                # more than max num concepts for image => Error
                if len(row) > 1:
                    concepts = row[1].split(";");
                    if len(concepts) > max_num_concepts:
                        raise Exception("There must be between 0 and {} concepts per image {}"
                            .format(max_num_concepts,self.line_nbr_string(lineCnt)))

                # concept(s) specified more than once for an image => Error
                if len(concepts) != len(set(concepts)):
                    raise Exception("Same concept was specified more than once for image ID '{}' {}"
                        .format(image_id, self.line_nbr_string(lineCnt)))

                # image id occured at least twice in file => Error
                if image_id in occured_images:
                    raise Exception("Image ID '{}' was specified more than once in submission file {}"
                        .format(image_id, self.line_nbr_string(lineCnt)))

                occured_images.append(image_id)

                # Now add image with concepts to final dict
                # We have an ID and a set of concepts (possibly empty) => OK
                if len(row) == 2:
                    pairs[image_id] = row[1]
                # We only have an ID => OK
                elif len(row) == 1:
                    pairs[image_id] = ''

            # In case not all images from the testset are contained in the file => Error
            if(len(occured_images) != len (image_ids_gt)):
                raise Exception("Number of image IDs in submission file not equal to number of image IDs in testset")

        return pairs


    """
    Load and return groundtruth data
    """
    def load_gt(self):
        pairs = {}
        with open(self.answer_file_path) as csvfile:
            reader = csv.reader(csvfile, delimiter='\t', quoting=csv.QUOTE_NONE)
            for row in reader:
                # We have an ID and a set of concepts (possibly empty)
                if len(row) == 2:
                    pairs[row[0]] = row[1]
                # We only have an ID
                elif len(row) == 1:
                    pairs[row[0]] = ''
                else:
                    raise Exception("Answer file format is wrong. Organizer should check the answer file")
        return pairs


    """
    Compute and return the primary score
    Parameter 'predictions' : predictions object generated by the load_predictions method
    NO VALIDATION OF THE RUNFILE SHOULD BE IMPLEMENTED HERE
    We assume that the predictions in the parameter are valid
    Valiation should be handled in the load_predictions method
    """
    def compute_f1(self,candidate_pairs):
        # Define max score and current score
        max_score = len(self.gt_pairs) #nbr images
        current_score = 0

        # Check there are the same number of pairs between candidate and ground truth
        if len(candidate_pairs) != len(self.gt_pairs):
            raise Exception('ERROR : Candidate does not contain the same number of entries as the ground truth!')

        # Evaluate each candidate concept list against the ground truth
        i = 0
        for image_key in candidate_pairs:

            # Get candidate and GT concepts
            candidate_concepts = candidate_pairs[image_key].upper()
            gt_concepts = self.gt_pairs[image_key].upper()

            # Split concept string into concept array
            # Manage empty concept lists
            if gt_concepts.strip() == '':
                gt_concepts = []
            else:
                gt_concepts = gt_concepts.split(',')

            if candidate_concepts.strip() == '':
                candidate_concepts = []
            else:
                candidate_concepts = candidate_concepts.split(',')

            # Manage empty GT concepts (ignore in evaluation)
            if len(gt_concepts) == 0:
                max_score -= 1
            # Normal evaluation
            else:
                # Global set of concepts
                all_concepts = sorted(list(set(gt_concepts + candidate_concepts)))

                # Calculate F1 score for the current concepts
                y_true = [int(concept in gt_concepts) for concept in all_concepts]
                y_pred = [int(concept in candidate_concepts) for concept in all_concepts]

                f1score = f1_score(y_true, y_pred, average='binary')

                # Increase calculated score
                current_score += f1score

        return current_score/max_score


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
    submission_file_path = "runs/00_perfect_run.csv"
    #submission_file_path = "runs/01_more_than_2_tokens.csv"
    #submission_file_path = "runs/02_zero_tokens.csv"
    #submission_file_path = "runs/03_wrong_image_id.csv"
    #submission_file_path = "runs/04_more_than_max_num_concepts.csv"
    #submission_file_path = "runs/05_same_concept_more_than_once_for_image.csv"
    #submission_file_path = "runs/06_same_image_more_than_once.csv"
    #submission_file_path = "runs/07_not_all_images_contained.csv"

    _client_payload = {}
    _client_payload["submission_file_path"] = submission_file_path
    #Create instance of Evaluator
    evaluator = ConceptDetectionEvaluator(gt_file_path)
    #Call _evaluate method
    result = evaluator._evaluate(_client_payload)
    print(result)
