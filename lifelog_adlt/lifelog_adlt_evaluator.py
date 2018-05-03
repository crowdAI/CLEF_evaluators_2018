import csv

"""
Evaluator class
Evaluates one single runfile
_evaluate method is called by the CrowdAI framework and returns an object holding up to 2 different scores
"""


class LifelogAdltEvaluator:
    """
    Constructor
    Parameter 'answer_file_path': Path of file containing ground truth
    """

    def __init__(self, answer_file_path, allowed_image_ids_file_path,debug_mode=False):
        # Ground truth file
        self.answer_file_path = answer_file_path
        # Ground truth data
        self.gt = self.load_gt()

        self.allowed_image_ids_file_path = allowed_image_ids_file_path
        # ...

    """
    This is the only method that will be called by the framework
    Parameter 'submission_file_path': Path of the submitted runfile
    returns a _result_object that can contain up to 2 different scores
    """
    def _evaluate(self, client_payload, context={}):
        submission_file_path = client_payload['submission_file_path']
        # Load predictions
        predictions = self.load_predictions(submission_file_path)
        # Compute first score
        percentage_dissimilarity = self.compute_percentage_dissimilarity(predictions)
        # Compute second score
        #secondary_score = self.compute_secondary_score(predictions)

        # Create object that is returned to the CrowdAI framework
        #_result_object = {
        #    "percentage_dissimilarity": percentage_dissimilarity,
        #    "score_secondary": 0
        #}

        _result_object = {
            "score": percentage_dissimilarity,
            "score_secondary": 0
        }

        return _result_object

    """
    Load and return groundtruth data
    """

    def load_gt(self):
        gt = {}
        with open(self.answer_file_path) as csvfile:
            reader = csv.reader(csvfile, delimiter=',', quoting=csv.QUOTE_NONE)
            for row in reader:
                gt[row[0]] = (row[1], row[2])
        return gt

    """
    Load and return allowed image ids
    """

    def load_allowed_image_ids(self):
        image_ids = set()
        with open(self.allowed_image_ids_file_path) as f:
            for image_id in f.readlines():
                image_ids.add(image_id.rstrip("\n"))
        return image_ids

    """
    Loads and returns a predictions object (dictionary) that contains the submitted data that will be used in the _evaluate method
    Parameter 'submission_file_path': Path of the submitted runfile
    Validation of the runfile format will also be handled here
    THE VALIDATION PART CAN BE IMPLEMENTED BY IVAN IF YOU WISH (ivan.eggel@hevs.ch)
    """

    # Format: topic_id, nbr_times, nbr_minutes
    def load_predictions(self, submission_file_path):
        predictions = {}
        topic_ids_gt = list(self.gt.keys())
        allowed_image_ids = self.load_allowed_image_ids()

        in_mandatory_section = True
        with open(submission_file_path) as csvfile:
            reader = csv.reader(csvfile, delimiter=',', quoting=csv.QUOTE_NONE)
            lineCnt = 0

            for row in reader:
                lineCnt += 1

                if (len(row) != 3) and (len(row) == 1 and not self.is_mandatory_section_separator(row)):
                    raise Exception(
                        "Wrong format: Line must consist of 3 comma-separated tokens or '*****' (separator after mandatory lines). {}"
                        .format(self.line_nbr_string(lineCnt)))
                elif len(row) == 1 and self.is_mandatory_section_separator(row):
                    break;  # reached optional section, ignore validation for all following lines

                # Not 3 comma separated tokens on line => Error
                if len(row) != 3:
                    raise Exception(
                        "Wrong format: Each line in the mandatory section must consist of a topic ID followed by a comma, nbr of times, a comma, and the number of minutes ({}). {}"
                        .format("<topic_id>,<nbr_of_times>,<nbr_of_minutes>", self.line_nbr_string(lineCnt)))

                topic_id = row[0]
                # Topic ID does not exist in testset => Error
                if topic_id not in topic_ids_gt:
                    raise Exception("Topic ID '{}' in submission file does not exist in testset {}"
                                    .format(topic_id, self.line_nbr_string(lineCnt)))

                # Topic ID already specified in runfile => Error
                if topic_id in predictions:
                    raise Exception("Topic ID '{}' specified more than once in submission file {}"
                                    .format(topic_id, self.line_nbr_string(lineCnt)))

                # nbr of times not a number or not >= 1 => Error
                try:
                    nbr_times = int(row[1])
                    if nbr_times < 1:
                        raise ValueError
                except ValueError:
                    raise Exception("'Number of times' (2nd column) must be an integer > 0 {}"
                                    .format(self.line_nbr_string(lineCnt)))

                # Nbr of minutes not a number or >= 1 => Error
                try:
                    nbr_minutes = int(row[2])
                    if nbr_minutes < 1:
                        raise ValueError
                except ValueError:
                    raise Exception("'Number of minutes' (3rd column) must be an integer > 0 {}"
                                    .format(self.line_nbr_string(lineCnt)))

                predictions[topic_id] = (nbr_times, nbr_minutes)

            # nbr topics in gt != nbr topics in submission file => Error
            if len(topic_ids_gt) != len(predictions):
                raise Exception("Not all topics from testset included in submission file")

        return predictions

    """
    Checks if a line (row) is contained in the mandatory section or not
    """

    def is_mandatory_section_separator(self, row):
        return True if row[0] == "*****" else False

    """
    Compute and return the primary score
    Parameter 'predictions' : predictions object generated by the load_predictions method
    NO VALIDATION OF THE RUNFILE SHOULD BE IMPLEMENTED HERE
    We assume that the predictions in the parameter are valid
    Valiation should be handled in the load_predictions method
    """

    def compute_percentage_dissimilarity(self, predictions):
        # ...
        # return primary_score
        no_topics = 10
        final_score = 0
        for i in range(1, no_topics + 1):
            times_gt = int(self.gt[str(i)][0])
            minutes_gt = int(self.gt[str(i)][0])

            time_input = int(predictions[str(i)][0])
            minutes_input = int(predictions[str(i)][0])

            score_time = max(0, 1 - abs(times_gt - time_input)/times_gt)
            score_minute = max(0, 1 - abs(minutes_gt - minutes_input)/minutes_gt)

            final_score += 0.5 * score_time + 0.5*score_minute

        return final_score/no_topics

    """
    Compute and return the secondary score
    Parameter 'predictions' : predictions object generated by the load_predictions method
    NO VALIDATION OF THE RUNFILE SHOULD BE IMPLEMENTED HERE
    We assume that the predictions in the parameter are valid
    Valiation should be handled in the load_predictions method
    """

    def compute_secondary_score(self, predictions):
        # ...
        # return secondary_score
        pass

    def line_nbr_string(self, line_nbr):
        return "(Line nbr {})".format(line_nbr)


"""
Test evaluation a runfile
provide path to ground truth file in constructor
call _evaluate method with path of submitted file as argument
"""
if __name__ == "__main__":
    # Ground truth file
    gt_file_path = "gt_file.csv"
    # Allowed image ids file
    allowed_image_ids_file_path = "allowed_image_ids.txt"
    # Submission file
    submission_file_path = "runs/00_perfect_run_no_asterisks.csv"  # pass
    #submission_file_path = "runs/01_perfect_run_with_asterisks.csv"  # pass
    #submission_file_path = "runs/02_perfect_run_with_asterisks_and_additional_lines.csv"  # pass
    #submission_file_path = "runs/03_wrong_mand_section_separator.csv"
    #submission_file_path = "runs/04_not_3_tokens.csv"
    #submission_file_path = "runs/05_wrong_topic_id.csv"
    #submission_file_path = "runs/06_same_topic_id_more_than_once.csv"
    #submission_file_path = "runs/07_nbr_times_not_an_integer.csv"
    #submission_file_path = "runs/08_nbr_times_smaller_than_1.csv"
    #submission_file_path = "runs/09_nbr_minutes_not_an_integer.csv"
    #submission_file_path = "runs/10_nbr_minutes_smaller_than_1.csv"
    #submission_file_path = "runs/11_not_all_topics_included.csv"
    #submission_file_path = "runs/12_mandatory_section_separator_too_early.csv"  # => should error: not all topics included

    _client_payload = {}
    _client_payload["submission_file_path"] = submission_file_path
    # Create instance of Evaluator
    evaluator = LifelogAdltEvaluator(gt_file_path, allowed_image_ids_file_path)
    # Call _evaluate method
    result = evaluator._evaluate(_client_payload)
    print(result)
