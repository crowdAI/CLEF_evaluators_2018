import csv

"""
Evaluator class
Evaluates one single runfile
_evaluate method is called by the CrowdAI framework and returns an object holding up to 2 different scores
"""


class LifelogLmrtEvaluator:
    """
    Constructor
    Parameter 'answer_file_path': Path of file containing ground truth
    """

    def __init__(self, answer_file_path, clusters_gt_file_path, allowed_image_ids_file_path,
                        debug_mode=False):
        # Ground truth file
        self.answer_file_path = answer_file_path
        # Clusters ground truth file
        self.clusters_gt_file_path = clusters_gt_file_path
        # Allowed image ids file
        self.allowed_image_ids_file_path = allowed_image_ids_file_path
        # Ground truth data
        self.gt, self.gt_topic, self.gt_image_id, self.gt_cluster_id = self.load_gt()
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
        f1_10 = self.compute_f1_at_10(predictions)
        # Compute second score
        #secondary_score = self.compute_secondary_score(predictions)

        # Create object that is returned to the CrowdAI framework
        #_result_object = {
        #    "f1_@_10": f1_10,
        #    "score_secondary": 0
        #}

        _result_object = {
            "score": f1_10,
            "score_secondary": 0
        }

        return _result_object

    """
    Load and return groundtruth data
    """

    # @Duc-Tien, just load your gt as you wish, i only included the loading of the gt file here
    # The loading of the clusters_gt_file must be implemented somehow as well
    def load_gt(self):
        gt = {}
        gt_topic = []
        gt_image_id = []
        gt_cluster_id = []
        with open(self.answer_file_path) as csvfile:
            reader = csv.reader(csvfile, delimiter=',', quoting=csv.QUOTE_NONE)
            for row in reader:
                gt[row[0].strip()] = (row[1].strip(), row[2].strip())
                gt_topic += [int(row[0].strip())]
                gt_image_id += [row[1].strip()]
                gt_cluster_id += [int(row[2].strip())]

        return gt, gt_topic, gt_image_id, gt_cluster_id

    """
    Load and return allowed image ids
    """

    def load_allowed_image_ids(self):
        image_ids = set()
        with open(self.allowed_image_ids_file_path) as f:
            for image_id in f.readlines():
                image_id = image_ids.add(image_id.rstrip("\n"))

        return image_ids

    """
    Loads and returns a predictions object (dictionary) that contains the submitted data that will be used in the _evaluate method
    Parameter 'submission_file_path': Path of the submitted runfile
    Validation of the runfile format will also be handled here
    THE VALIDATION PART CAN BE IMPLEMENTED BY IVAN IF YOU WISH (ivan.eggel@hevs.ch)
    """

    # Format : topic_id, image_id, score
    def load_predictions(self, submission_file_path):
        predictions = {}
        topic_ids_gt = list(self.gt.keys())
        allowed_image_ids = self.load_allowed_image_ids()
        images_in_topic = {}
        with open(submission_file_path) as csvfile:
            reader = csv.reader(csvfile, delimiter=',', quoting=csv.QUOTE_NONE)
            lineCnt = 0

            for row in reader:
                lineCnt += 1

                # Not 3 comma separated tokens on line => Error
                if len(row) != 3:
                    raise Exception(
                        "Wrong format: Each line must consist of a topic ID followed by a comma, an image ID, a comma, and a score ({}). {}"
                            .format("<topic_id>,<image_id>,<confidence_score>", self.line_nbr_string(lineCnt)))

                topic_id = row[0].strip()
                # Topic ID does not exist in testset => Error
                if topic_id not in topic_ids_gt:
                    raise Exception("Topic ID '{}' in submission file does not exist in testset {}"
                                    .format(topic_id, self.line_nbr_string(lineCnt)))

                image_id = row[1].strip()
                # Image ID does not exist in testset => Error
                if image_id not in allowed_image_ids:
                    raise Exception("'{}' is not a valid image ID {}"
                                    .format(image_id, self.line_nbr_string(lineCnt)))

                # Image ID occured more than once for a given topic => Error
                occured_images_for_topic = images_in_topic.get(topic_id, list())
                if image_id in occured_images_for_topic:
                    raise Exception("Image ID '{}' specified more than once for topic ID {}. {}"
                                    .format(image_id, topic_id, self.line_nbr_string(lineCnt)))

                occured_images_for_topic.append(image_id)
                images_in_topic[topic_id] = occured_images_for_topic

                # Score not a number or not between 0 and 1 => Error
                try:
                    score = float(row[2])
                    if score < 0 or score > 1:
                        raise ValueError
                except ValueError:
                    raise Exception("Score must be a number between 0 and 1 {}"
                                    .format(self.line_nbr_string(lineCnt)))

                values_for_topic = predictions.get(topic_id, list())
                values_for_topic.append((image_id, score))
                predictions[topic_id] = values_for_topic

            # nbr topics in gt != nbr topics in submission file => Error
            if len(topic_ids_gt) != len(images_in_topic):
                raise Exception("Not all topics from testset included in submission file {}"
                                .format(self.line_nbr_string(lineCnt)))

        return predictions

    """
    Compute and return the primary score
    Parameter 'predictions' : predictions object generated by the load_predictions method
    NO VALIDATION OF THE RUNFILE SHOULD BE IMPLEMENTED HERE
    We assume that the predictions in the parameter are valid
    Valiation should be handled in the load_predictions method
    """

    def compute_f1_at_10(self, predictions):
        # ...
        # return primary_score

        cut_off = 10
        number_of_topics = 1
        fscore_final = 0

        for topic in range(1, number_of_topics+1):
            image_cluster = {}
            for t in range(len(self.gt_topic)):
                if self.gt_topic[t] == topic:
                    image_cluster[self.gt_image_id[t]] = self.gt_cluster_id[t]
            no_clusters = image_cluster[max(image_cluster)]
            cluster_found = [0] * no_clusters
            precision = 0
            for i in range(cut_off):
                image_id = predictions[str(topic)][i][0]
                if image_id in image_cluster:
                    precision += 1
                    cluster_found[image_cluster[image_id]-1] = 1
            precision = precision / cut_off
            recall = sum(cluster_found) / no_clusters
            if precision + recall > 0:
                fscore = 2 * precision * recall / (precision + recall)
            else:
                fscore = 0.0

            fscore_final += fscore

        return fscore_final/number_of_topics

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
    # Clusters ground truth file
    clusters_gt_file_path = "clusters_gt_file.csv"
    # Allowed image ids file
    allowed_image_ids_file_path = "allowed_image_ids.txt"

    # Submission file
    #submission_file_path = "runs/09_gt_file.csv" # => Did not manage to get a perfect run from the roganisers
    submission_file_path = "runs/01_run_ok.csv" #pass but score 0
    #submission_file_path = "runs/02_not_3_tokens.csv"
    #submission_file_path = "runs/03_wrong_topic_id.csv"
    #submission_file_path = "runs/04_wrong_image_id.csv"
    #submission_file_path = "runs/05_same_image_id_more_than_once.csv"
    #submission_file_path = "runs/06_score_no_number.csv"
    #submission_file_path = "runs/07_score_not_btwn_0_and_1.csv"
    #submission_file_path = "runs/08_not_all_topics_included.csv"

    _client_payload = {}
    _client_payload["submission_file_path"] = submission_file_path

    # Create instance of Evaluator
    evaluator = LifelogLmrtEvaluator(gt_file_path, clusters_gt_file_path, allowed_image_ids_file_path)
    # Call _evaluate method
    result = evaluator._evaluate(_client_payload)
    print(result)
