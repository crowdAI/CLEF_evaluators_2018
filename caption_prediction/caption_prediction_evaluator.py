import csv
import string
import nltk
import warnings
from nltk.translate.bleu_score import SmoothingFunction
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
"""
Evaluator class
Evaluates one single runfile
_evaluate method is called by the CrowdAI framework and returns an object holding up to 2 different scores
"""
class CaptionPredictionEvaluator:

    remove_stopwords = True
    stemming = True
    case_sensitive = False


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
        bleu_score = self.compute_bleu(candidate_pairs)

        #_result_object = {
        #  "score": bleu_score,
        #  "score_secondary" : 0
        #}

        _result_object = {
          "score": bleu_score,
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
        with open(submission_file_path) as csvfile:
            reader = csv.reader(csvfile, delimiter='\t', quoting=csv.QUOTE_NONE)
            lineCnt = 0
            occured_images = []
            for row in reader:
                lineCnt += 1

                # less than two tab separated tokens on line => Error
                if(len(row) < 2):
                    raise Exception("Wrong format: Each line must consist of an image ID followed by a tab and a caption ({}) {}"
                        .format("<imageID><TAB><caption>", self.line_nbr_string(lineCnt)))

                image_id = row[0]

                # Image ID does not exist in testset => Error
                if image_id not in image_ids_gt:
                    raise Exception("Image ID '{}' in submission file does not exist in testset {}"
                        .format(image_id,self.line_nbr_string(lineCnt)))

                # image id occured at least twice in file => Error
                if image_id in occured_images:
                    raise Exception("Image ID '{}' was specified more than once in submission file {}"
                        .format(image_id, self.line_nbr_string(lineCnt)))

                occured_images.append(image_id)

                pairs[row[0]] = row[1]

            # In case not all images from the testset are contained in the file => Error
            if(len(occured_images) != len (image_ids_gt)):
                raise Exception("Number of image IDs in submission file not equal to number of image IDs in testset")

        return pairs

    def compute_bleu(self, candidate_pairs):
        # Hide warnings
        warnings.filterwarnings('ignore')

        # NLTK
        # Download Punkt tokenizer (for word_tokenize method)
        # Download stopwords (for stopword removal)
        nltk.download('punkt')
        nltk.download('stopwords')

        # English Stopwords
        stops = set(stopwords.words("english"))

        # Stemming
        stemmer = SnowballStemmer("english")

        # Remove punctuation from string
        translator = str.maketrans('', '', string.punctuation)

        # Define max score and current score
        max_score = len(self.gt_pairs)
        current_score = 0

        i = 0
        for image_key in candidate_pairs:

            # Get candidate and GT caption
            candidate_caption = candidate_pairs[image_key]
            gt_caption = self.gt_pairs[image_key]

            # Optional - Go to lowercase
            if not CaptionPredictionEvaluator.case_sensitive:
                candidate_caption = candidate_caption.lower()
                gt_caption = gt_caption.lower()

            # Split caption into individual words (remove punctuation)
            candidate_words = nltk.tokenize.word_tokenize(candidate_caption.translate(translator))
            gt_words = nltk.tokenize.word_tokenize(gt_caption.translate(translator))


            # Optional - Remove stopwords
            if CaptionPredictionEvaluator.remove_stopwords:
                candidate_words = [word for word in candidate_words if word.lower() not in stops]
                gt_words = [word for word in gt_words if word.lower() not in stops]

            # Optional - Apply stemming
            if CaptionPredictionEvaluator.stemming:
                candidate_words = [stemmer.stem(word) for word in candidate_words]
                gt_words = [stemmer.stem(word) for word in gt_words]

            # Calculate BLEU score for the current caption
            try:
                # If both the GT and candidate are empty, assign a score of 1 for this caption
                if len(gt_words) == 0 and len(candidate_words) == 0:
                    bleu_score = 1
                # Calculate the BLEU score
                else:
                    bleu_score = nltk.translate.bleu_score.sentence_bleu([gt_words], candidate_words, smoothing_function=SmoothingFunction().method0)
            # Handle problematic cases where BLEU score calculation is impossible
            except ZeroDivisionError:
                pass
                #raise Exception('Problem with {} {}', gt_words, candidate_words)

            # Increase calculated score
            current_score += bleu_score

        return current_score / max_score


    """
    Load and return groundtruth data
    """
    def load_gt(self):
        pairs = {}
        with open(self.answer_file_path) as csvfile:
            reader = csv.reader(csvfile, delimiter='\t', quoting=csv.QUOTE_NONE)
            for row in reader:
                pairs[row[0]] = row[1]
        return pairs

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
    #submission_file_path = "runs/01_less_than_2_tokens.csv"
    #submission_file_path = "runs/02_wrong_image_id.csv"
    #submission_file_path = "runs/03_same_image_more_than_once.csv"
    #submission_file_path = "runs/04_not_all_images_contained.csv"
    #submission_file_path = "runs/05_imperfect_run.csv" #should return score < 1

    _client_payload = {}
    _client_payload["submission_file_path"] = submission_file_path
    #Create instance of Evaluator
    evaluator = CaptionPredictionEvaluator(gt_file_path)
    #Call _evaluate method
    result = evaluator._evaluate(_client_payload)
    print(result)
