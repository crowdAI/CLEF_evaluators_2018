import csv
"""
Evaluator class
Evaluates one single runfile
_evaluate method is called by the CrowdAI framework and returns an object holding up to 2 different scores
"""
class GeoEvaluator:

	"""
	Constructor
	Parameter 'answer_file_path': Path of file containing ground truth
	"""
	def __init__(self, answer_file_path, allowed_classes_file_path, debug_mode=False):
		#Ground truth file
		self.answer_file_path = answer_file_path
		#Ground truth data
		self.gt = self.load_gt()
		#allowed ids files in the predictions files
		self.allowed_classes_file_path = allowed_classes_file_path
	"""
	This is the only method that will be called by the framework
	Parameter 'submission_file_path': Path of the submitted runfile
	returns a _result_object that can contain up to 2 different scores
	"""
	def _evaluate(self, client_payload, context={}):
		submission_file_path = client_payload['submission_file_path']
		#Load predictions
		predictions = self.load_predictions(submission_file_path)
		# Metric :MRR
		mrr = self.compute_top_1_experts(predictions)
		#Create object that is returned to the CrowdAI framework
		#_result_object = { "MRR": mrr }

		_result_object = {
            "score": mrr,
            "score_secondary" : 0
        }
		return _result_object

	"""
	Load and return groundtruth data
	"""
	def load_gt(self):
		gt = {}
		with open(self.answer_file_path) as f:
			for line in f.readlines():
				linef = line.rstrip("\n")
				query = linef.split(';')[0]
				classid = linef.split(';')[1]
				#source = linef.split(';')[2]
				#gt[query] = [classid,source]
				gt[query] = classid
		return gt
	"""
	Load and return allowed class ids in the predictions files
	"""
	def load_allowed_classes(self):
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
	#Format : query_id;clas_id;score;rank
	def load_predictions(self, submission_file_path):
		#returns predictions
		#is_valid = self.check_predictions(submission_file_path)
		allowed_queries = self.gt.keys()
		allowed_classes = self.load_allowed_classes()
		max_rank = 100 #max nbr of classes for observation
		query_to_correct_classid_rank = {}
		#absent_queries=[]
		with open(submission_file_path) as csvfile:
			reader = csv.reader(csvfile, delimiter=';', quoting=csv.QUOTE_NONE)
			lineCnt = 0
			occured_observations = {}

			for row in reader:
				lineCnt += 1
				if len(row) != 4:
					raise Exception("Wrong format: Each line must consist of a query ID, class ID, score and a rank separated by semicolons ({}) {}"
						.format("<query_id>;<class_id><score>;<rank>", self.line_nbr_string(lineCnt)))
				query_id = row[0]
				# Query ID not in testset => Error
				if query_id not in allowed_queries:
					raise Exception("Query ID '{}' in submission file does not exist in testset {}"
						.format(query_id, self.line_nbr_string(lineCnt)))
				class_id = row[1]
				# Query ID not in testset => Error
				if class_id not in allowed_classes:
					raise Exception("'{}' is not a valid class ID {}"
						.format(class_id, self.line_nbr_string(lineCnt)))
				# 3rd value in line is not a number  => Error
				try:
					probability = float(row[2])
				except ValueError:
					raise Exception("Score must be a number {}"
						.format(self.line_nbr_string(lineCnt)))
				# Rank not an int between 1 and 100 => Error
				try:
					rank = int(row[3])
					if rank < 1 or rank > max_rank:
						raise ValueError
				except ValueError:
					raise Exception("Rank 'must be an integer between 1 and 100 {}"
						.format(self.line_nbr_string(lineCnt)))
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

				if class_id == self.gt[query_id]:
					query_to_correct_classid_rank[query_id] = rank

			for q_id in occured_observations:
				if q_id not in query_to_correct_classid_rank.keys():
					query_to_correct_classid_rank[q_id] = 0

				observation_values_sorted = sorted(occured_observations[q_id], key=lambda tup: (tup[2],tup[1]) )
				last_rank = 0
				for values in observation_values_sorted:
					curr_class_id, curr_score, curr_rank, curr_line = values
					#Ranking for query_id not consecutive => Error
					if curr_rank != (last_rank+1):
						raise Exception("Ranking must be consecutive {}"
							.format(self.line_nbr_string(curr_line)))
					last_rank, last_score = curr_rank, curr_score

		return query_to_correct_classid_rank

	"""
	Compute and return the primary score
	Parameter 'predictions' : predictions object generated by the load_predictions method
	NO VALIDATION OF THE RUNFILE SHOULD BE IMPLEMENTED HERE
	We assume that the predictions in the parameter are valid
	Valiation should be handled in the load_predictions method
	"""
	def compute_top_1_experts(self, predictions):
		sum = 0
		for query in predictions:
			if predictions[query]!=0:
				sum += 1. / (1.* predictions[query])
			#print('good class not in the predictions list')
			#print('query :'+str(query)+' rank of good class:'+str(predictions[query]))
		mrr = sum / float(len(self.gt))
		return mrr

	"""
	Compute and return the secondary score
	Parameter 'predictions' : predictions object generated by the load_predictions method
	NO VALIDATION OF THE RUNFILE SHOULD BE IMPLEMENTED HERE
	We assume that the predictions in the parameter are valid
	Valiation should be handled in the load_predictions method
	"""
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
	#allowed classids
	allowed_classes_file_path = 'allowed_classes.txt'
	#Submission file
	submission_file_path = "runs/00_perfect_run.csv"
	#submission_file_path = "runs/01_ideal_run.csv"
	#submission_file_path = "runs/02_random_run.csv"
	#submission_file_path = "runs/03_not_4_tokens.csv"
	#submission_file_path = "runs/04_wrong_query_id.csv"
	#submission_file_path = "runs/05_wrong_class_id.csv"
	#submission_file_path = "runs/06_score_no_number.csv"
	#submission_file_path = "runs/07_rank_no_number.csv"
	#submission_file_path = "runs/08_rank_greater_max_rank.csv"
	#submission_file_path = "runs/09_same_prediction_more_than_once.csv"
	#submission_file_path = "runs/10_ranking_broken.csv"

	print(submission_file_path)

	_client_payload = {}
	_client_payload["submission_file_path"] = submission_file_path

	#Create instance of Evaluator
	evaluator = GeoEvaluator(gt_file_path, allowed_classes_file_path)
	#Call _evaluate method
	result = evaluator._evaluate(_client_payload)
	print(result)
