import numpy as np

class AHP:
    def __init__(self, pairwise_matrix):
        self.pairwise_matrix = pairwise_matrix
        self.priority_vector = self.calculate_priority_vector(pairwise_matrix)
        self.consistency_ratio = self.check_consistency(pairwise_matrix, self.priority_vector)

    def normalize(self, matrix):
        column_sums = np.sum(matrix, axis=0)
        normalized_matrix = matrix / column_sums[np.newaxis, :]
        return normalized_matrix

    def calculate_priority_vector(self, pairwise_matrix):
        normalized_matrix = self.normalize(pairwise_matrix)
        priority_vector = np.mean(normalized_matrix, axis=1)
        return priority_vector

    def check_consistency(self, pairwise_matrix, priority_vector):
        weighted_sum_vector = np.dot(pairwise_matrix, priority_vector)
        lambda_max = np.mean(weighted_sum_vector / priority_vector)
        ci = (lambda_max - len(priority_vector)) / (len(priority_vector) - 1)
        ri = {1: 0.00, 2: 0.00, 3: 0.58, 4: 0.90, 5: 1.12, 6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45, 10: 1.49}
        cr = ci / ri[len(priority_vector)]
        return cr

    def score_alternatives(self, alternatives):
        if not isinstance(alternatives, np.ndarray):
            alternatives = np.array(alternatives)
        
        normalized_alternatives = self.normalize(alternatives)
        scores = np.dot(normalized_alternatives, self.priority_vector)
        return scores
