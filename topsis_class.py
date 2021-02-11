from pprint import pprint

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math


def init_candidates_df(data):
    df_candidates = pd.read_excel(data).iloc[:-1]
    df_persons = df_candidates.astype('int').set_index('No')
    return df_persons


class TopsisGroup:
    # Arxikopoihsh twn dedomenwn
    def __init__(self, candidates_data, interview_data, weights_data):
        self.df_candidates = self._init_candidates_df(candidates_data)
        self.df_interview = self._init_interview_df(interview_data)
        self.df_weights = self._init_weights_df(weights_data)
        self.total_DMs = self.df_weights.columns.size
        self.total_candidates = self.df_candidates.index.size
        self.total_criteria = 0
        self.decision_matrix = {}
        self.normalized_matrix = {}
        self.positive_ideal_sol = {}
        self.negative_ideal_sol = {}
        self.euclidean_measure_pis = {}
        self.manhattan_measure_pis = {}
        self.euclidean_measure_nis = {}
        self.manhattan_measure_nis = {}
        self.eucl_pis_arithmetic = np.empty(self.total_candidates, dtype=float)
        self.eucl_pis_geometric = np.empty(self.total_candidates, dtype=float)
        self.manhattan_pis_arithmetic = np.empty(self.total_candidates, dtype=float)
        self.manhattan_pis_geometric = np.empty(self.total_candidates, dtype=float)
        self.eucl_nis_arithmetic = np.empty(self.total_candidates, dtype=float)
        self.eucl_nis_geometric = np.empty(self.total_candidates, dtype=float)
        self.manhattan_nis_arithmetic = np.empty(self.total_candidates, dtype=float)
        self.manhattan_nis_geometric = np.empty(self.total_candidates, dtype=float)
        self.rel_close_eucl_arithmetic = np.empty(self.total_candidates, dtype=float)
        self.rel_close_eucl_geometric = np.empty(self.total_candidates, dtype=float)
        self.rel_close_manh_arithmetic = np.empty(self.total_candidates, dtype=float)
        self.rel_close_manh_geometric = np.empty(self.total_candidates, dtype=float)

    # Arxikopoihsh twn dataframes
    @staticmethod
    def _init_candidates_df(data):
        df_candidates = pd.read_excel(data).iloc[:]
        df_persons = df_candidates.astype('int').set_index('No')
        return df_persons

    @staticmethod
    def _init_interview_df(data):
        df_interview = pd.read_excel("Interview_Results.xlsx", header=1).set_index('No')
        return df_interview

    @staticmethod
    def _init_weights_df(data):
        df_weights_per_DM = pd.read_excel("Weights_Matrix.xlsx").iloc[:7].drop(['No'], axis=1).set_index('Attributes')
        return df_weights_per_DM

    # Calculations
    # Step 1:Decision Matrix
    def create_decision_matrix(self):
        for x in range(self.total_DMs):
            self.decision_matrix["DM" + str(x + 1)] = self.df_candidates.copy()

        for x in range(self.total_DMs):
            if x == 0:
                self.decision_matrix["DM" + str(x + 1)][['Panel Interview', '1-on-1 Interview']] = self.df_interview[
                    ['Panel Interview', '1-on-1 Interview']]
            else:
                self.decision_matrix["DM" + str(x + 1)][['Panel Interview', '1-on-1 Interview']] = self.df_interview[
                    ['Panel Interview.' + str(x), '1-on-1 Interview.' + str(x)]]
        self.total_criteria = self.decision_matrix["DM1"].columns.size

    # Step 2:Normalized Matrix
    def create_normalization_matrix(self):
        self.normalized_matrix = self.decision_matrix.copy()
        for x in range(self.total_DMs):
            for idx, col in enumerate(self.decision_matrix["DM" + str(x + 1)].columns):
                sum_of_squares = math.sqrt(float(self.decision_matrix["DM" + str(x + 1)].pow(2).sum().get(idx)))
                for index, row in enumerate(self.decision_matrix["DM" + str(x + 1)].index):
                    curr_value = self.decision_matrix["DM" + str(x + 1)].loc[row, col]
                    normalized_value = curr_value / sum_of_squares
                    self.normalized_matrix["DM" + str(x + 1)].iloc[
                        index, self.normalized_matrix["DM" + str(x + 1)].columns.get_loc(col)] = normalized_value

    # Step 3: Find PIS and NIS distances
    def determine_ideal_solutions(self):
        for x in range(self.total_DMs):
            self.positive_ideal_sol["DM" + str(x + 1)] = np.empty(
                self.normalized_matrix["DM" + str(x + 1)].columns.size,
                dtype=float)
            self.negative_ideal_sol["DM" + str(x + 1)] = np.empty(
                self.normalized_matrix["DM" + str(x + 1)].columns.size,
                dtype=float)
            for idx, col in enumerate(self.normalized_matrix["DM" + str(x + 1)].columns):
                max_value = max(self.normalized_matrix["DM" + str(x + 1)][col])
                min_value = min(self.normalized_matrix["DM" + str(x + 1)][col])
                self.positive_ideal_sol["DM" + str(x + 1)][idx] = max_value
                self.negative_ideal_sol["DM" + str(x + 1)][idx] = min_value

    # Step 4 and 5a: Calculate distances
    def calculate_distances(self):
        for x in range(self.total_DMs):
            self.euclidean_measure_pis["DM" + str(x + 1)] = np.empty(self.total_candidates, dtype=float)
            self.manhattan_measure_pis["DM" + str(x + 1)] = np.empty(self.total_candidates, dtype=float)
            self.euclidean_measure_nis["DM" + str(x + 1)] = np.empty(self.total_candidates, dtype=float)
            self.manhattan_measure_nis["DM" + str(x + 1)] = np.empty(self.total_candidates, dtype=float)
            for index, row in enumerate(self.normalized_matrix["DM" + str(x + 1)].index):
                temp_euclidean_pis = 0.0
                temp_euclidean_nis = 0.0
                temp_manhattan_pis = 0.0
                temp_manhattan_nis = 0.0
                temp_value = 0.0
                for idx, col in enumerate(self.normalized_matrix["DM" + str(x + 1)].columns):
                    curr_weight = self.df_weights.loc[col, "DM" + str(x + 1)]
                    curr_pis = self.positive_ideal_sol["DM" + str(x + 1)][idx]
                    curr_nis = self.negative_ideal_sol["DM" + str(x + 1)][idx]
                    curr_value = self.normalized_matrix["DM" + str(x + 1)].loc[row, col]
                    # Euclidean
                    temp_value = curr_weight * ((curr_value - curr_pis) ** 2)
                    temp_euclidean_pis = temp_euclidean_pis + temp_value
                    temp_value = curr_weight * ((curr_value - curr_nis) ** 2)
                    temp_euclidean_nis = temp_euclidean_nis + temp_value
                    # Manhattan
                    temp_value = curr_weight * (curr_pis - curr_value)
                    temp_manhattan_pis = temp_manhattan_pis + temp_value
                    temp_value = curr_weight * (curr_value - curr_nis)
                    temp_manhattan_nis = temp_manhattan_nis + temp_value
                self.euclidean_measure_pis["DM" + str(x + 1)][index] = math.sqrt(temp_euclidean_pis)
                self.manhattan_measure_pis["DM" + str(x + 1)][index] = temp_manhattan_pis
                self.euclidean_measure_nis["DM" + str(x + 1)][index] = math.sqrt(temp_euclidean_nis)
                self.manhattan_measure_nis["DM" + str(x + 1)][index] = temp_manhattan_nis

    # Step 5b: Calculate mean of distances
    def calculate_means(self):
        for index, row in enumerate(self.normalized_matrix["DM1"].index):
            temp_pow = 1 / self.total_DMs
            # Arithmetic mean
            # PIS
            temp_value = 0
            for x in range(self.total_DMs):
                temp_value = temp_value + self.euclidean_measure_pis["DM" + str(x + 1)][index]
            self.eucl_pis_arithmetic[index] = temp_value / self.total_DMs
            temp_value = 0

            for x in range(self.total_DMs):
                temp_value = temp_value + self.manhattan_measure_pis["DM" + str(x + 1)][index]
            self.manhattan_pis_arithmetic[index] = temp_value / self.total_DMs
            # NIS
            temp_value = 0
            for x in range(self.total_DMs):
                temp_value = temp_value + self.euclidean_measure_nis["DM" + str(x + 1)][index]
            self.eucl_nis_arithmetic[index] = temp_value / self.total_DMs
            temp_value = 0

            for x in range(self.total_DMs):
                temp_value = temp_value + self.manhattan_measure_nis["DM" + str(x + 1)][index]
            self.manhattan_nis_arithmetic[index] = temp_value / self.total_DMs

            # Geometric mean
            # PIS
            temp_value = 1
            for x in range(self.total_DMs):
                temp_value = temp_value * self.euclidean_measure_pis["DM" + str(x + 1)][index]
            self.eucl_pis_geometric[index] = temp_value ** temp_pow
            temp_value = 1
            for x in range(self.total_DMs):
                temp_value = temp_value * self.manhattan_measure_pis["DM" + str(x + 1)][index]
            self.manhattan_pis_geometric[index] = temp_value ** temp_pow
            # NIS
            temp_value = 1
            for x in range(self.total_DMs):
                temp_value = temp_value * self.euclidean_measure_nis["DM" + str(x + 1)][index]
            self.eucl_nis_geometric[index] = temp_value ** temp_pow
            temp_value = 1
            for x in range(self.total_DMs):
                temp_value = temp_value * self.manhattan_measure_nis["DM" + str(x + 1)][index]
            self.manhattan_nis_geometric[index] = temp_value ** temp_pow

    # Step 6: Calculate relative closeness coefficients
    def calculate_relative_coefficients(self):
        for index, row in enumerate(self.normalized_matrix["DM1"].index):
            self.rel_close_eucl_arithmetic[index] = self.eucl_nis_arithmetic[index] / (
                    self.eucl_nis_arithmetic[index] + self.eucl_pis_arithmetic[index])
            self.rel_close_eucl_geometric[index] = self.eucl_nis_geometric[index] / (
                    self.eucl_nis_geometric[index] + self.eucl_pis_geometric[index])
            self.rel_close_manh_arithmetic[index] = self.manhattan_nis_arithmetic[index] / (
                    self.manhattan_nis_arithmetic[index] + self.manhattan_pis_arithmetic[index])
            self.rel_close_manh_geometric[index] = self.manhattan_nis_geometric[index] / (
                    self.manhattan_nis_geometric[index] + self.manhattan_pis_geometric[index])

    # Figures
    def create_figures(self):
        print("#" * 78)
        # Ranks Euclidean-Arithmetic
        array1 = self.rel_close_eucl_arithmetic.copy()
        temp = (-array1).argsort()
        ranks_eucl_arithmetic = np.arange(len(array1))[temp.argsort()] + 1
        pprint("Ranking Euclidean-Arithmetic:")
        print(ranks_eucl_arithmetic)

        # Ranks Euclidean-Geometric
        array2 = self.rel_close_eucl_geometric.copy()
        temp = (-array2).argsort()
        ranks_eucl_geometric = np.arange(len(array2))[temp.argsort()] + 1
        pprint("Ranking Euclidean-Geomemtric:")
        print(ranks_eucl_geometric)

        # Ranks Manhattan-Arithmetic
        array3 = self.rel_close_manh_arithmetic.copy()
        temp = (-array3).argsort()
        ranks_manh_arithmetic = np.arange(len(array3))[temp.argsort()] + 1
        pprint("Ranking Manhattan-Arithmetic:")
        print(ranks_manh_arithmetic)

        # Ranks Manhattan-Geometric
        array4 = self.rel_close_manh_geometric.copy()
        temp = (-array4).argsort()
        ranks_manh_geometric = np.arange(len(array4))[temp.argsort()] + 1
        pprint("Ranking Manhattan-Geometric:")
        print(ranks_manh_geometric)

        x_axis = ["Option " + str(x + 1) for x in range(self.normalized_matrix["DM1"].index.size)]

        data = pd.DataFrame(self.rel_close_eucl_arithmetic, index=x_axis)
        plots = data.plot(kind='bar', figsize=(14, 9), xlabel='Candidates', ylabel='Ranking Score',
                          title="Evaluation using Euclidean-Arithmetic technique")
        for index, bar in enumerate(plots.patches):
            plots.annotate(ranks_eucl_arithmetic[index],
                           (bar.get_x() + bar.get_width() / 2,
                            bar.get_height()), ha='center', va='center',
                           size=11, xytext=(0, 8),
                           textcoords='offset points')
        plt.legend().remove()
        plt.savefig("Evaluation-Euclidean-Arithmetic.svg")
        plt.show()

        data = pd.DataFrame(self.rel_close_eucl_geometric, index=x_axis)
        plots = data.plot(kind='bar', figsize=(14, 9), color='red', xlabel='Candidates', ylabel='Ranking Score',
                          title="Evaluation using Euclidean-Geometric technique")
        for index, bar in enumerate(plots.patches):
            plots.annotate(ranks_eucl_geometric[index],
                           (bar.get_x() + bar.get_width() / 2,
                            bar.get_height()), ha='center', va='center',
                           size=11, xytext=(0, 8),
                           textcoords='offset points')
        plt.legend().remove()
        plt.savefig("Evaluation-Euclidean-Geometric.svg")
        plt.show()

        data = pd.DataFrame(self.rel_close_manh_arithmetic, index=x_axis)
        plots = data.plot(kind='bar', figsize=(14, 9), color='green', xlabel='Candidates', ylabel='Ranking Score',
                          title="Evaluation using Manhattan-Arithmetic technique")
        for index, bar in enumerate(plots.patches):
            plots.annotate(ranks_manh_arithmetic[index],
                           (bar.get_x() + bar.get_width() / 2,
                            bar.get_height()), ha='center', va='center',
                           size=11, xytext=(0, 8),
                           textcoords='offset points')
        plt.legend().remove()
        plt.savefig("Evaluation-Manhattan-Arithmetic.svg")
        plt.show()

        data = pd.DataFrame(self.rel_close_manh_geometric, index=x_axis)
        plots = data.plot(kind='bar', figsize=(14, 9), color='olive', xlabel='Candidates', ylabel='Ranking Score',
                          title="Evaluation using Manhattan-Geometric technique")
        for index, bar in enumerate(plots.patches):
            plots.annotate(ranks_manh_geometric[index],
                           (bar.get_x() + bar.get_width() / 2,
                            bar.get_height()), ha='center', va='center',
                           size=11, xytext=(0, 8),
                           textcoords='offset points')
        plt.legend().remove()
        plt.savefig("Evaluation-Manhattan-Geometric.svg")
        plt.show()
