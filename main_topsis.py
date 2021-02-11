from topsis_class import *
from pathlib import Path
import pandas as pd
import warnings

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)


def intro_menu():
    print("#"*78)
    print("#        Topsis group implementation for Group Decision Support Systems      #")
    print("#        Problem: Group decision for hiring a candidate into a company       #")
    print("#                              Manousos Braoudakis                           #")
    print("#"*78)


# Sinartisi gia na diavazei ta paths twn arxeiwn poy tha ginei i epeksergasia
def get_path(import_settings):
    if import_settings == "Candidates_Results":
        file_path = Path(input("Insert data path for Candidates Results:"))
        while not file_path.exists():
            file_path = Path(
                input("Wrong file path! Insert correct data path for Candidates Results:"))
    elif import_settings == "Interview_Results":
        file_path = Path(input("Insert data path for Interview Results::"))
        while not file_path.exists():
            file_path = Path(
                input("Wrong file path! Insert correct data path for Interview Results:"))
    else:
        file_path = Path(input("Insert data path for Weights Matrix for all members of group:"))
        while not file_path.exists():
            file_path = Path(
                input("Wrong file path! Insert correct data path for Weights Matrix for all members of group:"))
    return file_path


# Kiria methodos ektelesis tou programmatos
if __name__ == '__main__':
    intro_menu()
    canditates_path = get_path("Candidates_Results")
    interview_path = get_path("Interview_Results")
    weights_path = get_path("Weights_Matrix")
    topsis_system = TopsisGroup(canditates_path, interview_path, weights_path)
    # Step 1: Decision Matrix
    topsis_system.create_decision_matrix()
    # Step 2: Normalized Matrix
    topsis_system.create_normalization_matrix()
    # Step 3: Ideal solution
    topsis_system.determine_ideal_solutions()
    # Step 4 and 5a: Calculate distances
    topsis_system.calculate_distances()
    # Step 5b: Calculate means
    topsis_system.calculate_means()
    # Step 6: Calculate relative closeness coefficients
    topsis_system.calculate_relative_coefficients()
    # Ranking and Create Figures
    topsis_system.create_figures()
