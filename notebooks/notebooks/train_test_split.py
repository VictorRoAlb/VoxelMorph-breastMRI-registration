import os
from sklearn.model_selection import train_test_split

# === DIRECTORIES ===
# Set your local directories here before running the script.
# Replace the paths below with your actual data locations.
DTI_DIR  = '/path/to/preprocessed_registered_volumes'     # Directory containing registered DTI volumes after preprocesing
PERF_DIR = '/path/to/downsampled_normalized_perfusion'    # Directory containing normalized perfusion volumes
SAVE_DIR = '/path/to/save_partitions_lists'               # Directory where the train/test lists will be saved

SEED = 42  # Fixed random seed for reproducibility of splits

# Ensure output directory exists; create it if necessary
os.makedirs(SAVE_DIR, exist_ok=True)

# === PATIENT LIST ===
# List of patient IDs to include in the dataset
# Note: patients 9 and 22 are excluded due to missing data.
all_ids = [f"{i:02d}" for i in range(1, 61) if i not in [9, 22]]

# === SPLIT DATA ===
# Randomly split patient IDs into training and test sets.
# Here, 9 patients (~15% of total 58) are assigned to the test set.
# This proportion can be adjusted depending on the dataset size and application.
train_ids, test_ids = train_test_split(all_ids, test_size=9, random_state=SEED)

# === WRITE SPLIT FILES ===
def write_list(ids, filename):
    """
    Create a text file listing paired paths to DTI and perfusion volumes.
    Each line corresponds to one patient and contains the paths separated by a comma.
    Paths are constructed based on the patient IDs and base directories.
    """
    # Safely open the file and write patient paths, auto-closing afterward
    with open(os.path.join(SAVE_DIR, filename), 'w') as f:
        for pid in ids:
            dti_path  = os.path.join(DTI_DIR,  f'dti_{pid}_reg_final.nii')
            perf_path = os.path.join(PERF_DIR, f'perf_{pid}_TRJ_norm.nii')
            f.write(f"{dti_path},{perf_path}\n")

# Write training and test lists
write_list(train_ids, 'train_list.txt')
write_list(test_ids, 'test_list.txt')

# === WRITE TEST IDS FILE ===
# Additionally, save a simple list of test patient IDs (one per line).
with open(os.path.join(SAVE_DIR, 'test_ids.txt'), 'w') as f:
    f.writelines(pid + '\n' for pid in test_ids)

print("Lists generated successfully without spaces in paths:")
print(" - train_list.txt")
print(" - test_list.txt")
print(" - test_ids.txt")

