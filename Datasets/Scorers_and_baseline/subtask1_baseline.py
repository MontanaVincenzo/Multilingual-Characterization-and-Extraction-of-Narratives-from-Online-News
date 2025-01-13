import sys, csv, random, os
from collections import Counter
import argparse

# Set a fixed seed for reproducibility of random operations
random.seed(42)



# Define roles and sub-roles for random guessing
ROLES = ['Protagonist', 'Antagonist', 'Innocent']
PROTAGONISTS = ['Guardian', 'Martyr', 'Peacemaker', 'Rebel', 'Underdog', 'Virtuous']
ANTAGONISTS = ['Instigator', 'Conspirator', 'Tyrant', 'Foreign Adversary', 'Traitor', 'Spy', 'Saboteur', 'Corrupt', 'Incompetent', 'Terrorist', 'Deceiver', 'Bigot']
INNOCENTS = ['Forgotten', 'Exploited', 'Victim', 'Scapegoat']


def process_tsv_file(file_path):
    """
    Reads a TSV file as a single string and manually processes it based on tabs and newlines.

    Parameters:
    - file_path: Path to the TSV file.

    Returns:
    - A list of rows, where each row is a list of columns.
    """
    try:
        # Read the file as a single string
        with open(file_path, mode='r', encoding='utf-8') as file:
            file_content = file.read()

        # Split content into lines
        lines = file_content.split('\n')

        # Process each line into columns based on tabs
        rows = []
        for line in lines:
            if line.strip():  # Skip empty lines
                columns = line.split('\t')
                rows.append(columns)

        return rows
    except Exception as e:
        print(f"Error reading or processing file {file_path}: {e}")
        return None
    

def random_guess():
    """
    Assigns a random role and sub-role to an entity.
    
    Returns:
    - A tuple containing the randomly chosen role and sub-role
    """    
    role = random.choice(ROLES)
    if role == 'Protagonist':
        sub_role = random.choice(PROTAGONISTS)
    elif role == 'Antagonist':
        sub_role = random.choice(ANTAGONISTS)
    else:
        sub_role = random.choice(INNOCENTS)
    return role, sub_role

def majority_votes(train_file):
    """
    Assigns the most common role and sub-role based on training data.
    
    Parameters:
    - train_file: Path to the training data file
    
    Returns:
    - A tuple containing the most common role and sub-role from the training file
    """
    # Load roles and sub-roles from the training file to find the most common values    
    rows = process_tsv_file(train_file)

    roles = []
    sub_roles = []
    for row in rows:
        roles.append(row[4])
        sub_roles.append(row[5])
    # Return the most common role and sub-role
    return Counter(roles).most_common(1)[0][0], Counter(sub_roles).most_common(1)[0][0]

def main(dev_file, output_dir, baseline_type, train_file=None):
    """
    Processes the input file to generate baseline predictions for entity roles.
    
    Parameters:
    - dev_file: Path to the input file containing entities to label
    - output_dir: Directory to save the output file
    - baseline_type: Type of baseline to use ('random' or 'majority')
    - train_file: Optional path to the training file, required if using 'majority' baseline
    """    
    # Ensure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Set output file name based on baseline type
    output_file = os.path.join(output_dir, f"baseline_{baseline_type}.txt")

    # Precompute role and sub-role if using the majority baseline
    if baseline_type == "majority":
        if train_file is None:
            print("ERROR: train_file is required for majority baseline")
            return
        role, sub_role = majority_votes(train_file)
    elif baseline_type == "random":
        role, sub_role = None, None  # These will be generated on each iteration

    try:
        with open(dev_file, mode='r', encoding='utf-8') as file:
            with open(output_file, mode='w', encoding='utf-8') as fout:
                tsv_reader = csv.reader(file, delimiter='\t')
                writer = csv.writer(fout, delimiter='\t')
                for row in tsv_reader:
                    # Extract necessary fields from each row
                    file_id = row[0]
                    obj = row[1]
                    span_start = row[2]
                    span_end = row[3]
                    
                    # Generate role and sub-role for random baseline
                    if baseline_type == "random":
                        role, sub_role = random_guess()
                    
                    # Write the predicted role and sub-role to the output file
                    writer.writerow([file_id, obj, span_start, span_end, role, sub_role])
    except FileNotFoundError:
        print(f"ERROR: File not found '{dev_file}'")
    except Exception as e:
        print(f"ERROR: An unexpected error occurred: {e}")
        
if __name__ == "__main__":
    # Argument parsing for command-line inputs
    parser = argparse.ArgumentParser(description="Run baselines for role annotation")
    parser.add_argument('--dev_file', type=str, required=True, help="Path to the input file")
    parser.add_argument('--output_dir', type=str, required=True, help="Directory to save the output file")
    parser.add_argument('--baseline_type', type=str, choices=['random', 'majority'], required=True, help="Baseline type: 'random' or 'majority'")
    parser.add_argument('--train_file', type=str, help="Path to the training file (required for majority baseline)")

    args = parser.parse_args()
    # Call the main function with parsed arguments
    main(args.dev_file, args.output_dir, args.baseline_type, args.train_file)
