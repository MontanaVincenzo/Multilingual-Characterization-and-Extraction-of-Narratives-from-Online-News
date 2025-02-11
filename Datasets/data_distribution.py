import collections
import csv
import matplotlib.pyplot as plt

def parse_file(file_path):
    main_roles_mapping = {
        "protagonist": ["Guardian", "Martyr", "Peacemaker", "Rebel", "Underdog", "Virtuous"],
        "antagonist": [
            "Instigator", "Conspirator", "Tyrant", "Foreign Adversary", "Traitor",
            "Spy", "Saboteur", "Corrupt", "Incompetent", "Terrorist", "Deceiver", "Bigot"
        ],
        "innocent": ["Forgotten", "Exploited", "Victim", "Scapegoat"]
    }
    
    main_role_counts = collections.Counter()
    refined_role_counts = collections.defaultdict(collections.Counter)
    
    with open(file_path, 'r', encoding='utf-8') as file:
        reader = csv.reader(file, delimiter='\t')
        for row in reader:
            if len(row) < 5:
                continue  # Skip malformed lines
            
            file_id, entity, start_offset, end_offset, main_role, *refined_roles = row
            main_role = main_role.lower()
            
            if main_role in main_roles_mapping:
                main_role_counts[main_role] += 1
                for refined_role in refined_roles:
                    if refined_role in main_roles_mapping[main_role]:
                        refined_role_counts[main_role][refined_role] += 1
    
    return main_role_counts, refined_role_counts

def display_distributions(main_role_counts, refined_role_counts):
    print("\nMain Role Distribution:")
    for role, count in main_role_counts.items():
        print(f"{role.capitalize()}: {count}")
    
    print("\nRefined Role Distribution per Main Role:")
    for main_role, role_counts in refined_role_counts.items():
        print(f"\n{main_role.capitalize()}:")
        for refined_role, count in role_counts.items():
            print(f"  {refined_role}: {count}")

def plot_main_role_distribution(main_role_counts):
    plt.figure(figsize=(8, 6))
    plt.bar(main_role_counts.keys(), main_role_counts.values(), color=['blue', 'red', 'green'])
    plt.xlabel('Main Roles')
    plt.ylabel('Count')
    plt.title('Distribution of Main Roles')
    plt.show()

if __name__ == "__main__":
    file_path = "./Train/EN/subtask-1-annotations.txt" 
    main_role_counts, refined_role_counts = parse_file(file_path)
    display_distributions(main_role_counts, refined_role_counts)
    plot_main_role_distribution(main_role_counts)
