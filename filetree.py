#!/usr/bin/env python3
import os

def print_directory_tree(root_path, indent="", ignore_list=None):
    """
    Recursively prints the directory tree, skipping ignored folders.
    """
    if ignore_list is None:
        ignore_list = [".git", "__pycache__", "venv311", "venv", ".cache"]

    base_name = os.path.basename(root_path)
    if not base_name and root_path.endswith(os.sep): # Handle root edge case
        base_name = root_path
        
    print(f"{indent}└── {base_name}/")

    try:
        entries = sorted(os.listdir(root_path))
        for i, entry in enumerate(entries):
            if entry in ignore_list:
                continue
                
            entry_path = os.path.join(root_path, entry)
            is_last = (i == len(entries) - 1)
            
            # Use visual branch markers for better readability
            connector = "    " if indent else ""
            
            if os.path.isdir(entry_path):
                print_directory_tree(entry_path, indent + "  ", ignore_list)
            else:
                print(f"{indent}    ├── {entry}")
    except PermissionError:
        print(f"{indent}    [Permission Denied]")

if __name__ == "__main__":
    # Get the directory where the script is located
    base_directory = os.path.dirname(os.path.abspath(__file__))
    
    print("\nProject Structure:")
    print("-" * 30)
    print_directory_tree(base_directory)
    print("-" * 30)