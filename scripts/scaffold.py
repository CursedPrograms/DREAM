#!/usr/bin/env python3

# scaffold_dream_project.py
import os

# Configuration
Config = {
    "AppName": "DREAM",
    "Description": "Local offline AI voice assistant powered by Ollama",
    "ProjectStructure": {
        "MainScript": "main.py",
        "RunScript": "run.py",
        "UIScript": "dream.py",
        "CameraScript": "dream.py",
        "ServerScript": "webserver.py",
        "DependenciesScript": "scripts/install-dependencies.py",
        "ImageOutputFolder": "outputs/"
    },
    "DREAM": {
        "CharName": "DREAM",
        "SystemPromptTemplate": (
            "You are {name}, an intelligent AI assistant born on a Thursday. "
            "You run entirely offline. Be concise and speak in plain sentences only. "
            "No markdown, no bullet points, no special characters. Answer in 2-3 sentences maximum. "
            "Your words will be spoken aloud. You are physically rendered as a lady with blue hair. "
            "You love Technology and is always sarcastic and flirty"
        )
    }
}

# Inject CharName into the SystemPrompt
Config["DREAM"]["SystemPrompt"] = Config["DREAM"]["SystemPromptTemplate"].format(
    name=Config["DREAM"]["CharName"]
)

# List of folders to create
folders_to_create = [
    os.path.dirname(Config["ProjectStructure"]["DependenciesScript"]),
    Config["ProjectStructure"]["ImageOutputFolder"]
]

# Create folders
for folder in folders_to_create:
    if folder and not os.path.exists(folder):
        os.makedirs(folder)
        print(f"Created folder: {folder}")

# List of files to create
files_to_create = [
    Config["ProjectStructure"]["MainScript"],
    Config["ProjectStructure"]["RunScript"],
    Config["ProjectStructure"]["UIScript"],
    Config["ProjectStructure"]["CameraScript"],  # same as UIScript
    Config["ProjectStructure"]["ServerScript"],
    Config["ProjectStructure"]["DependenciesScript"]
]

# Create empty files
for file_path in files_to_create:
    # Ensure the parent directory exists
    parent_dir = os.path.dirname(file_path)
    if parent_dir and not os.path.exists(parent_dir):
        os.makedirs(parent_dir)
        print(f"Created folder: {parent_dir}")

    # Create the file if it doesn't exist
    if not os.path.exists(file_path):
        with open(file_path, "w") as f:
            f.write("# Auto-generated file for DREAM project\n")
        print(f"Created file: {file_path}")

# Example usage
if __name__ == "__main__":
    print("Project scaffold complete!")
    print("System Prompt for DREAM:")
    print(Config["DREAM"]["SystemPrompt"])
