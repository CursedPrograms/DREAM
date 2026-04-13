import os
import subprocess
import json
import sys

def get_venv_python():
    """Detects the virtual environment python executable based on OS."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Matching your previous setup: using 'venv311'
    if os.name == "nt":  # Windows
        path = os.path.join(current_dir, "venv311", "Scripts", "python.exe")
    else:  # Linux / macOS
        path = os.path.join(current_dir, "venv311", "bin", "python")
        
    return path

def main():
    # Load config with error handling
    try:
        with open('config.json') as json_file:
            config_data = json.load(json_file)
            app_name = config_data.get('Config', {}).get('AppName', 'AI Command Centre')
    except FileNotFoundError:
        app_name = "AI Command Centre (Config not found)"

    print(f"--- {app_name} ---")

    scripts = {
        "1": {"name": "Run 'DREAM'", "file_name": "scripts/dream.py", "desc": "Main DREAM script"},
        "2": {"name": "Run 'Chatbot'", "file_name": "scripts/base_chatbot.py", "desc": "CPU Chatbot"},
        "3": {"name": "Run 'Chatbot GPU'", "file_name": "scripts/base_chatbot_gpu.py", "desc": "CUDA Chatbot"},
        "4": {"name": "Run 'DREAM Low FPS'", "file_name": "scripts/_dream.py", "desc": "JPEG-based DREAM"},
        "5": {"name": "Run 'WebServer'", "file_name": "app.py", "desc": "Flask Server"},
        "6": {"name": "Run 'Speak'", "file_name": "scripts/speak.py", "desc": "Piper TTS Test"},
        "00": {"name": "Update Deps", "file_name": "scripts/install_dependencies.py", "desc": "Install requirements"}
    }

    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    venv_python = get_venv_python()

    if not os.path.exists(venv_python):
        print(f"⚠️  Virtual environment not found at: {venv_python}")
        print("Falling back to system Python...")
        venv_python = sys.executable 

    while True:
        print("\n" + "="*30)
        print(f"  {app_name.upper()} MENU")
        print("="*30)
        for key, info in scripts.items():
            print(f"[{key.ljust(2)}] {info['name']} - {info['desc']}")
        print("[q ] Quit")

        try:
            user_choice = input("\nSelect an option: ").strip()
            
            if user_choice.lower() == 'q':
                print("Goodbye!")
                break

            if user_choice in scripts:
                selected = scripts[user_choice]
                script_path = os.path.join(current_script_dir, selected["file_name"])

                if os.path.exists(script_path):
                    print(f"\n>> Launching {selected['name']}...")
                    # cwd=current_script_dir ensures relative paths inside scripts work
                    subprocess.run([venv_python, script_path], cwd=current_script_dir)
                else:
                    print(f"❌ Error: File not found: {selected['file_name']}")
            else:
                print("❌ Invalid choice.")

        except KeyboardInterrupt:
            print("\nReturning to menu...")

if __name__ == "__main__":
    main()