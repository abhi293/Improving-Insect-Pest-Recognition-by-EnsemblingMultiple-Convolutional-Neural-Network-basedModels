"""
Configuration script to enable/disable hybrid training in Trainmain.py
"""

import re

def toggle_hybrid_training(enable=True):
    """
    Toggle hybrid training in Trainmain.py
    """
    file_path = "Trainmain.py"
    
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Find the hybrid_training line
        if enable:
            # Enable hybrid training
            new_content = re.sub(
                r'hybrid_training = False  # Set to True to enable hybrid GPU/CPU training',
                'hybrid_training = True  # Set to False to disable hybrid GPU/CPU training',
                content
            )
            status = "ENABLED"
        else:
            # Disable hybrid training
            new_content = re.sub(
                r'hybrid_training = True  # Set to False to disable hybrid GPU/CPU training',
                'hybrid_training = False  # Set to True to enable hybrid GPU/CPU training',
                content
            )
            status = "DISABLED"
        
        # Write back if changed
        if new_content != content:
            with open(file_path, 'w') as f:
                f.write(new_content)
            print(f"✅ Hybrid training {status} in {file_path}")
            return True
        else:
            print(f"⚠️  Hybrid training already {status} in {file_path}")
            return False
            
    except FileNotFoundError:
        print(f"❌ File {file_path} not found")
        return False
    except Exception as e:
        print(f"❌ Error modifying {file_path}: {e}")
        return False

def get_hybrid_status():
    """
    Check current hybrid training status
    """
    file_path = "Trainmain.py"
    
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        
        if 'hybrid_training = True' in content:
            return True
        elif 'hybrid_training = False' in content:
            return False
        else:
            return None
            
    except FileNotFoundError:
        print(f"❌ File {file_path} not found")
        return None

if __name__ == "__main__":
    print("🔧 Hybrid Training Configuration Tool")
    print("="*50)
    
    current_status = get_hybrid_status()
    if current_status is None:
        print("❌ Could not determine current status")
    else:
        status_text = "ENABLED" if current_status else "DISABLED"
        print(f"Current status: {status_text}")
        
        print("\nOptions:")
        print("1. Enable hybrid training")
        print("2. Disable hybrid training")
        print("3. Exit")
        
        choice = input("\nEnter your choice (1-3): ").strip()
        
        if choice == "1":
            toggle_hybrid_training(True)
        elif choice == "2":
            toggle_hybrid_training(False)
        elif choice == "3":
            print("Exiting...")
        else:
            print("Invalid choice")
    
    print("\n📝 Note:")
    print("- SampleTrain.py always has hybrid training ENABLED for testing")
    print("- Trainmain.py hybrid training can be toggled with this script")
    print("- Hybrid training uses both GPU and CPU simultaneously")
    print("- Useful for maximizing hardware utilization")
