# compile_dataset_json.py
# Author: Boobalan Arjunan (https://github.com/BoobalanArjunan)
# Part of: ACE-Step 1.5 South Indian Parai LoRA Playground
#
import os
import json
import argparse
from pathlib import Path

def compile_dataset(audio_dir, output_file):
    dataset = []
    
    print(f"Scanning directory: {audio_dir}")
    
    # Iterate through all files in the directory
    for file_name in os.listdir(audio_dir):
        if file_name.endswith('.wav'):
            base_name = os.path.splitext(file_name)[0]
            json_path = os.path.join(audio_dir, f"{base_name}.json")
            
            # If a matching sidecar JSON exists
            if os.path.exists(json_path):
                with open(json_path, 'r', encoding='utf-8') as f:
                    try:
                        metadata = json.load(f)
                        
                        # Copy all metadata and ensure the 'audio' key exists for the training pipeline
                        entry = metadata.copy()
                        # Keep it as just the filename, dataset_builder usually resolves it relative to audio_dir
                        entry["audio"] = file_name 
                        
                        dataset.append(entry)
                    except json.JSONDecodeError:
                        print(f"❌ Error: Could not parse JSON in {json_path}")
            else:
                print(f"⚠️ Warning: No matching JSON found for {file_name}")
                
    # Save the master JSON array
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, indent=4)
        
    print(f"✅ Successfully compiled {len(dataset)} dataset entries into '{output_file}'!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compile sidecar JSONs into a master dataset.json")
    parser.add_argument("--audio-dir", required=True, help="Directory containing .wav and .json files")
    parser.add_argument("--output", default="dataset.json", help="Output master JSON file path")
    args = parser.parse_args()
    
    compile_dataset(args.audio_dir, args.output)
