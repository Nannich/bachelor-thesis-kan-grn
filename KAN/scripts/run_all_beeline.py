import os
import subprocess
import time
from pathlib import Path
import datetime

def main():
    base_inputs = Path("data/BEELINE-data/inputs")
    
    # Find all expression files to identify valid run directories
    expr_files = list(base_inputs.rglob("ExpressionData.csv"))
    
    runs = []
    for expr_file in expr_files:
        run_dir = expr_file.parent
        group_dir = run_dir.parent
        
        # Skip R script pre-processing folders
        if "scRNAseq_preprocessing" in str(run_dir):
            continue
            
        data_dir = str(group_dir)
        dataset_name = run_dir.name
        dataset_group = group_dir.name
        runs.append((data_dir, dataset_name, dataset_group))

    print(f"{len(runs)} dataset")
    
    total_start_time = time.time()
    
    for data_dir, dataset_name, dataset_group in runs:
        print(f"Processing: {dataset_group} / {dataset_name}")
        start_time = time.time()
        
        cmd_train = ["python", "main.py", "--data_dir", data_dir, "--dataset", dataset_name, "train", "--model", "effkan"]        
        model_name = f"effkan_{dataset_name}_all.pth"
        cmd_grn   = ["python", "main.py", "--data_dir", data_dir, "--dataset", dataset_name, "grn", model_name]
        
        try:
            subprocess.run(cmd_train, check=True)
            subprocess.run(cmd_grn, check=True)
                        
        except subprocess.CalledProcessError as e:
            print(f"Error on {dataset_name}")
            continue
        except KeyboardInterrupt:
            break
            
    total_elapsed = time.time() - total_start_time
    print(f"Finished after {total_elapsed}")

if __name__ == "__main__":
    main()