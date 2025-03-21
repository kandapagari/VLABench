import os
import argparse
import json
from typing import Optional, List

# setting environment variable
os.environ["MUJOCO_GL"] = "egl"  
from data_producing_utils import *

dim2task_config_path = os.path.join(os.getenv("VLABENCH_ROOT"), "configs/evaluation/dim2task.json")
with open(dim2task_config_path, 'r') as json_file:
    dim2task_config = json.load(json_file)
task2dim_config = {}
all_task_list = []
for dim in dim2task_config:
    for task in dim2task_config[dim]:
        all_task_list.append(task)
        if task not in task2dim_config:
            task2dim_config[task] = dim
all_task_list = list(set(all_task_list))


def parse_args():
    """parse the args"""
    parser = argparse.ArgumentParser(description='Dataset Generation Pipeline')
    parser.add_argument('--task_list_path', type=str, default=None,
                       help='Path to task list JSON config (default: None)')
    parser.add_argument('--example_num', type=int, default=100,
                       help='Number of examples to generate per task')
    parser.add_argument('--data_path', type=str, default='./dataset',
                       help='Root directory for dataset storage')
    return parser.parse_args()

def load_task_list(config_path: Optional[str]) -> List[str]:
    """loading task list"""
    if config_path:
        try:
            with open(config_path, 'r') as f:
                return json.load(f)['tasks']
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Error loading task list: {str(e)}")
            print("Falling back to directory listing")
    
    # default loading from config
    return all_task_list

def validate_paths(data_path: str):
    """path validation"""
    if not os.path.exists(data_path):
        os.makedirs(data_path, exist_ok=True)
        print(f"Created dataset directory at {data_path}")

def main():
    args = parse_args()
    validate_paths(args.data_path)
    
    # loading task list
    task_list = load_task_list(args.task_list_path)
    print(f"Loaded {len(task_list)} tasks from {'config' if args.task_list_path else 'directory'}")
    
    error_list = []
    example_template = "example{}"  # template
    
    for task in task_list:
        assert  task in task2dim_config ,"Task not in task2dim_config error"
        task_path = os.path.join(args.data_path, task2dim_config[task] ,task)
        os.makedirs(task_path, exist_ok=True)
        
        print(f"\n\nProcessing task: {task.center(40, '-')}")
        
        for idx in range(args.example_num):
            example_name = example_template.format(idx)
            example_path = os.path.join(task_path, example_name)
            
            print(f"producing: [{task:^40}] on example: {example_name:<20} , idx: {idx:>5}", end="\r")
            
            if check_example_visual_complement(example_path):
                continue
                
            try:
                info, state = single_example_visual_produce(task, idx, args.data_path)
                if state == "ERROR":
                    error_list.append((task, idx))
                    print(f"\nERROR: {task} {idx} - {info}")
            except Exception as e:
                error_list.append((task, idx))
                print(f"\nCRITICAL ERROR: {task} {idx} - {str(e)}")
    
    print("\n\nGeneration Summary:")
    print(f"Total tasks processed: {len(task_list)}")
    print(f"Total examples attempted: {len(task_list)*args.example_num}")
    print(f"Errors encountered: {len(error_list)}")
    
    # if error_list:
    #     print("\nError Details:")
    #     for task, idx in error_list:
    #         print(f"Task: {task:<20} Example: {example_template.format(idx)}")

if __name__ == "__main__":
    main()