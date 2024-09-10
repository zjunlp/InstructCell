import json 
import os 
from rouge_score import rouge_scorer
from collections import defaultdict
import numpy as np 
import tiktoken
from metadata import TRAIN_SIZE, SEED 
from sklearn.model_selection import train_test_split
from pathlib import Path
import argparse 

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dirs", required=True, type=str, nargs='+', help="A list of output directories.")
    parser.add_argument("--model", default="gpt-4o", type=str, help="The model used generate templates.")
    parser.add_argument("--max_input_length", default=85, type=int, help="The maximum length of input (exclusive).")
    parser.add_argument("--max_output_length", default=70, type=int, help="The maximum length of output (exclusive).")
    args = parser.parse_args()

    output_dirs = args.output_dirs
    model = args.model
    max_input_length = args.max_input_length
    max_output_length = args.max_output_length

    encoding = tiktoken.encoding_for_model(model)
    generator = np.random.default_rng(0)

    dummy_inputs = {
        "cell type annotation": { 
            "sequencing_method": '',
            "tissue": '',
            "species": '',
            "input": '', 
            "choices": '', 
        },
        "drug sensitivity prediction": { 
            "sequencing_method": '',
            "tissue": '',
            "species": '',
            "drug": '', 
            "input": '', 
            "choices": '', 
        },
        "conditional pseudo cell generation": {
            "cell_type": '', 
            "sequencing_method": '',
            "tissue": '',
            "species": '', 
        },
    }

    get_length = np.vectorize(lambda text: len(encoding.encode(text)))
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=False)

    for output_dir in output_dirs:
        output_file = os.path.join(output_dir, "filtered_templates.json")
        with open(output_file, 'r') as f:
            previous_templates = json.load(f)
        task_names = list(previous_templates.keys())
        total_num_templates = sum(len(previous_templates[task_name]["instruction"]) for task_name in task_names)
        filtered_templates = {} 
        for task_name in task_names:
            filtered_templates[task_name] = defaultdict(list)
            for i in range(len(previous_templates[task_name]["instruction"])):
                instruction = previous_templates[task_name]["instruction"][i]
                response = previous_templates[task_name]["response"][i]
                input_length = len(encoding.encode(instruction))
                output_length = len(encoding.encode(response))
                if output_length < max_output_length and input_length < max_input_length:
                    for key in previous_templates[task_name]:
                        filtered_templates[task_name][key].append(previous_templates[task_name][key][i])
        previous_templates = filtered_templates 
        filtered_templates = {} 
        for task_name in task_names:
            filtered_templates[task_name] = defaultdict(list)
            for i in range(len(previous_templates[task_name]["instruction"])):
                instruction = previous_templates[task_name]["instruction"][i]
                response = previous_templates[task_name]["response"][i]
                try: 
                    instruction_ = instruction.format(**dummy_inputs[task_name])
                    dummy_outputs = {
                        "output": '', 
                        **dummy_inputs[task_name]
                    }
                    if "input" in dummy_outputs:
                        del dummy_outputs["input"]
                    response_ = response.format(**dummy_outputs)
                    if instruction.count("{input}") < 2 and response.count("{output}") == 1:
                        for key in previous_templates[task_name]:
                            filtered_templates[task_name][key].append(previous_templates[task_name][key][i])
                    else:
                        raise ValueError("The number of input is more than 1 or the number of output is not equal to 1")
                except Exception as e:
                    # print('-' * 50)
                    # print(f"Error message: {e}")
                    # print(f"input: {instruction}\noutput: {response}")
                    # print('-' * 50)
                    pass
        num_templates = sum(len(filtered_templates[task_name]["instruction"]) for task_name in task_names) 
        print(f"The ratio of valid templates for {output_dir} is {num_templates / total_num_templates}")
        formated_file = os.path.join(output_dir, "formatted_templates.json")
        templates = {}
        for task_name in task_names:
            task_filtered_templates = filtered_templates[task_name]
            templates[task_name] = []
            for i in range(len(task_filtered_templates["instruction"])):
                item = {
                    key: task_filtered_templates[key][i] for key in task_filtered_templates
                }
                templates[task_name].append(item)
        with open(formated_file, 'w') as f:
            json.dump(templates, f, indent=4)

    all_templates = [] 
    for output_dir in output_dirs:
        input_file = os.path.join(output_dir, "formatted_templates.json")
        with open(input_file, 'r') as f:
            input_templates = json.load(f)
        all_templates.append(input_templates) 
    min_task_templates = {task_name: min(len(templates[task_name]) for templates in all_templates) for task_name in task_names}
    print("For each template dataset:")
    for task_name in task_names:
        print(f"- Task: {task_name}, number of templates: {min_task_templates[task_name]}")
    # for each task, make sure each template set has the equal number of templates  
    for templates, output_dir in zip(all_templates, output_dirs):
        for task_name in min_task_templates:
            templates[task_name] = templates[task_name][: min_task_templates[task_name]] 
        for task_name in task_names:
            task_templates = [] 
            for item in templates[task_name]:
                task_templates.append(
                    {
                        "instruction": item["instruction"],
                        "response": item["response"],
                    }
                )
            train_templates, test_templates = train_test_split(
                task_templates, 
                train_size=TRAIN_SIZE, 
                random_state=SEED
            )
            valid_templates, test_templates = test_templates[: len(test_templates) // 2], test_templates[len(test_templates) // 2: ]
            for split, split_templates in zip(["train", "valid", "test"], [train_templates, valid_templates, test_templates]):
                dir_path = os.path.join(output_dir, f"{split}_templates", task_name)
                dir_path = Path(dir_path)
                dir_path.mkdir(parents=True, exist_ok=True)
                with open(os.path.join(str(dir_path), "templates.json"), 'w') as f:
                    json.dump(split_templates, f, indent=4)