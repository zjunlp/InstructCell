import json 
import os 
from rouge_score import rouge_scorer
import numpy as np 
from collections import defaultdict
import argparse 

if __name__ == "__main__":
    parser = argparse.ArgumentParser() 
    parser.add_argument("--input_dirs", required=True, type=str, nargs='+', help="A list of input directories.")
    parser.add_argument("--output_dir", required=True, type=str, help="The output directory.")
    parser.add_argument("--similarity_threshold", type=float, default=0.75, help="The similarity threshold.")
    args = parser.parse_args()

    input_dirs = args.input_dirs
    output_dir = args.output_dir
    all_templates = {} 
    for input_dir in input_dirs:
        file_path = os.path.join(input_dir, "raw_templates.json")
        if os.path.exists(file_path):
            with open(file_path, "r") as f:
                raw_templates = json.load(f)
                for task_name in raw_templates:
                    if task_name not in all_templates:
                        all_templates[task_name] = defaultdict(list)
                    for key in raw_templates[task_name]:
                        all_templates[task_name][key].extend(raw_templates[task_name][key])
    if len(all_templates) == 0:
        raise ValueError("No templates found in the input directories.")
    
    # recompute the rouge scores
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=False)
    new_all_templates = {}
    total_templates = 0 
    num_low_similarity_templates = 0
    os.makedirs(output_dir, exist_ok=True)
    for task_name in all_templates:
        task_templates = all_templates[task_name]
        new_all_templates[task_name] = defaultdict(list)
        instructions = np.array(task_templates["instruction"]) 
        responses = np.array(task_templates["response"])
        total_templates += len(instructions)
        for i in range(len(instructions)):
            instruction = instructions[i]
            if i == 0:
                scores = np.array([0.0])
            else:
                scorer_func = np.vectorize(
                    lambda prediction: scorer.score(
                        instruction, 
                        prediction
                    )['rougeL'].fmeasure
                )
                scores = scorer_func(instructions[:i])
            max_scores = np.max(scores)
            if max_scores <= args.similarity_threshold:
                response = responses[i]
                new_all_templates[task_name]["instruction"].append(instruction)
                new_all_templates[task_name]["response"].append(response)
                new_all_templates[task_name]["instruction_score"].append(max_scores)
                scorer_func = np.vectorize(
                    lambda prediction: scorer.score(
                        response, 
                        prediction
                    )['rougeL'].fmeasure
                )
                if i > 0:
                    response_scores = scorer_func(responses[: i])
                else:
                    response_scores = np.array([0.0])
                response_max_score = np.max(response_scores)
                new_all_templates[task_name]["response_score"].append(response_max_score)
                if "instruction_most_similar_candidates" in task_templates: 
                    new_all_templates[task_name]["instruction_most_similar_candidates"].append(
                        [instructions[k] for k in np.argsort(scores)[-3:]] if i > 0 else [] 
                    )
                if "response_most_similar_candidates" in task_templates:
                    new_all_templates[task_name]["response_most_similar_candidates"].append(
                        [responses[k] for k in np.argsort(response_scores)[-3:]] if i > 0 else []
                    )
            else:
                num_low_similarity_templates += 1 
    
    print(f"Total number of templates: {total_templates}")
    print(f"Number of low similarity templates: {num_low_similarity_templates}")
    with open(os.path.join(output_dir, "filtered_templates.json"), "w") as f:
        json.dump(new_all_templates, f, indent=4)