import argparse
import json

import yaml
from tqdm import tqdm

from .core import Extractor
from .utils import DataProcessor
import numpy as np 
from sklearn.metrics import accuracy_score, f1_score
from collections import defaultdict 
import json 

def check_config(config):
    if 'data_path' not in config:
        raise ValueError(
            "Error: 'data_path' not found in the configuration file.")
    if 'xfinder_model' not in config:
        raise ValueError(
            "Error: 'xfinder_model' not found in the configuration file.")
    if 'model_name' not in config['xfinder_model']:
        raise ValueError(
            "Error: 'model_name' of xfinder not found in the configuration file."
        )
    if 'model_path' not in config['xfinder_model'] and 'url' not in config[
            'xfinder_model']:
        raise ValueError(
            "Error: 'model_path' or 'url' of xfinder not found in the configuration file."
        )


def calc_acc(config_path: str) -> None:
    """Calculate the accuracy given the file to be evaluated.
    
    Args:
        config_path (str): Path to the configuration file.
    """
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    check_config(config)

    data_processor = DataProcessor()
    extractor = Extractor(
        model_name=config['xfinder_model']['model_name'],
        model_path=config['xfinder_model']['model_path']
        if 'model_path' in config['xfinder_model'] else None,
        url=config['xfinder_model']['url']
        if 'url' in config['xfinder_model'] else None,
    )

    # Get extracted answers
    ori_data = data_processor.read_data(config['data_path'])
    split_indices = new_ori_data = None 
    if isinstance(ori_data, dict):
        split_indices = [0] 
        new_ori_data = [] 
        for prompt in ori_data:
            split_indices.append(len(ori_data[prompt]) + split_indices[-1])
            new_ori_data.extend(ori_data[prompt])
    if new_ori_data is not None:
        ori_data = new_ori_data 
    outputs = [] 
    inputs = [] 
    res, targets = [], [] 
    for item in tqdm(ori_data):
        user_input = extractor.prepare_input(item)
        extracted_answer = extractor.gen_output(user_input)
        correct_answer = item["correct_answer"]
        res.append(extracted_answer.strip().rstrip(".").lower())
        targets.append(correct_answer.strip().rstrip(".").lower())
        outputs.append(item["llm_output"])
        inputs.append(item["question"])

    res, targets = np.array(res), np.array(targets)
    if split_indices is None:
        outputs = np.array(outputs)
        print(f"Accuracy: {accuracy_score(targets, res)}")
        print(f"The number of invaild answers/totel: {np.sum(res == '[no valid answer]')}/{len(res)}")
        for i, output in enumerate(outputs[res == "[no valid answer]"]):
            print(f"Q: {inputs[i]}")
            print(f"A: {output}") 
        indices = res != "[no valid answer]"
        source_predictions, source_targets = res[indices], targets[indices]
        print(f"Accuracy: {accuracy_score(source_targets, source_predictions)}")
        print(f"Average F1 score: {f1_score(source_targets, source_predictions, average='macro')}")
        print(f"Weighted F1 score: {f1_score(source_targets, source_predictions, average='weighted')}")
    else:
        metric_dict = defaultdict(list)
        for i in range(len(split_indices) - 1):
            s, e = split_indices[i], split_indices[i + 1]
            cur_res, cur_targets = res[s: e], targets[s: e]
            metric_dict["true_accuracy"].append(accuracy_score(cur_targets, cur_res))
            indices = cur_res != "[no valid answer]"
            source_predictions, source_targets = cur_res[indices], cur_targets[indices]
            metric_dict["accuracy"].append(accuracy_score(source_targets, source_predictions))
            metric_dict["average_f1"].append(f1_score(source_targets, source_predictions, average='macro'))
            metric_dict["weighted_f1"].append(f1_score(source_targets, source_predictions, average='weighted'))
        with open(config["data_path"], 'w') as f:
            json.dump(metric_dict, f, indent=4)
        
    return

def main():
    parser = argparse.ArgumentParser(description='Run xFinder evaluation.')
    parser.add_argument(
        'config_path',
        nargs='?',
        default=None,
        help='Path to the configuration file')
    args = parser.parse_args()

    config_path = args.config_path
    if not config_path:
        print("Error: No configuration path provided.")
        parser.print_help()
        return

    return calc_acc(config_path)


if __name__ == "__main__":
    main()
