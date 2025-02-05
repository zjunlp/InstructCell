import ast
import json


class DataProcessor:

    def __init__(self):
        pass

    def _read_custom_data(self, data):

        for key in data:
            for item in data[key]:
                if isinstance(item["standard_answer_range"],
                          str) and item["key_answer_type"] != "math":
                    try:
                        item["standard_answer_range"] = ast.literal_eval(
                            item["standard_answer_range"])
                    except Exception as e:
                        print(f"Error: {e}")
                        print(
                            "Please check if you provide the correct form of the 'standard_answer_range': ",
                            item)
                        exit(0)

                item["standard_answer_range"] = str(item["standard_answer_range"])
                item["key_answer_type"] = str(item["key_answer_type"])
        
        return data 


    def read_data(self, path):
        with open(path, 'r', encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, dict):
            keys = list(data.keys())
            if len(keys) > 1:
                return self._read_custom_data(data)
            data = data[keys[0]]

        for item in data:
            if isinstance(item["standard_answer_range"], str) and item["key_answer_type"] != "math":
                try:
                    item["standard_answer_range"] = ast.literal_eval(
                        item["standard_answer_range"])
                except Exception as e:
                    print(f"Error: {e}")
                    print(
                        "Please check if you provide the correct form of the 'standard_answer_range': ",
                        item)
                    exit(0)

            item["standard_answer_range"] = str(item["standard_answer_range"])
            item["key_answer_type"] = str(item["key_answer_type"])

        return data
