import re
import os
import json

root = os.path.join('data', 'MATH')

pattern = re.compile(r'\\boxed\s*(?:{([^}]*)}|(\S))')

boxed_single = re.compile(r'\\boxed\s+(\S)')  # matches \boxed x
boxed_start = re.compile(r'\\boxed\s*{')      # detects \boxed{ start

def extract_answer(text):
    results = []
    i = 0
    while i < len(text):
        m = boxed_single.match(text, i)
        if m:
            results.append(m.group(1))
            i = m.end()
            continue

        m = boxed_start.match(text, i)
        if m:
            i = m.end()
            depth = 1
            start = i
            while i < len(text) and depth > 0:
                if text[i] == '{':
                    depth += 1
                elif text[i] == '}':
                    depth -= 1
                i += 1
            results.append(text[start:i-1])
            continue

        i += 1
    return results[0]

def normalize_data(data: dict):
    """
    Normalizes the data by extracting the answer from the solution.
    :param data: The JSON data to be normalized.
    :return: A dictionary with the normalized question, solution, and answer.
    """
    try:
        answer = extract_answer(data['solution'])
        return {
            'question': data['problem'],
            'solution': data['solution'],
            'answer': answer
        }
    except ValueError as e:
        print(data)
        print(f"Error normalizing data: {e}")
        return None

def extract_folder(mode='train'):
    """
    Extracts the JSON files from the specified folder and normalizes the data.
    :param mode: The mode of the dataset (train, test, or val).
    :return: A list of normalized data dictionaries.
    """
    data_list = []

    for dirpath, _, files in os.walk(os.path.join(root, mode)):
        for file in files:
            if file.endswith('.json'):
                file_path = os.path.join(dirpath, file)
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    normalized_data = normalize_data(data)
                    if normalized_data:
                        data_list.append(normalized_data)

    return data_list

def jsonl_dump(data_list, file_path):
    """
    Dumps the data list into a JSONL file.
    :param data_list: The list of data dictionaries to be dumped.
    :param file_path: The path to the output JSONL file.
    """
    with open(file_path, 'w', encoding='utf-8') as f:
        for data in data_list:
            f.write(json.dumps(data) + '\n')

if __name__ == '__main__':
    # Example usage
    train_data = extract_folder('train')
    test_data = extract_folder('test')
    jsonl_dump(train_data, os.path.join(root, 'train.jsonl'))
    jsonl_dump(test_data, os.path.join(root, 'test.jsonl'))
