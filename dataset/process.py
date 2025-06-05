from dapo_math import process_dapo_math
from open_r1_math import process_open_r1_math
from gsm8k import process_gsm8k
from math_dataset import process_math_dataset

if __name__ == '__main__':
    process_math_dataset()
    print('MATH dataset processed successfully.')
    process_gsm8k()
    print('GSM8K dataset processed successfully.')
    process_dapo_math()
    print('DAPO-Math-17k dataset processed successfully.')
    process_open_r1_math()
    print('Open-R1-Math dataset processed successfully.')
    import os
    if os.path.exists('data/AIME'):
        for file in os.listdir('data/AIME'):
            if file.endswith('.jsonl'):
                os.rename(
                    os.path.join('data', 'AIME', file),
                    os.path.join('data', file)
                )
    print('AIME dataset processed successfully.')
    # Now zip all jsonl file
    import zipfile
    with zipfile.ZipFile('data/math_datasets.zip', 'w') as zipf:
        for file in os.listdir('data'):
            if file.endswith('.jsonl'):
                zipf.write(os.path.join('data', file), file)
    print('All datasets zipped successfully.')