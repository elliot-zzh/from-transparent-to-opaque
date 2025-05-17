# https://www.kaggle.com/code/elliotzh/open-thought-data-collect

import polars as pl
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen3-1.7B')

data = pl.scan_parquet(
    'hf://datasets/open-thoughts/OpenThoughts-114k/data/train-*.parquet'
)  # lazy load


def combine_conversations(x):
    return (
        (x[0]['value'] + '\n' + x[1]['value'])
        .replace('<|begin_of_thought|>', '<think>')
        .replace('<|end_of_thought|>', '</think>')
        .replace('<|begin_of_solution|>', '')
        .replace('<|end_of_solution|>', '')
    )


def gather_questions(x):
    return x[0]['value'] + '<think>'


def ans_pos(x):
    return len(tokenizer(x[0]['value']).input_ids)


max_string_length = 1024 * 50
shorter_max_length = int(max_string_length * 0.5)
token_max_length = 2048

data = data.filter(
    pl.col('conversations').list.get(1).struct.field('value').str.len_chars()
    < max_string_length
).collect()
data = data.with_columns(
    pl.col('conversations')
    .map_elements(combine_conversations, return_dtype=pl.String)
    .alias('text')
)
data = data.with_columns(
    pl.col('conversations')
    .map_elements(gather_questions, return_dtype=pl.String)
    .alias('questions')
)
data = data.with_columns(
    pl.col('conversations')
    .map_elements(ans_pos, return_dtype=pl.Int16)
    .alias('ans_pos')
)

data = data.drop(['system', 'conversations'])
data = data.filter(pl.col('text').str.len_chars() < max_string_length)
print(data)
data.write_parquet('res-longer.parquet')

data = data.filter(pl.col('text').str.len_chars() < shorter_max_length)
print(data)
data.write_parquet('res-shorter.parquet')

seed = 42
subset_size = 1024
data = data.sample(seed=seed, n=subset_size)
print(data)
data.write_parquet('res-sampled.parquet')
