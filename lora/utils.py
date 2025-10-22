from datasets import load_dataset
from datasets import IterableDataset
import re

SYSTEM_PROMPT = """you're a helpful assistant."""

XML_COT_FORMAT = """
{think}

boxed{{{answer}}}

"""

def extract_boxed_answer(s: str) -> str | None:
    if not s:
        return None
    s = s.replace('\\!', '')
    m = re.findall(r'\\?boxed\{([^}]*)\}', s)
    if not m:
        return None
    ans = m[-1].strip()
    ans = re.sub(r'[^0-9\-/\.]', '', ans)
    if ans != '' and ans[-1] == '/':
        ans = ans[:-1]
    return ans or None

def get_math_dataset(split="train", sft=False, cache_dir=None) -> IterableDataset:
    # Define the file paths for the local datasets
    local_data_paths = {
        'train': '/root/nlpexp/lora/data/hendrycks_math/algebra/train-00000-of-00001.parquet',
        'test': '/root/nlpexp/lora/data/hendrycks_math/algebra/test-00000-of-00001.parquet'
    }
    
    # Load the dataset from the local file
    data = load_dataset('parquet', data_files=local_data_paths[split])["train"]

    if not sft:
        data = data.map(lambda x: {
            'prompt': [
                {'role': 'system', 'content': SYSTEM_PROMPT},
                {'role': 'user', 'content': x['problem']}
            ],
            'answer': extract_boxed_answer(x['solution'])
        },remove_columns=['problem','solution','type','level'])
    else:
        data = data.map(lambda x: {
            'messages': [
                {'role': 'system', 'content': SYSTEM_PROMPT},
                {'role': 'user', 'content': x['problem']},
                {'role': 'assistant', 'content': x['solution']},
            ]
        },remove_columns=['problem','solution','type','level'])
    # print(data)
    return data

if __name__ == "__main__":
    
    dataset = get_math_dataset("train")
    print(dataset[0])
