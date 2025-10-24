from tokenizers import Tokenizer
from tokenizers.models import Unigram
from tokenizers.trainers import UnigramTrainer
from tokenizers.pre_tokenizers import Whitespace, BertPreTokenizer
from tokenizers.normalizers import NFKC, Lowercase, Sequence
from tokenizers.processors import TemplateProcessing
from tqdm import tqdm
import os

def train_unigram_tokenizer(corpus_path, vocab_size, output_path, is_chinese=False):
    """
    corpus_path: 语料文件路径
    vocab_size: 词汇表大小
    output_path: 输出 json 路径
    is_chinese: 是否是中文语料
    """
    print(f"🚀 开始训练 {'中文' if is_chinese else '英文'} Tokenizer...")

    # 初始化 tokenizer
    tokenizer = Tokenizer(Unigram())

    # 设置 normalizer（统一格式）
    if is_chinese:
        # 中文不转小写，但做兼容归一化
        tokenizer.normalizer = NFKC()
        # 中文使用 BertPreTokenizer，可以处理中日韩字符分词
        tokenizer.pre_tokenizer = BertPreTokenizer()
    else:
        tokenizer.normalizer = Sequence([NFKC(), Lowercase()])
        tokenizer.pre_tokenizer = Whitespace()

    # 设置训练器
    trainer = UnigramTrainer(
        vocab_size=vocab_size,
        unk_token="[UNK]",
        special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"],
    )

    # 训练
    with open(corpus_path, "r", encoding="utf-8") as f:
        tokenizer.train_from_iterator(f, trainer=trainer, length=None)

    # 设置 post-processor，使编码输出带 [CLS] 和 [SEP]
    tokenizer.post_processor = TemplateProcessing(
        single="[CLS] $A [SEP]",
        pair="[CLS] $A [SEP] $B [SEP]",
        special_tokens=[
            ("[CLS]", tokenizer.token_to_id("[CLS]")),
            ("[SEP]", tokenizer.token_to_id("[SEP]")),
        ],
    )
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # 保存
    tokenizer.save(output_path)
    print(f"✅ 已保存至 {output_path}\n")


if __name__ == "__main__":
    train_unigram_tokenizer("./data/train.en", vocab_size=10000, output_path="tokenizers/unigram_tokenizer_en.json", is_chinese=False)
    train_unigram_tokenizer("./data/train.zh", vocab_size=8000, output_path="tokenizers/unigram_tokenizer_zh.json", is_chinese=True)
