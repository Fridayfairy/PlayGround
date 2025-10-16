import json
import os
import random
from transformers import AutoTokenizer, PreTrainedTokenizerFast

from tokenizers import (
    decoders,
    models,
    pre_tokenizers,
    trainers,
    Tokenizer)

tokenizer = Tokenizer.from_file("data/tokenizer-wiki.json")



