"""Tokenizer functionality.

Note: CANNOT name this file "tokenizers.py ;>
"""

from transformers import AutoTokenizer, PreTrainedTokenizerFast
from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, decoders, trainers, Regex, processors


def load_tokenizer(tokenizer_path_or_name, seq_length=512, vocab_size=None, cache_dir=None):
    """Load a tokenizer from disk/huggingface. This will never construct a new tokenizer."""
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path_or_name, model_max_length=seq_length)
    except FileNotFoundError:
        tokenizer = _download_tokenizer(tokenizer_path_or_name, seq_length, cache_dir)
    if vocab_size is not None and tokenizer.vocab_size != vocab_size:
        raise ValueError(f"Loaded tokenizer with vocab_size {tokenizer.vocab_size} incompatible with given vocab.")
    return tokenizer


def construct_tokenizer(raw_datasets, cfg_data, path, known_tokens=[]):
    """Construct a new tokenizer. This may include downloading from huggingface."""
    if cfg_data.tokenizer not in ["BPE", "Unigram", "WordLevel", "WordPiece", "WordPieceBERT", "SentencePieceUnigram", "SentencePieceBPE"]:
        tokenizer = _download_tokenizer(cfg_data.tokenizer, cfg_data.seq_length, cache_dir=path)
    else:
        tokenizer = _construct_tokenizer(raw_datasets, cfg_data, known_tokens)
    tokenizer.name = f"{cfg_data.tokenizer}-{cfg_data.name}-{cfg_data.vocab_size}.json"
    return tokenizer


def _download_tokenizer(tokenizer_path_or_name, seq_length, cache_dir=None):
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path_or_name, cache_dir=cache_dir)
        tokenizer.model_max_length = seq_length
    except OSError as error_msg:
        raise OSError(f"Invalid huggingface tokenizer {tokenizer_path_or_name} given: {error_msg}")
    return tokenizer


def _get_sane_token_args():
    return dict(
        pad_token="<pad>",
        bos_token="<s>",
        eos_token="</s>",
        unk_token="<unk>",
        cls_token="<cls>",
        sep_token="<sep>",
        mask_token="<mask>",
    )


def _get_sane_normalizers(force_english_keyboard=False, force_lowercase=False, strip_accents=False, whitespace_escape=False, sanity=False):
    """original rules as in XLNET with optional modifications. force_english_keyboard is actually an ascii normalization."""
    if sanity:
        return normalizers.BertNormalizer(lowercase=force_lowercase)
    normalize_ops = []
    normalize_ops.append(normalizers.Replace("``", '"'))
    normalize_ops.append(normalizers.Replace("''", '"'))
    normalize_ops.append(normalizers.NFD() if strip_accents else normalizers.NFKC())
    if force_lowercase:
        normalize_ops.append(normalizers.Lowercase())
    if strip_accents:
        normalize_ops.append(normalizers.StripAccents())
    normalize_ops.append(normalizers.Replace(Regex(" {2,}"), " "))
    if force_english_keyboard:
        normalize_ops.append(normalizers.Replace(Regex(r"[^\x00-\x7F]+"), ""))  # start from 00 instead of 1F to include tab
    if whitespace_escape:
        normalize_ops.append(normalizers.Replace(Regex(" "), "一"))  # ▁ this might kill some of the tokenization schemes...
        # using yi in the in the previous regex because huggingface does not split on yi, but would split on bigunderscore
    return normalizers.Sequence(normalize_ops)


def _construct_tokenizer(raw_datasets, cfg_data, known_tokens=[]):
    """The actual generation instructions for a new tokenizer. Might make this more scriptable in the future...

    Follows closely along with https://huggingface.co/course/chapter6"""
    try:
        len_dataset = len(raw_datasets)

        def batch_iterator(batch_size=4096):
            for i in range(0, len_dataset, batch_size):
                yield raw_datasets[i : i + batch_size]["text"]

    except TypeError:
        # streaming dataset
        len_dataset = int(cfg_data.max_entries_in_dataset)

        def batch_iterator():
            for entry in iter(raw_datasets):
                yield entry["text"]

    special_token_args = _get_sane_token_args()
    normalizer_sequence = _get_sane_normalizers(**cfg_data.normalizer)
    # Outline tokenizer rules:
    if cfg_data.tokenizer == "Unigram":  # without the sentencepice part
        tokenizer = Tokenizer(models.Unigram())
        tokenizer.add_tokens(known_tokens)
        tokenizer.normalizer = normalizer_sequence
        tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
        # tokenizer.decoder = None
        special_tokens = list(set(v for k, v in special_token_args.items()))

        trainer = trainers.UnigramTrainer(
            vocab_size=cfg_data.vocab_size,
            special_tokens=special_tokens,
            unk_token=special_token_args["unk_token"],
        )
    elif cfg_data.tokenizer == "BPE":
        tokenizer = Tokenizer(models.BPE())
        tokenizer.add_tokens(known_tokens)

        tokenizer.normalizer = normalizer_sequence
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
        tokenizer.decoder = decoders.ByteLevel()

        trainer = trainers.BpeTrainer(
            vocab_size=cfg_data.vocab_size, min_frequency=2, special_tokens=list(set(special_token_args.values()))
        )
    elif cfg_data.tokenizer == "WordPiece":
        tokenizer = Tokenizer(models.WordPiece(unk_token=special_token_args["unk_token"]))
        tokenizer.add_tokens(known_tokens)

        tokenizer.normalizer = normalizer_sequence
        tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
        tokenizer.decoder = decoders.WordPiece(prefix="##")

        trainer = trainers.WordPieceTrainer(vocab_size=cfg_data.vocab_size, special_tokens=list(set(special_token_args.values())))
    elif cfg_data.tokenizer == "WordPieceBERT":
        # Sanity check tokenizer
        tokenizer = Tokenizer(models.WordPiece(unk_token="<unk>"))
        tokenizer.add_tokens(known_tokens)
        tokenizer.normalizer = normalizers.BertNormalizer()
        tokenizer.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
        tokenizer.decoder = decoders.WordPiece(prefix="##")

        trainer = trainers.WordPieceTrainer(vocab_size=cfg_data.vocab_size, special_tokens=list(set(special_token_args.values())))
    elif cfg_data.tokenizer == "WordLevel":
        tokenizer = Tokenizer(models.WordLevel(unk_token=special_token_args["unk_token"]))
        tokenizer.add_tokens(known_tokens)
        tokenizer.normalizer = normalizer_sequence
        tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
        trainer = trainers.WordLevelTrainer(vocab_size=cfg_data.vocab_size, special_tokens=list(set(special_token_args.values())))
    elif cfg_data.tokenizer == "SentencePieceBPE":
        """ref https://github.com/huggingface/tokenizers/blob/main/bindings/python/py_src/tokenizers/implementations/sentencepiece_bpe.py"""
        tokenizer = Tokenizer(models.BPE())
        tokenizer.add_tokens(known_tokens)

        tokenizer.normalizer = normalizer_sequence
        tokenizer.pre_tokenizer = pre_tokenizers.Metaspace(replacement="▁", add_prefix_space=True)
        tokenizer.decoder = decoders.Metaspace(replacement="▁", add_prefix_space=True)

        trainer = trainers.BpeTrainer(
            vocab_size=cfg_data.vocab_size, min_frequency=2, special_tokens=list(set(special_token_args.values()))
        )
    elif cfg_data.tokenizer == "SentencePieceUnigram":
        tokenizer = Tokenizer(models.Unigram())
        tokenizer.add_tokens(known_tokens)
        tokenizer.normalizer = normalizer_sequence
        tokenizer.pre_tokenizer = pre_tokenizers.Metaspace(replacement="▁", add_prefix_space=True)
        tokenizer.decoder = decoders.Metaspace(replacement="▁", add_prefix_space=True)
        special_tokens = list(set(v for k, v in special_token_args.items()))

        trainer = trainers.UnigramTrainer(
            vocab_size=cfg_data.vocab_size,
            special_tokens=special_tokens,
            unk_token=special_token_args["unk_token"],
        )
    else:
        raise ValueError(f"Invalid tokenization strategy {cfg_data.tokenizer} given.")

    # Construct tokenizer
    tokenizer.train_from_iterator(batch_iterator(), trainer=trainer, length=len_dataset)

    if tokenizer.get_vocab_size() != cfg_data.vocab_size:
        raise RuntimeError(f"Tokenizer generation failure. Vocab size of trained tokenizer is {tokenizer.get_vocab_size()}.")

    # Postprocess:
    cls_token_id = tokenizer.token_to_id("<cls>")
    sep_token_id = tokenizer.token_to_id("<sep>")

    # Generate template:
    type_id = str(1) if cfg_data.use_type_ids else str(0)
    single_template = "$A"
    if cfg_data.include_cls_token_in_corpus:
        single_template = "<cls> " + single_template
    if cfg_data.include_sep_token_in_corpus:
        single_template = single_template + " <sep>"
    tokenizer.post_processor = processors.TemplateProcessing(
        single=single_template,
        pair=f"<cls>:0 $A:0 <sep>:0 $B:{type_id} <sep>:{type_id}",
        special_tokens=[("<cls>", cls_token_id), ("<sep>", sep_token_id)],
    )
    # Wrap into fast codebase
    wrapped_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        model_max_length=cfg_data.seq_length,
        **special_token_args,
    )
    return wrapped_tokenizer
