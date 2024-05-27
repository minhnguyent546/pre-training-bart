

WORD_PIECE_SUBWORD_PREFIX = '##'

# it's like a underscore, but will not be effected by tokenizer of sacrebleu
# when working with Vietnamese (underscore is used in word segmentation, e.g. underthesea, pyvi)
LOWER_ONE_EIGHTH_BLOCK = u'\u2581'  # "▁"

class SpecialToken:
    MASK = '[MASK]'
    SOS = '<s>'
    EOS = '</s>'
    UNK = '[UNK]'
    PAD = '[PAD]'
    PLACEHOLDER = '[PLACEHOLDER]'

    @classmethod
    def all(cls) -> list[str]:
        return [
            SpecialToken.MASK,
            SpecialToken.SOS,
            SpecialToken.EOS,
            SpecialToken.UNK,
            SpecialToken.PAD,
        ]
