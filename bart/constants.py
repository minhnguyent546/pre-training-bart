

WORD_PIECE_SUBWORD_PREFIX = '##'

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
