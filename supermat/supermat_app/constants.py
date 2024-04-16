class Constants:
    def __init__(self):
        pass
    TIME_STAMP_REGEX = "(\b\d{1,2}[:]\d{2}\s(?:AM|PM)\sET\b)"
    HUGGING_FACE_MODEL_NAME = "ml6team/keyphrase-extraction-distilbert-inspec"
    SPEAKER_REGEX = "(Operator|Speaker|Moderator|Guest Speaker):"
    SPLIT_SENTENCE_REGEX = "(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s"
    ADOBE_CLIENT_ID = "3d9beffdbaef4c54be15b6eada4347ff"
    ADOBE_CLIENT_SECRET = "p8e-BTNEazAwCgKMUSn9IkMKVx2I8uEA_0rB"