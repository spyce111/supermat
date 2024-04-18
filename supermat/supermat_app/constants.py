import logging
ERROR_NAME = 'Error_Log'
LOG_PATH = '/rest/logs'
ERROR_FILE = '/rest/logs/supermat.log'
SUCCESS_NAME = 'Success_Log'
SUCCESS_FILE = '/rest/logs/supermat.log'
ERROR_LOG = logging.getLogger(ERROR_NAME)
SUCCESS_LOG = logging.getLogger(SUCCESS_NAME)

class Constants:
    
    def __init__(self):
        pass

    TIME_STAMP_REGEX = "(\b\d{1,2}[:]\d{2}\s(?:AM|PM)\sET\b)"
    HUGGING_FACE_MODEL_NAME = "ml6team/keyphrase-extraction-distilbert-inspec"
    SPEAKER_REGEX = "(Operator|Speaker|Moderator|Guest Speaker):"
    SPLIT_SENTENCE_REGEX = "(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s"
    ADOBE_CLIENT_ID = "3d9beffdbaef4c54be15b6eada4347ff"
    ADOBE_CLIENT_SECRET = "p8e-BTNEazAwCgKMUSn9IkMKVx2I8uEA_0rB"
    ERR_STRING_EXTRACT_ROLES = "Error while extracting roles"
    ERR_STRING_EXTRACT_TIMESTAMP = "Error while extracting Timestamp"
    ERR_STRING_HUGGING_FACE_EXTRACT = 'Error while extracting the key words: Hugging face extract'
    ERR_STRING_NLTK_EXTRACT = "Error while extracting the keywords using NLTK"
    ERR_STRING_SPACY_EXTRACT = "Error while extracting the keywords using Spacy"
    ERR_STRING_SPEAKER_EXTRACT = "Error while extracting speaker"
    ERR_STRING_PDF_CHECK = "Error while cheking the PDF type"
    ERR_STING_PDF_PARSER = "Error while parsing the adobe generated json"
    ERR_STRING_ADOBE_PARSER = 'Something went wrong while extracting pdf structure from adobe'
    ERR_STR_LOG_CLOSE = "Error while closing the log file"
    ERR_STR_LOG_CREATE = "Error while creating the log file"
    ERR_STR_PARSE_PDF= "JSON generated Successfully, request_id: {request_id}"
    ERR_STR_REMOVE_SPECIAL_CHAR = "Error while removing the special chanracters from the keywords"
    ERR_STR_GENERIC = "Something went wrong please try again later"
