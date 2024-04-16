import PyPDF2
import re
import json
import logging
import io
from io import BytesIO
import zipfile
import spacy
import nltk
import yaml
from adobe.pdfservices.operation.auth.credentials import Credentials
from adobe.pdfservices.operation.exception.exceptions import ServiceApiException, ServiceUsageException, SdkException
from adobe.pdfservices.operation.pdfops.options.extractpdf.extract_pdf_options import ExtractPDFOptions
from adobe.pdfservices.operation.pdfops.options.extractpdf.extract_element_type import ExtractElementType
from adobe.pdfservices.operation.execution_context import ExecutionContext
from adobe.pdfservices.operation.io.file_ref import FileRef
from adobe.pdfservices.operation.pdfops.extract_pdf_operation import ExtractPDFOperation
from nltk import WordNetLemmatizer, word_tokenize
from nltk.corpus import stopwords
# from spacy.cli import download
# download("en_core_web_sm")
from pypdf import PdfWriter
from transformers import (
    TokenClassificationPipeline,
    AutoModelForTokenClassification,
    AutoTokenizer,
)
from transformers.pipelines import AggregationStrategy
import numpy as np

# with open('C:\\supermat\\supermat\\supermat_app\\config.yml', 'r') as fp:
#     config = yaml.load(fp, yaml.SafeLoader)
# client_id = config['AdobeCredentials']['client_id']
# client_secret = config['AdobeCredentials']['client_secret']
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')


def get_pdf_encoding(file_path):
    with open(file_path.name, 'r', encoding='utf-8'):
        reader = PyPDF2.PdfFileReader(file_path.file)
        try:
            info = reader.getDocumentInfo()
            encoding = info.encoding if hasattr(info, 'encoding') else None
        except UnicodeDecodeError:
            # If the encoding cannot be determined from the metadata, try reading the file with a specific encoding
            encoding = 'utf-8'  # You can change this to another encoding if needed
    return encoding


def extract_keywords(sentence):
    # Extract keywords based on the words in the sentence
    keywords = re.findall(r'\b\w+\b', sentence)
    return keywords


def extract_roles(sentence):
    # Define the patterns for different roles
    patterns = {
        "author": r'(author):',
        "operator": r'(operator):',
        "speaker": r'(speaker):',
        "moderator": r'(moderator):',
        "guest": r'(guest):'
    }
    roles = {}
    for role, pattern in patterns.items():
        match = re.search(pattern, sentence, re.IGNORECASE)
        if match:
            # Extract the role name after the colon
            roles[role] = sentence[match.end():].strip()
    return roles


def extract_timestamp(text):
    # Extract timestamp from text
    pattern = r'(\b\d{1,2}[:]\d{2}\s(?:AM|PM)\sET\b)'
    match = re.search(pattern, text)
    return match.group(1) if match else None


def extract_entities(text_):
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text_)
    entities_ = set()
    for entity in doc.ents:
        entities_.add(entity)
        # spacy.explain(entity.text)
    return entities_


def extract_keywords_spacy(sentence):
    try:
        nlp = spacy.load("en_core_web_sm")
        doc = nlp(sentence)

        # Extract meaningful words (nouns, adjectives, verbs, and adverbs)
        keywords = [token.lemma_ for token in doc if token.pos_ in ["NOUN", "ADJ", "VERB", "ADV"]]

        # Remove duplicates and return
        return list(set(keywords))
    except Exception as e:
        print(str(e))
        raise Exception('Error while extracting the key words: Hugging face extract')


def extract_keywords_nltk(sentence):
    # Tokenize the sentence
    tokens = word_tokenize(sentence.lower())

    # Remove stopwords and punctuation
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word.isalnum() and word not in stop_words]

    # Lemmatize words
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]

    return tokens


# Define keyphrase extraction pipeline
class KeyphraseExtractionPipeline(TokenClassificationPipeline):
    def __init__(self, model, *args, **kwargs):
        super().__init__(
            model=AutoModelForTokenClassification.from_pretrained(model),
            tokenizer=AutoTokenizer.from_pretrained(model),
            *args,
            **kwargs
        )

    def postprocess(self, all_outputs):
        results = super().postprocess(
            all_outputs=all_outputs,
            aggregation_strategy=AggregationStrategy.FIRST,
        )
        return np.unique([result.get("word").strip() for result in results])


def hugging_face_extractor(sentence):
    try:
        model_name = "ml6team/keyphrase-extraction-distilbert-inspec"
        extractor = KeyphraseExtractionPipeline(model=model_name)
        sentence = sentence.replace("\n", " ")
        keywords = extractor(sentence)
        return keywords
    except Exception as e:
        raise Exception('Error while extracting the keywords')


def extract_speaker(text):
    # Extract speaker from text
    pattern = r'(Operator|Speaker|Moderator|Guest Speaker):'
    match = re.search(pattern, text)
    return match.group(1) if match else None


def key_exist_in_json(json_, key_):
    if key_ in json_.keys():
        return True
    return False


def is_pdf(file_path):
    response = file_path.name.lower().endswith('.pdf')
    if response:
        return True
    return False


def parse_file(parsed_json, file_name):
    try:
        section_number = 0
        passage_number = 0
        modified_sentences = []
        figure_count = 0
        output = []
        existing_texts = []

        def add_prefix_to_sentences(section_number_, passage_data, element_):
            sentence_number = 0

            properties = {}
            if key_exist_in_json(element_, 'Bounds'):
                properties['Bounds'] = element_['Bounds']
            if key_exist_in_json(element_, 'Font'):
                properties['Font'] = element_['Font']
            if key_exist_in_json(element_, 'HasClip'):
                properties['HasClip'] = element_['HasClip']
            if key_exist_in_json(element_, 'Lang'):
                properties['Lang'] = element_['Lang']
            if key_exist_in_json(element_, 'ObjectID'):
                properties['ObjectID'] = element_['ObjectID']
            if key_exist_in_json(element_, 'Page'):
                properties['Page'] = element_['Page']
            if key_exist_in_json(element_, 'Path'):
                properties['Path'] = element_['Path']
            if key_exist_in_json(element_, 'TextSize'):
                properties['TextSize'] = element_['TextSize']
            if key_exist_in_json(element_, 'attributes'):
                properties['attributes'] = element_['attributes']
            output.append(
                {
                    'type': 'Text',
                    'structure': f'{section_number_}.{passage_number}.0',
                    'text': passage_data,
                    'key': list(
                        set(hugging_face_extractor(passage_data)).union(set(extract_keywords_spacy(passage_data)))),
                    'properties': properties,
                    'sentences': [],
                    'speaker': extract_roles(passage_data),
                    'document': file_name,
                    'timestamp': extract_timestamp(passage_data)
                }
            )
            # Split the sentences from a passage
            sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', passage_data)
            if "" in sentences:
                sentences.remove("")
            if len(sentences) > 1:
                for sentence in sentences:
                    sentence.strip()
                    sentence_number += 1
                    output[-1]['sentences'].append(
                        {
                            'type': 'Text',
                            'structure': f'{section_number_}.{passage_number}.{sentence_number}',
                            'text': sentence,
                            'key': list(
                                set(hugging_face_extractor(sentence)).union(set(extract_keywords_spacy(sentence)))),
                            'properties': properties
                        })

        for element in parsed_json['elements']:
            # Checking for Headers
            if element['Path'][11] == 'H':
                section_number += 1
                passage_number = 0
                add_prefix_to_sentences(section_number, element['Text'], element)
            # Checking for Passages
            elif element['Path'][11] == 'P':
                passage_number += 1
                if element['Path'].endswith('ParagraphSpan'):
                    text = element['Text']
                    pattern = re.compile(rf"{re.escape(element['Path'])}\[\d+\]")
                    for element1 in parsed_json['elements']:
                        if pattern.match(element1['Path']) and element1['Path'] not in existing_texts:
                            text += element1['Text']
                            existing_texts.append(element1['Path'])
                    add_prefix_to_sentences(section_number, text, element)
                else:
                    if 'ParagraphSpan' not in element['Path']:
                        add_prefix_to_sentences(section_number, element['Text'], element)
            # Checking for Images
            elif 'Figure' in element['Path']:
                passage_number += 1
                output.append(
                    {
                        'type': 'Image',
                        'structure': f'{section_number}.{passage_number}.0',
                        'figure': f'FIGURE {figure_count}',
                        'Bounds': element["Bounds"],
                        'ObjectId': element["ObjectID"],
                        'Page': element["Page"],
                        'Path': element["Path"],
                        'attributes': element["attributes"]
                    }
                )
                figure_count += 1
            # Checking for List Items
            elif element['Path'][11] == 'L':
                match = re.match(r'^.*?/LI', element['Path'])
                if match:
                    length_until_li = len(match.group())
                text1 = element['Text']
                existing_texts.append(element['Path'])
                for element2 in parsed_json['elements']:
                    if element2['Path'][11] == 'L':
                        if element2['Path'][:length_until_li] + "/Lbl" == element['Path'] and element2[
                            'Path'] not in existing_texts:
                            section_number += 1
                            passage_number = 0
                            text1 += element2['Text']
                            existing_texts.append(element2['Path'])
                            add_prefix_to_sentences(section_number, text1, element)
            # All the elements having text
            else:
                if 'Text' in element.keys() and element['Path'] not in existing_texts:
                    text = element['Text']
                    pattern = re.compile(rf"{re.escape(element['Path'])}\[\d+\]")
                    for element1 in parsed_json['elements']:
                        if pattern.match(element1['Path']) and 'Text' in element1.keys() and element1['Path'] not in \
                                existing_texts:
                            text += element1['Text']
                            existing_texts.append(element1['Path'])
                    passage_number += 1
                    add_prefix_to_sentences(section_number, text, element)
        return output
    except Exception as e:
        print(str(e))
        import traceback;
        error = traceback.format_exc()
        print(error)
        raise Exception('Something went wrong while parsing the pdf')


def adobe_pdf_parser(upload_file):
    try:
        # get base path.
        uploaded_file = upload_file.file
        file_data = uploaded_file.read()
        bytes_data = BytesIO(file_data)
        # Initial setup, create credentials instance.
        credentials = Credentials.service_principal_credentials_builder(). \
            with_client_id("3d9beffdbaef4c54be15b6eada4347ff"). \
            with_client_secret("p8e-BTNEazAwCgKMUSn9IkMKVx2I8uEA_0rB"). \
            build()

        # Create an ExecutionContext using credentials and create a new operation instance.
        execution_context = ExecutionContext.create(credentials)
        extract_pdf_operation = ExtractPDFOperation.create_new()

        # Set operation input from a source file.
        # source = FileRef.create_from_local_file(file_data)

        source = FileRef.create_from_stream(bytes_data, "application/pdf")
        extract_pdf_operation.set_input(source)

        # Build ExtractPDF options and set them into the operation
        extract_pdf_options: ExtractPDFOptions = ExtractPDFOptions.builder() \
            .with_element_to_extract(ExtractElementType.TEXT) \
            .build()
        extract_pdf_operation.set_options(extract_pdf_options)

        # Execute the operation.
        result: FileRef = extract_pdf_operation.execute(execution_context)
        # Get json structure as binary
        binary_stream = result.get_as_stream()
        bytes_stream = io.BytesIO(binary_stream)
        with zipfile.ZipFile(bytes_stream, 'r') as zip_file:
            file_name = zip_file.namelist()[0]
            file_content = zip_file.read(file_name)
        json_data = json.loads(file_content)
        parsed_json = parse_file(json_data, upload_file.name)
        return parsed_json
    except (ServiceApiException, ServiceUsageException, SdkException):
        logging.exception("Exception encountered while executing operation")
