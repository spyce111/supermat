import re
import json
from io import BytesIO
import zipfile
import spacy
from tqdm import tqdm
from .constants import *
import uuid
import traceback
import string
import difflib
import base64
import shutil
import networkx as nx
import json 
import plotly.graph_objects as go
from plotly.offline import plot

# from transformers.pipelines import AggregationStrategy
from adobe.pdfservices.operation.auth.credentials import Credentials
from adobe.pdfservices.operation.exception.exceptions import ServiceApiException, ServiceUsageException, SdkException
from adobe.pdfservices.operation.pdfops.options.extractpdf.extract_pdf_options import ExtractPDFOptions
from adobe.pdfservices.operation.pdfops.options.extractpdf.extract_element_type import ExtractElementType
from adobe.pdfservices.operation.execution_context import ExecutionContext
from adobe.pdfservices.operation.io.file_ref import FileRef
from adobe.pdfservices.operation.pdfops.extract_pdf_operation import ExtractPDFOperation
from nltk import WordNetLemmatizer, word_tokenize
from nltk.corpus import stopwords
from nltk.tag import pos_tag
from adobe.pdfservices.operation.pdfops.options.extractpdf.extract_renditions_element_type import ExtractRenditionsElementType
# from transformers import (
#     TokenClassificationPipeline,
#     AutoModelForTokenClassification,
#     AutoTokenizer,
# )
import os
from PIL import Image as PILImage
from spire.presentation import *
from spire.presentation.common import *
from spire.xls import *
from spire.xls.common import *
from unidecode import unidecode
from bs4 import BeautifulSoup
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
nlp = spacy.load("en_core_web_sm")
import unicodedata

class CustomLogger:

    def set_process_id(self,process_id):
        self.process_id = process_id

    # Creates Log handlers
    def create_log_file(self,logger_name, log_file):
        try:
            logger = logging.getLogger(logger_name)
            logger.setLevel(logging.DEBUG)
            formatter = logging.Formatter('%(asctime)s %(levelname)s '+ self.process_id +' %(message)s')
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
            return logger
        except Exception as e:
            raise Exception(SupermatConstants.ERR_STRING_EXTRACT_ROLES)

    def log_closer(self,logger):
        try:
            handlers = logger.handlers[:]
            for handler in handlers:
                handler.close()
                logger.removeHandler(handler)
        except Exception as e:
            raise Exception(SupermatConstants.ERR_STR_LOG_CLOSE)

def log_create():
    try:
        pid = str(uuid.uuid4().hex)
        custom_process = CustomLogger()
        custom_process.set_process_id(pid)
        SUCCESS_LOG = custom_process.create_log_file(SUCCESS_NAME,SUCCESS_FILE)
        ERROR_LOG = custom_process.create_log_file(ERROR_NAME,ERROR_FILE)
        return SUCCESS_LOG, ERROR_LOG
    except Exception as e:
        raise Exception(SupermatConstants.ERR_STR_LOG_CREATE)

def convert_file_to_pdf(file_path):
    try:
        conversion_map = {
            '.pptx': convert_pptx_to_pdf,
            '.ppt': convert_pptx_to_pdf,
            '.doc': convert_docx_to_pdf,
            '.docx': convert_docx_to_pdf,
            '.odt': convert_docx_to_pdf  # Assuming the same function for odt as docx
        }
        
        _, file_extension = os.path.splitext(file_path.name)
        
        if file_extension in conversion_map:
            return conversion_map[file_extension](file_path)
        else:
            raise Exception('Unsupported file format')
    except Exception as e:
        raise Exception(str(e))

def convert_docx_to_pdf(file_path):
    try:
        document = Document()
        document.LoadFromFile(file_path)
        parameter = ToPdfParameterList()
        # parameter.DisableLink = True
        # parameter.IsEmbeddedAllFonts = True
        temporary_location = os.path.abspath(os.path.join(os.path.dirname(__name__), 'temporary_files'))
        output_file_path = os.path.join(temporary_location, 'new_' + os.path.basename(file_path).replace('.docx', '.pdf'))
        document.SaveToFile(output_file_path, parameter)
        document.Close()
        return True, output_file_path
    except Exception as e:
        print(str(e))
        raise Exception(str(e))

def convert_pptx_to_pdf(file_path):
    try:
        # Create an object of Presentation class
        presentation = Presentation()

        # Load a PPT or PPTX file
        presentation.LoadFromFile(file_path)
        temporary_location = os.path.abspath(os.path.join(os.path.dirname(__name__), 'temporary_files'))
        output_file_path = os.path.join(temporary_location, 'new_' + os.path.basename(file_path).replace('.docx', '.pdf'))
        # Convert the presentation file to PDF and save it
        presentation.SaveToFile(output_file_path, FileFormat.PDF)
        presentation.Dispose()
        return True, output_file_path
    except Exception as e:
        print(str(e))
        raise Exception(str(e))

def resize_image(image_path, max_width=500):
    try:
        with PILImage.open(image_path) as img:
            ratio = max_width / float(img.size[0])
            height = int(float(img.size[1]) * ratio)
            resized_img = img.resize((max_width, height))
            resized_img.save(image_path)
    except Exception as e:
        raise Exception(str(e))

def has_different_unicode(sentence):
    # Create a set to store unique Unicode characters
    unique_chars = set()

    # Iterate over each character in the sentence
    for char in sentence:
        unique_chars.add(char)

    # Compare the length of the set to the length of the sentence
    if len(unique_chars) == len(sentence):
        return False  # All characters are unique
    else:
        return True   # Sentence contains different Unicode characters

#match left and right single quotes
single_quote_expr = re.compile(r'[\u2018\u2019]', re.U)
#match all non-basic latin unicode
unicode_chars_expr = re.compile(r'[\u0080-\uffff]', re.U)

def unicode_to_native_string(unicode_sequence):
    parts = unicode_sequence.split("\\")
    native_string = parts[0]
    for part in parts[1:]:
        if part.startswith("u") or part.startswith("U"):
            code_point = int(part[1:5], 16)
            native_string += chr(code_point)
            if len(part) > 5:
                native_string += "\\" + part[5:]
        else:
            native_string += "\\" + part
    return native_string

def convert_unicode_in_sentence(sentence):
    converted_sentence = ""
    current_sequence = ""
    in_sequence = False
    for char in sentence:
        if char == "\\":
            if in_sequence:
                converted_sequence = unicode_to_native_string(current_sequence)
                converted_sentence += converted_sequence
                current_sequence = ""
            in_sequence = not in_sequence
        elif in_sequence:
            current_sequence += char
        else:
            converted_sentence += char
    # Check if there's a sequence at the end
    if current_sequence:
        converted_sequence = unicode_to_native_string(current_sequence)
        converted_sentence += converted_sequence
    return converted_sentence

# def clean_unicode_string(sentence):
#     if not has_different_unicode(sentence):
#         return sentence
#     # from unicodedata import normalize
#     # import unicodedata
#     # sentence = normalize('NFKD', sentence).encode('ascii','ignore')
#     # # sentence = sentence.encode().decode('unicode_escape')
#     # sentence = sentence.encode('ascii', 'ignore')
#     # sentence = sentence.decode()
#     # sentence = unicodedata.normalize('NFKD', sentence)
#     # sentence = u"".join(c for c in sentence if not unicodedata.combining(c))
#     # sentence  = cleanse_unicode(sentence)
#     sentence = convert_unicode_in_sentence(sentence)
#     # Replace the problematic Unicode characters with spaces
#     cleaned_sentence = ""
#     for char in sentence:
#         if char == '\udc3c' or char == '\ue020' \
#             or char == '\u2022' or char == '\u2019' \
#             or char == '\u201c' or char == '\u2013' \
#             or char == '\ud835' or char == '\udc66' \
#             or char == '\udc61' or char == '\udc51' \
#             or char == '\udc50' or char == '\udc47' \
#             or char == '\u00b7' or char == '\udc51' \
#             or char == '\udc60' or char == '' \
#             or char == '\u201d' or '\\u' in repr(char):
#             print("Before: "+char)
#             cleaned_sentence += " "
#             print("After: "+cleaned_sentence)
#         else:
#             print("Before: "+char)
#             cleaned_sentence += str(char)
#             print("After: "+cleaned_sentence)

    
#     # print(cleaned_sentence)
#     return cleaned_sentence
#     # return sentence

def clean_unicode_string(sentence):
    cleaned_sentence = ""
    for char in sentence:
        try:
            # Try to normalize the character
            normalized_chars = unicodedata.normalize('NFKD', char)
            for normalized_char in normalized_chars:
                # Check if the character is printable ASCII
                if ord(normalized_char) < 128 and normalized_char.isprintable():
                    cleaned_sentence += normalized_char
                else:
                    # If the character is not printable ASCII, replace it with a space
                    cleaned_sentence += ' '
        except ValueError:
            # If normalization fails, replace the character with a space
            cleaned_sentence += ' '

    return cleaned_sentence


# Function to read JSON file and return content as a list of strings
def read_json_file(json_file):
    try:
        with open(json_file, 'r') as file:
            data = json.load(file)
        return [json.dumps(entry, indent=4) for entry in data]
    except FileNotFoundError:
        raise Exception(f"Error: File '{json_file}' not found.")
    except json.JSONDecodeError:
        raise Exception(f"Error: Failed to decode JSON from '{json_file}'.")

# Function to close unclosed HTML tags
def close_unclosed_tags(html_content):
    try:
        soup = BeautifulSoup(html_content, 'html.parser')
        return str(soup)
    except:
        return html_content

def create_pdf_from_list(pdf_file, lines):
    try:
        doc = SimpleDocTemplate(pdf_file, pagesize=letter)
        styles = getSampleStyleSheet()

        story = []

        for line in lines:
            # Append line directly to the story list with styles['Normal']
            line = close_unclosed_tags(line)
            story.append(Paragraph(line, styles['Normal']))
            story.append(Spacer(1, 12))  # Add some space between paragraphs

        doc.build(story)
        print(f"PDF created successfully: {pdf_file}")
    except Exception as e:
        raise Exception(str(e))

def write_json_file(result,output_file):
    try:
        json_data = json.dumps(result, indent=4)
        with open(output_file, 'w') as json_file:
            json_file.write(json_data)
    except Exception as e:
        raise Exception(str(e))

def extract_roles(sentence):
    try:
        # Define the patterns for different roles
        # SUCCESS_LOG, ERROR_LOG = log_create()
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
    except Exception as e:
        trace_back = traceback.format_exc()
        ERROR_LOG.error("Error while Extracing the role: "+str(e)+". Trace Back: "+str(trace_back))
        raise Exception(SupermatConstants.ERR_STRING_EXTRACT_ROLES)

def extract_timestamp(text):
    try:
        # Extract timestamp from text
        SUCCESS_LOG, ERROR_LOG = log_create()
        pattern = rf'{SupermatConstants.TIME_STAMP_REGEX}'
        match = re.search(pattern, text)
        return match.group(1) if match else None
    except Exception as e:
        trace_back = traceback.format_exc()
        ERROR_LOG.error("Error while Extracing the timestamp" + str(e)+". Trace Back: "+ str(trace_back))
        raise Exception(SupermatConstants.ERR_STRING_EXTRACT_TIMESTAMP)

def extract_keywords_spacy(sentence):
    try:
        SUCCESS_LOG, ERROR_LOG = log_create()
        doc = nlp(sentence)
        # Extract words with more than 4 characters, numerics, nouns, verbs, adverbs, and adjectives excluding pronouns
        keywords = [token.text for token in doc if ((token.is_alpha and \
                                                     len(token.text) > 4) \
                                                    or (token.is_digit)) \
                                                    and token.pos_ in \
                                                    ['NUM', 'NOUN', 'VERB', 'ADV', 'ADJ']]
        # SUCCESS_LOG.info("Spacy Extraction Successful")
        return list(set(keywords))
    except Exception as e:
        trace_back = traceback.format_exc()
        ERROR_LOG.error("Error while Extracing the keywords using Spacy: "+str(e)+". Trace Back: "+ str(trace_back))
        raise Exception(SupermatConstants.ERR_STRING_SPACY_EXTRACT)

def extract_meaningful_words(sentence):
    try:
        SUCCESS_LOG, ERROR_LOG = log_create()
        # Tokenize the sentence
        tokens = word_tokenize(sentence)
        # Perform POS tagging
        tagged_tokens = pos_tag(tokens)
        # Extract words with more than 4 characters, numerics, nouns, verbs, adverbs, and adjectives excluding pronouns
        keywords = [word for word, tag in tagged_tokens \
                    if ((tag.startswith(('NN', 'VB', 'JJ', 'RB')) and\
                          len(word) > 4) or (tag == 'CD')) and \
                            word.lower() != 'i']
        # SUCCESS_LOG.info("Meaningfull words Extraction Successful")
        return list(set(keywords))
    except Exception as e:
        trace_back = traceback.format_exc()
        ERROR_LOG.error("Error while Extracing the keywords using NLTK: "+str(e)+". Trace Back: "+ str(trace_back))
        raise Exception(SupermatConstants.ERR_STRING_NLTK_EXTRACT)

def extract_keywords_nltk(sentence):
    try:
        # SUCCESS_LOG, ERROR_LOG = log_create()
        # Tokenize the sentence
        tokens = word_tokenize(sentence.lower())
        # Remove stopwords and punctuation
        stop_words = set(stopwords.words('english'))
        tokens = [word for word in tokens if word.isalnum() and word not in stop_words]
        # Lemmatize words
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(word) for word in tokens]
        # SUCCESS_LOG.info("NLTK extraction Successful")
        return tokens
    except Exception as e:
        trace_back = traceback.format_exc()
        # ERROR_LOG.error("Error while Extracing the keywords using NLTK: "+str(e)+". Trace Back: "+ str(trace_back))
        raise Exception("Error while Extracing the keywords using NLTK: "+str(e)+". Trace Back: "+ str(trace_back))

# Define keyphrase extraction pipeline
# class KeyphraseExtractionPipeline(TokenClassificationPipeline):
#     def __init__(self, model_name, *args, **kwargs):
#         super().__init__(
#             model=AutoModelForTokenClassification.from_pretrained(model_name),
#             tokenizer=AutoTokenizer.from_pretrained(model_name),
#             *args,
#             **kwargs
#         )

#     def postprocess(self, all_outputs):
#         results = super().postprocess(
#             all_outputs=all_outputs,
#             aggregation_strategy=AggregationStrategy.FIRST,
#         )
#         return np.unique([result.get("word").strip() for result in results])

# # Load the model outside the function to reuse it
# MODEL_NAME = SupermatConstants.HUGGING_FACE_MODEL_NAME
# EXTRACTOR = KeyphraseExtractionPipeline(MODEL_NAME)

# def hugging_face_extractor(sentence):
#     try:
#         cleaned_sentence = sentence.replace("\n", " ")
#         # Process the sentence
#         keywords = EXTRACTOR(cleaned_sentence)
#         return keywords
#     except Exception as e:
#         trace_back = traceback.format_exc()
#         ERROR_LOG.error(f"Error while extracting keywords using Hugging Face: {str(e)}. Trace Back: {str(trace_back)}")
#         raise Exception(SupermatConstants.ERR_STRING_HUGGING_FACE_EXTRACT)


def remove_similar_words(word_list, threshold=0.8):
    try:
        SUCCESS_LOG, ERROR_LOG = log_create()
        cleaned_list = []
        word_set = set(word_list)

        for word in word_set:
            if all(difflib.SequenceMatcher(None, word, w).ratio() < threshold for w in cleaned_list):
                cleaned_list.append(word)
        return list(set(cleaned_list))
    except Exception as e:
        trace_back = traceback.format_exc()
        ERROR_LOG.error(f"Error while removing the similar words from the keywords : {str(e)}. Trace Back: {str(trace_back)}")
        raise Exception(SupermatConstants.ERR_STR_REMOVE_SPECIAL_CHAR)

def extract_annotations(sentence):
    try:
        # Process the sentence
        doc = nlp(sentence)
        # Extract annotations (entities)
        annotations = [ent.text for ent in doc.ents]
        return list(set(annotations))
    except Exception as e:
        trace_back = traceback.format_exc()
        ERROR_LOG.error(f"Error extracting annotations: {str(e)}. Trace Back: {str(trace_back)}")
        raise Exception(SupermatConstants.ERR_STR_EXTRACt_ANNOT)

def remove_special_characters_from_list(lst):
    try:
        SUCCESS_LOG, ERROR_LOG = log_create()
        # Create translation table
        table = str.maketrans(string.punctuation, ' ' * len(string.punctuation))
        # Remove special characters from each string in the list
        cleaned_lst = [s.translate(table) for s in lst]
        # Remove extra spaces
        cleaned_lst = [' '.join(s.split()) for s in cleaned_lst]
        # Remove empty strings
        cleaned_lst = [s for s in cleaned_lst if s]
        # Remove similar words
        cleaned_lst = remove_similar_words(list(set(cleaned_lst)))
        return cleaned_lst
    except Exception as e:
        trace_back = traceback.format_exc()
        ERROR_LOG.error(f"Error while removing the special characters from the keywords: {str(e)}. Trace Back: {str(trace_back)}")
        raise Exception(SupermatConstants.ERR_STR_REMOVE_SPECIAL_CHAR)

def is_pdf(file_path):
    try:
        SUCCESS_LOG, ERROR_LOG = log_create()
        response = file_path.name.lower().endswith('.pdf')
        if file_path.content_type == 'application/pdf' and response:
            return True
        return False
    except Exception as e:
        trace_back = traceback.format_exc()
        ERROR_LOG.error("Error while checking the type: "+ str(e)+ ". Trace Back: "+str(trace_back))
        raise Exception(SupermatConstants.ERR_STRING_PDF_CHECK)

def parse_file(parsed_json, file_name, request_id, pdf_path,image_json, image_count):
    try:
        SUCCESS_LOG, ERROR_LOG = log_create()
        output = []
        existing_texts = set()
        def add_prefix_to_sentences(section_number_, passage_data, element_):
            keys_to_check = ['Bounds', 'Font', 'HasClip', 'Lang', 'ObjectID', 'Page', 'Path', 'TextSize', 'attributes']
            properties = {key: element_.get(key) for key in keys_to_check if key in element_}
            keywords = set(extract_keywords_spacy(clean_unicode_string(passage_data))).union(extract_meaningful_words(clean_unicode_string(passage_data)))
            # keywords = set(extract_keywords_spacy(passage_data)).union(extract_meaningful_words(passage_data))
            passage_data = clean_unicode_string(passage_data)
            output.append({
                'type': 'Text',
                'structure': f'{section_number_}.{passage_number}.0',
                'text': clean_unicode_string(passage_data),
                'key': list(keywords),
                'properties': properties,
                'sentences': [],
                'speaker': extract_roles(passage_data),
                'document': file_name,
                'timestamp': extract_timestamp(passage_data),
                'annotations': extract_annotations(clean_unicode_string(passage_data))
            })
            sentences = re.split(rf'{SupermatConstants.SPLIT_SENTENCE_REGEX}', passage_data)
            sentences = [s.strip() for s in sentences if s]
            if len(sentences) > 1:
                for i, sentence in enumerate(sentences, start=1):
                    sentence = clean_unicode_string(sentence)
                    keywords = set(extract_keywords_spacy(sentence)).union(extract_meaningful_words(sentence))
                    output[-1]['sentences'].append({
                        'type': 'Text',
                        'structure': f'{section_number_}.{passage_number}.{i}',
                        'text': clean_unicode_string(sentence),
                        'key': list(keywords),
                        'properties': properties
                    })
        section_number = 0
        passage_number = 0
        figure_count = 0
        for element in tqdm(parsed_json.get('elements',''), desc="Processing"):
            path_type = element.get('Path','')[11]
            
            if 'filePaths' in element.keys():
                passage_number += 1
                figure = None
                print('Total Images using Figure directory: ', str(len(image_json)))
                reverse_lookup = {val: (key, dictionary) for dictionary in image_json for key, val in dictionary.items()}
                print(figure_count)
                if figure_count in reverse_lookup:
                    key, found_dict = reverse_lookup.get(figure_count)
                    figure =  found_dict.get('base64_string')
                if not figure:
                    print('-----------------START---------------------')
                    # print("Image Count from Json Dict", image_data.get('image_number'))
                    print('Missing Image')
                    print(f"Figure count of {figure_count} ")
                    print("-------------------END---------------------")

                output.append({
                    'type': 'Image',
                    'structure': f'{section_number}.{passage_number}.0',
                    'figure': f'FIGURE {figure_count}',
                    'figure-object': figure,
                    'Bounds': element.get('Bounds',''),
                    'ObjectId': element.get('ObjectID',''),
                    'Page': element.get('Page',''),
                    'Path': element.get('Path',''),
                    'attributes': element.get('attributes','')
                })
                print('Figure Count from adobe: ', figure_count)
                figure_count += 1
            elif path_type == 'H':
                section_number += 1
                passage_number = 0
                add_prefix_to_sentences(section_number, element.get('Text',''), element)
            elif path_type == 'P':
                passage_number += 1
                text = element.get('Text','')
                if element.get('Path','').endswith('ParagraphSpan'):
                    pattern = re.compile(rf"{re.escape(element.get('Path',''))}\[\d+\]")
                    text += ''.join(elem.get('Text','') for elem in parsed_json.get('elements','') if pattern.match(elem.get('Path','')) and elem.get('Path','') not in existing_texts)
                add_prefix_to_sentences(section_number, text, element)
            elif path_type == 'L':
                match = re.match(r'^.*?/LI', element.get('Path',''))
                if match:
                    length_until_li = len(match.group())
                    text = element.get('Text','')
                    for elem in parsed_json.get('elements',''):
                        if elem.get('Path','')[11] == 'L' and elem.get('Path','')[:length_until_li] + "/Lbl" == element.get('Path',''):
                            section_number += 1
                            passage_number = 0
                            text += elem.get('Text','')
                    add_prefix_to_sentences(section_number, text, element)

            else:
                if 'Text' in element and element.get('Path','') not in existing_texts:
                    text = element.get('Text','')
                    pattern = re.compile(rf"{re.escape(element.get('Path',''))}\[\d+\]")
                    text += ''.join(elem.get('Text','') for elem in parsed_json.get('elements','') if pattern.match(elem.get('Path','')) and 'Text' in elem and elem.get('Path','') not in existing_texts)
                    passage_number += 1
                    add_prefix_to_sentences(section_number, text, element)
        SUCCESS_LOG.info(SupermatConstants.ERR_STR_PARSE_PDF.format(request_id=request_id))
        return output
    except Exception as e:
        trace_back = traceback.format_exc()
        ERROR_LOG.error(f"Error while parsing the file: parse_file {str(e)}. Traceback: {str(trace_back)}")
        raise Exception(SupermatConstants.ERR_STING_PDF_PARSER)

def merge_directories(source_dir1, source_dir2, destination_dir):
    try:
        # Create the destination directory if it doesn't exist
        if not os.path.exists(destination_dir):
            os.makedirs(destination_dir)
        if not os.path.exists(source_dir1):
            os.makedirs(source_dir1)
        if not os.path.exists(source_dir2):
            os.makedirs(source_dir2)
        
        # Check if the source directories exist
        if os.path.exists(source_dir1):
            # Iterate over the files in the first source directory
            for root, dirs, files in os.walk(source_dir1):
                for file in files:
                    # Construct the source and destination paths
                    source_path = os.path.join(root, file)
                    destination_path = os.path.join(destination_dir, file)
                    # Copy the file to the destination directory
                    shutil.copy2(source_path, destination_path)
        
        if os.path.exists(source_dir2):
            # Iterate over the files in the second source directory
            for root, dirs, files in os.walk(source_dir2):
                for file in files:
                    # Construct the source and destination paths
                    source_path = os.path.join(root, file)
                    destination_path = os.path.join(destination_dir, file)
                    # Copy the file to the destination directory
                    shutil.copy2(source_path, destination_path)
    except FileNotFoundError as e:
        print(f"Error: {e}")


def image_to_base64(image_path):
    try:
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
        return encoded_string
    except Exception as e:
        raise Exception(str(e))

def extract_integer(filename):
    # Define a regular expression pattern to match integers in filenames
    pattern = r'\d+'
    # Use re.findall to extract all integers from the filename
    integers = re.findall(pattern, filename)
    # If integers are found, convert the first one to an integer and return it
    if integers:
        return int(integers[0])
    else:
        return None  # Return None if no integers are found

def images_in_directory_to_base64(directory_path, destination_path):
    try:
        base64_images = []
        image_count = 0
        directory_name, _ = os.path.split(directory_path)
        table_directory = directory_name+'/tables'
        # Get a list of all files in the directory sorted based on the integers in their filenames
        
        merge_directories(directory_path, table_directory, DESTINATION_DIRECTORY)
        sorted_files = sorted(os.listdir(DESTINATION_DIRECTORY), key=extract_integer)
        for filename in sorted_files:
            image_path = os.path.join(DESTINATION_DIRECTORY, filename)
            if os.path.isfile(image_path):  # Check if it's a file
                print("Processing file:", image_path)  # Add this line for debug
                with open(image_path, 'rb') as file:
                    base64_string = base64.b64encode(file.read()).decode('utf-8')
                # Check if the extracted integer matches the image count
                extracted_integer = extract_integer(filename)
                if extracted_integer != image_count:
                    image_count = extracted_integer
                    base64_images.append({'image_number': extracted_integer, 'base64_string': base64_string, 'file_name': filename})
                else:
                    base64_images.append({'image_number': image_count, 'base64_string': base64_string, 'file_name': filename})
                print(f'Inside the base 64 function {image_count} for the file_name : {filename}')
                image_count += 1
        
                
        try:
            if os.path.exists(directory_path):
                directory_name, _ = os.path.split(directory_path)
                shutil.rmtree(directory_path)
                shutil.rmtree(table_directory)
                shutil.rmtree(DESTINATION_DIRECTORY)
            if os.path.exists(destination_path):
                os.remove(destination_path)
        except Exception as e:
            print(str(e))
            raise Exception(str(e))
        print("Total images processed:", len(base64_images))
        return base64_images, image_count  # Return both base64 images and total count
    except Exception as e:
        raise Exception(str(e))


def adobe_pdf_parser(upload_file, request_id, is_original=False):
    try:
        SUCCESS_LOG, ERROR_LOG = log_create()
        if is_original:
            # Read the uploaded file
            uploaded_file_data = upload_file.file.read()
            file_name_ = upload_file.name
        else:
            with open(upload_file, 'rb') as file:
                uploaded_file_data = file.read()
                file_name_ = file.name
        
        # Setup Adobe credentials
        credentials = Credentials.service_principal_credentials_builder() \
            .with_client_id(SupermatConstants.ADOBE_CLIENT_ID) \
            .with_client_secret(SupermatConstants.ADOBE_CLIENT_SECRET) \
            .build()

        # Create an ExecutionContext using credentials and create a new operation instance
        execution_context = ExecutionContext.create(credentials)
        extract_pdf_operation = ExtractPDFOperation.create_new()

        # Set operation input from a source file
        source = FileRef.create_from_stream(BytesIO(uploaded_file_data), "application/pdf")
        extract_pdf_operation.set_input(source)

        # Build ExtractPDF options and set them into the operation
        extract_pdf_options = ExtractPDFOptions.builder() \
            .with_element_to_extract(ExtractElementType.TEXT)\
            .with_elements_to_extract_renditions([ExtractRenditionsElementType.TABLES,
                                              ExtractRenditionsElementType.FIGURES]) \
            .build()
        extract_pdf_operation.set_options(extract_pdf_options)
        
        # Execute the operation and get the result
        result = extract_pdf_operation.execute(execution_context)
        binary_stream = result.get_as_stream()
        source_path = result._file_path
        base_name, file_extension = os.path.splitext(source_path)
        file_base_name = base_name.split('/')[-1]
        data_path_zip = os.path.abspath(os.path.join(os.path.dirname(__name__)))
        destination_path = data_path_zip+'/'+str(file_base_name)+request_id+file_extension
        shutil.copy(source_path,destination_path)
        
        with zipfile.ZipFile(destination_path, 'r') as zip_ref:
            zip_ref.extractall(data_path_zip+'/')
        figure_directory = data_path_zip+ '/figures'
        image_json, image_count= images_in_directory_to_base64(figure_directory, destination_path)
        # print(len(image_json))
        # Extract JSON data from the zip file
        with zipfile.ZipFile(BytesIO(binary_stream), 'r') as zip_file:
            file_name = zip_file.namelist()[0]
            file_content = zip_file.read(file_name)
        
        json_data = json.loads(file_content)

        parsed_json = parse_file(json_data, file_name_, request_id, uploaded_file_data, image_json, image_count)
        # pdf_json_text = parse_pdf_file(json_data, file_name, request_id, uploaded_file_data)
        if not is_original:
            try:
                if os.path.exists(file.name):
                    os.remove(file.name)
            except Exception as e:
                trace_back = traceback.format_exc()
                ERROR_LOG.critical(f"Error deleting {file.name}: {e}. Traceback: {str(trace_back)}")
        
        SUCCESS_LOG.info(SupermatConstants.ERR_STR_PARSE_PDF.format(request_id=request_id))
        # write_json_file(pdf_json_text,'output_file.json')
        # lines = read_json_file('output_file.json')
        return parsed_json
    except (ServiceApiException, ServiceUsageException, SdkException) as e:
        trace_back = traceback.format_exc()
        ERROR_LOG.critical(f"Adobe Internal Exception: adobe_pdf_parser {str(e)}. Traceback: {str(trace_back)}")
        raise Exception(SupermatConstants.ERR_STRING_ADOBE_PARSER)
    except Exception as e:
        trace_back = traceback.format_exc()
        ERROR_LOG.error(f"Error while parsing the pdf: adobe_pdf_parser {str(e)}. Traceback: {str(trace_back)}")
        raise Exception(SupermatConstants.ERR_STRING_ADOBE_PARSER)
    


def create_graph(data,output_file):
    try:
        
        SUCCESS_LOG, ERROR_LOG = log_create()



# Load JSON data from the file
        with open(data,encoding='utf8') as f:
            data = json.load(f)

        # sentences=data["results"]
        # print(data)

        

        node_list_head = [d['key'] for d in data if 'key' in d and d['key']]
        # print(node_list_head)
        edges_list = [d['structure'] for d in data if 'key'in d and d['key'] in node_list_head and 'structure' in d]
        # node_list_tail=[]
        # print(len(node_list_head))
        # print(len(edges_list))
        
        

        G = nx.DiGraph()

    # Add edges to the graph
        for node, edge in zip(node_list_head, edges_list):
            if isinstance(node, list):
                node = str(tuple(node))
            if isinstance(edge, list):
                edge = str(tuple(edge))
            G.add_edge(node, edge)



        # Get positions for the nodes in G
        pos = nx.spring_layout(G)

        # Create Edges
        edge_x = []
        edge_y = []
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.append(x0)
            edge_x.append(x1)
            edge_x.append(None)
            edge_y.append(y0)
            edge_y.append(y1)
            edge_y.append(None)

        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=0.5, color='#888'),
            hoverinfo='none',
            mode='lines')

        # Create Nodes
        node_x = [pos[i][0] for i in G.nodes()]
        node_y = [pos[i][1] for i in G.nodes()]

        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers',
            hoverinfo='text',
            marker=dict(
                showscale=True,
                colorscale='YlGnBu',
                reversescale=True,
                color=[],
                size=10,
                colorbar=dict(
                    thickness=15,
                    title='Node Connections',
                    xanchor='left',
                    titleside='right'
                ),
                line_width=2))

        # Add hover text
        node_text = [str(i) for i in G.nodes()]
        node_trace.text = node_text

        # Color node points by the number of connections.
        node_adjacencies = []
        for node, adjacencies in enumerate(G.adjacency()):
            node_adjacencies.append(len(adjacencies[1]))

        node_trace.marker.color = node_adjacencies

        fig = go.Figure(data=[edge_trace, node_trace],
                layout=go.Layout(
                    title='<br>Network graph made with Python',
                    titlefont_size=16,
                    showlegend=False,
                    hovermode='closest',
                    margin=dict(b=20,l=5,r=5,t=40),
                    annotations=[ dict(
                        
                        showarrow=False,
                        xref="paper", yref="paper",
                        x=0.005, y=-0.002 ) ],
                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                    )
        plot(fig, filename=output_file)
    except Exception as e:
        trace_back = traceback.format_exc()
        ERROR_LOG.error("Error while generating the Knowledge graph" + str(e)+". Trace Back: "+ str(trace_back))
        raise Exception(SupermatConstants.ERR_STR_GRAPH_CREATE)

    