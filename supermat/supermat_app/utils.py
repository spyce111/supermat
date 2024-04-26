import re
import json
from io import BytesIO
import zipfile
import spacy
# import nltk
from tqdm import tqdm
from .constants import *
import uuid
import traceback
import string
import difflib
import base64
import io
# import numpy as np
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
import pdfplumber
from bs4 import BeautifulSoup
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
nlp = spacy.load("en_core_web_sm")

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

# def clean_unicode_string(sentence):
#     if not has_different_unicode(sentence):
#         return sentence
#     return unidecode(sentence)

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

def clean_unicode_string(sentence):
    if not has_different_unicode(sentence):
        return sentence
    
    # Replace the problematic Unicode characters with spaces
    cleaned_sentence = ""
    for char in sentence:
        if char == '\u2022\ue020' or char == '\ue020' \
            or char == '\u2022' or char == '\u2019' \
            or char == '\u201c' or char == '\u2013' \
            or char == '\u201d' or '\\u' in repr(char):
            # print("Before: "+char)
            cleaned_sentence += " "
            # print("After: "+cleaned_sentence)
        else:
            cleaned_sentence += char
    
    # print(cleaned_sentence)
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


# Function to create PDF from list of strings
def create_pdf_from_list(pdf_file, lines):
    try:
        doc = SimpleDocTemplate(pdf_file, pagesize=letter)
        styles = getSampleStyleSheet()

        # Define custom style for the PDF content
        custom_style = ParagraphStyle(
            'custom_style',
            parent=styles['Normal'],
            fontName='Helvetica',
            fontSize=10,
            textColor=colors.black,
            spaceAfter=12,
        )

        story = []

        for line in lines:
            # Close unclosed HTML tags
            line = close_unclosed_tags(line)

            try:
                # Try to create a Paragraph with the line
                story.append(Paragraph(line, custom_style))
            except ValueError as e:
                # If ValueError occurs, log the error and add a modified version of the line
                print(f"Error processing line: {e}. Trying to fix the line.")

                # Add a closing </para> tag to try to fix the issue
                fixed_line = f"{line}</para>"
                story.append(Paragraph(fixed_line, custom_style))

            story.append(Spacer(1, 12))  # Add some space between paragraphs

        doc.build(story)
        print(f"PDF created successfully: {pdf_file}")
    except Exception as e:
        raise Exception(f"Error: An error occurred while creating the PDF: {e}")

# Function to create PDF from list of strings
# def create_pdf_from_list(pdf_file, lines):
#     try:
#         doc = SimpleDocTemplate(pdf_file, pagesize=letter)
#         styles = getSampleStyleSheet()
#
#         # Define custom style for the PDF content
#         custom_style = ParagraphStyle(
#             'custom_style',
#             parent=styles['Normal'],
#             fontName='Helvetica',
#             fontSize=10,
#             textColor=colors.black,
#             spaceAfter=12,
#         )
#
#         story = []
#
#         for line in lines:
#             story.append(Paragraph(line, custom_style))
#             story.append(Spacer(1, 12))  # Add some space between paragraphs
#
#         doc.build(story)
#         print(f"PDF created successfully: {pdf_file}")
#     except Exception as e:
#         raise Exception(f"Error: An error occurred while creating the PDF: {e}")

def write_json_file(result,output_file):
    try:
        json_data = json.dumps(result, indent=4)
        with open(output_file, 'w') as json_file:
            json_file.write(json_data)
    except Exception as e:
        raise Exception(str(e))



# def clean_unicode_string(sentence):
#     try:

#         # Replace unicode characters with spaces
#         clean_string = sentence.replace("\ue020", " ")
#         clean_str = clean_string.replace("\u2019","")
#         cleaned_string = clean_str.replace("\u201","")
        
#         # Remove any extra spaces
#         cleaned_string = " ".join(cleaned_string.split())
        
#         return cleaned_string
#     except Exception as e:
#         raise Exception(str(e))

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

def get_image_from_pdf(pdf_path, page_number, bbox):
    """
    Extract an image from a PDF.
    
    Args:
    - pdf_path (str): Path to the local PDF file.
    - page_number (int): Page number from which to extract the image (0-based index).
    - bounds (dict): Dictionary containing 'left', 'top', 'width', and 'height' of the bounding box.
    
    Returns:
    - PIL.Image.Image: Extracted and cropped image.
    """
    try:
        # Convert PDF page to image
        pdf_stream = io.BytesIO(pdf_path)
        pdf_obj = pdfplumber.open(pdf_stream)
        page = pdf_obj.pages[page_number]
        images_in_page = page.images
        for image in images_in_page:
            if int(image['x0']) == int(bbox[0]) and int(image['y0']) == int(bbox[1]) and int(image['x1']) == int(
                    bbox[2]) and int(image['y1']) == int(bbox[3]):
                page_height = page.height
                x0, x1, y0, y1 = [abs(coord) for coord in (image.get('x0',0), image.get('x1',0), image.get('y0',0), image.get('y1',0))]
                # image = images_in_page[0]  # assuming images_in_page has at least one element, only for understanding purpose.
                image_bbox = (x0, min(page_height,abs(page_height - y1)), x1, min(page_height,abs(page_height - y0)))
                cropped_page = page.crop(image_bbox)
                image_obj = cropped_page.to_image(resolution=400)
                return image_obj

    except Exception as e:
        raise Exception(str(e))

def image_to_base64(image):
    """
    Convert the given image to a Base64 encoded string.
    
    Args:
    - image (PIL.Image.Image): Image to be converted.
    
    Returns:
    - str: Base64 encoded string of the image.
    """
    try:
        # Convert image to bytes
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='PNG')
        img_byte_arr = img_byte_arr.getvalue()
        
        # Encode bytes to Base64
        base64_str = base64.b64encode(img_byte_arr).decode('utf-8')
        
        return base64_str
    except Exception as e:
        raise Exception(str(e))


def parse_file(parsed_json, file_name, request_id, pdf_path):
    try:
        SUCCESS_LOG, ERROR_LOG = log_create()
        output = []
        existing_texts = set()
        def add_prefix_to_sentences(section_number_, passage_data, element_):
            keys_to_check = ['Bounds', 'Font', 'HasClip', 'Lang', 'ObjectID', 'Page', 'Path', 'TextSize', 'attributes']
            properties = {key: element_.get(key) for key in keys_to_check if key in element_}
            keywords = set(extract_keywords_spacy(clean_unicode_string(passage_data))).union(extract_meaningful_words(clean_unicode_string(passage_data)))
            # keywords = set(extract_keywords_spacy(passage_data)).union(extract_meaningful_words(passage_data))
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
            if path_type == 'H':
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
            elif 'Figure' in element.get('Path',''):
                passage_number += 1
                bound_img = None
                image_bound = element.get('attributes', {})
                if image_bound:
                    bound_img = image_bound.get('BBox', None)
                if bound_img:
                    figure = get_image_from_pdf(pdf_path, element.get('Page',0), bound_img)
                output.append({
                    'type': 'Image',
                    'structure': f'{section_number}.{passage_number}.0',
                    'figure': f'FIGURE {figure_count}',
                    'figure-object': image_to_base64(figure) if figure else None,
                    'Bounds': element.get('Bounds',''),
                    'ObjectId': element.get('ObjectID',''),
                    'Page': element.get('Page',''),
                    'Path': element.get('Path',''),
                    'attributes': element.get('attributes','')
                })
                figure_count += 1
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

def adobe_pdf_parser(upload_file, request_id, is_original=False):
    try:
        SUCCESS_LOG, ERROR_LOG = log_create()
        if is_original:
            # Read the uploaded file
            uploaded_file_data = upload_file.file.read()
            file_name = upload_file.name
        else:
            with open(upload_file, 'rb') as file:
                uploaded_file_data = file.read()
                file_name = file.name
        
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
            .with_element_to_extract(ExtractElementType.TEXT) \
            .build()
        extract_pdf_operation.set_options(extract_pdf_options)

        # Execute the operation and get the result
        result = extract_pdf_operation.execute(execution_context)
        binary_stream = result.get_as_stream()

        # Extract JSON data from the zip file
        with zipfile.ZipFile(BytesIO(binary_stream), 'r') as zip_file:
            file_name = zip_file.namelist()[0]
            file_content = zip_file.read(file_name)
        
        json_data = json.loads(file_content)

        parsed_json = parse_file(json_data, file_name, request_id, uploaded_file_data)

        if not is_original:
            try:
                if os.path.exists(file.name):
                    os.remove(file.name)
            except Exception as e:
                trace_back = traceback.format_exc()
                ERROR_LOG.critical(f"Error deleting {file.name}: {e}. Traceback: {str(trace_back)}")
        
        SUCCESS_LOG.info(SupermatConstants.ERR_STR_PARSE_PDF.format(request_id=request_id))
        return parsed_json
    except (ServiceApiException, ServiceUsageException, SdkException) as e:
        trace_back = traceback.format_exc()
        ERROR_LOG.critical(f"Adobe Internal Exception: adobe_pdf_parser {str(e)}. Traceback: {str(trace_back)}")
        raise Exception(SupermatConstants.ERR_STRING_ADOBE_PARSER)
    except Exception as e:
        trace_back = traceback.format_exc()
        ERROR_LOG.error(f"Error while parsing the pdf: adobe_pdf_parser {str(e)}. Traceback: {str(trace_back)}")
        raise Exception(SupermatConstants.ERR_STRING_ADOBE_PARSER)