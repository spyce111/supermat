import PyPDF2
import re
import json
import logging
import io
from io import BytesIO
import zipfile
from adobe.pdfservices.operation.auth.credentials import Credentials
from adobe.pdfservices.operation.exception.exceptions import ServiceApiException, ServiceUsageException, SdkException
from adobe.pdfservices.operation.pdfops.options.extractpdf.extract_pdf_options import ExtractPDFOptions
from adobe.pdfservices.operation.pdfops.options.extractpdf.extract_element_type import ExtractElementType
from adobe.pdfservices.operation.execution_context import ExecutionContext
from adobe.pdfservices.operation.io.file_ref import FileRef
from adobe.pdfservices.operation.pdfops.extract_pdf_operation import ExtractPDFOperation
from pypdf import PdfWriter



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


def extract_timestamp(text):
    # Extract timestamp from text
    pattern = r'(\b\d{1,2}[:]\d{2}\s(?:AM|PM)\sET\b)'
    match = re.search(pattern, text)
    return match.group(1) if match else None


def extract_speaker(text):
    # Extract speaker from text
    pattern = r'(Operator|Speaker|Moderator|Guest Speaker):'
    match = re.search(pattern, text)
    return match.group(1) if match else None

def generate_json_structure(pdf_path, encoding):
    # Open the PDF file
    with open(pdf_path.name, 'r', encoding=encoding, errors='ignore'):
        # Create a PDF reader object
        reader = PyPDF2.PdfReader(pdf_path.file)

        # Initialize JSON structure
        json_structure = {
            "document": pdf_path.name,
            "speaker": None,
            "timestamp": None,
            "annotations": [],
            "sections": []
        }

        section_number = 1
        paragraph_number = 1
        sentence_number = 1

        # Loop through each page in the PDF
        for page_num in range(len(reader.pages)):
            page = reader.pages[page_num]
            text = page.extractText()

            # Extract timestamp and speaker
            if not json_structure["timestamp"]:
                json_structure["timestamp"] = extract_timestamp(text)

            if not json_structure["speaker"]:
                json_structure["speaker"] = extract_speaker(text)

            # Split the text into sentences
            sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', text)

            # Create a new section
            section_title = "Section {}".format(section_number)
            section = {
                "section_number": str(section_number),
                "section_title": section_title,
                "paragraphs": []
            }

            # Create a new paragraph
            paragraph = {
                "paragraph_number": "{}.{}".format(section_number, paragraph_number),
                "sentences": []
            }

            # Loop through each sentence
            for sentence in sentences:
                if sentence.strip():
                    keywords = extract_keywords(sentence)

                    # Update the overall key list
                    json_structure["annotations"].extend(keywords)

                    # Add sentence to paragraph
                    paragraph["sentences"].append({
                        "sentence_number": "{}.{}.{}".format(section_number, paragraph_number, sentence_number),
                        "text": sentence.strip(),
                        "keywords": keywords
                    })

                    sentence_number += 1

                    # If paragraph has reached its limit, start a new one
                    if sentence_number > 3:
                        section["paragraphs"].append(paragraph)
                        paragraph_number += 1
                        sentence_number = 1
                        paragraph = {
                            "paragraph_number": "{}.{}.{}".format(section_number, paragraph_number, sentence_number),
                            "sentences": []
                        }

            # Add the last paragraph to the section
            if paragraph["sentences"]:
                section["paragraphs"].append(paragraph)

            # Add the section to the JSON structure
            json_structure["sections"].append(section)
            section_number += 1
            paragraph_number = 1
            sentence_number = 1

        return json_structure

def is_pdf(file_path):
    response = file_path.name.lower().endswith('.pdf')
    if response:
        return True
    return False


def parse_file(parsed_json):
    section_number = 0
    passage_number = 0
    modified_sentences = []

    def add_prefix_to_sentences(section_number, passage_data):
        sentence_number = 0
        # Split the sentences from a passage
        sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', passage_data)
        for sentence in sentences:
            if sentence != "":
                modified_sentences.append(
                    {sentence: {'structure': "{}.{}.{}".format(section_number, passage_number, sentence_number)}})
                sentence_number += 1

    for element in parsed_json['elements']:
        if element['Path'][11] == 'H':
            section_number += 1
            passage_number = 0
            add_prefix_to_sentences(section_number, element['Text'])
        elif element['Path'][11] == 'P':
            passage_number += 1
            add_prefix_to_sentences(section_number, element['Text'])
    return modified_sentences


def adobe_pdf_parser(uploaded_file):
    try:
        # get base path.
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
        parsed_json = parse_file(json_data)
        return parsed_json
    except (ServiceApiException, ServiceUsageException, SdkException):
        logging.exception("Exception encountered while executing operation")