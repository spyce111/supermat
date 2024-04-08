import PyPDF2
import re
import json
from io import BytesIO
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

# def generate_json_structure(pdf_path, encoding):
#     # Open the PDF file
#     with open(pdf_path.name, 'r', encoding=encoding, errors='ignore'):
#         # Create a PDF reader object
#         reader = PyPDF2.PdfFileReader(pdf_path.file)
        
#         # Initialize JSON structure
#         json_structure = {
#             "document": pdf_path.name,
#             "speaker": None,
#             "timestamp": None,
#             "annotations": [],
#             "sections": []
#         }

#         section_number = 1
#         paragraph_number = 1
#         sentence_number = 1

#         # Loop through each page in the PDF
#         for page_num in range(reader.numPages):
#             page = reader.getPage(page_num)
#             text = page.extractText()

#             # Extract timestamp and speaker
#             if not json_structure["timestamp"]:
#                 json_structure["timestamp"] = extract_timestamp(text)
            
#             if not json_structure["speaker"]:
#                 json_structure["speaker"] = extract_speaker(text)

#             # Split the text into sentences
#             sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', text)

#             # Create a new section
#             section_title = "Section {}".format(section_number)
#             section = {
#                 "section_number": str(section_number),
#                 "section_title": section_title,
#                 "paragraphs": []
#             }

#             # Create a new paragraph
#             paragraph = {
#                 "paragraph_number": "{}.{}".format(section_number, paragraph_number),
#                 "sentences": []
#             }

#             # Loop through each sentence
#             for sentence in sentences:
#                 if sentence.strip():
#                     keywords = extract_keywords(sentence)
                    
#                     # Update the overall key list
#                     json_structure["annotations"].extend(keywords)

#                     # Add sentence to paragraph
#                     paragraph["sentences"].append({
#                         "sentence_number": "{}.{}.{}".format(section_number, paragraph_number,sentence_number),
#                         "text": sentence.strip(),
#                         "keywords": keywords
#                     })

#                     sentence_number += 1

#                     # If paragraph has reached its limit, start a new one
#                     if sentence_number > 3:
#                         section["paragraphs"].append(paragraph)
#                         paragraph_number += 1
#                         sentence_number = 1
#                         paragraph = {
#                             "paragraph_number": "{}.{}.{}".format(section_number, paragraph_number),
#                             "sentences": []
#                         }

#             # Add the last paragraph to the section
#             if paragraph["sentences"]:
#                 section["paragraphs"].append(paragraph)

#             # Add the section to the JSON structure
#             json_structure["sections"].append(section)
#             section_number += 1
#             paragraph_number = 1
#             sentence_number = 1

#         return json_structure

def is_pdf(file_path):
    response = file_path.name.lower().endswith('.pdf')
    if response:
        return True
    return False


# def parse_file(json_file_path):
#     with open(json_file_path,"r") as fp:
#         parsed_json = json.load(fp)
#     section_number = 0
#     passage_number = 0
#     modified_sentences = []


#     def add_prefix_to_sentences(section_number, passage_data):
#         sentence_number = 0
#         sentences = passage_data.split('.')
#         for sentence in sentences:
#             modified_sentences.append({sentence: {'structure': "{}.{}.{}".format(section_number,passage_number,sentence_number)}})
#             sentence_number += 1

#     for element in parsed_json['elements']:
#         if element['Path'][11] == 'H':
#             section_number += 1
#             passage_number = 0
#             add_prefix_to_sentences(section_number, element['Text'])
#         elif element['Path'][11] == 'P':
#             passage_number += 1
#             add_prefix_to_sentences(section_number, element['Text'])

#     print(modified_sentences)
#     return modified_sentences
