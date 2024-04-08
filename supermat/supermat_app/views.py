from django.shortcuts import render
from django.views.generic import View
from django.http.response import JsonResponse, HttpResponse
from .utils import *


# Create your views here.

class BaseView(View):

    def __init__(self):
        self.response = {}
        self.response['res_code'] = '1'
        self.response['res_str'] = 'Processed Successfully'
        self.response['res_data'] = {}


class UploadParse(BaseView):
    def get(self, request):
        print("GET")
        return JsonResponse(data=self.response, safe=False, status=201)

    def post(self, request):
        print('POST')
        # params = request.POST

        # Path to the PDF file
        uploaded_file = request.FILES.get('file_path')
        # if is_pdf(pdf_path):
        #     pass
        # encoding = get_pdf_encoding(pdf_path)
        # if not encoding:
        #     encoding = 'utf-8'
        # Generate the JSON structure
        json_structure = adobe_pdf_parser(uploaded_file)
        print(json_structure)
        self.response['res_data'] = json_structure
        # parsed_data = parse_file('file_name')
        return JsonResponse(data=self.response, safe=False, status=201)
