import uuid
from django.shortcuts import render
from django.views.generic import View
from django.http.response import JsonResponse, HttpResponse
from .utils import *
from .constants import Constants

# Create your views here.
class BaseView(View):

    def __init__(self):
        self.response = {'res_code': '1', 'res_str': 'Processed Successfully', 'res_data': {}}


class UploadParse(BaseView):
    def get(self, request):
        print("GET")
        return JsonResponse(data=self.response, safe=False, status=201)

    def post(self, request):
        try:
            # Path to the PDF file
            from datetime import datetime
            start = datetime.now()
            pdf_path = request.FILES.get('file_path')
            request_id = str(uuid.uuid4().hex)
            if not is_pdf(pdf_path):
                raise Exception('Please Provide a PDF file')
            result = adobe_pdf_parser(pdf_path, request_id)
            end = datetime.now()
            print("Execution Time: "+ str(end-start))
            # Generate the JSON structure
            self.response['res_data']['results'] = result
            self.response['res_data']['request_id'] = {f"{request_id}": f"{pdf_path}"}
            return JsonResponse(data=self.response, safe=False, status=200)
        except Exception as e:
            self.response['res_data'] = {}
            self.response['res_str'] = Constants.ERR_STR_GENERIC
            return JsonResponse(data=self.response, safe=True, status=400)
