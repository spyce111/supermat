from django.shortcuts import render
from django.views.generic import View
from django.http.response import JsonResponse,HttpResponse
from .utils import *
# Create your views here.

class BaseView(View):

    def __init__(self):
        self.response = {}
        self.response['res_code'] = '1'
        self.response['res_str'] = 'Processed Successfully'
        self.response['res_data'] = {}


class UploadParse(BaseView):
    def get(self,request):
        print("GET")
        return JsonResponse(data=self.response, safe=False,status=201)
    def post(self,request):
        try:
            # Path to the PDF file
            pdf_path = request.FILES.get('file_path')
            if not is_pdf(pdf_path):
                raise Exception('Please Provide a PDF file')
            result = adobe_pdf_parser(pdf_path.file)
            # import pdb;pdb.set_trace()
            # Generate the JSON structure
            # for res in result:
            #     for key,val in res.items():   
            #         res[key]['document'] = str(pdf_path.name)
            for res in result:
                key = next(iter(res.keys()))
                keywords = extract_keywords(key)
                speaker = extract_roles(key)
                if not speaker:
                    speaker = {'speaker':'author'}
                res[key]['document'] = str(pdf_path.name)
                res[key]['keywords'] = keywords
                # extract = hugging_face_extraction(key)
                # print(extract)
                res[key]['speaker'] =  [j for i,j in speaker.items()][0]
            # result.append({'document': str(pdf_path.name)})
            self.response['res_data'] = result
            return JsonResponse(data=self.response, safe=False,status=200)
        except Exception as e:
            self.response['res_data'] = {}
            self.response['res_str'] = 'Somethingwent wrong please try again later'
            return JsonResponse(data=self.response, safe=True,status=400)