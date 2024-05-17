import uuid
from django.shortcuts import render
from django.views.generic import View
from django.http.response import JsonResponse, HttpResponse
from .utils import *
from .constants import SupermatConstants
import traceback
from datetime import datetime
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
            SUCCESS_LOG, ERROR_LOG = log_create()
            # Path to the PDF file
            start = datetime.now()
            file_path = request.FILES.get('file_path')
            DATA_PATH = os.path.abspath(os.path.join(os.path.dirname(__name__)))
            if not os.path.exists(DATA_PATH):
                os.makedirs(DATA_PATH, exist_ok=True)
            temporary_file_location = os.path.join(DATA_PATH)
            request_id = str(uuid.uuid4().hex)
            is_original = True
            if not is_pdf(file_path):
                is_original = False
                file_path = convert_file_to_pdf(file_path)
            result = adobe_pdf_parser(file_path, request_id, is_original)
            end = datetime.now()
            print("Execution Time: "+ str(end-start))
            file_base_name, file_extension = os.path.splitext(file_path.name )
            output_file = temporary_file_location+'/'+str(file_base_name)+request_id+'_'+'response.json'
            write_json_file(result,output_file)
            lines = read_json_file(output_file)
            pdf_file = temporary_file_location+'/'+str(file_base_name)+request_id+'_'+'response.pdf'
            # Create PDF from list of strings
            # pdf_start = datetime.now()
            # create_pdf_from_list(pdf_file, lines)
            # pdf_end = datetime.now()
            # print(f"PDF creation time: {str((pdf_end-pdf_start))}")
            graph_file_path =f"{temporary_file_location}/{file_base_name}_{request_id}_knowledge_graph.html"
            create_graph(output_file,graph_file_path)
            self.response['res_data']['results'] = result
            self.response['res_data']['request_id'] = {f"{request_id}": f"{file_path}"}
            self.response['res_data']['response_file'] = {'response_json':output_file,"response_pdf":pdf_file,"response_graph":graph_file_path}

            return JsonResponse(data=self.response, safe=False, status=200)
        except Exception as e:
            self.response['res_data'] = {}
            self.response['res_str'] = SupermatConstants.ERR_STR_GENERIC
            trace_back = traceback.format_exc()
            ERROR_LOG.error("Error while UploadParse : "+ str(e)+ ". Trace Back: "+str(trace_back)) 
            return JsonResponse(data=self.response, safe=True, status=400)
