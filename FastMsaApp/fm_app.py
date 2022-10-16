import logging

from django.http import JsonResponse
from msa_predict import FastMsaApp
from concurrent.futures import ThreadPoolExecutor

import logging, logging.handlers


res =  {
        "data": None,
        "msg": "",
        "code": -1,
}

LOG_FILE = "log/fast_msa_server.log"
handler = logging.handlers.RotatingFileHandler(LOG_FILE, maxBytes = 20*1024*1024, backupCount = 3)
fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s - [%(filename)s:%(lineno)s]"
formatter = logging.Formatter(fmt)  # 实例化formatter  
handler.setFormatter(formatter)     # 为handler添加formatter

handler2 = logging.StreamHandler()
handler.setFormatter(formatter)

logger = logging.getLogger('msaRunner') # 获取名为xzs的logger  
logger.addHandler(handler)
logger.addHandler(handler2);           # 为logger添加handler
logger.setLevel(logging.DEBUG)


# g_threadpool = ThreadPool()
executor = ThreadPoolExecutor(max_workers=32)
fma=FastMsaApp(init_flag=True)


def go_run_predict(input, output, tarnum):
    logger.info("Task Submited...")
    # fma.submit0(input, output)
    executor.submit(fma.run_predict, inputp=input, outputp=output, tarnump=tarnum)
    logger.info("Task finished.")


def query_msa_server_status(request):
    try:
        print("executor._threads len=%d" % len(executor._threads))
        print("executor._qsize=%d" % executor._work_queue.qsize())
        ret={}
        ret['working'] = len(executor._threads)
        ret['waiting'] = executor._work_queue.qsize()
        res['code'] = 0
        res['msg'] = 'MSA Service: {}'.format(fma.status)
        return JsonResponse(res)
    except Exception as ex:
        logger.error(ex)
        res['code'] = -1
        return JsonResponse(res)


def handle_fast_msa(request):
    try:
        if request.method != 'POST':
            res['code'] = 1000001
            res['msg'] = "request method not is POST!"
            return JsonResponse(res)
        input = request.POST['input']
        output = request.POST['output']
        tar_num = request.POST['tarnum']
        ##确保字段不为空
        if input == '' or output == '':
            res['code'] = 1000003
            res['msg'] = "please check body!"
            return JsonResponse(res)
        input_fasta = input
        output_dir = output
        tarnum = int(tar_num)
        go_run_predict(input_fasta, output_dir, tarnum)
        res['code'] = 0
        res['msg'] = 'ok'
        return JsonResponse(res)
    except Exception as ex:
        res['code'] = -1
        res['msg'] = str(ex)
        logging.error(ex)
        return JsonResponse(res)
