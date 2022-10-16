from concurrent.futures.thread import ThreadPoolExecutor

class ThreadPool(object):
    def __init__(self):
        # 线程池
        self.executor = ThreadPoolExecutor(1)
        # 用于存储每个项目批量任务的期程
        self.future_dict = {}

    def _sumbit(self, fn):
        try:
            self.executor.submit(fn)
        except Exception as ex:
            import traceback
            traceback.print_exc(ex)

    def check(self):
        print("executor._threads len=%d" % len(self.executor._threads))
        print("executor._qsize=%d" % self.executor._work_queue.qsize())
        ret={}
        ret['working'] = len(self.executor._threads)
        ret['waiting'] = self.executor._work_queue.qsize()
        print(ret)
        return ret

    def __del__(self):
        self.executor.shutdown()
