import time
import asyncio


class Meter():
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f',id=None):
        self.name = name
        self.fmt = fmt
        self.reset()
        self.id = id

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.history = []
        self.epoch = 0
        self._start_time = 0

        return self
    def reset_time(self):
        self._start_time = time.time()

    def update(self, val,epoch=0, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        self.epoch = epoch
        self.time = time.time() - self._start_time
        self.history.append({
            "epoch":self.epoch,
            "val":self.val,
            "sum":self.sum,
            "avg":self.avg,
            "time":self.time,
            "count":self.count,
        })
            
            
    
    def save_history(self,path=""):
        file_path = path + "/{}_{}.csv".format(self.name,self.id)
        if len(self.history) == 0:
            print(f"No {self.name} data updated!")
            return False
        t = int(time.time())
        keys = self.history[0].keys()
        with open(file_path, "w",encoding="utf-8") as f:
            f.write(",".join([str(i) for i in keys]) + "\n")
            for i in self.history:
                f.write(",".join([str(i[key]) for key in keys]) + "\n")
        print("[saved {} {}]:to {}".format(self.name,t,file_path))

    def __str__(self):
        fmtstr = '[{name} {val' + self.fmt + '} ({avg' + self.fmt + '}) time:{time}]'
        return fmtstr.format(**self.__dict__)