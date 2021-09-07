import torch

class data_prefetcher():
    def __init__(self, loader):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        # With Amp, it isn't necessary to manually convert data to half.
        # if args.fp16:
        #     self.mean = self.mean.half()
        #     self.std = self.std.half()
        self.preload()

    def preload(self):
        try:
            self.next_input, self.next_target, self.next_si = next(self.loader)
        except StopIteration:
            self.next_input = None
            self.next_target = None
            self.next_si = None
            return
        with torch.cuda.stream(self.stream):
            self.next_input = self.next_input.cuda(non_blocking=True)
            self.next_target = self.next_target.cuda(non_blocking=True)
            self.next_si = self.next_si.cuda(non_blocking=True)
            self.next_input = self.next_input.float()
            self.next_target = self.next_target.float()
            self.next_si = self.next_si.float()
            
    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        input = self.next_input
        target = self.next_target
        si = self.next_si
        self.preload()
        return input, target, si
            
    
            
    