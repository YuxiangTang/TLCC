from builtins import object
import torchvision.transforms as transforms

class PairedData(object):
    def __init__(self, data_loader_A, data_loader_B, max_dataset_size):
        self.data_loader_A = data_loader_A
        self.data_loader_B = data_loader_B
        self.stop_A = False
        self.stop_B = False
        self.max_dataset_size = max_dataset_size

    def __iter__(self):
        self.stop_A = False
        self.stop_B = False
        self.data_loader_A_iter = iter(self.data_loader_A)
        self.data_loader_B_iter = iter(self.data_loader_B)
        self.iter = 0
        return self

    def __next__(self):
        jpg_img, jpg_gt, jpg_camera = None, None, None
        raw_img, raw_gt, raw_camera = None, None, None
        try:
            jpg_img, jpg_gt, jpg_camera = next(self.data_loader_A_iter)
        except StopIteration:
            if jpg_img is None or jpg_gt is None:
                self.stop_A = True
                self.data_loader_A_iter = iter(self.data_loader_A)
                jpg_img, jpg_gt, jpg_camera = next(self.data_loader_A_iter)

        try:
            raw_img, raw_gt, raw_camera = next(self.data_loader_B_iter)
        except StopIteration:
            if raw_img is None or raw_gt is None:
                self.stop_B = True
                self.data_loader_B_iter = iter(self.data_loader_B)
                raw_img, raw_gt, raw_camera = next(self.data_loader_B_iter)

        if (self.stop_A and self.stop_B) or self.iter > self.max_dataset_size:
            self.stop_A = False
            self.stop_B = False
            raise StopIteration()
        else:
            self.iter += 1
            return {'jpg_img': jpg_img, 'jpg_gt': jpg_gt, 'jpg_camera': jpg_camera,
                     'raw_img': raw_img, 'raw_gt': raw_gt, 'raw_camera': raw_camera}
