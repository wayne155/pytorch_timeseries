# https://github.com/Mcompetitions/M4-methods/tree/master/Dataset
from .UEA import UEA

class FaceDetection(UEA):
    """
        https://www.timeseriesclassification.com/description.php?Dataset=FaceDetection
    """
    name: str = "FaceDetection"
    def __init__(self,root='./data'):
        super(FaceDetection, self).__init__(self.name, root)
        
