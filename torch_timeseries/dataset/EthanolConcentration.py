# https://github.com/Mcompetitions/M4-methods/tree/master/Dataset
from .UEA import UEA

class EthanolConcentration(UEA):
    """
https://www.timeseriesclassification.com/description.php?Dataset=EthanolConcentration
    """
    name: str = "EthanolConcentration"
    def __init__(self,root='./data'):
        super(EthanolConcentration, self).__init__(self.name, root)
        
