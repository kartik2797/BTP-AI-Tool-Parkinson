import numpy as np
import pandas as pd
import os

class Custom():

    def __init__(self,selections):
        """ Inits the Class """
        self.file = os.path.join(os.getcwd(),'custom.py')
        self.model = selections['model']
        self.preprocess = selections['preprocess']
        self.foot = selections['foot']

    def generate(self):
        """ Main Method for Generating """
    
        if os.path.exists(self.file):
            os.remove(self.file)
            print("Old File deleted")
        else:
            print("File doesnt Existing! Creating One for you")

        f = open(self.file,"a")
        f.write(self.import_library())
        f.write(self.load_data())
        f.write(self.select_foot())
        f.write(self.select_preprocess())
        f.write(self.select_model())
        f.close()
        

    def import_library(self):
        """ Library Imports """
        lib = "import numpy as np\nimport pandas as pd\nfrom keras import Input\nfrom keras import optimizers\nfrom keras.models import Model,Sequential\nfrom keras.layers import Dense,LSTM,Dropout, Flatten, Convolution2D, MaxPooling2D, Dense,Conv1D\nfrom keras.backend import l2_normalize\nimport os\nfrom keras.utils import to_categorical\nimport random\nimport sys\n"
        return lib

    def load_data(self):
        """ Load the Data """
        data_str = "datas = np.loadtxt('sample.txt')\n\tdatas = datas[:,np.arange(1,19)]\n"
        return data_str

    def select_foot(self):
        """ Writes Code for the Foot """
        foot_str = ""
        
        
        return foot_str
    
    def select_preprocess(self):
        """ Writes Code for the Preprocessing """
        pre_str = ""
        return pre_str
                

    def select_model(self):
        """ Writes Code for the Model Selected """
        return ""

