import glob
class Constants:

    def __init__(self):
        self.batch_size = 128

        self.num_training = 4000 #34500
        self.num_dev = 34500
        self.num_test = 4

        self.output_classes = 4

        self.image_height = 255
        self.image_width = 255


        self.raw_data_directory = '/home/stephan/Documents/Data/Doodles/raw/*.ndjson'
        self.record_output_dir = '/home/stephan/Documents/Data/Doodles/tfrecords/'
        self.train_record_output_dir = self.record_output_dir + 'train'

        self.file_list = glob.glob(self.raw_data_directory)
        self.class_list = []

        for element in self.file_list:
            self.class_list.append(element[element.rfind('/') + 1:element.rfind('.')]) # Get all element names from the files to map classes to numbers

