import pandas as pd
import re
import os
from sklearn.model_selection import train_test_split

class ImageDataProcessor:
    def __init__(self, folder_path, output_csv_path):
        self.folder_path = folder_path
        self.output_csv_path = output_csv_path
        self.img_names = []
        self.sex_values = []
        self.age_values = []
        self.height_values = []
        self.weight_values = []

    def process_images(self):
        for filename in os.listdir(self.folder_path):
            if filename.endswith('.jpg') or filename.endswith('.png'):
                ret = re.match(r"\d+?_([FMfm])_(\d+?)_(\d+?)_(\d+).+", filename)
                if ret:
                    sex = 0 if (ret.group(1) == 'F' or ret.group(1) == 'f') else 1
                    age = int(ret.group(2))
                    height = int(ret.group(3)) / 100000
                    weight = int(ret.group(4)) / 100000

                    self.img_names.append(filename)
                    self.sex_values.append(sex)
                    self.age_values.append(age)
                    self.height_values.append(height)
                    self.weight_values.append(weight)
                else:
                    print('Error: Could not parse filename: ' + filename)

    def create_dataframe(self):
        data = {
            'img_name': self.img_names,
            'sex': self.sex_values,
            'age': self.age_values,
            'height': self.height_values,
            'weight': self.weight_values
        }

        df = pd.DataFrame(data)
        train, test = train_test_split(df, test_size=0.2, random_state=42)
        train, validation = train_test_split(train, test_size=0.2, random_state=42)
        df['split'] = 'train'  # Set default value to 'train'
        df.loc[df.index.isin(test.index), 'split'] = 'test'
        df.loc[df.index.isin(validation.index), 'split'] = 'validation'
        df.to_csv(self.output_csv_path, index=False)

# # Usage
# folder_path = 'datasets/Images'
# output_csv_path = 'datasets/updated_data.csv'

# processor = ImageDataProcessor(folder_path, output_csv_path)
# processor.process_images()
# processor.create_dataframe()
