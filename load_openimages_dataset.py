import os
import fiftyone as fo
import fiftyone.zoo as foz
from fiftyone import ViewField as F

# Load or download the dataset
# This will download the specified dataset split if it is not already downloaded
dataset = foz.load_zoo_dataset('open-images-v7', split='validation')
#print(dataset.head(5))

# Define your electronics categories
electronics_categories = ['Laptop', 'Mobile phone', 'Camera', 'Television', 'Printer', 'Washing machine']
# Assuming the relevant field is 'detections'
electronics_view = dataset.filter_labels("detections", F("label").is_in(electronics_categories))

# Launch the app
#session = fo.launch_app(electronics_view)
#session.wait()

# Directory where the exported dataset will be saved
export_dir = os.path.join('.', 'dataset')

# Export the dataset in YOLO format
electronics_view.export(
    export_dir=export_dir,
    dataset_type=fo.types.YOLOv5Dataset,  # Using YOLOv5 format, adjust if YOLOv8 has a different format
    label_field="detections",  # This is the field that contains your labels. Adjust if necessary.
    classes=electronics_categories
)

print("Dataset exported successfully to:", export_dir)