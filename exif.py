import PIL.Image
from pytorch_toolbelt.utils import fs
from tqdm import tqdm
import PIL.ExifTags

dataset = fs.find_images_in_dir("d:\\datasets\\ALASKA2\\Cover")

for x in tqdm(dataset):
    img = PIL.Image.open(x)
    exif_data = img._getexif()

    if exif_data is not None:
        exif = {
            PIL.ExifTags.TAGS[k]: v
            for k, v in exif_data.items()
            if k in PIL.ExifTags.TAGS
        }

        print(exif)
