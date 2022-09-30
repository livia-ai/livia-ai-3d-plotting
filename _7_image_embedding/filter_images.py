import pandas as pd
import shutil
import os

museum_data = pd.read_csv("data_local/mak/mak.csv")

print(museum_data.columns)

#def alter_path(img_path):
#    img_path = img_path.split("/")[-1]
#    img_path = img_path.split(".")
#    img_path.insert(1, "224")
#    img_path = ".".join(img_path)
#    return img_path

#museum_data["img_path"] = museum_data["reproduction"].apply(alter_path)


##shutil.copy( dir_src + filename, dir_dst)

#os.chdir("..")
##print(os.listdir("livia_images/images-mak"))
#src_dir = "livia_images/images-mak/images-cropped/"
#dst_dir = "livia-ai-scripts/data_local/images/mak_cropped/"

#for img_path in museum_data["img_path"]:
#    shutil.copy( src_dir + img_path, dst_dir)