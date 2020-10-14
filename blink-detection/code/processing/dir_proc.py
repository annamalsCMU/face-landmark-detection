import os
import shutil
from tqdm import tqdm
import glob
import cv2

def convertPNGtoJPGandSave(src_png_path, dest_jpg_path):
    im = cv2.imread(src_png_path)
    cv2.imwrite(dest_jpg_path, im, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
    pass

def copyImagesAndAnnotations(final_data_path, image_files, annot_files, img_nm_sfx, filetype=".jpg"):

    assert (len(image_files) == len(annot_files))

    files_zip_ls = list(zip(image_files, annot_files))

    for img_path, annot_path in tqdm(files_zip_ls):
        img_nm = img_path.split('\\')[-1]
        annot_nm = annot_path.split('\\')[-1]
        assert (img_nm.replace(filetype, "") == annot_path.split('\\')[-1].replace(".pts", ""))

        if os.path.exists(os.path.join(img_path)) and os.path.exists(os.path.join(annot_path)):
            # copy files into master final dataset
            if filetype == ".jpg":
                shutil.copy(os.path.join(img_path),
                            os.path.join(final_data_path, "-".join([img_nm_sfx, img_nm.split('\\')[-1]])))
            elif filetype == ".png":
                jpg_nm = img_nm.split('\\')[-1].replace(".png", ".jpg")
                convertPNGtoJPGandSave(src_png_path=img_path,
                                       dest_jpg_path=os.path.join(final_data_path, "-".join([img_nm_sfx, jpg_nm])))

            shutil.copy(os.path.join(annot_path),
                        os.path.join(final_data_path, "-".join([img_nm_sfx, annot_nm.split('\\')[-1]])))
        # break
    pass

def processAndSaveData(src_data_path, final_data_path):
    total_files = 0
    for f in os.listdir(src_data_path):
        ds_path = os.path.join(src_data_path, f)
        print("=" * 50)
        print(f"f: {f}")
        if os.path.isdir(ds_path):
            if ("300VW" in f) or ("helen" in f) or ("lfpw" in f) or ("300W" in f):
                for dir in os.listdir(ds_path):
                    print("="*20)
                    print(f"dir: {dir}")
                    if "300VW" in f:
                        image_files = glob.glob(os.path.join(ds_path, dir, "images", "*"))
                        annot_files = glob.glob(os.path.join(ds_path, dir, "annot", "*"))
                        img_nm_sfx = "-".join(["300VW", dir])
                        copyImagesAndAnnotations(final_data_path, image_files, annot_files, img_nm_sfx, filetype=".jpg")
                        pass
                    elif "helen" in f:
                        image_files = glob.glob(os.path.join(ds_path, dir, "*.jpg"))
                        annot_files = glob.glob(os.path.join(ds_path, dir, "*.pts"))
                        img_nm_sfx = "-".join([f, dir])
                        copyImagesAndAnnotations(final_data_path, image_files, annot_files, img_nm_sfx, filetype=".jpg")
                    else:
                        image_files = glob.glob(os.path.join(ds_path, dir, "*.png"))
                        annot_files = glob.glob(os.path.join(ds_path, dir, "*.pts"))
                        img_nm_sfx = "-".join([f, dir])
                        copyImagesAndAnnotations(final_data_path, image_files, annot_files, img_nm_sfx, filetype=".png")

                    print(f"#FILES: {len(image_files)}")
                    total_files += len(image_files)

            elif ("afw" in f) or ("ibug" in f):
                print("="*20)
                print(f"dir: {f}")
                image_files = glob.glob(os.path.join(ds_path, "*.jpg"))
                annot_files = glob.glob(os.path.join(ds_path, "*.pts"))
                img_nm_sfx = f

                copyImagesAndAnnotations(final_data_path, image_files, annot_files, img_nm_sfx, filetype=".jpg")

                print(f"#FILES: {len(image_files)}")
                total_files += len(image_files)


    return total_files

if __name__ == "__main__":
    src_data_path = os.path.join("..", "..", "data")
    final_data_path = os.path.join("..", "..", "final_data")
    total_files = processAndSaveData(src_data_path, final_data_path)
    print(total_files)
