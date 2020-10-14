import os
import subprocess as sbp
from tqdm import tqdm


def extract_images_from_video(parent_dir, video_name, dest_dir):
    for f in tqdm(os.listdir(parent_dir)):
        print(f"======== DIRECTORY: {f} ========")
        errors = []
        query = None
        dest_path = os.path.join(parent_dir, f, dest_dir)
        video_path = os.path.join(parent_dir, f, video_name)

        if not os.path.exists(dest_path):
            try:
                os.makedirs(dest_path)
            except Exception as e:
                print(e)
                errors.append(f"Error in creating directory {dest_path}")

        if os.path.exists(dest_path):
            query = "ffmpeg -i " + video_path + " -qscale:v 1 -f image2 " + os.path.join(dest_path, "%06d.jpg")
            # print(query)

        if query:
            try:
                resp = sbp.Popen(query, shell=True, stdout=sbp.PIPE).stdout.read()
                print(resp)
            except Exception as e:
                print(e)
                errors.append(f"ERROR in directory: {f}")

    if len(errors) > 0:
        return errors

    return None

if __name__ == "__main__":
    data_dir = os.path.join("..", "..", "..", "blink-detection", "data", "300VW_Dataset_2015_12_14")
    dest_folder_name = 'images'
    video_name = "vid.avi"

    errs = extract_images_from_video(data_dir, video_name, dest_folder_name)
    print(errs)
