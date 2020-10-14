import os
from pytube import YouTube

def downloadYTvideo(link, dest_path):
    #ask for the link from user
    yt = YouTube(link)

    #Showing details
    print("Title: ",yt.title)
    yt.streams \
        .filter(progressive=True, file_extension='mp4') \
        .order_by('resolution') \
        .desc() \
        .first() \
        .download(dest_path)

if __name__ == "__main__":
    dest_path = os.path.join("..","flm_detection_pipeline", "videos")
    downloadYTvideo(link="https://www.youtube.com/watch?v=FEeTLopLkEo&ab_channel=TEDxTalks",
                    dest_path=dest_path)