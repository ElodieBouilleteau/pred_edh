import argparse
import cv2
import numpy as np
import os
import math

def run(input_video_file, output_video_file):
    """Run transform video right format.
    
	Parameters
    ----------
    input_video_file : str
        Path to the video file.
    output_video_file : str
        Path to the right format video file '.mp4'.

    """
    print("Debut de la transformation du format de la video")
    #récupération de la vidéo
    video = cv2.VideoCapture(input_video_file)
    #fps de la vidéo
    fps = video.get(cv2.CAP_PROP_FPS)
    #largeur des images de la vidéo
    width_video = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    #hauteur des images de la vidéo
    height_video = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    #nombre d'images dans la vidéo
    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    #durée de la vidéo
    duration = frame_count/fps
    #nouvelle durée de la vidéo (on arrondi)
    new_duration = math.floor(duration)
    #nouveau fps de la vidéo
    new_fps = float(round(fps))
    #appliquer le nouveau fps
    video.set(cv2.CAP_PROP_FPS,new_fps)
    #appliquer la nouvelle durée
    print(new_duration)
    print(new_fps)
    print(new_duration*new_fps)
    new_frame_count = new_duration*new_fps
    video.set(cv2.CAP_PROP_FRAME_COUNT,new_duration*new_fps)
    #déffinition du format de la vidéo en sortie
    video_out = cv2.VideoWriter(output_video_file,0x7634706d,new_fps,(width_video,height_video),True)
    
    count = 0
    #ouverture de la vidéo
    while(video.isOpened()):
        #lecture image par image
        ret, frame = video.read()
        if ret==True:

            #ecriture de l'image dans la vidéo en sortie
            video_out.write(frame)
            count = count + 1
            
            if (count > (new_frame_count-1)):
                # Libérer la vidéo
                video.release()
                break
        else:
            break

    print("fin de la transformation")
    #fermer les vidéos
    video.release()
    video_out.release()
    
def parse_args():
    """ Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description = 'Transform Video Right format')
    parser.add_argument(
        "--input_video_file", help="Path to video file",
        default=None, required=True)
    parser.add_argument(
        "--output_video_file", help="Path to the right format video file '.mp4'",
        default=None, required=True)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    run(args.input_video_file, args.output_video_file)