import argparse
import cv2
import numpy as np
import os
import math
import configparser

def run(video_file, output_dir):
    """Run transform video file into directory MOT format.
	Directory : img1 -> all video frame (000001 to maxframe)
				seqinfo.ini -> video information
    
	Parameters
    ----------
    video_file : str
        Path to the video file.
    output_dir : str
        Path to the MOT format directory.

    """
    
    # Récupérer le nom du répertoire
    list_output_dir = output_dir.split('\\')
    name = list_output_dir[len(list_output_dir)-1]
    # Récupérer la vidéo
    video = cv2.VideoCapture(video_file)
    # Récupérer le fps
    fps = video.get(cv2.CAP_PROP_FPS)
    print(video.get(cv2.CAP_PROP_FPS))
    #largeur des images de la vidéo
    width_video = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    #hauteur des images de la vidéo
    height_video = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # Récupérer le nombre d'images
    length_video = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    print ("Number of frames: ", length_video)
    count = 0
    print ("Converting video..\n")
    # Commencer à convertir la vidéo
    while video.isOpened():
        # Extraire les images
        ret, frame = video.read()
        # Ecrire le résultat dans la sortie.
        if not os.path.exists(output_dir + "/img1/"):
            os.makedirs(output_dir + "/img1/")
        cv2.imwrite(output_dir + "/img1/%d.jpg" % (count+1), frame)
        count = count + 1
        # Si il n'y a plus d'images
        if (count > (length_video-1)):
            # Libérer la vidéo
            video.release()
            break
    
    # Ecrire dans le fichier seqinfo.ini
    cfg = configparser.ConfigParser()
    
    S = 'Sequence'
    cfg.add_section(S)
    
    cfg.set(S,'name',name)
    cfg.set(S,'imDir','img1')
    cfg.set(S,'frameRate',str(int(fps)))
    cfg.set(S,'seqLength',str(length_video))
    cfg.set(S,'imWidth',str(width_video))
    cfg.set(S,'imHeight',str(height_video))
    cfg.set(S,'imExt','.jpg')
    
    cfg.write(open(output_dir+"/seqinfo.ini",'w'))

	
def parse_args():
    """ Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description = 'Generate MOTFormat file from video')
    parser.add_argument(
        "--video_file", help="Path to video file",
        default=None, required=True)
    parser.add_argument(
        "--output_dir", help="Path to MOT format file output",
        default=None, required=True)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    run(args.video_file, args.output_dir)