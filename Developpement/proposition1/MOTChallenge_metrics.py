import argparse
import cv2
import motmetrics as mm
import numpy as np
import os
import math
import configparser

def bb_intersection_over_union(boxA, boxB):
	# determine the (x, y)-coordinates of the intersection rectangle
	xA = max(boxA[0], boxB[0])
	yA = max(boxA[1], boxB[1])
	xB = min(boxA[2], boxB[2])
	yB = min(boxA[3], boxB[3])
 
	# compute the area of intersection rectangle
	interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
 
	# compute the area of both the prediction and ground-truth
	# rectangles
	boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
	boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
 
	# compute the intersection over union by taking the intersection
	# area and dividing it by the sum of prediction + ground-truth
	# areas - the interesection area
	iou = round(interArea / float(boxAArea + boxBArea - interArea),2)
 
	# return the intersection over union value
	return iou

def iou(list_coord_gt, list_coord_hyp):
    result = [[0 for x in range(len(list_coord_hyp))] for y in range(len(list_coord_gt))] 
    for i in range(0,len(list_coord_gt)):
        for j in range(0,len(list_coord_hyp)):
            result[i][j] = bb_intersection_over_union(list_coord_gt[i],list_coord_hyp[j])
    return result

def run(hypotheses_file, sequence_dir, output_file):
    """Run calcul MOTChallenge metrics from 2 files "Ground truth objects" and "Detector hypotheses".
	Directory : hypotheses -> text file detections hypotheses
				ground truth -> text file detections ground truth
    
	Parameters
    ----------
    hypotheses_file : str
        Path to the hypotheses detections file.
    sequence_dir : str
        Path to the MOT dir.
	output_file : str
        Path to the MOTChallenge metrics file.
    """
    
    list_coord_hyp = {}
    # Récupérer les éléments des fichiers hypotheses et ground truth
    hyp_file = open(hypotheses_file, 'r')
    for line in hyp_file:
        idframe = int(line.split(',')[0])
        list_coord_hyp.setdefault(idframe,[]).append([float(line.split(',')[2]),float(line.split(',')[3]),float(line.split(',')[4]),float(line.split(',')[5])])
    
    list_coord_gt = {}
    # Récupérer les éléments des fichiers hypotheses et ground truth
    gt_file = open(sequence_dir+"\gt\gt.txt", 'r')
    for line in gt_file:
        idframe = int(line.split(',')[0])
        list_coord_gt.setdefault(idframe,[]).append([float(line.split(',')[2]),float(line.split(',')[3]),float(line.split(',')[4]),float(line.split(',')[5])])

    # Intersection over union norm for 2D rectangles
    #test = 0
    #for key_gt in list_coord_gt:
        #if key_gt in list_coord_hyp:
            #test = np.zeros(len(list_coord_hyp[key_gt]),len(list_coord_gt[key_gt]))
            #print(test)
            #for coord_hyp in list_coord_hyp[key_gt]:
                
        #else:
          
    a = [[0, 0, 20, 100],[0, 0, 0.8, 1.5]]
    b = [[0, 0, 1, 2],[0, 0, 1, 1],[0.1, 0.2, 2, 2]]
    result = iou(a,b)
    #print(result)
    #print(bb_intersection_over_union([0, 0, 20, 100],[0, 0, 1, 2]))
    a = np.array([
    [0, 0, 20, 100],    # Format X, Y, Width, Height
    [0, 0, 0.8, 1.5],
    ])

    b = np.array([
    [0, 0, 1, 2],
    [0, 0, 1, 1],
    [0.1, 0.2, 2, 2],
    ])
    #print(mm.distances.iou_matrix(a, b, max_iou=0.5))
    
    # Create an accumulator that will be updated during each frame
    acc = mm.MOTAccumulator(auto_id=True)

    # Call update once for per frame. For now, assume distances between
    # frame objects / hypotheses are given.
    acc.update(
        ['a', 'b'],                 # Ground truth objects in this frame
        [1, 2, 3],                  # Detector hypotheses in this frame
    [
        [0.1, np.nan, 0.3],     # Distances from object 'a' to hypotheses 1, 2, 3
        [0.5,  0.2,   0.3]      # Distances from object 'b' to hypotheses 1, 2, 3
    ]
    )

    print(acc.events)
    print(acc.mot_events)
	
    frameid = acc.update(
        ['a', 'b'],
        [1],
        [
            [0.2], 
            [0.4]
        ]
    )
    print(acc.mot_events.loc[frameid])
    
    frameid = acc.update(
    ['a', 'b'],
    [1, 3],
    [
        [0.6, 0.2],
        [0.1, 0.6]
    ]
    )
    print(acc.mot_events.loc[frameid])
	
    mh = mm.metrics.create()
    summary = mh.compute(acc, metrics=['num_frames', 'mota', 'motp'], name='acc')
    print(summary)
	
    summary = mh.compute_many(
    [acc, acc.events.loc[0:1]], 
    metrics=mm.metrics.motchallenge_metrics, 
    names=['full', 'part'])

    strsummary = mm.io.render_summary(
    summary, 
    formatters=mh.formatters, 
    namemap=mm.io.motchallenge_metric_names
    )
    print(strsummary)
    

def parse_args():
    """ Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description = 'Generate MOTChallenge metrics file')
    parser.add_argument(
        "--hypotheses_file", help="Path to the hypotheses detections file",
        default=None, required=True)
    parser.add_argument(
        "--sequence_dir", help="Path to the MOT dir",
        default=None, required=True)
    parser.add_argument(
        "--output_file", help="Path to the MOTChallenge metrics file",
        default=None, required=True)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    run(args.hypotheses_file, args.sequence_dir, args.output_file)