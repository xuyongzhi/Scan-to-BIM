
# Angle order

## (1) opencv rotation order  
visual_utils.py / draw_XYLgWsA / cv2.boxPoints(rect)  
rect = ( center, size, angle )

Positive: Clock wise in Img coordinate
          Anti-clock wise in Euclidean coordinate.

## (2) torch.cross
Same with opencv

## (2) geometry_utils.py / angle_from_vecs_to_vece
Same with opencv

## (3) obj_utils.py / XYZLgWsHA          
Same with opencv
