
# Angle order

Positive: Clock wise in Img coordinate
          Anti-clock wise in Euclidean coordinate.

## (1) opencv rotation order  (checked ok)
visual_utils.py / draw_XYLgWsA / cv2.boxPoints(rect)  
rect = ( center, size, angle )  
check: test_rotation_order()  

## (2) torch.cross (checked ok)
Same with opencv

## (2) geometry_utils.py / angle_from_vecs_to_vece (checked ok)
Same with opencv

## (3) obj_utils.py / XYZLgWsHA    (checked ok)
Same with opencv
