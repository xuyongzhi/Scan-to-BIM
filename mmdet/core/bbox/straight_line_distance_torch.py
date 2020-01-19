import torch
import cv2
import numpy as np
from .geometric_torch import angle_from_vec0_to_vec1
import os

def line_overlaps(lines0, lines1, weight=2):
  diss = straight_lines_dis(lines0, lines1)
  overlaps = torch.exp(-diss * weight)
  return overlaps

def straight_lines_dis(lines0, lines1):
  '''
  Input:
    line0: [n,2,2]
    line1: [m,2,2]
  Output:
    distance: [n,m]
  '''
  n = lines0.shape[0]
  m = lines1.shape[0]
  dis = torch.zeros([n,m])
  for i in range(n):
    for j in range(m):
      dis[i,j] = straight_2lines_dis( lines0[i], lines1[j] )
  return dis

def perpendicular_dis(point, line):
  vec0 = point - line[0]
  vec1 = line[1] - line[0]
  angle = angle_from_vec0_to_vec1(vec0, vec1)
  dis_perp = vec0.norm() * torch.sin(angle)
  return dis_perp

def closet_dis_point_line(point, line):
  vec_A1 = point - line[1]
  vec_B1 = line[0] - line[1]
  angle_1 = angle_from_vec0_to_vec1(vec_A1.unsqueeze(0), vec_B1.unsqueeze(0))[0]

  vec_A0 = point - line[0]
  vec_B0 = line[1] - line[0]
  angle_0 = angle_from_vec0_to_vec1(vec_A0.unsqueeze(0), vec_B0.unsqueeze(0))[0]

  if torch.isnan(angle_0) or torch.isnan(angle_1):
    return torch.zeros([], device=point.device, dtype=point.dtype)

  if angle_0 > np.pi / 4.0 or angle_1 > np.pi/4.0:
    closest_dis = min( vec_A1.norm(), vec_A0.norm() )
  else:
    closest_dis =  vec_A0.norm() * torch.sin(angle_0)

  return closest_dis

def closet_dis_line_line(line0, line1):
  c_dis0 = closet_dis_point_line(line0[0], line1)
  c_dis1 = closet_dis_point_line(line0[1], line1)
  c_dis2 = closet_dis_point_line(line1[0], line0)
  c_dis3 = closet_dis_point_line(line1[1], line0)
  closest_dis = torch.stack([c_dis0, c_dis1, c_dis2, c_dis3]).min()
  return closest_dis

def straight_2lines_dis(line0, line1):
  '''
  ref: Evaluation of Established line segment distance functions
        (S.Wirtz and D.Paulus)
  Input:
    line0: [2,2]
    line1: [2,2]
  Output:
    distance: [n,m]
  '''
  vector0 = line0[0] - line0[1]
  vector1 = line1[0] - line1[1]

  # d_translation
  d1 = (line0[0] - line1[0]).norm()
  d2 = (line0[0] - line1[1]).norm()
  d3 = (line0[1] - line1[0]).norm()
  d4 = (line0[1] - line1[1]).norm()

  l1 = vector0.norm()
  l2 = vector1.norm()

  d_translation = (d1+d2+d3+d4)/4-(l1+l2)/4

  # d_angle
  cos_a = (vector0*vector1).sum()/(l1*l2)
  sin_a = (1 - cos_a**2).sqrt()
  d_angle = min(l1,l2) * sin_a
  if torch.isnan(d_angle):
    d_angle = torch.zeros([], device=line0.device, dtype=line0.dtype)

  # d_closest_point
  closest_dis = closet_dis_line_line(line0, line1)

  dis = d_translation + 0.25 * d_angle + closest_dis

  # normalize dis
  dis_norm = dis / (l1+l2) * 2
  #print(l1, l2)
  return  dis_norm

def test():
  lines0 = []
  for i in range(5):
    line0 = np.array([ [30, 230], [300,10*i+80] ], dtype=np.float).reshape([1,2,2])
    lines0.append(line0)
  lines0 = np.concatenate(lines0, axis=0)

  lines1 = []
  for i in range(10):
    line1 = np.array([ [30, i*15+220], [300,100] ], dtype=np.float).reshape([1,2,2])
    lines1.append(line1)
  lines1 = np.concatenate(lines1, axis=0)
  lines0 = torch.from_numpy(lines0)
  lines1 = torch.from_numpy(lines1)

  diss = straight_lines_dis(lines0, lines1)
  sims = torch.exp(-diss * 3)

  res_dir = './line_dis_res'
  if not os.path.exists(res_dir):
    os.makedirs(res_dir)

  n = lines0.shape[0]
  m = lines1.shape[0]
  for i in range(n):
    for j in range(m):
      line0 = lines0[i].data.numpy().astype(np.int)
      line1 = lines1[j].data.numpy().astype(np.int)
      img = np.zeros([400,400,3], dtype=np.uint8)
      img[:,:,2] = 100
      draw_line(img, line0)
      draw_line(img, line1)
      dis = diss[i,j]
      sim = sims[i,j]
      cv2.imwrite(f'{res_dir}/dis_{dis:.2}_sim_{sim:0.2}.png', img)
  print(f'save line dis res')

def draw_line(img, line, color=(0,255,0)):
  s, e = line
  cv2.line(img, (s[0], s[1]), (e[0], e[1]), color, 3)

if __name__ == '__main__':
  test()

