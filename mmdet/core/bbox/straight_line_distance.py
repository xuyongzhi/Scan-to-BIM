# xyz 2020 1 21

import torch
import cv2
import numpy as np
import sys
from .geometric_utils import angle_from_vecs_to_vece
import os


def line_overlaps(lines0, lines1):
  assert lines0.dim() == 2
  assert lines1.dim() == 2
  assert lines0.shape[1]  == 4
  assert lines1.shape[1]  == 4
  lines0 = lines0.view(-1,2,2)
  lines1 = lines1.view(-1,2,2)
  diss = straight_lines_dis(lines0, lines1)
  overlaps = dis_to_overlap(diss)
  return overlaps

def dis_to_overlap(diss, afa=2):
  diss *= afa
  #overlaps = torch.exp(-diss)
  overlaps = (1-diss).clamp(min=0)
  return overlaps

def straight_lines_dis(lines0, lines1):
  '''
  Input:
    line0: [n,2,2]
    line1: [m,2,2]
  Output:
    distance: [n,m]
  '''
  assert lines0.dim() == 3
  assert lines1.dim() == 3
  assert lines0.shape[1:]  == torch.Size([2, 2])
  assert lines1.shape[1:]  == torch.Size([2, 2])

  n = lines0.shape[0]
  m = lines1.shape[0]

  vectors0 = lines0[:,0] - lines0[:,1]
  vectors1 = lines1[:,0] - lines1[:,1]
  vectors0 = vectors0.unsqueeze(1).repeat(1,m,1)
  vectors1 = vectors1.unsqueeze(0).repeat(n,1,1)

  # d_translation
  d1 = (lines0[:,0].unsqueeze(1) - lines1[:,0].unsqueeze(0)).norm(dim=-1)
  d2 = (lines0[:,0].unsqueeze(1) - lines1[:,1].unsqueeze(0)).norm(dim=-1)
  d3 = (lines0[:,1].unsqueeze(1) - lines1[:,0].unsqueeze(0)).norm(dim=-1)
  d4 = (lines0[:,1].unsqueeze(1) - lines1[:,1].unsqueeze(0)).norm(dim=-1)

  l1 = vectors0.norm(dim=-1)
  l2 = vectors1.norm(dim=-1)

  d_translation = (d1+d2+d3+d4)/4-(l1+l2)/4

  # d_angle
  cos_a = (vectors0*vectors1).sum(dim=-1)/(l1*l2)
  sin_a = (1 - cos_a**2).sqrt()
  d_angle = torch.min(l1,l2) * sin_a
  mask = torch.isnan(d_angle)
  d_angle[mask] = 0

  # d_closest_point
  closest_dis = closet_dis_line_line(lines0, lines1)

  dis = d_translation + 0.25 * d_angle + closest_dis

  # normalize dis
  dis_norm = dis / (l1+l2) * 2
  #print(l1, l2)
  return  dis_norm

def perpendicular_dis(point, line):
  vec0 = point - line[0]
  vec1 = line[1] - line[0]
  angle = angle_from_vecs_to_vece(vec0, vec1)
  dis_perp = vec0.norm() * torch.sin(angle)
  return dis_perp

def closet_dis_point_line(points, lines):
  pn = points.shape[0]
  ln = lines.shape[0]

  angles = []
  vectors = []
  for i in range(2):
    vec_A1_ = points.unsqueeze(1) - lines[:,i].unsqueeze(0)
    assert vec_A1_.shape == torch.Size([pn, ln, 2])
    vec_A1 = vec_A1_.view(-1,2)
    vectors.append(vec_A1)
    vec_B1_ = (lines[:,1-i] - lines[:,i]).unsqueeze(0).repeat(pn,1,1)
    vec_B1 = vec_B1_.view(-1,2)
    angle_1 = angle_from_vecs_to_vece(vec_A1, vec_B1, scope_id=0)
    #angle_1 = angle_1.view(pn, ln)
    angles.append(angle_1)


  nan_mask = torch.isnan(angles[0]) + torch.isnan(angles[1])

  perpendicular_out_line_mask = (angles[0] > np.pi / 2.0) + \
    (angles[1] > np.pi/2.0)
  pout_mask = (perpendicular_out_line_mask > 0).type(points.dtype)

  vec_norm0 = vectors[0].norm(dim=-1)
  vec_norm1 = vectors[1].norm(dim=-1)
  closest_dis_A = torch.min( vec_norm0, vec_norm1 )
  closest_dis_B = vec_norm0 * torch.sin(angles[0])

  closest_dis = closest_dis_A * pout_mask + closest_dis_B * (1-pout_mask)
  closest_dis[nan_mask] = 0
  closest_dis = closest_dis.view(pn, ln)

  return closest_dis

def closet_dis_line_line(lines0, lines1):
  c_dis0 = closet_dis_point_line(lines0[:,0], lines1).unsqueeze(0)
  c_dis1 = closet_dis_point_line(lines0[:,1], lines1).unsqueeze(0)
  c_dis2 = closet_dis_point_line(lines1[:,0], lines0).permute(1,0).unsqueeze(0)
  c_dis3 = closet_dis_point_line(lines1[:,1], lines0).permute(1,0).unsqueeze(0)
  closest_dis = torch.cat([c_dis0, c_dis1, c_dis2, c_dis3], 0).min(dim=0)[0]
  return closest_dis

def test():
  lines0 = []
  for i in range(10):
    line0 = np.array([ [50, 20], [240,200+i*30] ], dtype=np.float).reshape([1,2,2])
    lines0.append(line0)
  lines0 = np.concatenate(lines0, axis=0)

  lines1 = []
  for i in range(5):
    line1 = np.array([ [250, i*5+210], [230+i*10,210] ], dtype=np.float).reshape([1,2,2])
    lines1.append(line1)
  lines1 = np.concatenate(lines1, axis=0)
  lines0 = torch.from_numpy(lines0)
  lines1 = torch.from_numpy(lines1)

  diss = straight_lines_dis(lines0, lines1)
  overlaps = dis_to_overlap(diss)

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
      overlap = overlaps[i,j]
      cv2.imwrite(f'{res_dir}/dis_{dis:.2}_overlap_{overlap:0.2}.png', img)
  print(f'save line dis res')

def draw_line(img, line, color=(0,255,0)):
  s, e = line
  cv2.line(img, (s[0], s[1]), (e[0], e[1]), color, 3)

if __name__ == '__main__':
  test()
