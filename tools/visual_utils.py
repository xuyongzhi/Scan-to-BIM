import open3d as o3d
import torch
import torch.nn.functional as F
import numpy as np
import mmcv
from mmcv.image import imread, imwrite
import cv2
from MinkowskiEngine import SparseTensor

from tools.color import color_val, get_random_color, _label2color
from configs.common import DEBUG_CFG, DIM_PARSE
from obj_geo_utils.obj_utils import OBJ_REPS_PARSE


ADD_FRAME = 0
BOX_TYPE = ['line_set', 'line_mesh', 'surface_mesh'][0]
BOX_LINE_RADIUS = 0.02

#-2d general------------------------------------------------------------------------------
def _show_objs_ls_points_ls_torch(img,
                            objs_ls=None,
                            obj_rep='XYXYSin2',
                            points_ls=None,
                            obj_colors='random',
                            obj_scores_ls=None,
                            point_colors='red',
                            out_file=None,
                            obj_thickness=1,
                            point_thickness=1,
                            only_save=False,
                            ):
  if isinstance(img, torch.Tensor):
    img = img.cpu().numpy()
  if objs_ls is not None:
    objs_ls = [o.cpu().data.numpy() for o in objs_ls]
  if points_ls is not None:
    points_ls = [o.cpu().data.numpy() for o in points_ls]
  if obj_scores_ls is not None:
    obj_scores_ls = [s.cpu().data.numpy() for s in obj_scores_ls]
  _show_objs_ls_points_ls(img, objs_ls, obj_rep,
                          points_ls=points_ls,
                          obj_colors=obj_colors,
                          obj_scores_ls=obj_scores_ls,
                          point_colors=point_colors,
                          out_file=out_file,
                          point_thickness=point_thickness,
                          only_save=only_save)

def _show_3d_bboxes_ids(bboxes, obj_rep):
  '''
  img: [h,w,3] or [h,w,1], or [h,w] or (h_size, w_size)
  '''
  bboxes = bboxes.copy()
  if obj_rep == 'XYXYSin2':
    min_xy = bboxes[:,:4].reshape(-1,2).min(0)
    bboxes[:,[0,2]] -= min_xy[0]
    bboxes[:,[1,3]] -= min_xy[1]
  voxel_size = 0.01
  bboxes[:,:4] /= voxel_size
  w, h = bboxes[:,:4].reshape(-1,2).max(0).astype(np.int32) + 100
  bids = np.arange(bboxes.shape[0])
  _show_objs_ls_points_ls((h,w), [bboxes], obj_rep=obj_rep, obj_scores_ls=[bids])
  pass

def _show_objs_ls_points_ls(img,
                            objs_ls=None,
                            obj_rep='XYXYSin2',
                            points_ls=None,
                            obj_colors='random',
                            obj_scores_ls=None,
                            point_colors='red',
                            point_scores_ls=None,
                            out_file=None,
                            obj_thickness=1,
                            point_thickness=1,
                            only_save=False,
                            ):
  '''
  img: [h,w,3] or [h,w,1], or [h,w] or (h_size, w_size)
  '''
  assert obj_rep in OBJ_REPS_PARSE._obj_reps
  img = _draw_objs_ls_points_ls(img, objs_ls, obj_rep, points_ls, obj_colors,
      point_colors=point_colors, out_file=out_file, obj_thickness=obj_thickness,
      point_thickness=point_thickness, obj_scores_ls=obj_scores_ls,
      point_scores_ls=point_scores_ls)
  if not only_save:
    mmcv.imshow(img)

def _draw_objs_ls_points_ls(img,
                            objs_ls=None,
                            obj_rep='XYXYSin2',
                            points_ls=None,
                            obj_colors='green',
                            point_colors='red',
                            out_file=None,
                            obj_thickness=1,
                            point_thickness=1,
                            obj_scores_ls=None,
                            point_scores_ls=None,
                            obj_cats_ls=None,
                            text_colors_ls='green',
                            ):
  if objs_ls is not None:
    assert isinstance(objs_ls, list)
    if objs_ls is not None and isinstance(obj_colors, str):
      obj_colors = [obj_colors] * len(objs_ls)
    if not isinstance(obj_thickness, list):
      obj_thickness = [obj_thickness] * len(objs_ls)
    if not isinstance(obj_scores_ls, list):
      obj_scores_ls = [obj_scores_ls] * len(objs_ls)
    if not isinstance(obj_cats_ls, list):
      obj_cats_ls = [obj_cats_ls] * len(objs_ls)
    if not isinstance(text_colors_ls, list):
      text_colors_ls = [text_colors_ls] * len(objs_ls)

  if points_ls is not None:
    assert isinstance(points_ls, list)
    if points_ls is not None and isinstance(point_colors, str):
      point_colors = [point_colors] * len(points_ls)
    if not isinstance(point_thickness, list):
      point_thickness = [point_thickness] * len(points_ls)
    if not isinstance(point_scores_ls, list):
      point_scores_ls = [point_scores_ls] * len(points_ls)

  img = _read_img(img)
  if objs_ls is not None:
    for i, objs in enumerate(objs_ls):
      img = draw_objs(img, objs, obj_rep, obj_colors[i],
                      obj_thickness = obj_thickness[i],
                      scores = obj_scores_ls[i],
                      cats = obj_cats_ls[i],
                      text_color=text_colors_ls[i],
                       )

  if points_ls is not None:
    for i, points in enumerate(points_ls):
      img = _draw_points(img, points, point_colors[i], point_thickness[i], point_scores=point_scores_ls[i])
  if out_file is not None:
    mmcv.imwrite(img, out_file)
    print('\n',out_file)
  return img

def _draw_points(img, points, color, point_thickness, point_scores=None, font_scale=0.5, text_color='green'):
    points = np.round(points).astype(np.int32)
    h, w = img.shape[:2]
    text_color = _get_color(text_color)
    for i in range(points.shape[0]):
      p = points[i]
      c = _get_color(color)
      cv2.circle(img, (p[0], p[1]), 2, c, thickness=point_thickness)

      if point_scores is not None:
        try:
          label_text = '{:d}'.format(point_scores[i]) # score
          #label_text = '{:.01f}'.format(point_scores[i]) # score
        except:
          label_text = str(point_scores[i]) # score

        ofs = 10
        x = min(max(p[0] - 2, ofs), w-ofs)
        y = min(max(p[1] - 4, ofs), h-ofs)
        cv2.putText(img, label_text, (x, y),
                    cv2.FONT_HERSHEY_COMPLEX, font_scale, text_color)
    return img

def _draw_lines(img, lines, color, point_thickness, line_scores=None, font_scale=0.5, text_color='green'):
    lines = np.round(lines).astype(np.int32)
    h, w = img.shape[:2]
    text_color = _get_color(text_color)
    for i in range(lines.shape[0]):
      p0, p1 = lines[i]
      c = _get_color(color)
      cv2.line(img, (p0[0], p0[1]), (p1[0], p1[1]), c, thickness=point_thickness)

      if line_scores is not None:
        try:
          label_text = '{:d}'.format(line_scores[i]) # score
          #label_text = '{:.01f}'.format(point_scores[i]) # score
        except:
          label_text = str(line_scores[i]) # score

        ofs = 10
        x = min(max(p[0] - 2, ofs), w-ofs)
        y = min(max(p[1] - 4, ofs), h-ofs)
        cv2.putText(img, label_text, (x, y),
                    cv2.FONT_HERSHEY_COMPLEX, font_scale, text_color)
    return img

def draw_objs(img, objs, obj_rep, color, obj_thickness=1, scores=None,
              cats=None, font_scale=0.5, text_color='green'):
  if obj_rep != 'XYLgWsA':
    objs = OBJ_REPS_PARSE.encode_obj(objs, obj_rep, 'XYLgWsA')
  draw_XYLgWsA(img, objs, color, obj_thickness=obj_thickness, scores=scores, cats=cats, font_scale=font_scale, text_color=text_color)
  return img

def draw_XYLgWsA(img, objs, color, obj_thickness=1, scores=None, cats=None, font_scale=0.5, text_color='green'):
    '''
    img: [h,w,3]
    objs: [n,5/6]  of XYLgWsA
    color: 'red' or labels of objs
    '''
    assert objs.ndim == 2
    assert objs.shape[1]==5
    rotations = objs[:,4]
    text_color = _get_color(text_color)
    h,w = img.shape[:2]

    n = objs.shape[0]
    if isinstance(color, np.ndarray) and len(color) == n:
      colors = _label2color(color)
    else:
      assert isinstance(color, str)
      colors = [_get_color(color) for i in range(n)]

    if cats is not None and not isinstance(cats, list):
      cats = [cats] * n
    if scores is not None:
      scores = scores.reshape(-1)
      assert scores.shape[0] == n

    boxes = []
    for i in range(n):
        center, size, angle = objs[i][:2], objs[i][2:4], objs[i][4]
        #print(f'size: {size}')
        angle *= 180 / np.pi
        rect = ( center, size, angle )
        box = cv2.boxPoints(rect)
        box = np.int0(np.ceil(box))
        boxes.append(box)
        c = colors[i]
        cv2.drawContours(img, [box],0, c, obj_thickness)

        label_text = ''
        if cats is not None:
          label_text += cats[i] + ' '
        if scores is not None:
          try:
            label_text += '{:.01f}'.format(scores[i]) # score
          except:
            label_text += str(scores[i]) # score

        if label_text != '':
          cen = center.astype(np.int)
          ofs = 20
          x = min(max(cen[0] - 4, ofs), w-ofs)
          y = min(max(cen[1] - 4, ofs), h-ofs)
          cv2.putText(img, label_text, (x, y),
                      cv2.FONT_HERSHEY_COMPLEX, font_scale, text_color)

    return img

def draw_XYXYSin2(img, objs, color, obj_thickness=1, font_scale=0.5, text_color='green'):
    '''
    img: [h,w,3]
    objs: [n,5/6]
    color: 'red'
    '''
    assert objs.ndim == 2
    assert objs.shape[1]==5 or objs.shape[1]==6
    if objs.shape[1] == 6:
      scores = objs[:,5]
      objs = objs[:,:5]
    else:
      scores = None
    rotations = objs[:,4]
    text_color = _get_color(text_color)

    objs = decode_line_rep(objs, 'lscope_istopleft').reshape(-1,2,2)
    objs = OBJ_REPS_PARSE.encode_obj(objs, 'XYXYSin2', 'RoLine2D_2p')
    for i in range(objs.shape[0]):
        s, e = np.round(objs[i]).astype(np.int32)
        c = _get_color(color)
        cv2.line(img, (s[0], s[1]), (e[0], e[1]), c, thickness=obj_thickness)

        #label_text = class_names[label] if class_names is not None else 'cls {}'.format(label)
        label_text = ''
        if DEBUG_CFG.OBJ_LEGEND == 'score':
          if scores is not None:
            label_text += '{:.01f}'.format(scores[i]) # score
        else:
          label_text += '{:.01f}'.format(rotations[i]) # rotation
        m = np.round(((s+e)/2)).astype(np.int32)
        if label_text != '':
          cv2.putText(img, label_text, (m[0]-4, m[1] - 4),
                      cv2.FONT_HERSHEY_COMPLEX, font_scale, text_color)
    return img

def draw_AlBox2D_UpRight_xyxy(img, objs, color, obj_thickness=1, font_scale=0.5, text_color='green'):
    '''
    img: [h,w,3]
    objs: [n,5/6]
    color: 'red'
    '''
    from beike_data_utils.line_utils import decode_line_rep
    assert objs.ndim == 2
    assert objs.shape[1]==5 or objs.shape[1]==6
    if objs.shape[1] == 6:
      scores = objs[:,5]
      objs = objs[:,:5]
    else:
      scores = None
    rotations = objs[:,4]
    text_color = _get_color(text_color)

    objs = decode_line_rep(objs, 'lscope_istopleft').reshape(-1,2,2)
    for i in range(objs.shape[0]):
        s, e = np.round(objs[i]).astype(np.int32)
        c = _get_color(color)
        cv2.rectangle(img, (s[0], s[1]), (e[0], e[1]), c, thickness=obj_thickness)

        #label_text = class_names[label] if class_names is not None else 'cls {}'.format(label)
        label_text = ''
        if DEBUG_CFG.OBJ_LEGEND == 'score':
          if scores is not None:
            label_text += '{:.01f}'.format(scores[i]) # score
        else:
          label_text += '{:.01f}'.format(rotations[i]) # rotation
        m = np.round(((s+e)/2)).astype(np.int32)
        if label_text != '':
          cv2.putText(img, label_text, (m[0]-4, m[1] - 4),
                      cv2.FONT_HERSHEY_COMPLEX, font_scale, text_color)
    return img

def _read_img(img_in):
  '''
  img_in: [h,w,3] or [h,w,1], or [h,w] or (h_size, w_size)
  img_out: [h,w,3]
  '''
  if isinstance(img_in, tuple):
    assert len(img_in) == 2
    img_size = img_in + (3,)
    img_out = np.zeros(img_size, dtype=np.float32)
  else:
    assert isinstance(img_in, np.ndarray)
    if img_in.ndim == 2:
      img_in = np.expand_dims(img_in, 2)
    assert img_in.ndim == 3
    if img_in.shape[2] == 1:
      img_in = np.tile(img_in, (1,1,3))
    img_out = img_in.copy()

    if img_out.max() <= 1:
      img_out *= 255
    img_out = img_out.astype(np.uint8)
  return img_out

def _get_color(color_str):
  assert isinstance( color_str, str), print(color_str)
  if color_str == 'random':
    c = get_random_color()
  else:
    c = color_val(color_str)
  return c

#-3d------------------------------------------------------------------------------

def _show_sparse_coords(x, gt_bboxes=None):
  C = x.C.cpu().data.numpy()
  F = x.F.cpu().data.numpy()
  if gt_bboxes is not None:
    gt_bboxes = [g.cpu().data.numpy() for g in gt_bboxes]

  batch_size = C[:,0].max()+1
  for i in range(batch_size):
    mask = C[:,0] == i
    Ci = C[mask][:,1:]
    Fi_ = F[mask]
    ci = Fi_.shape[1]
    Fi0 = Fi_[:,:ci//3].sum(1).reshape(-1, 1)
    Fi1 = Fi_[:,ci//3:ci//3*2].sum(1).reshape(-1, 1)
    Fi2 = Fi_[:,ci//3*2:ci].sum(1).reshape(-1, 1)
    Fi = np.concatenate([Fi0, Fi1, Fi2], axis=1)
    Fi = (Fi / Fi.max() * 255).astype(np.int)
    if gt_bboxes is None:
      _show_3d_points_bboxes_ls([Ci])
    else:
      _show_3d_points_lines_ls([Ci], [Fi], lines_ls = [gt_bboxes[i]])


#-3d general------------------------------------------------------------------------------
def _show_3d_points_objs_ls(points_ls=None, point_feats=None,
              objs_ls=None, obj_rep=None, obj_colors='random', box_types=None,
              polygons_ls=None, polygon_colors='random'):
  if objs_ls is not None:
    if obj_rep == 'XYXYSin2':
      if points_ls is not None:
        for i in range(len(points_ls)):
          if points_ls[i].shape[1] == 2:
            n = points_ls[i].shape[0]
            z = np.zeros([n,1])
            points_ls[i] = np.concatenate([points_ls[i], z], axis=1)

      if objs_ls is not None:
        for i in range(len(objs_ls)):
          n = objs_ls[i].shape[0]
          tzz = np.ones([n,3])
          tzz[:,0] = 0.01
          tzz[:,1] = 0
          tzz[:,2] = 0.01
          objs_ls[i] = np.concatenate([objs_ls[i], tzz], axis=1)
      obj_rep = 'XYXYSin2WZ0Z1'
    bboxes_ls = [OBJ_REPS_PARSE.encode_obj(o, obj_rep, 'XYZLgWsHA') for o in objs_ls]
  else:
    bboxes_ls = None
  mesh_ls = _show_3d_points_bboxes_ls(points_ls, point_feats, bboxes_ls, obj_colors, box_oriented=True, box_types=box_types)

  if polygons_ls is not None:
    pls = _make_polygon_surface_ls(polygons_ls, polygon_colors)
    mesh_ls += pls

  #o3d.visualization.draw_geometries( show_ls )
  custom_draw_geometry_with_key_callback( mesh_ls )

def _show_3d_points_bboxes_ls(points_ls=None, point_feats=None,
             bboxes_ls=None, b_colors='random', box_oriented=False, point_normals=None, box_types=None):
  show_ls = []
  if points_ls is not None:
    assert isinstance(points_ls, list)
    n = len(points_ls)
    if point_feats is not None:
      assert isinstance(point_feats, list)
    else:
      point_feats = [None] * n
    if point_normals is not None:
      assert isinstance(point_normals, list)
    else:
      point_normals = [None] * n
    for points, feats, normals in zip(points_ls, point_feats, point_normals):
      pcd = _make_pcd(points, feats, normals)
      show_ls.append(pcd)

  if bboxes_ls is not None:
    assert isinstance(bboxes_ls, list)
    if not isinstance(b_colors, list):
      b_colors = [b_colors] * len(bboxes_ls)
    if not isinstance(box_types, list):
      box_types = [box_types] * len(bboxes_ls)
    for i,bboxes in enumerate(bboxes_ls):
      bboxes_o3d = _make_bboxes_o3d(bboxes, box_oriented, b_colors[i], box_types[i])
      show_ls = show_ls + bboxes_o3d

  if ADD_FRAME:
    if points_ls is not None:
      center = points_ls[0][:,:3].mean(axis=0)
      center = points_ls[0].min(0)
      fsize = (points_ls[0].max() - points_ls[0].min())*0.1
    else:
      center =  (0,0,0)
      fsize = 1.0
    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=fsize, origin=center)
    show_ls.append(mesh_frame)

  return show_ls


def custom_draw_geometry_with_key_callback(pcd_ls):

    def change_background_to_black(vis):
        opt = vis.get_render_option()
        opt.background_color = np.asarray([0, 0, 0])
        return False

    def change_background_to_white(vis):
        opt = vis.get_render_option()
        opt.background_color = np.asarray([255, 255, 255])
        return False

    def load_render_option(vis):
        vis.get_render_option().load_from_json(
            "../../TestData/renderoption.json")
        return False

    def capture_depth(vis):
        depth = vis.capture_depth_float_buffer()
        plt.imshow(np.asarray(depth))
        plt.show()
        return False

    def capture_image(vis):
        image = vis.capture_screen_float_buffer()
        plt.imshow(np.asarray(image))
        plt.show()
        return False

    key_to_callback = {}
    key_to_callback[ord("K")] = change_background_to_black
    key_to_callback[ord("W")] = change_background_to_white
    #key_to_callback[ord("R")] = load_render_option
    #key_to_callback[ord(",")] = capture_depth
    #key_to_callback[ord(".")] = capture_image
    o3d.visualization.draw_geometries_with_key_callbacks(pcd_ls, key_to_callback)

def _make_bboxes_o3d(bboxes, box_oriented, color, box_type):
  assert bboxes.ndim==2
  if box_type == None:
    box_type = BOX_TYPE
  if box_oriented:
    assert bboxes.shape[1] == 7
  else:
    assert bboxes.shape[1] == 6
  if isinstance(color, np.ndarray) and color.shape[0] == bboxes.shape[0]:
    colors = _label2color(color)
  elif isinstance(color, list) and len(color) == bboxes.shape[0]:
    colors = [ _get_color(c) for c in color]
  else:
    colors = [ _get_color(color) for i in range(bboxes.shape[0]) ]

  min_size = bboxes[:,3:6].max() / 100
  bboxes[:,3:6] = np.clip(bboxes[:,3:6], a_min=min_size, a_max=None)
  bboxes_ = []
  for i,bbox in enumerate(bboxes):
    if box_type == 'line_mesh':
      bboxes_.append( _make_bbox_line_mesh(bbox, box_oriented, colors[i]) )
    elif box_type == 'line_set':
      bboxes_.append( _make_bbox_line_set(bbox, box_oriented, colors[i]) )
    elif box_type == 'surface_mesh':
      bboxes_.append( _make_bbox_surface_mesh(bbox, box_oriented, colors[i]) )
    else:
      raise NotImplementedError
  return bboxes_

def _make_bbox_line_set(bbox, box_oriented, color):
  '''
  box_oriented = True
  bbox: [7], :3 is center, 3:6 is size, 6:7 is angle
  '''
  assert bbox.ndim==1
  if not box_oriented:
    assert bbox.shape == (6,)
    bbox_ = o3d.geometry.AxisAlignedBoundingBox(bbox[:3], bbox[3:6])
  else:
    assert bbox.shape == (7,)
    center = bbox[:3]
    extent = bbox[3:6]
    axis_angle = np.array([0,0,bbox[6]])

    R = o3d.geometry.get_rotation_matrix_from_yxz(axis_angle)
    bbox_ = o3d.geometry.OrientedBoundingBox( center, R, extent )
  bbox_.color = color
  return bbox_

def _make_bbox_surface_mesh(bbox, box_oriented, color):
    assert bbox.shape == (7,)
    bbox = bbox.copy()
    center = bbox[:3]
    extent = bbox[3:6]
    axis_angle = np.array([0,0,bbox[6]])
    R = o3d.geometry.get_rotation_matrix_from_yxz(axis_angle)
    w,h,d = extent

    bbox_ = o3d.geometry.TriangleMesh.create_box(w, h, d)
    transformation = np.identity(4)
    transformation[:3,:3] = R
    transformation[:3,3] = R @( -extent/2)
    bbox_.transform(transformation)

    transformation = np.identity(4)
    center = center
    transformation[:3,3] = center
    bbox_.transform(transformation)

    bbox_.paint_uniform_color(color)
    return bbox_

def _make_bbox_line_mesh(bbox, box_oriented, color):
    assert box_oriented
    assert bbox.shape == (7,)
    radius = BOX_LINE_RADIUS
    #radius = bbox[3:6].max()/200

    lines = OBJ_REPS_PARSE.get_12_lines(bbox.reshape(1,7), 'XYZLgWsHA')[0] # [12,2,3]
    centroids = lines.mean(1)
    directions = lines[:,1,:]-lines[:,0,:]
    heights = np.linalg.norm(directions,axis=1)
    heights = np.clip(heights, a_min=0.001, a_max=None)
    directions = directions/heights.reshape([-1,1])
    mesh = []
    for i in range(12):
        cylinder_i = o3d.geometry.TriangleMesh.create_cylinder(radius=radius, height=heights[i])
        cylinder_i.paint_uniform_color(color)
        transformation = np.identity(4)
        transformation[:3,3] = centroids[i]
        transformation[:3,2] = directions[i]
        cylinder_i.transform(transformation)
        mesh.append(cylinder_i)

    cm = mesh[0]
    for i in range(1,12):
      cm += mesh[i]
    return cm

def _make_pcd(points, colors=None, normals=None):
    if points.shape[1] == 2:
      points = np.concatenate([points, points[:,0:1]*0], 1)
    if points.shape[1] == 6 and colors is None:
      colors = points[:,3:6]
      points = points[:,:3]
    assert points.shape[1] == 3
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    if colors is not None:
      assert colors.shape[0] == points.shape[0]
      colors = colors.reshape(points.shape[0], -1)
      if  colors.shape[1] == 3:
        if colors.max() > 1:
          colors = colors / 255
        pcd.colors = o3d.utility.Vector3dVector(colors)
      elif colors.shape[1] == 1:
        labels = colors
        pcd.colors = o3d.utility.Vector3dVector( _label2color(labels))

    if normals is not None:
      pcd.normals = o3d.utility.Vector3dVector(normals)
    return pcd

def points_to_bboxes(points, size=0.1):
  '''
  XYZLgWsHA
  '''
  n = points.shape[0]
  bboxes = np.zeros([n,7])
  bboxes[:,:3] = points
  bboxes[:,3:6] = size
  return bboxes


def _show_polygon_surface(points, color='red'):
    color = _get_color(color)
    mesh = _make_polygon_surface(points, color)
    custom_draw_geometry_with_key_callback([mesh])

def _make_polygon_surface_ls(points_ls, colors_ls):
    assert isinstance(points_ls, list)
    n = len(points_ls)
    if not isinstance(colors_ls, list):
      c = colors_ls
      colors_ls = [ _get_color( c )] * n
    else:
      colors_ls = [_get_color(c) for c in colors_ls]
    polygons_ls = []
    for i in range(n):
      pmesh =  _make_polygon_surface(points_ls[i], colors_ls[i])
      polygons_ls.append(pmesh)
    return polygons_ls

def _make_polygon_surface(points, color):
    pmin = points.min(0,keepdims=True)
    pmax = points.max(0,keepdims=True)
    center = (pmin + pmax)/2
    points = np.concatenate([points, center], 0)

    vertices = o3d.utility.Vector3dVector(points)
    n = points.shape[0] -1
    triangles = []
    for i in range(n-1):
      ids = np.array([i,i+1, n])
      ids = ids % n
      triangles.append( ids )
    triangles = np.array(triangles)
    triangles = o3d.utility.Vector3iVector(triangles)
    polygon = o3d.geometry.TriangleMesh( vertices, triangles )
    polygon.paint_uniform_color(color)
    return polygon

def _show_3d_as_img(bboxes3d, points_ls=None, obj_rep='RoBox3D_UpRight_xyxy_sin2a_thick_Z0Z1'):
    lines2d = bboxes3d[:,:5]
    voxel_size = 0.004
    lines2d[:,:4] /= voxel_size
    scope_min = lines2d[:,:4].reshape(-1,2).min(0, keepdims=True)
    lines2d[:,:4] = (lines2d[:,:4].reshape(-1,2) - scope_min).reshape(-1,4)
    scope_max = lines2d[:,:4].reshape(-1,2).max(0, keepdims=True)
    w, h = np.ceil(scope_max +10).astype(np.int32)[0]
    _show_objs_ls_points_ls( (h,w), [lines2d], 'XYXYSin2', points_ls)
#-feature------------------------------------------------------------------------------
def _show_feats(feats, gt_bboxes, stride):
  '''
  feats: [batch_size, c, h, w]
  gt_bboxes: []*batch_size
  '''
  feats = F.interpolate(feats, scale_factor=stride, mode='bilinear')
  feats =  feats.cpu().data.numpy()
  gt_bboxes =  [gt.cpu().data.numpy() for gt in gt_bboxes]

  batch_size = feats.shape[0]
  for i in range(batch_size):
    img = np.moveaxis(feats[i], [0,1,2], [2,0,1])
    cn = img.shape[2]
    img0 = img[:,:,:cn//3].sum(2)[:,:,None]
    img1 = img[:,:,cn//3:cn//2*2].sum(2)[:,:,None]
    img2 = img[:,:,cn//3*2:].sum(2)[:,:,None]
    img_3 = np.concatenate([img0, img1, img2], axis=2)
    gt = gt_bboxes[i]
    _show_objs_ls_points_ls( img_3, [gt], obj_rep='XYXYSin2', obj_colors='yellow' )
    pass

def test_rotation_order():
  u = np.pi/180
  XYLgWsA= np.array( [
    [200, 200, 200, 10, 0],
    [200, 200, 200, 30, 30*u],
  ] )
  obj_rep = 'XYLgWsA'
  img = np.zeros([512,512,3], dtype=np.uint8)
  #img = draw_XYLgWsA( img, XYLgWsA, 'red')
  #mmcv.imshow(img)

  corners = OBJ_REPS_PARSE.encode_obj(XYLgWsA, 'XYLgWsA', 'RoLine2D_2p').reshape(-1,2)
  XYZLgWsHA = OBJ_REPS_PARSE.encode_obj(XYLgWsA, 'XYLgWsA', 'XYZLgWsHA')
  XYZLgWsHA[:,5] = 10
  corners_3d = OBJ_REPS_PARSE.encode_obj(XYZLgWsHA, 'XYZLgWsHA', 'Bottom_Corners').reshape(-1,3)

  #_show_objs_ls_points_ls( (512,512), [XYLgWsA], obj_rep, points_ls=[corners])
  _show_3d_points_objs_ls([corners_3d], objs_ls=[XYZLgWsHA], obj_rep='XYZLgWsHA' )


  pass

def test_show_box():
  u = np.pi/180
  XYZLgWsHA= np.array( [
    [00, 200, 0, 200, 50, 5, 0*u],
    [300, 200, 0, 200, 50, 5, 30*u],
  ] )
  #lines = OBJ_REPS_PARSE.get_12_lines(XYZLgWsHA, 'XYZLgWsHA')
  _show_3d_points_objs_ls(objs_ls=[XYZLgWsHA], obj_rep='XYZLgWsHA')


if __name__  == '__main__':
  #test_rotation_order()
  test_show_box()

