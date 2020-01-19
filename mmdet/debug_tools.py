import torch

def show_multi_ls_shapes(in_ls, names_ls, env):
  for i,ls in enumerate(in_ls):
    show_shapes(ls, env + ' - ' + names_ls[i])

def show_shapes(tensor_ls, flag=''):
  print(f'\n{flag}:')
  _show_tensor_ls_shapes(tensor_ls, i='', pre='')
  print(f'\n')

def _show_tensor_ls_shapes(tensor_ls, flag='', i='', pre=''):
  if isinstance(tensor_ls, torch.Tensor):
    shape = tensor_ls.shape
    print(f'{pre} {i} \t{shape}')
  else:
    pre += '  '
    for i,tensor in enumerate(tensor_ls):
      _show_tensor_ls_shapes(tensor, flag, i, pre)

