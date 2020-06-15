
classes_order = ['background', 'wall', 'door', 'window', 'room', 'other']
def sort_classes(classes_in):
      assert all([c in classes_order for c in classes_in])
      classes = [c for c in classes_order if c in classes_in]
      return classes
