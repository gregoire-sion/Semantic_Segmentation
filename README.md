
(Environnement) scvmpr10.fr.mbda.priv:/home/gsionsua/Work_bis/Bases/Projet $python simulation.py 
Scénario A : tous capteurs, biais estimé...
Scénario B : tous capteurs, biais NON estimé...
Scénario C : sans distances, biais NON estimé...
Scénario D : sans distances, biais estimé...

=== MSE position drone 2 ===
  A - Complet + biais estimé               : 4.3417
  B - Complet, biais non estimé            : 6.7388
  C - Sans distances, biais non estimé     : 498.5122
  D - Sans distances, biais estimé         : 986.0845
Exception in Tkinter callback
Traceback (most recent call last):
  File "/home/gsionsua/Work_bis/Environnement/lib/python3.10/site-packages/matplotlib/lines.py", line 207, in _mark_every_path
    return Path(verts[markevery], _slice_or_none(codes, markevery))
  File "/home/gsionsua/Work_bis/Environnement/lib/python3.10/site-packages/matplotlib/path.py", line 130, in __init__
    _api.check_shape((None, 2), vertices=vertices)
  File "/home/gsionsua/Work_bis/Environnement/lib/python3.10/site-packages/matplotlib/_api/__init__.py", line 162, in check_shape
    raise ValueError(
ValueError: 'vertices' must be 2D with shape (N, 2), but your input has shape (1, 1, 2)

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "/home/gsionsua/Work_bis/Environnement/lib/python3.10/tkinter/__init__.py", line 1921, in __call__
    return self.func(*args)
  File "/home/gsionsua/Work_bis/Environnement/lib/python3.10/tkinter/__init__.py", line 839, in callit
    func(*args)
  File "/home/gsionsua/Work_bis/Environnement/lib/python3.10/site-packages/matplotlib/backends/_backend_tk.py", line 274, in idle_draw
    self.draw()
  File "/home/gsionsua/Work_bis/Environnement/lib/python3.10/site-packages/matplotlib/backends/backend_tkagg.py", line 10, in draw
    super().draw()
  File "/home/gsionsua/Work_bis/Environnement/lib/python3.10/site-packages/matplotlib/backends/backend_agg.py", line 382, in draw
    self.figure.draw(self.renderer)
  File "/home/gsionsua/Work_bis/Environnement/lib/python3.10/site-packages/matplotlib/artist.py", line 94, in draw_wrapper
    result = draw(artist, renderer, *args, **kwargs)
  File "/home/gsionsua/Work_bis/Environnement/lib/python3.10/site-packages/matplotlib/artist.py", line 71, in draw_wrapper
    return draw(artist, renderer)
  File "/home/gsionsua/Work_bis/Environnement/lib/python3.10/site-packages/matplotlib/figure.py", line 3257, in draw
    mimage._draw_list_compositing_images(
  File "/home/gsionsua/Work_bis/Environnement/lib/python3.10/site-packages/matplotlib/image.py", line 134, in _draw_list_compositing_images
    a.draw(renderer)
  File "/home/gsionsua/Work_bis/Environnement/lib/python3.10/site-packages/matplotlib/artist.py", line 71, in draw_wrapper
    return draw(artist, renderer)
  File "/home/gsionsua/Work_bis/Environnement/lib/python3.10/site-packages/matplotlib/axes/_base.py", line 3210, in draw
    mimage._draw_list_compositing_images(
  File "/home/gsionsua/Work_bis/Environnement/lib/python3.10/site-packages/matplotlib/image.py", line 134, in _draw_list_compositing_images
    a.draw(renderer)
  File "/home/gsionsua/Work_bis/Environnement/lib/python3.10/site-packages/matplotlib/artist.py", line 71, in draw_wrapper
    return draw(artist, renderer)
  File "/home/gsionsua/Work_bis/Environnement/lib/python3.10/site-packages/matplotlib/lines.py", line 853, in draw
    subsampled = _mark_every_path(
  File "/home/gsionsua/Work_bis/Environnement/lib/python3.10/site-packages/matplotlib/lines.py", line 209, in _mark_every_path
    raise ValueError(
ValueError: markevery=[(53,)] is iterable but not a valid numpy fancy index
Exception in Tkinter callback
Traceback (most recent call last):
  File "/home/gsionsua/Work_bis/Environnement/lib/python3.10/site-packages/matplotlib/lines.py", line 207, in _mark_every_path
    return Path(verts[markevery], _slice_or_none(codes, markevery))
  File "/home/gsionsua/Work_bis/Environnement/lib/python3.10/site-packages/matplotlib/path.py", line 130, in __init__
    _api.check_shape((None, 2), vertices=vertices)
  File "/home/gsionsua/Work_bis/Environnement/lib/python3.10/site-packages/matplotlib/_api/__init__.py", line 162, in check_shape
    raise ValueError(
ValueError: 'vertices' must be 2D with shape (N, 2), but your input has shape (1, 1, 2)

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "/home/gsionsua/Work_bis/Environnement/lib/python3.10/tkinter/__init__.py", line 1921, in __call__
    return self.func(*args)
  File "/home/gsionsua/Work_bis/Environnement/lib/python3.10/tkinter/__init__.py", line 839, in callit
    func(*args)
  File "/home/gsionsua/Work_bis/Environnement/lib/python3.10/site-packages/matplotlib/backends/_backend_tk.py", line 274, in idle_draw
    self.draw()
  File "/home/gsionsua/Work_bis/Environnement/lib/python3.10/site-packages/matplotlib/backends/backend_tkagg.py", line 10, in draw
    super().draw()
  File "/home/gsionsua/Work_bis/Environnement/lib/python3.10/site-packages/matplotlib/backends/backend_agg.py", line 382, in draw
    self.figure.draw(self.renderer)
  File "/home/gsionsua/Work_bis/Environnement/lib/python3.10/site-packages/matplotlib/artist.py", line 94, in draw_wrapper
    result = draw(artist, renderer, *args, **kwargs)
  File "/home/gsionsua/Work_bis/Environnement/lib/python3.10/site-packages/matplotlib/artist.py", line 71, in draw_wrapper
    return draw(artist, renderer)
  File "/home/gsionsua/Work_bis/Environnement/lib/python3.10/site-packages/matplotlib/figure.py", line 3257, in draw
    mimage._draw_list_compositing_images(
  File "/home/gsionsua/Work_bis/Environnement/lib/python3.10/site-packages/matplotlib/image.py", line 134, in _draw_list_compositing_images
    a.draw(renderer)
  File "/home/gsionsua/Work_bis/Environnement/lib/python3.10/site-packages/matplotlib/artist.py", line 71, in draw_wrapper
    return draw(artist, renderer)
  File "/home/gsionsua/Work_bis/Environnement/lib/python3.10/site-packages/matplotlib/axes/_base.py", line 3210, in draw
    mimage._draw_list_compositing_images(
  File "/home/gsionsua/Work_bis/Environnement/lib/python3.10/site-packages/matplotlib/image.py", line 134, in _draw_list_compositing_images
    a.draw(renderer)
  File "/home/gsionsua/Work_bis/Environnement/lib/python3.10/site-packages/matplotlib/artist.py", line 71, in draw_wrapper
    return draw(artist, renderer)
  File "/home/gsionsua/Work_bis/Environnement/lib/python3.10/site-packages/matplotlib/lines.py", line 853, in draw
    subsampled = _mark_every_path(
  File "/home/gsionsua/Work_bis/Environnement/lib/python3.10/site-packages/matplotlib/lines.py", line 209, in _mark_every_path
    raise ValueError(
ValueError: markevery=[(53,)] is iterable but not a valid numpy fancy index
