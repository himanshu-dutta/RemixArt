import os
import string


def listdepth(path,depth,type='f')
    path = os.path.normpath(path)
    res = {}
    for root,dirs,files in os.walk(path, topdown=True):
        dep = root[len(path) + len(os.path.sep):].count(os.path.sep)
        if type == 'f': df = files
        else: df = dirs
        if dep == depth:
            res.update({f[:-5]:os.path.join(root, f) for f in df})
            dirs[:] = [] # Don't recurse any deeper
    return res