# -*- coding: utf-8 -*-
import numpy as np
import visdom


class VisdomShow(object):
    def __init__(self):
        self.vis = visdom.Visdom()
        self.vis.close()

    def line(self, X, Y, win, opts):
        if isinstance(Y, float):
            Y = np.expand_dims(Y, axis=0)
        if isinstance(X, int):
            X = np.expand_dims(X, axis=0)
        win_res = self.vis.line(X=X, Y=Y, win=win, update='append')
        if win_res != win:
            self.vis.line(X=X, Y=Y, win=win, opts=opts)


viz = VisdomShow()
