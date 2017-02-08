#! /usr/bin/env python

""" Helper functions for dynamic plotting of Bokeh plot (frontend x server side) """

import numpy as np
from bokeh.models import ColumnDataSource

class DynamicMap:

    """
        Server-side dynamic Bokeh map using numpy data sources.

        Needs to update resolution on zoom / pan.
    """

    def __init__(self, bokeh_plot, data, x_key):
        """
            Initialize and setup callback

            Args:
                bokeh_plot (bokeh.plotting.figure) : plot for downsampling
                data (dict) : data source of the plots, contains all samples. Arrays
                              are expected to be numpy
                x_key (str): key for x axis in data
        """

        self.bokeh_plot = bokeh_plot
        self.x_key = x_key
        self.data = data
        self.last_step_size = 1

        # Copy of initial data
        self.init_data = {}
        self.cur_data = {}
        for k in data:
            self.init_data[k] = data[k]
            self.cur_data[k] = data[k]

        # Register the callbacks
        bokeh_plot.x_range.on_change('start', self.x_range_change_cb)
        bokeh_plot.x_range.on_change('end', self.x_range_change_cb)

        bokeh_plot.y_range.on_change('start', self.y_range_change_cb)
        bokeh_plot.y_range.on_change('end', self.y_range_change_cb)

    def x_range_change_cb(self, attr, old, new):
        """ Bokeh server-side callback when x range changes (i.e. zooming) """

        new_x_range = [self.bokeh_plot.x_range.start, self.bokeh_plot.x_range.end]
        if None in new_x_range:
            return

        plot_width = self.bokeh_plot.plot_width
        init_x = self.init_data[self.x_key]
        cur_x = self.cur_data[self.x_key]
        cur_x_range = [cur_x[0], cur_x[-1]]

        need_update = False
        if (new_x_range[0] < cur_x_range[0] and cur_x_range[0] > init_x[0]) or \
                (new_x_range[1] > cur_x_range[1] and cur_x_range[1] < init_x[-self.last_step_size]):
            need_update = True # zooming out / panning

        if need_update:
            drange = new_x_range[1] - new_x_range[0]
            new_x_range[0] -= drange * self.range_margin
            new_x_range[1] += drange * self.range_margin
            #num_data_points = plot_width * self.init_density * (1 + 2 * self.range_margin)
            indices = np.logical_and(init_x > new_x_range[0], init_x < new_x_range[1])

            self.cur_data = {}
            for k in self.init_data:
                self.cur_data[k] = self.init_data[k][indices]

            self.data_source.data = self.cur_data

    def y_range_change_cb(self, attr, old, new):
        """ Bokeh server-side callback when y range changes (i.e. zooming) """

        new_y_range = [self.bokeh_plot.y_range.start, self.bokeh_plot.y_range.end]
        if None in new_y_range:
            return

        plot_width = self.bokeh_plot.plot_width
        init_y = self.init_data[self.y_key]
        cur_y = self.cur_data[self.y_key]
        cur_y_range = [cur_y[0], cur_y[-1]]

        need_update = False
        if (new_y_range[0] < cur_y_range[0] and cur_y_range[0] > init_y[0]) or \
                (new_y_range[1] > cur_y_range[1] and cur_y_range[1] < init_y[-self.last_step_size]):
            need_update = True # zooming out / panning

        if need_update:
            drange = new_y_range[1] - new_y_range[0]
            new_y_range[0] -= drange * self.range_margin
            new_y_range[1] += drange * self.range_margin
            #num_data_points = plot_width * self.init_density * (1 + 2 * self.range_margin)
            indices = np.logical_and(init_y > new_y_range[0], init_y < new_y_range[1])

            self.cur_data = {}
            for k in self.init_data:
                self.cur_data[k] = self.init_data[k][indices]

            self.data_source.data = self.cur_data
