#! /usr/bin/env python

""" EWAIM source code: an extensible web app for interactive mapping

    Links to helper functions and Bokeh plotting wrappers
"""

import argparse
from unittest import TestCase
import math
import csv
import numpy as np
import pandas as pd
import datashader as ds
import datashader.transfer_functions as tf
import dask.dataframe as dd
import dask
from bokeh.models import ColumnDataSource


""" Helper functions for dynamic plotting of Bokeh plot (frontend x server side) """

class DynamicMap:

    """
        Server-side dynamic Bokeh map using numpy data sources.

        Needs to update resolution on zoom / pan.
    """

    def __init__(self, bokeh_plot):
        """
            Initialize and setup callback

            Args:
                bokeh_plot (bokeh.plotting.figure) : plot for downsampling
                data (dict) : data source of the plots, contains all samples. Arrays
                              are expected to be numpy
                x_key (str): key for x axis in data
        """

        self.bokeh_plot = bokeh_plot
        #self.x_key = x_key
        #self.data = data
        self.last_step_size = 1

        # Copy of initial data
        #self.init_data = {}
        #self.cur_data = {}
        #for k in data:
        #    self.init_data[k] = data[k]
        #    self.cur_data[k] = data[k]

        # Register the callbacks
        #bokeh_plot.x_range.on_change('start', self.x_range_change_cb)
        #bokeh_plot.x_range.on_change('end', self.x_range_change_cb)
        #bokeh_plot.y_range.on_change('start', self.y_range_change_cb)
        #bokeh_plot.y_range.on_change('end', self.y_range_change_cb)

        bokeh_plot.x_range.on_change('start', self.image_callback)
        bokeh_plot.x_range.on_change('end', self.image_callback)
        #bokeh_plot.y_range.on_change('start', self.image_callback)
        #bokeh_plot.y_range.on_change('end', self.image_callback)

    def image_callback(self, attr, old, new):
        """ Bokeh server-side callback when x range changes (i.e. zooming) """

        x_range = self.bokeh_plot.x_range
        y_range = self.bokeh_plot.y_range
        w = self.bokeh_plot.plot_width
        h = self.bokeh_plot.plot_height
        #cvs = ds.Canvas(plot_width = w, plot_height = h, x_range = x_range, y_range)
        #agg = cvs.points(df, 'meterswest', 'metersnorth', ds.count_cat('race'))
        #img = tf.shade(agg, color_key = color_key, how = 'log')
        #return tf.dynspread(img,threshold = 0.75, max_px = 8)

        print("\n-------\nWithin image_callback....\n---------\n")

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


def check_args():
    parser = argparse.ArgumentParser(description = 'EWAIM: an extensible web app for interactive mapping')
    parser.add_argument('-sm', action = 'store', dest = 'simple_math',
                        help = 'Store a flag')
    parser.add_argument('-ff', action = 'store_true', default = False,
                        dest = 'fflag',
                        help = 'Store a flag')
    parser.add_argument('--version', action = 'version', version = '%(prog)s 1.0')

    results = parser.parse_args()
    return(results.simple_math. results.fflag)

def calculate(simple_math, f_flag = False):
    calc_return = eval(simple_math)
    return calc_return

def get_csv(csv_path = "./static/csv/carbon_sample_sm.csv"):
    #csv_path = "./static/csv/la-riots-deaths.csv"
    #csv_path = "./static/csv/carbon_sample_sm.csv"
    csv_file = open(csv_path, 'r')
    csv_obj = csv.DictReader(csv_file)
    return list(csv_obj)

def mean_lat_long(obj_csv):
    print("obj_csv: ", obj_csv)

    #return list(lat_mean)

"""

    WGS84 and map_projection functions from flight_review web app

"""

def WGS84_to_mercator(lon, lat):
    """ Convert lon, lat in [deg] to Mercator projection """
# alternative that relies on the pyproj library:
# import pyproj # GPS coordinate transformations
#    wgs84 = pyproj.Proj('+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs')
#    mercator = pyproj.Proj('+proj=merc +a=6378137 +b=6378137 +lat_ts=0.0 ' +
#       '+lon_0=0.0 +x_0=0.0 +y_0=0 +units=m +k=1.0 +nadgrids=@null +no_defs')
#    return pyproj.transform(wgs84, mercator, lon, lat)

    semimajor_axis = 6378137.0  # WGS84 spheriod semimajor axis
    east = lon * 0.017453292519943295
    north = lat * 0.017453292519943295
    northing = 3189068.5 * np.log((1.0 + np.sin(north)) / (1.0 - np.sin(north)))
    easting = semimajor_axis * east
    return easting, northing

def map_projection(lat, lon, anchor_lat, anchor_lon):
    """ convert lat, lon in [rad] to x, y in [m] with an anchor position """
    sin_lat = np.sin(lat)
    cos_lat = np.cos(lat)
    cos_d_lon = np.cos(lon - anchor_lon)
    sin_anchor_lat = np.sin(anchor_lat)
    cos_anchor_lat = np.cos(anchor_lat)

    arg = sin_anchor_lat * sin_lat + cos_anchor_lat * cos_lat * cos_d_lon
    arg[arg > 1] = 1
    arg[arg < -1] = -1

    np.set_printoptions(threshold=np.nan)
    c = np.arccos(arg)
    k = np.copy(lat)
    for i in range(len(lat)):
        if np.abs(c[i]) < np.finfo(float).eps:
            k[i] = 1
        else:
            k[i] = c[i] / np.sin(c[i])

    CONSTANTS_RADIUS_OF_EARTH = 6371000
    x = k * (cos_anchor_lat * sin_lat - sin_anchor_lat * cos_lat * cos_d_lon) * \
        CONSTANTS_RADIUS_OF_EARTH
    y = k * cos_lat * np.sin(lon - anchor_lon) * CONSTANTS_RADIUS_OF_EARTH
    return x, y


if __name__ == '__main__':
    cli_args = check_args()
    #raw_result = calculate(cli_args[0], f_flag = cli_args[1])
    #print('\n-----------------')
    #print('The result is: ', raw_result)
    #print('-----------------\n')
    calculate(cli_args[0], f_flag = cli_args[1])
