import math
import numpy as np
import os.path as osp
import matplotlib.pyplot as plt
import matplotlib as mpl

import os.path as osp
import tempfile
import xml.etree.ElementTree as ET
from core.serializable import Serializable
import os

mpl.use('Agg')

# code adapted from rllab: https://rllab.readthedocs.io/en/latest/index.html#


def line_intersect(pt1, pt2, ptA, ptB):
    """
    Taken from https://www.cs.hmc.edu/ACM/lectures/intersections.html

    this returns the intersection of Line(pt1,pt2) and Line(ptA,ptB)

    returns a tuple: (xi, yi, valid, r, s), where
    (xi, yi) is the intersection
    r is the scalar multiple such that (xi,yi) = pt1 + r*(pt2-pt1)
    s is the scalar multiple such that (xi,yi) = pt1 + s*(ptB-ptA)
    valid == 0 if there are 0 or inf. intersections (invalid)
    valid == 1 if it has a unique intersection ON the segment
    """

    DET_TOLERANCE = 0.00000001

    # the first line is pt1 + r*(pt2-pt1)
    # in component form:
    x1, y1 = pt1
    x2, y2 = pt2
    dx1 = x2 - x1
    dy1 = y2 - y1

    # the second line is ptA + s*(ptB-ptA)
    x, y = ptA
    xB, yB = ptB
    dx = xB - x
    dy = yB - y

    # we need to find the (typically unique) values of r and s
    # that will satisfy
    #
    # (x1, y1) + r(dx1, dy1) = (x, y) + s(dx, dy)
    #
    # which is the same as
    #
    #    [ dx1  -dx ][ r ] = [ x-x1 ]
    #    [ dy1  -dy ][ s ] = [ y-y1 ]
    #
    # whose solution is
    #
    #    [ r ] = _1_  [  -dy   dx ] [ x-x1 ]
    #    [ s ] = DET  [ -dy1  dx1 ] [ y-y1 ]
    #
    # where DET = (-dx1 * dy + dy1 * dx)
    #
    # if DET is too small, they're parallel
    #
    DET = (-dx1 * dy + dy1 * dx)

    if math.fabs(DET) < DET_TOLERANCE:
        return (0, 0, 0, 0, 0)

    # now, the determinant should be OK
    DETinv = 1.0 / DET

    # find the scalar amount along the "self" segment
    r = DETinv * (-dy * (x - x1) + dx * (y - y1))

    # find the scalar amount along the input line
    s = DETinv * (-dy1 * (x - x1) + dx1 * (y - y1))

    # return the average of the two descriptions
    xi = (x1 + r * dx1 + x + s * dx) / 2.0
    yi = (y1 + r * dy1 + y + s * dy) / 2.0
    return (xi, yi, 1, r, s)


def ray_segment_intersect(ray, segment):
    """
    Check if the ray originated from (x, y) with direction theta intersects the line segment (x1, y1) -- (x2, y2),
    and return the intersection point if there is one
    """
    (x, y), theta = ray
    # (x1, y1), (x2, y2) = segment
    pt1 = (x, y)
    len = 1
    pt2 = (x + len * math.cos(theta), y + len * math.sin(theta))
    xo, yo, valid, r, s = line_intersect(pt1, pt2, *segment)
    if valid and r >= 0 and 0 <= s <= 1:
        return (xo, yo)
    return None


def point_distance(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5


def convert_point_to_plot(point, size_scaling):
    return (np.array(point) * np.array([2/size_scaling, -2/size_scaling])) + np.array([3, 3])


def plot_ray(reading, structure, size_scaling, robot_xy, ori, sensor_range, sensor_span, n_bins, xy_goal, plot_file="plot.png", color='r'):
    # duplicate cells to plot the maze
    # print(np.array(structure))

    structure_plot = np.zeros(
        ((len(structure)) * 2, (len(structure[0])) * 2))
    for i in range(len(structure)):
        for j in range(len(structure[0])):
            cell = structure[i][j]
            if type(cell) is not int:
                cell = 0.3 if cell == 'r' else 0.7

            structure_plot[2 * i:2 * (i+1), 2 * j: 2 * (j + 1)] = cell

    fig, ax = plt.subplots()
    im = ax.pcolor(-np.array(structure_plot), cmap='tab20c',
                   edgecolor='black', linestyle=':', lw=1)
    x_labels = np.array(
        range(len(structure[0])+1))*size_scaling-size_scaling*1.5
    y_labels = -np.array(range(len(structure)+1)) * \
        size_scaling+size_scaling*1.5

    #from matplotlib.ticker import MultipleLocator
    # spacing = 0.5  # This can be your user specified spacing.
    #minorLocator = MultipleLocator(spacing)
    # Set minor tick locations.
    # ax.yaxis.set_minor_locator(minorLocator)
    # ax.xaxis.set_minor_locator(minorLocator)
    # Set grid to use minor tick locations.
    # ax.grid(which='minor')
    ax.grid(True)  # elimiate this to avoid inner lines

    # the coordinates of this are wrt the init!!
    # for Ant this is computed with atan2, which gives [-pi, pi]

    # compute origin cell i_o, j_o coordinates and center of it x_o, y_o (with 0,0 in the top-right corner of struc)
    # this is self.init_torso_x, self.init_torso_y !!: center of the cell xy!
    o_xy = robot_xy
    # this is the position in the grid (check if correct..)

    # o_xy_plot = o_xy / size_scaling * 2
    robot_xy_plot = convert_point_to_plot(o_xy, size_scaling)

    # plot goal
    xy_goal_plot = convert_point_to_plot(xy_goal, size_scaling)

    for ray_idx in range(n_bins):
        length_wall = reading[ray_idx] * sensor_range

        ray_ori = ori - sensor_span * 0.5 + 1.0 * \
            (2 * ray_idx + 1) / (2 * n_bins) * sensor_span

        # if ray_ori > math.pi:
        #    ray_ori -= 2 * math.pi
        # elif ray_ori < - math.pi:
        #    ray_ori += 2 * math.pi

        # find the end point wall
        end_xy = (robot_xy + length_wall *
                  np.array([math.cos(ray_ori), math.sin(ray_ori)]))
        end_xy_plot = convert_point_to_plot(end_xy, size_scaling)
        plt.plot([robot_xy_plot[0], end_xy_plot[0]], [
            robot_xy_plot[1], end_xy_plot[1]], color=color, )

    draw_circle = plt.Circle(robot_xy_plot, 0.25, color='b')
    ax.add_artist(draw_circle)

    # plot vector
    plt.quiver(*robot_xy_plot, 3*math.cos(ori), 3 *
               math.sin(ori), color='b', scale=25)

    # visualisation
    ax.xaxis.set(ticks=2 * np.arange(len(x_labels)), ticklabels=x_labels)
    ax.yaxis.set(ticks=2 * np.arange(len(y_labels)), ticklabels=y_labels)
    ax.set_aspect(1)

    plt.gca().invert_yaxis()

    ax.set_title('sensors debug')
    # print('plotting now, close the window')
    plt.savefig(plot_file)
    # plt.show(fig)
    plt.close()


class MazeEnv(Serializable):

    def __init__(
            self,
            maze_height=0.5,
            maze_size_scaling=3,
            *args,
            **kwargs):
        Serializable.quick_init(self, locals())

        xml_path = os.path.join(os.path.dirname(
            __file__), 'assets', 'ant_hfield.xml')
        tree = ET.parse(xml_path)
        worldbody = tree.find(".//worldbody")

        self.maze_make_contact = True

        self.maze_height = maze_height
        self.maze_size_scaling = maze_size_scaling
        self.maze_structure = [
            [1, 1, 1, 1, 1],
            [0, 'r', 0, 0, 1],
            [1, 1, 1, 0, 1],
            [1, 0, 0, 'g', 1],
            [1, 1, 1, 1, 1],
        ]

        torso_x, torso_y = self._find_robot()
        self._init_torso_x = torso_x
        self._init_torso_y = torso_y

        for i in range(len(self.maze_structure)):
            for j in range(len(self.maze_structure[0])):
                if str(self.maze_structure[i][j]) == '1':
                    # offset all coordinates so that robot starts at the origin
                    ET.SubElement(
                        worldbody, "geom",
                        name="block_%d_%d" % (i, j),
                        pos="%f %f %f" % (j * self.maze_size_scaling - torso_x,
                                          -(i * self.maze_size_scaling - torso_y),
                                          self.maze_height),
                        size="%f %f %f" % (0.5 * self.maze_size_scaling,
                                           0.5 * self.maze_size_scaling,
                                           self.maze_height),
                        type="box",
                        material="",
                        contype="1",
                        conaffinity="1",
                        rgba="0.4 0.4 0.4 1"
                    )

        torso = tree.find(".//body[@name='torso']")
        goal = tree.find(".//body[@name='goal']")
        geoms = torso.findall(".//geom")
        goal_geom = goal.findall(".//geom")

        cameras = worldbody.findall(".//camera")
        for camera in cameras:
            if camera.get("name") == "top_fixed":
                cam_x = int((len(self.maze_structure)//2) *
                            self.maze_size_scaling-torso_x-self.maze_size_scaling/2)
                cam_y = int(
                    (-len(self.maze_structure[0]))*self.maze_size_scaling+torso_y + self.maze_size_scaling*2)

                camera_pos = str(cam_x)+" "+str(cam_y-2)+" 25"
                camera.set("pos", camera_pos)

        for geom in geoms:
            if 'name' not in geom.attrib:
                raise Exception("Every geom of the torso must have a name "
                                "defined")

        if self.maze_make_contact:
            contact = ET.SubElement(
                tree.find("."), "contact"
            )
            for i in range(len(self.maze_structure)):
                for j in range(len(self.maze_structure[0])):
                    if str(self.maze_structure[i][j]) == '1':
                        for geom in geoms:
                            ET.SubElement(
                                contact, "pair",
                                geom1=geom.attrib["name"],
                                geom2="block_%d_%d" % (i, j)
                            )
            # floor contact
            for geom in geoms:
                ET.SubElement(
                    contact, "pair",
                    geom1=geom.attrib["name"],
                    geom2="floor"
                )

        # adapt camera position

        #_, file_path = tempfile.mkstemp(text=True)
        # here we write a temporal file with the robot specifications. Why not the original one??
        self.maze_xmlfile_path = os.path.join(os.path.dirname(
            __file__), 'assets', 'ant_hfield_maze.xml')
        tree.write(self.maze_xmlfile_path)

    def _find_robot(self):
        structure = self.maze_structure
        size_scaling = self.maze_size_scaling
        for i in range(len(structure)):
            for j in range(len(structure[0])):
                if structure[i][j] == 'r':
                    return j * size_scaling, i * size_scaling
        assert False

    def _find_goal(self):
        structure = self.maze_structure
        size_scaling = self.maze_size_scaling
        for i in range(len(structure)):
            for j in range(len(structure[0])):
                if structure[i][j] == 'g':
                    return j * size_scaling, i * size_scaling
        assert False
