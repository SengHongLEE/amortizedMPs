# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

import numpy as np
from scipy import interpolate

from isaacgym import terrain_utils
from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg

class Terrain:
    def __init__(self, cfg: LeggedRobotCfg.terrain, num_robots) -> None:

        self.cfg = cfg
        self.num_robots = num_robots
        self.type = cfg.mesh_type
        if self.type in ["none", 'plane']:
            return
        self.env_length = cfg.terrain_length
        self.env_width = cfg.terrain_width
        self.proportions = [np.sum(cfg.terrain_proportions[:i+1]) for i in range(len(cfg.terrain_proportions))]

        self.cfg.num_sub_terrains = cfg.num_rows * cfg.num_cols
        self.env_origins = np.zeros((cfg.num_rows, cfg.num_cols, 3))

        self.width_per_env_pixels = int(self.env_width / cfg.horizontal_scale)
        self.length_per_env_pixels = int(self.env_length / cfg.horizontal_scale)

        self.border = int(cfg.border_size/self.cfg.horizontal_scale)
        self.tot_cols = int(cfg.num_cols * self.length_per_env_pixels) + (cfg.num_cols + 1) * self.border
        self.tot_rows = int(cfg.num_rows * self.width_per_env_pixels) + (cfg.num_rows + 1) * self.border

        self.height_field_raw = np.zeros((self.tot_rows , self.tot_cols), dtype=np.int16)
        for m in range(self.cfg.num_cols):
            for n in range(self.cfg.num_rows):
                (i, j) = np.unravel_index(m * self.cfg.num_rows + n, (self.cfg.num_cols, self.cfg.num_rows))
                terrain = terrain_utils.SubTerrain("terrain",
                        width=self.width_per_env_pixels,
                        length=self.length_per_env_pixels,
                        vertical_scale=self.cfg.vertical_scale,
                        horizontal_scale=self.cfg.horizontal_scale)

                tmp = np.random.randint(1,3)
                if tmp == 1:
                    max_height = np.random.choice([0.06])
                    random_uniform_terrain(terrain, max_height=max_height)
                elif tmp == 2:
                    hurdle_height = np.random.choice([0.08]) 
                    hurdle_terrain(terrain, [0.0, hurdle_height], 10, 0.1, 'random') 

                self.add_terrain_to_map(terrain, j, i, 0)  

        self.heightsamples = self.height_field_raw
        if self.type=="trimesh":
            self.vertices, self.triangles = terrain_utils.convert_heightfield_to_trimesh(   self.height_field_raw,
                                                                                            self.cfg.horizontal_scale,
                                                                                            self.cfg.vertical_scale,
                                                                                            self.cfg.slope_treshold)
    
    def randomized_terrain(self):
        for k in range(self.cfg.num_sub_terrains):
            # Env coordinates in the world
            (i, j) = np.unravel_index(k, (self.cfg.num_rows, self.cfg.num_cols))

            choice = np.random.uniform(0, 1)
            difficulty = np.random.choice([0.5, 0.75, 0.9])
            terrain = self.make_terrain(choice, difficulty)
            self.add_terrain_to_map(terrain, i, j)
        
    def curiculum(self):
        for j in range(self.cfg.num_cols):
            for i in range(self.cfg.num_rows):
                difficulty = i / self.cfg.num_rows
                choice = j / self.cfg.num_cols + 0.001

                terrain = self.make_terrain(choice, difficulty)
                self.add_terrain_to_map(terrain, i, j)

    def selected_terrain(self):
        terrain_type = self.cfg.terrain_kwargs.pop('type')
        for k in range(self.cfg.num_sub_terrains):
            # Env coordinates in the world
            (i, j) = np.unravel_index(k, (self.cfg.num_rows, self.cfg.num_cols))

            terrain = terrain_utils.SubTerrain("terrain",
                              width=self.width_per_env_pixels,
                              length=self.length_per_env_pixels,
                              vertical_scale=self.cfg.vertical_scale,
                              horizontal_scale=self.cfg.horizontal_scale)

            eval(terrain_type)(terrain, **self.cfg.terrain_kwargs.terrain_kwargs)
            self.add_terrain_to_map(terrain, i, j)
    
    def make_terrain(self, choice, difficulty):
        terrain = terrain_utils.SubTerrain(   "terrain",
                                width=self.width_per_env_pixels,
                                length=self.length_per_env_pixels,
                                vertical_scale=self.cfg.vertical_scale,
                                horizontal_scale=self.cfg.horizontal_scale)
        slope = difficulty * 0.4
        step_height = 0.05 + 0.18 * difficulty
        discrete_obstacles_height = 0.05 + difficulty * 0.2
        stepping_stones_size = 1.5 * (1.05 - difficulty)
        stone_distance = 0.05 if difficulty==0 else 0.1
        gap_size = 1. * difficulty
        pit_depth = 1. * difficulty
        if choice < self.proportions[0]:
            if choice < self.proportions[0]/ 2:
                slope *= -1
            terrain_utils.pyramid_sloped_terrain(terrain, slope=slope, platform_size=3.)
        elif choice < self.proportions[1]:
            terrain_utils.pyramid_sloped_terrain(terrain, slope=slope, platform_size=3.)
            terrain_utils.random_uniform_terrain(terrain, min_height=-0.05, max_height=0.05, step=0.005, downsampled_scale=0.2)
        elif choice < self.proportions[3]:
            if choice<self.proportions[2]:
                step_height *= -1
            terrain_utils.pyramid_stairs_terrain(terrain, step_width=0.31, step_height=step_height, platform_size=3.)
        elif choice < self.proportions[4]:
            num_rectangles = 20
            rectangle_min_size = 1.
            rectangle_max_size = 2.
            terrain_utils.discrete_obstacles_terrain(terrain, discrete_obstacles_height, rectangle_min_size, rectangle_max_size, num_rectangles, platform_size=3.)
        elif choice < self.proportions[5]:
            terrain_utils.stepping_stones_terrain(terrain, stone_size=stepping_stones_size, stone_distance=stone_distance, max_height=0., platform_size=4.)
        elif choice < self.proportions[6]:
            gap_terrain(terrain, gap_size=gap_size, platform_size=3.)
        else:
            pit_terrain(terrain, depth=pit_depth, platform_size=4.)
        
        return terrain

    def add_terrain_to_map(self, terrain, row, col, env_origin_x = 0, env_origin_z = 0):
        i = row
        j = col
        # map coordinate system
        start_x = self.border + i * (self.width_per_env_pixels + self.border)
        end_x = (i + 1) * (self.width_per_env_pixels + self.border)
        start_y = self.border + j * (self.length_per_env_pixels + self.border)
        end_y = (j + 1) * (self.length_per_env_pixels + self.border)
        self.height_field_raw[start_x: end_x, start_y:end_y] = terrain.height_field_raw
        # self.height_field_raw[:, end_y:end_y + self.border] = -10000
        # self.height_field_raw[end_x:end_x + int(0.1 * self.border), :] = 1000

        env_origin_x += i * self.env_width + (i - 1) * self.cfg.border_size + 0.5
        # env_origin_y = (j + 0.5) * self.env_length
        # x1 = int((self.env_width/2. - 1) / terrain.horizontal_scale)
        # x2 = int((self.env_width/2. + 1) / terrain.horizontal_scale)
        # y1 = int((self.env_length/2. - 1) / terrain.horizontal_scale)
        # y2 = int((self.env_length/2. + 1) / terrain.horizontal_scale)
        # env_origin_z = np.max(terrain.height_field_raw[x1:x2, y1:y2])*terrain.vertical_scale
        # if j == 0:
        # self.height_field_raw[:, :self.border] = -10000
        # self.height_field_raw[:self.border, :] = -10000
        self.env_origins[i, j] = [env_origin_x, 0.5 * (start_y + end_y) * self.cfg.horizontal_scale - 1., env_origin_z]


def wide_gap_terrain(terrain, gap_size=[.5, 1.2], number_of_gap:int=1, platform_width=None, type:str="linear", verbose: bool = False):
    gap_size = np.array(gap_size) / terrain.horizontal_scale
    gap_width = np.linspace(gap_size[0], gap_size[1], num=number_of_gap) if type == "linear" else np.ones(number_of_gap) * np.random.choice(np.arange(gap_size[0], gap_size[1] + 1, 1))
    # gap_width = np.clip(gap_width,a_min=0,a_max=3)
    # gap_width = np.array([0.3,0.6,0.9,1.16])
    # gap_width = (gap_width * 1 / terrain.horizontal_scale)
    total_gap_width = np.sum(gap_width)
    if verbose:
        print(gap_width)
    platform_width = int((terrain.width - total_gap_width) / (number_of_gap + 1)) if platform_width == None else int(platform_width / terrain.horizontal_scale)
    # residual_platform_width = (terrain.width - total_gap_width) % (number_of_gap + 1)

    # if total_gap_width + (number_of_gap + 1) * platform_width + residual_platform_width != terrain.width:
    #     raise ValueError(f"The sum of total gap width and platforms does not equal the terrain width: "
    #                     f"{total_gap_width} + {number_of_gap + 1} * {platform_width} + {residual_platform_width} = "
    #                  f"{total_gap_width + (number_of_gap + 1) * platform_width + residual_platform_width}, expected {terrain.width}")

    half_residual_platform_width = int(0.5 * (terrain.width - total_gap_width - (number_of_gap + 1) * platform_width))
    # print(half_residual_platform_width)
    for i in range(number_of_gap):
        start_index = (i + 1) * platform_width + np.sum(gap_width[:i]) + half_residual_platform_width
        end_index = (i + 1) * platform_width + np.sum(gap_width[:(i+1)]) + half_residual_platform_width
        if end_index <= terrain.width:
            terrain.height_field_raw[int(start_index):int(end_index), :] = -10000
        else:
            raise IndexError(f"Calculated end index {end_index} exceeds terrain width {terrain.width}.")
    
    return

def continuous_gap_terrain(terrain, gap_size=[.5, 1.2], number_of_gap:int=1, platform_width=None, type:str="linear", verbose: bool = False):
    gap_width = np.ones(number_of_gap) * gap_size[0] if type == "linear" else np.ones(number_of_gap) * np.random.choice(np.arange(gap_size[0], gap_size[1] + 0.1, 0.1))
    gap_width = np.clip(gap_width,a_min=0,a_max=gap_size[1])
    gap_width = (gap_width * 1 / terrain.horizontal_scale)   
    if verbose:
        print(gap_width)
    total_gap_width = np.sum(gap_width)
    platform_width = int((terrain.width - total_gap_width) / (number_of_gap + 1)) if platform_width == None else platform_width / terrain.horizontal_scale
    # residual_platform_width = (terrain.width - total_gap_width) % (number_of_gap + 1)

    # if total_gap_width + (number_of_gap + 1) * platform_width + residual_platform_width != terrain.width:
    #     raise ValueError(f"The sum of total gap width and platforms does not equal the terrain width: "
    #                     f"{total_gap_width} + {number_of_gap + 1} * {platform_width} + {residual_platform_width} = "
    #                  f"{total_gap_width + (number_of_gap + 1) * platform_width + residual_platform_width}, expected {terrain.width}")

    half_residual_platform_width = int(0.5 * (terrain.width - total_gap_width - (number_of_gap + 1) * platform_width))
    # print(half_residual_platform_width)
    for i in range(number_of_gap):
        start_index = (i) * platform_width + np.sum(gap_width[:i]) + half_residual_platform_width
        end_index = (i) * platform_width + np.sum(gap_width[:(i+1)]) + half_residual_platform_width
        if end_index <= terrain.width:
            terrain.height_field_raw[int(start_index):int(end_index), :] = -10000
        else:
            raise IndexError(f"Calculated end index {end_index} exceeds terrain width {terrain.width}.")
    
    return

def hurdle_terrain(terrain, gap_size=[.5, 1.2], number_of_gap:int=1, platform_width=None, type:str="linear", verbose: bool = False):
    # hurdle_height = np.linspace(gap_size[0], gap_size[1], num=number_of_gap) if type == "linear" else np.random.uniform(gap_size[0], gap_size[1], size=number_of_gap)
    hurdle_height = np.ones(number_of_gap) * gap_size[0] if type == "linear" else np.ones(number_of_gap) * np.random.choice(np.arange(gap_size[0], gap_size[1], 0.02),number_of_gap)
    # hurdle_height = np.array([0.1, 0.15, 0.16, 0.18, 0.2])
    # hurdle_height = np.array([0.18, 0.18, 0.18, 0.18, 0.18])
    # hurdle_height = np.clip(hurdle_height,a_min=0,a_max=0.2)
    hurdle_height = (hurdle_height * 1 / terrain.vertical_scale)
    width_choice = [0.1]
    hurdle_width = np.ones(number_of_gap) * np.random.choice(width_choice) / terrain.horizontal_scale
    if verbose:
        print(hurdle_height)
        print(hurdle_width)
        print('---')
    total_hurdle_width = np.sum(hurdle_width)
    platform_width = int((terrain.width - total_hurdle_width) / (number_of_gap + 1)) if platform_width == None else int(platform_width / terrain.horizontal_scale)
    # breakpoint()
    # residual_platform_width = (terrain.width - total_gap_width) % (number_of_gap + 1)

    # if total_gap_width + (number_of_gap + 1) * platform_width + residual_platform_width != terrain.width:
    #     raise ValueError(f"The sum of total gap width and platforms does not equal the terrain width: "
    #                     f"{total_gap_width} + {number_of_gap + 1} * {platform_width} + {residual_platform_width} = "
    #                  f"{total_gap_width + (number_of_gap + 1) * platform_width + residual_platform_width}, expected {terrain.width}")

    half_residual_platform_width = int(0.5 * (terrain.width - total_hurdle_width - (number_of_gap + 1) * platform_width))
    # print(half_residual_platform_width)
    for i in range(number_of_gap):
        start_index = (i + 1) * platform_width + np.sum(hurdle_width[:i]) + half_residual_platform_width
        end_index = (i + 1) * platform_width + np.sum(hurdle_width[:(i+1)]) + half_residual_platform_width
        if end_index <= terrain.width:
            terrain.height_field_raw[int(start_index):int(end_index), :] = hurdle_height[i]
        else:
            raise IndexError(f"Calculated end index {end_index} exceeds terrain width {terrain.width}.")
    
    return

def plane(terrain, gap_size=0.5, platform_size=1.):
    return

def beam(terrain, beam_width):
    # terrain.height_field_raw[...] = 
    # terrain.height_field_raw[10:30,:12] = -10000
    # terrain.height_field_raw[10:30,18:] = -10000
    # terrain.height_field_raw[28:30,10:20] = 0
    begin = int(0.5 * (terrain.length - beam_width / terrain.horizontal_scale))
    end = int(0.5 * (terrain.length + beam_width / terrain.horizontal_scale))
    terrain.height_field_raw[:,:begin] = -10000
    terrain.height_field_raw[:,end:] = -10000


    return

def beam_2(terrain, number_of_range:int=3, step_size:float=0.1):
    # terrain.height_field_raw[...] = 
    terrain.height_field_raw[:100,:30] = -10000
    terrain.height_field_raw[:100,44:] = -10000

    terrain.height_field_raw[100:200,:31] = -10000
    terrain.height_field_raw[100:200,43:] = -10000

    terrain.height_field_raw[200:,:32] = -10000
    terrain.height_field_raw[200:,42:] = -10000

    return

def pit_terrain(terrain, depth=0.1, platform_size=1.):
    depth = int(depth / terrain.vertical_scale)
    platform_size = int(platform_size / terrain.horizontal_scale / 2)
    x1 = terrain.length // 2 - platform_size
    x2 = terrain.length // 2 + platform_size
    y1 = terrain.width // 2 - platform_size
    y2 = terrain.width // 2 + platform_size
    terrain.height_field_raw[x1:x2, y1:y2] = -depth

def sloped_terrain(terrain, slope=.4,init_height=0.,platform_width=.1, randomized_add_height=False):
    platform_width = int(platform_width / terrain.horizontal_scale)
    x = np.arange(0, 0.5 * (terrain.width - 3 * platform_width))
    max_height = int(slope * (terrain.horizontal_scale / terrain.vertical_scale) * 0.5 * (terrain.width - 3 * platform_width))
    # breakpoint()
    terrain.height_field_raw[:,:] += int(init_height)
    # breakpoint()
    sloped_pattern = int(init_height) + (max_height * x / (terrain.width - 3 * platform_width)).astype(terrain.height_field_raw.dtype)
    sloped_pattern_2d = np.tile(sloped_pattern[:, np.newaxis], (1, terrain.length))
    # breakpoint()
    terrain.height_field_raw[(platform_width-1):platform_width+sloped_pattern_2d.shape[0]-1,:] += sloped_pattern_2d
    terrain.height_field_raw[(platform_width+sloped_pattern_2d.shape[0]-1):(2*platform_width+sloped_pattern_2d.shape[0]-1),:] += sloped_pattern_2d[-1,:]
    terrain.height_field_raw[(2*platform_width+sloped_pattern_2d.shape[0]-1):(2*platform_width+2*sloped_pattern_2d.shape[0]-1),:] += np.flipud(sloped_pattern_2d)
    if randomized_add_height:
        shape = terrain.height_field_raw.shape
        randomized_height = np.random.normal(loc=0,scale=0.5/terrain.horizontal_scale,size=shape)
        terrain.height_field_raw[:,:] += randomized_height.astype(terrain.height_field_raw.dtype)

def pyramid_sloped_terrain(terrain, slope=1, platform_size=10.,init_height=0.,randomized_add_height=False):
    x = np.arange(0, terrain.width)
    y = np.arange(0, terrain.length)
    center_x = int(terrain.width / 2)
    center_y = int(terrain.length / 2)
    xx, yy = np.meshgrid(x, y, sparse=True)
    xx = (center_x - np.abs(center_x-xx)) / center_x
    yy = (center_y - np.abs(center_y-yy)) / center_y
    xx = xx.reshape(terrain.width, 1)
    yy = yy.reshape(1, terrain.length)
    max_height = int(slope * (terrain.horizontal_scale / terrain.vertical_scale) * (terrain.width / 2))
    terrain.height_field_raw += int(init_height)
    terrain.height_field_raw += (max_height * xx * yy).astype(terrain.height_field_raw.dtype)

    platform_size = int(platform_size / terrain.horizontal_scale / 2)
    x1 = terrain.width // 2 - platform_size
    y1 = terrain.length // 2 - platform_size

    min_h = min(terrain.height_field_raw[x1, y1], init_height)
    max_h = max(terrain.height_field_raw[x1, y1], init_height)
    terrain.height_field_raw = np.clip(terrain.height_field_raw, min_h, max_h)

    if randomized_add_height:
        shape = terrain.height_field_raw.shape
        randomized_height = np.random.normal(loc=0,scale=0.5/terrain.horizontal_scale,size=shape)
        terrain.height_field_raw[:,:] += randomized_height.astype(terrain.height_field_raw.dtype)



def wave_terrain(terrain, num_waves=.5, amplitude=1.5, platform_width=0.1):
    platform_width = int(platform_width / terrain.horizontal_scale * 2)
    amplitude = int(0.5*amplitude / terrain.vertical_scale)
    div = (terrain.width - platform_width) / (num_waves * np.pi * 2)
    # div = 1
    xx = np.arange(0, terrain.width - platform_width)

    wave_pattern = ((amplitude * np.sin(xx / div)).astype(terrain.height_field_raw.dtype))
    wave_pattern_2d = np.tile(wave_pattern[:, np.newaxis], (1, terrain.length))
    terrain.height_field_raw[int(0.5*platform_width-1):(terrain.width - int(0.5*platform_width)-1), :] += wave_pattern_2d
    # terrain.height_field_raw = np.clip(terrain.height_field_raw, 0, None)
    return

def random_uniform_terrain(terrain, min_height=0., max_height=0.02, step=0.01, init_height=0.,platform_width=.1):        #rough terrain
    min_height = int(min_height / terrain.vertical_scale)
    max_height = int(max_height / terrain.vertical_scale)
    step = int(step / terrain.vertical_scale)

    heights_range = np.arange(min_height, max_height + step, step)
    height_field_downsampled = np.random.choice(heights_range, (int(terrain.width), int(terrain.length )))
    platform_width = int(platform_width / terrain.horizontal_scale * 2)

    x = np.linspace(0, terrain.width * terrain.horizontal_scale, height_field_downsampled.shape[0])
    y = np.linspace(0, terrain.length * terrain.horizontal_scale, height_field_downsampled.shape[1])

    f = interpolate.interp2d(y, x, height_field_downsampled, kind='linear')

    x_upsampled = np.linspace(0, terrain.width * terrain.horizontal_scale, terrain.width)
    y_upsampled = np.linspace(0, terrain.length * terrain.horizontal_scale, terrain.length)
    z_upsampled = np.rint(f(y_upsampled, x_upsampled))
    terrain.height_field_raw = init_height
    terrain.height_field_raw += z_upsampled.astype(np.int16)
    terrain.height_field_raw[:int(0.5*platform_width),:] = init_height
    terrain.height_field_raw[(terrain.width - int(0.5*platform_width)):,:] = init_height

def stairs_terrain(terrain, step_width=0.2, step_height=0.1, init_height=0, platform_size=1., type:str="normal"):
    step_width = int(step_width / terrain.horizontal_scale)
    # step_width = 3
    step_height = int(step_height / terrain.vertical_scale)
    platform_size = int(platform_size / terrain.horizontal_scale)
    num_steps = int(0.5 * (terrain.width - 3 * platform_size) // step_width)

    def get_heights(num_steps, step_height, init_height, type):
        if type == "normal":
            heights = [init_height + step_height * (i + 1) for i in range(num_steps)]
        elif type == "linear":
            heights=[]
            delta = 10
            for i in range(num_steps):
                if len(heights) == 0:
                    heights.append(init_height + step_height)
                else:
                    height = step_height + delta * i
                    heights.append(heights[-1] + height if height < 80 else heights[-1] + 80)
        elif type == "random": 
            heights=[]
            for i in range(num_steps):
                if i == 0:
                    heights.append(init_height + step_height)
                else:
                    heights.append(heights[-1] + step_height + np.random.uniform(
                        low = -np.abs(step_height/2), high = np.abs(step_height/2)))

        return heights

    heights = get_heights(num_steps, step_height, init_height, type)
    while len(heights) != num_steps:
        heights.append(heights[-1])
    # print(heights)
    start_x = platform_size
    stop_x = terrain.width - platform_size
    start_y = 1
    stop_y = -1
    for i in range(num_steps):
        start_x += step_width
        stop_x -= step_width
        terrain.height_field_raw[start_x: stop_x, start_y: stop_y] = heights[i]

def pyramid_stairs_terrain(terrain, step_width=0.2, step_height=0.15, platform_size=5.):
    step_width = int(step_width / terrain.horizontal_scale)
    step_height = int(step_height / terrain.vertical_scale)
    platform_size = int(platform_size / terrain.horizontal_scale)

    height = 0
    start_x = 0
    stop_x = terrain.width
    start_y = 0
    stop_y = terrain.length
    while (stop_x - start_x) > platform_size and (stop_y - start_y) > platform_size:
        start_x += step_width
        stop_x -= step_width
        # start_y += step_width
        # stop_y -= step_width
        height += step_height
        terrain.height_field_raw[start_x: stop_x, start_y: stop_y] = height



def stepping_stones_terrain(terrain, stone_size=0.3, stone_distance=0.3, max_height=0., platform_width=1.55, type:str='random', verbose: bool = False):       #mei hua zhuang
    
    stone_size = int(stone_size / terrain.horizontal_scale)
    stone_size = 3
    stone_distance = int(stone_distance / terrain.horizontal_scale)
    stone_distance = 6
    max_height = int(max_height / terrain.vertical_scale)
    platform_width = int(platform_width / terrain.horizontal_scale * 2)
    start_x = int(0.5*platform_width - 1)
    stop_env = terrain.width - int(0.5*platform_width)
    start_y = 0
    terrain.height_field_raw[start_x:stop_env-1, :] = int(-1 / terrain.vertical_scale)
    if type == 'normal':
        height_value = max_height
    else:
        height_value = np.random.randint(0,1)  * 100
        if verbose:
            print(height_value)
    
    while start_x < stop_env:
        stop_x = min(stop_env, start_x + stone_size)
        start_y = 1
        stop_y = max(0, start_y + stone_size)
        terrain.height_field_raw[start_x: stop_x, start_y: stop_y] = height_value
        start_y += stone_size + 2
        while start_y < terrain.length:
            stop_y = min(terrain.length, start_y + stone_size)
            terrain.height_field_raw[start_x: stop_x, start_y: stop_y] = height_value  
            start_y += stone_size + 2
        # if type == 'normal':
        #     height_value = max_height
        # else:
        #     height_value += np.random.randint(-2,3)  * 50
        #     height_value = np.clip(height_value,a_min=-50,a_max=150)
        #     print(height_value)
        #     print(height_value)
        # height_value += 100
        # if height_value > 500:
        #     height_value -= height_value % 500            
        start_x += stone_size + stone_distance
    return

def discrete_obstacles_terrain(terrain, max_height=0.15, min_size=1., max_size=5., num_rects=20, platform_size=1.):
    max_height = int(max_height / terrain.vertical_scale)
    min_size = int(min_size / terrain.horizontal_scale)
    max_size = int(max_size / terrain.horizontal_scale)
    platform_size = int(platform_size / terrain.horizontal_scale)

    (i, j) = terrain.height_field_raw.shape
    height_range = [-max_height, -max_height // 2, max_height // 2, max_height]
    width_range = range(min_size, max_size, 4)
    length_range = range(min_size, max_size, 4)

    for _ in range(num_rects):
        width = np.random.choice(width_range)
        length = np.random.choice(length_range)
        start_i = np.random.choice(range(0, i-width, 4))
        start_j = np.random.choice(range(0, j-length, 4))
        terrain.height_field_raw[start_i:start_i+width, start_j:start_j+length] = np.random.choice(height_range)

    x1 = (terrain.width - platform_size) // 2
    x2 = (terrain.width + platform_size) // 2
    y1 = (terrain.length - platform_size) // 2
    y2 = (terrain.length + platform_size) // 2
    terrain.height_field_raw[x1:x2, y1:y2] = 0
