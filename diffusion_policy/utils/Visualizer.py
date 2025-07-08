import pygame
import moderngl
import numpy as np
from math import sin, cos, pi
from pyrr import Matrix44, Vector3
from pygame.locals import *
from dataclasses import dataclass
import time
from typing import Optional, Union, List, Tuple, Dict, Any, TypeVar, Protocol

Color = tuple[float, float, float]
Point = np.ndarray
Points = np.ndarray
PointsSequence = np.ndarray
TransparencyValue = Union[float, Tuple[float, float]]

@dataclass
class ShaderProgram:
    vertex: str
    fragment: str
    
    @classmethod
    def from_files(cls, vert_file, frag_file):
        with open(vert_file, 'r') as f:
            vertex = f.read()
        with open(frag_file, 'r') as f:
            fragment = f.read()
        return cls(vertex, fragment)

class InstanceBuffer:
    def __init__(self, ctx, initial_size, instance_format):
        self.ctx = ctx
        self.current_size = initial_size
        self.instance_format = instance_format
        
        components = sum(int(fmt.split('f')[0]) for fmt in instance_format.split() if 'f' in fmt)
        self.stride = components * 4
        
        buffer_size = initial_size * self.stride
        self.buffer = self.ctx.buffer(reserve=buffer_size)
    
    def ensure_capacity(self, count):
        required_size = count * self.stride
        if required_size > self.buffer.size:
            new_size = max(count, self.current_size * 2)
            return self.resize(new_size)
        return False
    
    def resize(self, new_size):
        if new_size <= self.current_size:
            return False
        
        old_buffer = self.buffer
        self.current_size = new_size
        self.buffer = self.ctx.buffer(reserve=new_size * self.stride)
        old_buffer.release()
        return True
    
    def update(self, data):
        if not data:
            return 0
        
        self.ensure_capacity(len(data))
        self.buffer.orphan()
        data_array = np.array(data, dtype=np.float32)
        self.buffer.write(data_array.tobytes())
        return len(data)

class RenderableObject:
    def __init__(self, object_type, **properties):
        self.object_type = object_type
        self.properties = properties
    
    def get_instance_data(self):
        raise NotImplementedError("Subclasses must implement get_instance_data")

class Sphere(RenderableObject):
    def __init__(self, position, radius=0.1, color=(1,0,0), transparency=0.8):
        super().__init__(
            object_type='sphere',
            position=np.array(position, dtype=np.float32),
            radius=radius,
            color=np.array(color, dtype=np.float32),
            transparency=transparency
        )
    
    def get_instance_data(self):
        position = np.array(self.properties['position'], dtype=np.float32).flatten()
        radius = float(self.properties['radius'])
        color = np.array(self.properties['color'], dtype=np.float32).flatten()
        transparency = float(self.properties['transparency']) if not isinstance(self.properties['transparency'], tuple) else float(self.properties['transparency'][0])
        
        return np.array([
            position[0], position[1], position[2],
            radius,
            color[0], color[1], color[2],
            transparency
        ], dtype=np.float32)

class Edge(RenderableObject):
    def __init__(self, start_pos, end_pos, radius=0.02, color=(1,0,0), transparency=0.8):
        super().__init__(
            object_type='edge',
            start_pos=np.array(start_pos, dtype=np.float32),
            end_pos=np.array(end_pos, dtype=np.float32),
            radius=radius,
            color=np.array(color, dtype=np.float32),
            transparency=transparency
        )
    
    def get_instance_data(self):
        start_pos = np.array(self.properties['start_pos'], dtype=np.float32).flatten()
        end_pos = np.array(self.properties['end_pos'], dtype=np.float32).flatten()
        radius = float(self.properties['radius'])
        color = np.array(self.properties['color'], dtype=np.float32).flatten()
        transparency = float(self.properties['transparency']) if not isinstance(self.properties['transparency'], tuple) else float(self.properties['transparency'][0])
        
        return np.array([
            start_pos[0], start_pos[1], start_pos[2],
            end_pos[0], end_pos[1], end_pos[2],
            radius,
            color[0], color[1], color[2],
            transparency
        ], dtype=np.float32)

class Arrow(RenderableObject):
    def __init__(self, position, direction, magnitude=1.0, color=(1,0,0), transparency=0.8):
        direction = np.array(direction, dtype=np.float32)
        if np.linalg.norm(direction) > 0:
            direction = direction / np.linalg.norm(direction)
            
        super().__init__(
            object_type='arrow',
            position=np.array(position, dtype=np.float32),
            direction=direction,
            magnitude=float(magnitude),
            color=np.array(color, dtype=np.float32),
            transparency=transparency
        )
    
    def get_instance_data(self):
        position = np.array(self.properties['position'], dtype=np.float32).flatten()
        direction = np.array(self.properties['direction'], dtype=np.float32).flatten()
        magnitude = float(self.properties['magnitude'])
        color = np.array(self.properties['color'], dtype=np.float32).flatten()
        transparency = float(self.properties['transparency']) if not isinstance(self.properties['transparency'], tuple) else float(self.properties['transparency'][0])
        
        return np.array([
            position[0], position[1], position[2],
            direction[0], direction[1], direction[2],
            magnitude,
            color[0], color[1], color[2],
            transparency
        ], dtype=np.float32)

class Shaders:
    SPHERE = ShaderProgram(
        vertex='''
            #version 330
            in vec3 in_position;
            in vec3 in_normal;
            in vec3 instance_position;
            in float instance_radius;
            in vec3 instance_color;
            in float instance_transparency;

            uniform mat4 model;
            uniform mat4 view;
            uniform mat4 projection;
            uniform vec3 light_pos;
            uniform vec3 view_pos;

            out vec3 v_normal;
            out vec3 frag_pos;
            out vec3 v_color;
            out float v_transparency;

            void main() {
                vec3 world_pos = instance_position + (in_position * instance_radius);
                gl_Position = projection * view * model * vec4(world_pos, 1.0);
                v_normal = mat3(model) * in_normal;
                frag_pos = world_pos;
                v_color = instance_color;
                v_transparency = instance_transparency;
            }
        ''',
        fragment='''
            #version 330
            in vec3 frag_pos;
            in vec3 v_normal;
            in vec3 v_color;
            in float v_transparency;

            uniform vec3 light_pos;
            uniform vec3 view_pos;
            out vec4 f_color;

            void main() {
                vec3 norm = normalize(v_normal);
                vec3 light_dir = normalize(light_pos - frag_pos);
                vec3 view_dir = normalize(view_pos - frag_pos);

                float ambient_strength = 0.3;
                vec3 ambient = ambient_strength * v_color;

                float diff = max(dot(norm, light_dir), 0.1);
                vec3 diffuse = diff * v_color;

                float specular_strength = 0.5;
                vec3 reflect_dir = reflect(-light_dir, norm);
                float spec = pow(max(dot(view_dir, reflect_dir), 0.0), 16);
                vec3 specular = specular_strength * spec * vec3(1.0);

                float backlight_factor = 0.2;
                vec3 backlight = backlight_factor * v_color * (1.0 - diff);

                vec3 result = ambient + diffuse + specular + backlight;
                f_color = vec4(result, v_transparency);
            }
        '''
    )

    EDGE = ShaderProgram(
        vertex='''
        #version 330
        in vec3 in_position;
        in vec3 in_normal;
        in vec3 instance_start;
        in vec3 instance_end;
        in float instance_radius;
        in vec3 instance_color;
        in float instance_transparency;

        uniform mat4 model;
        uniform mat4 view;
        uniform mat4 projection;
        uniform vec3 light_pos;
        uniform vec3 view_pos;

        out vec3 v_normal;
        out vec3 frag_pos;
        out vec3 v_color;
        out float v_transparency;

        void main() {
            vec3 direction = normalize(instance_end - instance_start);
            
            if (length(direction) < 0.0001) {
                direction = vec3(0.0, 0.0, 1.0);
            }
            
            vec3 up;
            if (abs(direction.z) < 0.9) {
                up = vec3(0.0, 0.0, 1.0);
            } else {
                up = vec3(0.0, 1.0, 0.0);
            }
            
            vec3 right = normalize(cross(direction, up));
            up = normalize(cross(right, direction));

            mat3 rotation = mat3(right, up, direction);

            float edge_length = length(instance_end - instance_start);
            vec3 scaled_pos = vec3(
                in_position.x * instance_radius,
                in_position.z * instance_radius,
                in_position.y * edge_length
            );

            vec3 world_pos = instance_start + (rotation * scaled_pos);
            gl_Position = projection * view * model * vec4(world_pos, 1.0);

            v_normal = normalize(rotation * in_normal);
            frag_pos = world_pos;
            v_color = instance_color;
            v_transparency = instance_transparency;
        }
        ''',
        fragment='''
        #version 330
        in vec3 v_normal;
        in vec3 frag_pos;
        in vec3 v_color;
        in float v_transparency;

        uniform vec3 light_pos;
        uniform vec3 view_pos;
        out vec4 f_color;

        void main() {
            vec3 norm = normalize(v_normal);
            vec3 light_dir = normalize(light_pos - frag_pos);
            vec3 view_dir = normalize(view_pos - frag_pos);

            float ambient_strength = 0.3;
            vec3 ambient = ambient_strength * v_color;

            float diff = max(dot(norm, light_dir), 0.1);
            vec3 diffuse = diff * v_color;

            float specular_strength = 0.5;
            vec3 reflect_dir = reflect(-light_dir, norm);
            float spec = pow(max(dot(view_dir, reflect_dir), 0.0), 32);
            vec3 specular = specular_strength * spec * vec3(1.0);

            vec3 result = ambient + diffuse + specular;
            f_color = vec4(result, v_transparency);
        }
        '''
    )

    GRID = ShaderProgram(
        vertex='''
            #version 330
            in vec3 in_vert;
            in vec3 in_color;
            out vec3 v_color;
            uniform mat4 view;
            uniform mat4 projection;
            void main() {
                gl_Position = projection * view * vec4(in_vert, 1.0);
                v_color = in_color;
            }
        ''',
        fragment='''
            #version 330
            in vec3 v_color;
            out vec4 fragColor;
            void main() {
                fragColor = vec4(v_color, 1.0);
            }
        '''
    )

    ARROW = ShaderProgram(
        vertex='''
        #version 330
        in vec3 in_position;
        in vec3 in_normal;
        in vec3 instance_position;
        in vec3 instance_direction;
        in float instance_magnitude;
        in vec3 instance_color;
        in float instance_transparency;

        uniform mat4 model;
        uniform mat4 view;
        uniform mat4 projection;
        uniform vec3 light_pos;
        uniform vec3 view_pos;

        out vec3 v_normal;
        out vec3 frag_pos;
        out vec3 v_color;
        out float v_transparency;

        mat3 rotation_matrix(vec3 axis, float angle) {
            axis = normalize(axis);
            float s = sin(angle);
            float c = cos(angle);
            float oc = 1.0 - c;
            return mat3(
                oc * axis.x * axis.x + c,           oc * axis.x * axis.y - axis.z * s,  oc * axis.z * axis.x + axis.y * s,
                oc * axis.x * axis.y + axis.z * s,  oc * axis.y * axis.y + c,           oc * axis.y * axis.z - axis.x * s,
                oc * axis.z * axis.x - axis.y * s,  oc * axis.y * axis.z + axis.x * s,  oc * axis.z * axis.z + c
            );
        }

        void main() {
            vec3 default_dir = vec3(0, 0, 1);
            vec3 dir = normalize(instance_direction);
            
            vec3 rot_axis;
            float rot_angle;
            
            if (abs(dot(dir, default_dir)) > 0.9999) {
                rot_axis = vec3(1, 0, 0);
                rot_angle = (dot(dir, default_dir) > 0) ? 0.0 : 3.14159265359;
            } else {
                rot_axis = normalize(cross(dir, default_dir));
                rot_angle = acos(dot(dir, default_dir));
            }

            mat3 rotation = rotation_matrix(rot_axis, rot_angle);
            vec3 scaled_pos = in_position * instance_magnitude;
            vec3 world_pos = instance_position + (rotation * scaled_pos);

            gl_Position = projection * view * model * vec4(world_pos, 1.0);
            frag_pos = world_pos;
            v_normal = normalize(rotation * in_normal);
            v_color = instance_color;
            v_transparency = instance_transparency;
        }
        ''',
        fragment='''
        #version 330
        in vec3 frag_pos;
        in vec3 v_normal;
        in vec3 v_color;
        in float v_transparency;

        uniform vec3 light_pos;
        uniform vec3 view_pos;
        out vec4 f_color;

        void main() {
            vec3 norm = normalize(v_normal);
            vec3 light_dir = normalize(light_pos - frag_pos);
            vec3 view_dir = normalize(view_pos - frag_pos);

            float ambient_strength = 0.3;
            vec3 ambient = ambient_strength * v_color;

            float diff = max(dot(norm, light_dir), 0.1);
            vec3 diffuse = diff * v_color;

            float specular_strength = 0.5;
            vec3 reflect_dir = reflect(-light_dir, norm);
            float spec = pow(max(dot(view_dir, reflect_dir), 0.0), 16);
            vec3 specular = specular_strength * spec * vec3(1.0);

            float rim = 1.0 - max(dot(norm, view_dir), 0.0);
            rim = smoothstep(0.6, 1.0, rim);
            vec3 rim_light = rim * v_color * 0.3;

            vec3 result = ambient + diffuse + specular + rim_light;
            f_color = vec4(result, v_transparency);
        }
        ''')
    
class Renderer:
    def __init__(self, ctx, shader_program, initial_instances=1000, instance_format=None):
        self.ctx = ctx
        self.prog = self.ctx.program(
            vertex_shader=shader_program.vertex, 
            fragment_shader=shader_program.fragment
        )
        
        self.setup_geometry()
        
        if instance_format:
            self.instance_format = instance_format
            self.setup_buffers(initial_instances)
        else:
            self.instance_format = None
            self.instance_buffer = None
    
    def setup_geometry(self):
        pass
    
    def setup_buffers(self, initial_instances):
        self.vbo = self.ctx.buffer(self.vertices.tobytes())
        self.ibo = self.ctx.buffer(self.indices.tobytes())
        
        self.instance_buffer = InstanceBuffer(
            self.ctx, initial_instances, self.instance_format)
        
        self.setup_vao()
    
    def setup_vao(self):
        pass
    
    def _set_uniforms(self, **uniforms):
        for name, value in uniforms.items():
            if name in self.prog:
                if isinstance(value, (float, int, bool)):
                    self.prog[name].value = value
                elif isinstance(value, np.ndarray):
                    self.prog[name].write(value.astype('f4').tobytes())
    
    def render_instances(self, view_matrix, projection_matrix, camera_pos, instances=None):
        self._set_uniforms(
            view=view_matrix,
            projection=projection_matrix,
            model=np.identity(4, dtype=np.float32),
            light_pos=np.array([10.0, 10.0, 10.0], dtype=np.float32),
            view_pos=camera_pos
        )
        
        if instances and self.instance_buffer:
            count = self.instance_buffer.update(instances)
            if count > 0:
                self.vao.render(instances=count)
        elif hasattr(self, 'render_direct'):
            self.render_direct()

class SphereRenderer(Renderer):
    def __init__(self, ctx, initial_instances=1000):
        super().__init__(ctx, Shaders.SPHERE, initial_instances, '3f 1f 3f 1f/i')
    
    def setup_geometry(self, segments=32):
        vertices = []
        indices = []
        
        for i in range(segments + 1):
            lat = np.pi * (-0.5 + float(i) / segments)
            for j in range(segments + 1):
                lon = 2 * np.pi * float(j) / segments
                x = np.cos(lon) * np.cos(lat)
                y = np.sin(lat)
                z = np.sin(lon) * np.cos(lat)
                vertices.extend([x, y, z, x, y, z])

        for i in range(segments):
            for j in range(segments):
                first = i * (segments + 1) + j
                second = first + segments + 1
                indices.extend([first, second, first + 1])
                indices.extend([second, second + 1, first + 1])

        self.vertices = np.array(vertices, dtype=np.float32)
        self.indices = np.array(indices, dtype=np.int32)
    
    def setup_vao(self):
        self.vao = self.ctx.vertex_array(
            self.prog,
            [
                (self.vbo, '3f 3f', 'in_position', 'in_normal'),
                (self.instance_buffer.buffer, self.instance_format, 
                 'instance_position', 'instance_radius', 'instance_color', 'instance_transparency')
            ],
            self.ibo
        )

class EdgeRenderer(Renderer):
    def __init__(self, ctx, initial_instances=1000):
        super().__init__(ctx, Shaders.EDGE, initial_instances, '3f 3f 1f 3f 1f/i')
    
    def setup_geometry(self, segments=24):
        vertices = []
        indices = []

        for y in [0, 1]:
            for i in range(segments):
                angle = 2 * np.pi * i / segments
                x = np.cos(angle)
                z = np.sin(angle)
                vertices.extend([x, y, z, x, 0, z])

        for i in range(segments):
            next_i = (i + 1) % segments
            v0 = i
            v1 = next_i
            v2 = i + segments
            v3 = next_i + segments
            indices.extend([v0, v2, v1, v1, v2, v3])

        self.vertices = np.array(vertices, dtype=np.float32)
        self.indices = np.array(indices, dtype=np.int32)
    
    def setup_vao(self):
        self.vao = self.ctx.vertex_array(
            self.prog,
            [
                (self.vbo, '3f 3f', 'in_position', 'in_normal'),
                (self.instance_buffer.buffer, self.instance_format, 
                 'instance_start', 'instance_end', 'instance_radius', 
                 'instance_color', 'instance_transparency')
            ],
            self.ibo
        )

class ArrowRenderer(Renderer):
    def __init__(self, ctx, initial_instances=1000):
        super().__init__(ctx, Shaders.ARROW, initial_instances, '3f 3f 1f 3f 1f/i')
    
    def setup_geometry(self, segments=12):
        vertices = []
        indices = []
        normals = []
        
        arrow_length = 1.0
        shaft_radius = 0.05
        head_radius = 0.15
        head_length = 0.3
        shaft_length = arrow_length - head_length

        for i in range(segments):
            angle = 2 * np.pi * i / segments
            x = np.cos(angle) * shaft_radius
            y = np.sin(angle) * shaft_radius
            nx = np.cos(angle)
            ny = np.sin(angle)
            
            vertices.extend([x, y, 0])
            normals.extend([nx, ny, 0])
            
            vertices.extend([x, y, shaft_length])
            normals.extend([nx, ny, 0])

        for i in range(segments):
            next_i = (i + 1) % segments
            v0 = i * 2
            v1 = i * 2 + 1
            v2 = next_i * 2
            v3 = next_i * 2 + 1
            indices.extend([v0, v1, v2])
            indices.extend([v2, v1, v3])

        head_base_start = len(vertices) // 3
        for i in range(segments):
            angle = 2 * np.pi * i / segments
            x = np.cos(angle) * head_radius
            y = np.sin(angle) * head_radius
            nx = np.cos(angle) * 0.5
            ny = np.sin(angle) * 0.5
            nz = 0.5
            
            vertices.extend([x, y, shaft_length])
            normals.extend([nx, ny, nz])

        tip_index = len(vertices) // 3
        vertices.extend([0, 0, arrow_length])
        normals.extend([0, 0, 1])

        for i in range(segments):
            next_i = (i + 1) % segments
            v0 = head_base_start + i
            v1 = head_base_start + next_i
            indices.extend([v0, v1, tip_index])

        combined_vertices = []
        for i in range(len(vertices) // 3):
            combined_vertices.extend(vertices[i*3:i*3+3])
            combined_vertices.extend(normals[i*3:i*3+3])
            
        self.vertices = np.array(combined_vertices, dtype=np.float32)
        self.indices = np.array(indices, dtype=np.int32)
    
    def setup_vao(self):
        self.vao = self.ctx.vertex_array(
            self.prog,
            [
                (self.vbo, '3f 3f', 'in_position', 'in_normal'),
                (self.instance_buffer.buffer, self.instance_format, 
                 'instance_position', 'instance_direction', 'instance_magnitude', 
                 'instance_color', 'instance_transparency')
            ],
            self.ibo
        )

class GridRenderer(Renderer):
    def __init__(self, ctx):
        super().__init__(ctx, Shaders.GRID)
    
    def setup_geometry(self):
        axis_vertices = np.array([
            [0, 0, 0, 1, 0, 0], [3, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0], [0, 3, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1], [0, 0, 3, 0, 0, 1],
        ], dtype=np.float32)

        grid_size = 10
        grid_spacing = .5
        grid_color = [0.5, 0.5, 0.5]
        grid_vertices = []

        for i in np.arange(-grid_size, grid_size + 0.1, grid_spacing):
            grid_vertices.extend([
                [i, -grid_size, 0] + grid_color,
                [i, grid_size, 0] + grid_color
            ])
            grid_vertices.extend([
                [-grid_size, i, 0] + grid_color,
                [grid_size, i, 0] + grid_color
            ])

        self.axis_vertices = axis_vertices
        self.grid_vertices = np.array(grid_vertices, dtype=np.float32)
        
        self.axis_vbo = self.ctx.buffer(self.axis_vertices)
        self.grid_vbo = self.ctx.buffer(self.grid_vertices)
        self.axis_vao = self.ctx.simple_vertex_array(self.prog, self.axis_vbo, 'in_vert', 'in_color')
        self.grid_vao = self.ctx.simple_vertex_array(self.prog, self.grid_vbo, 'in_vert', 'in_color')
    
    def render_direct(self):
        self.ctx.line_width = 2.0
        self.axis_vao.render(moderngl.LINES)
        self.ctx.line_width = 1.0
        self.grid_vao.render(moderngl.LINES)

class ObjectFactory:
    @staticmethod
    def handle_temporal_sequence(points, create_func, color, transparency, **kwargs):
        result = []
        points = np.array(points)
        
        if len(points.shape) == 3:
            n_objects = points.shape[0]
            n_steps = points.shape[1]
            
            for t in range(n_steps):
                progress = t / (n_steps - 1) if n_steps > 1 else 1.0
                current_transparency = (
                    transparency[0] + (transparency[1] - transparency[0]) * progress
                    if isinstance(transparency, tuple) and len(transparency) == 2
                    else transparency
                )
                
                for i in range(n_objects):
                    point = points[i, t]
                    current_color = color[i] if isinstance(color, list) and i < len(color) else color
                    result.append(create_func(point, current_color, current_transparency, **kwargs))
        else:
            n_objects = points.shape[0]
            for i in range(n_objects):
                point = points[i]
                current_color = color[i] if isinstance(color, list) and i < len(color) else color
                current_transparency = transparency[0] if isinstance(transparency, tuple) else transparency
                result.append(create_func(point, current_color, current_transparency, **kwargs))
                
        return result
    
    @staticmethod
    def create_spheres(points, color=(1,0,0), transparency=0.8, radius=0.1):
        def create_sphere(point, clr, trans, radius=radius):
            return Sphere(point, radius=radius, color=clr, transparency=trans)
        return ObjectFactory.handle_temporal_sequence(points, create_sphere, color, transparency)
    
    @staticmethod
    def create_arrows(positions, directions, magnitudes=None, min_length=0.2, max_length=2.0, 
                      color=(1,0,0), transparency=0.8):
        result = []
        positions = np.array(positions)
        directions = np.array(directions)
        
        if positions.shape[0] != directions.shape[0]:
            raise ValueError(f"Number of objects must match: {positions.shape[0]} vs {directions.shape[0]}")
        
        if len(positions.shape) == 3 and len(directions.shape) == 3:
            if positions.shape[1] != directions.shape[1]:
                raise ValueError(f"Number of timesteps must match: {positions.shape[1]} vs {directions.shape[1]}")
        
        if magnitudes is not None:
            magnitudes = np.array(magnitudes)
            
            if len(positions.shape) == 3 and len(magnitudes.shape) == 2:
                if positions.shape[0] != magnitudes.shape[0] or positions.shape[1] != magnitudes.shape[1]:
                    raise ValueError(f"Magnitudes shape {magnitudes.shape} doesn't match positions {positions.shape}")
            elif len(positions.shape) == 2 and len(magnitudes.shape) == 1:
                if positions.shape[0] != magnitudes.shape[0]:
                    raise ValueError(f"Magnitudes shape {magnitudes.shape} doesn't match positions {positions.shape}")
            
            if np.size(magnitudes) > 0:
                mag_min = np.min(magnitudes)
                mag_max = np.max(magnitudes)
                
                if mag_min == mag_max:
                    normalized_magnitudes = np.ones_like(magnitudes) * (min_length + max_length) / 2
                else:
                    normalized_magnitudes = (magnitudes - mag_min) / (mag_max - mag_min)
                    normalized_magnitudes = min_length + normalized_magnitudes * (max_length - min_length)
            else:
                normalized_magnitudes = magnitudes
        else:
            if len(positions.shape) == 3:
                normalized_magnitudes = np.ones((positions.shape[0], positions.shape[1]))
            else:
                normalized_magnitudes = np.ones(positions.shape[0])
        
        if len(positions.shape) == 3:
            n_objects = positions.shape[0]
            n_steps = positions.shape[1]
            
            for t in range(n_steps):
                progress = t / (n_steps - 1) if n_steps > 1 else 1.0
                current_transparency = (
                    transparency[0] + (transparency[1] - transparency[0]) * progress
                    if isinstance(transparency, tuple) and len(transparency) == 2
                    else transparency
                )
                
                for i in range(n_objects):
                    current_color = color[i] if isinstance(color, list) and i < len(color) else color
                    result.append(Arrow(
                        positions[i, t],
                        directions[i, t],
                        normalized_magnitudes[i, t],
                        current_color,
                        current_transparency
                    ))
        else:
            n_objects = positions.shape[0]
            for i in range(n_objects):
                current_color = color[i] if isinstance(color, list) and i < len(color) else color
                current_transparency = transparency[0] if isinstance(transparency, tuple) else transparency
                result.append(Arrow(
                    positions[i],
                    directions[i],
                    normalized_magnitudes[i],
                    current_color,
                    current_transparency
                ))
                    
        return result
    
    @staticmethod
    def create_edges(points, edges, color=(1,0,0), transparency=0.8, radius=0.02):
        result = []
        points = np.array(points)
        
        if len(points.shape) == 3:
            n_objects = points.shape[0]
            n_steps = points.shape[1]
            
            for t in range(n_steps):
                progress = t / (n_steps - 1) if n_steps > 1 else 1.0
                current_transparency = (
                    transparency[0] + (transparency[1] - transparency[0]) * progress
                    if isinstance(transparency, tuple) and len(transparency) == 2
                    else transparency
                )
                
                for e, (i, j) in enumerate(edges):
                    if 0 <= i < n_objects and 0 <= j < n_objects:
                        current_color = color[e] if isinstance(color, list) and e < len(color) else color
                        result.append(Edge(
                            points[i, t], points[j, t], 
                            radius, current_color, current_transparency
                        ))
        else:
            n_objects = points.shape[0]
            current_transparency = transparency[0] if isinstance(transparency, tuple) else transparency
            
            for e, (i, j) in enumerate(edges):
                if 0 <= i < n_objects and 0 <= j < n_objects:
                    current_color = color[e] if isinstance(color, list) and e < len(color) else color
                    result.append(Edge(
                        points[i], points[j], 
                        radius, current_color, current_transparency
                    ))
            
        return result

class Camera:
    def __init__(self):
        self.pos = Vector3([5.0, 5.0, 2.5])
        self.front = Vector3([-0.7, -0.7, -0.3])
        self.up = Vector3([0.0, 0.0, 1.0])
        self.right = self.front.cross(self.up).normalized
        self.speed = 0.005
        self.mouse_sensitivity = 0.001 
        self.zoom_speed = 0.5

    def get_view_matrix(self):
        return Matrix44.look_at(
            self.pos,
            self.pos + self.front,
            self.up
        )
    
    def move(self, dx, dy, rotate=False):
        if rotate:
            rotation_z = Matrix44.from_z_rotation(dx * self.mouse_sensitivity)
            self.front = (rotation_z * self.front).normalized
            self.right = (rotation_z * self.right).normalized

            right = self.front.cross(self.up).normalized
            angle = dy * self.mouse_sensitivity
            c, s = np.cos(angle), np.sin(angle)
            x, y, z = right
            rotation_matrix = Matrix44([
                [x*x*(1-c)+c,    x*y*(1-c)-z*s,  x*z*(1-c)+y*s,  0],
                [y*x*(1-c)+z*s,  y*y*(1-c)+c,    y*z*(1-c)-x*s,  0],
                [z*x*(1-c)-y*s,  z*y*(1-c)+x*s,  z*z*(1-c)+c,    0],
                [0,              0,              0,               1]
            ])
            
            new_front = (rotation_matrix * self.front).normalized
            
            angle_with_up = np.arccos(new_front.dot(self.up))
            if 0.1 < angle_with_up < np.pi - 0.1:
                self.front = new_front
        else:
            right = self.front.cross(self.up).normalized
            self.pos += -dx * self.speed * right + dy * self.speed * self.up
    
    def zoom(self, delta):
        self.pos += delta * self.zoom_speed * self.front

class Scene:
    def __init__(self, ctx):
        self.ctx = ctx
        self.renderers = {
            'sphere': SphereRenderer(ctx),
            'arrow': ArrowRenderer(ctx),
            'edge': EdgeRenderer(ctx),
            'grid': GridRenderer(ctx)
        }
        self.objects = []
    
    def add_objects(self, objects):
        self.objects.extend(objects)
    
    def clear(self):
        self.objects = []
    
    def render(self, view_matrix, projection_matrix, camera_pos):
        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
        
        self.renderers['grid'].render_instances(view_matrix, projection_matrix, camera_pos)
        
        by_type = {}
        for obj in self.objects:
            by_type.setdefault(obj.object_type, []).append(obj)
        
        for obj_type, objects in by_type.items():
            if obj_type not in self.renderers:
                print(f"Warning: No renderer found for object type '{obj_type}'")
                continue
                
            def get_sort_key(obj):
                if obj.object_type == 'edge':
                    start = obj.properties.get('start_pos')
                    end = obj.properties.get('end_pos')
                    center = [(start[i] + end[i])/2 for i in range(3)]
                else:
                    center = obj.properties.get('position')
                
                return -np.linalg.norm(np.array(center) - np.array(camera_pos))
            
            objects.sort(key=get_sort_key)
            
            instance_data = []
            for obj in objects:
                try:
                    instance_data.append(obj.get_instance_data())
                except Exception as e:
                    print(f"Error creating instance data for {obj.object_type}: {e}")
                    continue
            
            if instance_data:
                try:
                    self.renderers[obj_type].render_instances(
                        view_matrix, projection_matrix, camera_pos, instance_data
                    )
                except Exception as e:
                    print(f"Error rendering {obj_type} objects: {e}")

class Viewport:
    def __init__(self, ctx, rect):
        self.ctx = ctx
        self.rect = rect
        self.camera = Camera()
        self.scene = Scene(ctx)
        
        self.dragging = False
        self.rotating = False
        self.last_mouse_pos = None
        
        self.row = -1
        self.col = -1
    
    def handle_event(self, event):
        if event.type == MOUSEBUTTONDOWN:
            self.last_mouse_pos = pygame.mouse.get_pos()
            if event.button == 1:
                self.dragging = True
            elif event.button == 3:
                self.rotating = True
                
        elif event.type == MOUSEBUTTONUP:
            if event.button == 1 and self.dragging:
                self.dragging = False
            elif event.button == 3 and self.rotating:
                self.rotating = False
                
        elif event.type == MOUSEMOTION and (self.dragging or self.rotating):
            x, y = pygame.mouse.get_pos()
            if self.last_mouse_pos:
                dx, dy = x - self.last_mouse_pos[0], y - self.last_mouse_pos[1]
                self.camera.move(dx, dy, rotate=self.rotating)
            self.last_mouse_pos = (x, y)
            
        elif event.type == MOUSEWHEEL:
            self.camera.zoom(event.y)
    
    def render(self, width, height):
        x, y, w, h = self.rect
        
        pixel_x = int(x * width)
        pixel_y = int((1.0 - y - h) * height)
        pixel_w = int(w * width)
        pixel_h = int(h * height)
        
        self.ctx.viewport = (pixel_x, pixel_y, pixel_w, pixel_h)
        
        view = self.camera.get_view_matrix()
        projection = Matrix44.perspective_projection(
            45.0, pixel_w / pixel_h, 0.1, 100.0
        )
        
        self.scene.render(view, projection, self.camera.pos)
        
    def clear(self):
        self.scene.clear()
        
    def draw_spheres(self, points, edges=None, color=(1,0,0), transparency=0.8, radius=0.1, edge_radius=None):
        obj_list = ObjectFactory.create_spheres(points, color, transparency, radius)
        self.scene.add_objects(obj_list)
        
        if edges is not None and len(edges) > 0:
            er = edge_radius if edge_radius is not None else radius/5
            edge_objects = ObjectFactory.create_edges(points, edges, color, transparency, er)
            self.scene.add_objects(edge_objects)
    
    def draw_arrows(self, positions, directions, magnitudes=None, edges=None, color=(1,0,0), 
                  transparency=0.8, min_length=0.2, max_length=2.0, edge_radius=None):
        obj_list = ObjectFactory.create_arrows(
            positions, directions, magnitudes, min_length, max_length, color, transparency
        )
        self.scene.add_objects(obj_list)
        
        if edges is not None and len(edges) > 0:
            er = edge_radius if edge_radius is not None else min_length/10
            edge_objects = ObjectFactory.create_edges(positions, edges, color, transparency, er)
            self.scene.add_objects(edge_objects)

class Visualizer:
    def __init__(self, rows=1, cols=1, size=(800, 600)):
        pygame.init()
        self.width, self.height = size
        self.screen = pygame.display.set_mode((self.width, self.height), pygame.OPENGL)
        pygame.display.set_caption("3D Visualization")
        self.ctx = moderngl.create_context()
        self.ctx.enable(moderngl.DEPTH_TEST)
        
        self.rows = rows
        self.cols = cols
        self.viewport_list = []
        self.active_viewport = None
        
        self.subplots = []
        
        self._create_viewports()
        
        self.clock = pygame.time.Clock()
        self.frame_rate = 60
    
    def __getattr__(self, name):
        if self.rows == 1 and self.cols == 1:
            if hasattr(self.subplots[0][0], name):
                attr = getattr(self.subplots[0][0], name)
                if callable(attr):
                    def forwarded_method(*args, **kwargs):
                        return attr(*args, **kwargs)
                    return forwarded_method
                return attr
        raise AttributeError(f"{self.__class__.__name__} object has no attribute '{name}'")
    
    def __getitem__(self, row):
        if self.rows == 1 and self.cols == 1 and row == 0:
            return self.subplots[0][0]
        return self.subplots[self.rows - 1 - row]
 
    def _create_viewports(self):
        for i in range(self.rows):
            row_viewports = []
            
            for j in range(self.cols):
                rect = (
                    j / self.cols,
                    i / self.rows,
                    1 / self.cols,
                    1 / self.rows
                )
                
                viewport = Viewport(self.ctx, rect)
                viewport.row = (self.rows - 1 - i)
                viewport.col = j
                
                row_viewports.append(viewport)
                self.viewport_list.append(viewport)
                
                if self.active_viewport is None:
                    self.active_viewport = viewport
            
            self.subplots.append(row_viewports)
        
        self.subplots.reverse()
  
    def update(self, delay=0):
        self.ctx.clear(0.95, 0.97, 1.0)
        for viewport in self.viewport_list:
            viewport.render(self.width, self.height)
        pygame.display.flip()
        
        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()
                return False
                
            elif event.type == MOUSEBUTTONDOWN:
                mouse_x, mouse_y = pygame.mouse.get_pos()
                
                viewport_col = int(mouse_x / self.width * self.cols)
                viewport_row_screen = int(mouse_y / self.height * self.rows)
                
                viewport_row = self.rows - 1 - viewport_row_screen
                
                viewport_col = min(max(0, viewport_col), self.cols - 1)
                viewport_row = min(max(0, viewport_row), self.rows - 1)
                
                self.active_viewport = self.subplots[viewport_row][viewport_col]
                
                if self.active_viewport:
                    self.active_viewport.handle_event(event)
            
            elif self.active_viewport:
                self.active_viewport.handle_event(event)
        
        for viewport in self.viewport_list:
            viewport.clear()
        
        if delay > 0:
            pygame.time.wait(int(delay * 1000))
        
        return True
        
    def clear(self):
        if self.rows == 1 and self.cols == 1:
            self.subplots[0][0].clear()
        else:
            for viewport in self.viewport_list:
                viewport.clear()



def vis_traj():
    vis = Visualizer(2, 2, size=(1200, 900))
    # data = np.load('/move/u/takaraet/whole_body_tracking/parallel/lafan_walk_short.npz')
    data = np.load('/move/u/takaraet/whole_body_tracking/parallel/21_Turn_left____LAFAN_walk1_subject1_0_-1.npz')
    # data = np.load('/move/u/takaraet/whole_body_tracking/parallel/29_Walk_around____LAFAN_walk1_subject1_0_-1.npz')

    dof_names = ['left_hip_pitch_joint', 'left_hip_roll_joint',
       'left_hip_yaw_joint', 'left_knee_joint', 'left_ankle_pitch_joint',
       'left_ankle_roll_joint', 'right_hip_pitch_joint',
       'right_hip_roll_joint', 'right_hip_yaw_joint', 'right_knee_joint',
       'right_ankle_pitch_joint', 'right_ankle_roll_joint',
       'waist_yaw_joint', 'waist_roll_joint', 'waist_pitch_joint',
       'left_shoulder_pitch_joint', 'left_shoulder_roll_joint',
       'left_shoulder_yaw_joint', 'left_elbow_joint',
       'left_wrist_roll_joint', 'left_wrist_pitch_joint',
       'left_wrist_yaw_joint', 'right_shoulder_pitch_joint',
       'right_shoulder_roll_joint', 'right_shoulder_yaw_joint',
       'right_elbow_joint', 'right_wrist_roll_joint',
       'right_wrist_pitch_joint', 'right_wrist_yaw_joint']
    
    body_names = ['pelvis', 'left_hip_pitch_link', 'left_hip_roll_link', 'left_hip_yaw_link', 'left_knee_link', 'left_ankle_pitch_link', 'left_ankle_roll_link', 'right_hip_pitch_link', 'right_hip_roll_link', 'right_hip_yaw_link', 'right_knee_link', 'right_ankle_pitch_link', 'right_ankle_roll_link', 'waist_yaw_link', 'waist_roll_link', 'torso_link', 'left_shoulder_pitch_link', 'left_shoulder_roll_link', 'left_shoulder_yaw_link', 'left_elbow_link', 'left_wrist_roll_link', 'left_wrist_pitch_link', 'left_wrist_yaw_link', 'right_shoulder_pitch_link', 'right_shoulder_roll_link', 'right_shoulder_yaw_link', 'right_elbow_link', 'right_wrist_roll_link', 'right_wrist_pitch_link', 'right_wrist_yaw_link']
    
    dof_name2idx = {name: i for i, name in enumerate(dof_names) if name is not None}
    body_name2idx = {name: i for i, name in enumerate(body_names) if name is not None}

    dof_positions = data['dof_positions']
    dof_velocities = data['dof_velocities'] 
    body_positions = data['body_positions']
    body_rotations = data['body_rotations']
    body_linear_velocities= data['body_linear_velocities']
    body_angular_velocities = data['body_angular_velocities']
    fps = data['fps']
    print('fps', fps)
    body_names_plot = ['pelvis', 'left_hip_pitch_link','left_hip_roll_link','left_knee_link', 'left_ankle_roll_link', 'right_hip_pitch_link', 'right_hip_roll_link', 'right_knee_link', 'right_ankle_roll_link', 'torso_link',  'left_shoulder_roll_link','left_elbow_link',  'left_wrist_yaw_link',  'right_shoulder_roll_link', 'right_elbow_link', 'right_wrist_yaw_link']
    
    print('kept idxs', len(body_names_plot))
    keep_body_idx = [body_name2idx[name] for name in body_names_plot]

    frame=0
    length= dof_positions.shape[0]
    running=True 
    while running:
        frame_index = frame % length
        future_frames = min(frame_index + 10, length-1)
        if future_frames > frame_index:
            vis[0][1].draw_spheres(body_positions.transpose(1,0,2)[keep_body_idx ,frame_index ],color=(0, 1, 0), radius=0.02, transparency=(.8, 0.1))
            vis[0][1].draw_spheres(body_positions.transpose(1,0,2)[keep_body_idx,frame_index+1:future_frames], color=(0, 0, 1), radius=0.01, transparency=(.2, 0.05))
            

        frame+=1 
        running = vis.update(delay=1/fps)



if __name__ == "__main__":
    # main()
    vis_traj()
    # test_sphere_functionalities()
    # test_arrow_functionalities()
    # test_3x3_grid()