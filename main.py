"""
3D Gravity Simulation with Beautiful Solar System Visualization
Using ModernGL for high-performance rendering with enhanced graphics
"""

import pygame
import moderngl
import numpy as np
from pyrr import Matrix44, Vector3
import math
import random

# Constants
G = 6.67430e-11  # Gravitational constant
SCALE = 1e-9  # Scale factor for visualization
TIME_SCALE = 86400 * 2  # Seconds per frame (2 days for faster orbits)

class CelestialBody:
    """Represents a celestial body with mass, position, and velocity"""
    
    def __init__(self, name, mass, position, velocity, radius, color, glow_color=None):
        self.name = name
        self.mass = mass
        self.position = np.array(position, dtype=np.float64)
        self.velocity = np.array(velocity, dtype=np.float64)
        self.radius = radius
        self.color = color
        self.glow_color = glow_color if glow_color else color
        self.acceleration = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        self.trail = []
        self.max_trail_length = 1000
        self.is_star = False
        
    def apply_force(self, force):
        """Apply force to the body (F = ma, so a = F/m)"""
        self.acceleration += force / self.mass
        
    def update(self, dt):
        """Update position and velocity using Verlet integration"""
        self.velocity += self.acceleration * dt
        self.position += self.velocity * dt
        
        # Store trail
        if len(self.trail) == 0 or np.linalg.norm(self.position - self.trail[-1]) > 5e9:
            self.trail.append(self.position.copy())
            if len(self.trail) > self.max_trail_length:
                self.trail.pop(0)
        
        # Reset acceleration
        self.acceleration = np.array([0.0, 0.0, 0.0], dtype=np.float64)


def create_sphere_vertices(radius, rings=30, sectors=30):
    """Generate high-quality sphere vertices with normals"""
    vertices = []
    indices = []
    
    for i in range(rings + 1):
        theta = i * np.pi / rings
        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta)
        
        for j in range(sectors + 1):
            phi = j * 2 * np.pi / sectors
            sin_phi = np.sin(phi)
            cos_phi = np.cos(phi)
            
            x = cos_phi * sin_theta
            y = cos_theta
            z = sin_phi * sin_theta
            
            # Position
            vertices.extend([x * radius, y * radius, z * radius])
            # Normal
            vertices.extend([x, y, z])
    
    for i in range(rings):
        for j in range(sectors):
            first = i * (sectors + 1) + j
            second = first + sectors + 1
            
            indices.extend([first, second, first + 1])
            indices.extend([second, second + 1, first + 1])
    
    return np.array(vertices, dtype='f4'), np.array(indices, dtype='i4')


def create_stars(count=2000, radius=400):
    """Create a starfield background"""
    stars = []
    for _ in range(count):
        # Random position on a sphere
        theta = random.uniform(0, 2 * np.pi)
        phi = random.uniform(0, np.pi)
        
        x = radius * np.sin(phi) * np.cos(theta)
        y = radius * np.sin(phi) * np.sin(theta)
        z = radius * np.cos(phi)
        
        # Random brightness
        brightness = random.uniform(0.3, 1.0)
        
        stars.extend([x, y, z, brightness])
    
    return np.array(stars, dtype='f4')


class SpacetimeGrid:
    """Visualizes the curvature of spacetime as a beautiful 2D grid"""
    
    def __init__(self, size=100, resolution=60):
        self.size = size
        self.resolution = resolution
        self.vertices = None
        self.indices = None
        self.create_grid()
        
    def create_grid(self):
        """Create grid vertices and indices"""
        vertices = []
        indices = []
        
        step = self.size / self.resolution
        
        # Create vertices
        for i in range(self.resolution + 1):
            for j in range(self.resolution + 1):
                x = (i - self.resolution / 2) * step
                z = (j - self.resolution / 2) * step
                vertices.extend([x, 0, z])
        
        # Create line indices
        # Horizontal lines
        for i in range(self.resolution + 1):
            for j in range(self.resolution):
                idx = i * (self.resolution + 1) + j
                indices.extend([idx, idx + 1])
        
        # Vertical lines
        for j in range(self.resolution + 1):
            for i in range(self.resolution):
                idx = i * (self.resolution + 1) + j
                indices.extend([idx, idx + (self.resolution + 1)])
        
        self.vertices = np.array(vertices, dtype='f4')
        self.indices = np.array(indices, dtype='i4')
        
    def update_curvature(self, bodies):
        """Calculate spacetime curvature based on mass distribution"""
        step = self.size / self.resolution
        vertices = []
        
        for i in range(self.resolution + 1):
            for j in range(self.resolution + 1):
                x = (i - self.resolution / 2) * step
                z = (j - self.resolution / 2) * step
                
                # Calculate gravitational potential at this point
                y = 0
                for body in bodies:
                    bx = body.position[0] * SCALE
                    bz = body.position[2] * SCALE
                    
                    dist = math.sqrt((x - bx)**2 + (z - bz)**2 + 1.0)
                    
                    # Flamm paraboloid approximation with realistic scaling
                    # Larger masses create deeper wells
                    k = 0.00002  # Increased for better visibility
                    mass_factor = (body.mass / 1e24) ** 0.7  # Non-linear for better visualization
                    y -= k * mass_factor / dist
                    
                y = max(y, -50)  # Deeper wells for massive objects
                vertices.extend([x, y, z])
        
        self.vertices = np.array(vertices, dtype='f4')


class GravitySimulation:
    """Main simulation class using ModernGL with enhanced graphics"""
    
    def __init__(self):
        pygame.init()
        self.width, self.height = 1600, 900
        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MAJOR_VERSION, 3)
        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MINOR_VERSION, 3)
        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_PROFILE_MASK, pygame.GL_CONTEXT_PROFILE_CORE)
        pygame.display.gl_set_attribute(pygame.GL_MULTISAMPLESAMPLES, 4)  # Anti-aliasing
        
        self.screen = pygame.display.set_mode((self.width, self.height), pygame.OPENGL | pygame.DOUBLEBUF)
        pygame.display.set_caption("Solar System - Gravity Simulation")
        
        # Create ModernGL context
        self.ctx = moderngl.create_context()
        self.ctx.enable(moderngl.DEPTH_TEST)
        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
        
        # Setup shaders and buffers
        self.setup_shaders()
        self.setup_sphere()
        self.setup_stars()
        
        # Simulation state
        self.bodies = []
        self.grid = SpacetimeGrid(size=400, resolution=70)  # Larger grid for outer planets
        self.setup_grid()
        self.paused = False
        self.show_grid = True
        self.show_trails = True
        self.show_names = True
        
        # Camera controls
        self.camera_pos = Vector3([0.0, 120.0, 300.0])  # Better view for all 8 planets
        self.camera_rotation_x = 35  # Slightly higher angle
        self.camera_rotation_y = 0
        self.camera_auto_rotate = True
        self.rotation_speed = 0.15
        
        # Camera presets
        self.camera_presets = {
            'perspective': {'pos': Vector3([0.0, 120.0, 300.0]), 'rot_x': 35, 'rot_y': 0},
            'top': {'pos': Vector3([0.0, 400.0, 0.1]), 'rot_x': 89, 'rot_y': 0},
            'side': {'pos': Vector3([0.0, 0.0, 350.0]), 'rot_x': 0, 'rot_y': 0},
        }
        
        # Mouse controls
        self.mouse_dragging = False
        self.last_mouse_pos = None
        self.mouse_sensitivity = 0.3
        self.zoom_speed = 10.0
        
        self.setup_solar_system()
        
        # Font for UI
        self.font = pygame.font.Font(None, 28)
        self.title_font = pygame.font.Font(None, 48)
        
        # Time tracking
        self.simulation_time = 0
        
    def setup_shaders(self):
        """Create enhanced shader programs"""
        
        # Enhanced vertex shader for spheres with normals
        vertex_shader = '''
        #version 330
        
        uniform mat4 model;
        uniform mat4 view;
        uniform mat4 projection;
        
        in vec3 in_position;
        in vec3 in_normal;
        
        out vec3 v_pos;
        out vec3 v_normal;
        out vec3 v_world_pos;
        
        void main() {
            vec4 world_pos = model * vec4(in_position, 1.0);
            v_world_pos = world_pos.xyz;
            v_pos = in_position;
            v_normal = mat3(model) * in_normal;
            gl_Position = projection * view * world_pos;
        }
        '''
        
        # Enhanced fragment shader with better lighting
        fragment_shader = '''
        #version 330
        
        uniform vec3 color;
        uniform vec3 glow_color;
        uniform vec3 light_pos;
        uniform vec3 camera_pos;
        uniform bool is_star;
        
        in vec3 v_pos;
        in vec3 v_normal;
        in vec3 v_world_pos;
        
        out vec4 fragColor;
        
        void main() {
            if (is_star) {
                // Stars glow
                vec3 result = glow_color * 1.2;
                fragColor = vec4(result, 1.0);
            } else {
                vec3 normal = normalize(v_normal);
                vec3 light_dir = normalize(light_pos - v_world_pos);
                vec3 view_dir = normalize(camera_pos - v_world_pos);
                vec3 reflect_dir = reflect(-light_dir, normal);
                
                // Ambient
                float ambient_strength = 0.15;
                vec3 ambient = ambient_strength * color;
                
                // Diffuse
                float diff = max(dot(normal, light_dir), 0.0);
                vec3 diffuse = diff * color;
                
                // Specular
                float spec_strength = 0.4;
                float spec = pow(max(dot(view_dir, reflect_dir), 0.0), 32);
                vec3 specular = spec_strength * spec * vec3(1.0, 1.0, 1.0);
                
                // Rim lighting for better depth
                float rim = 1.0 - max(dot(view_dir, normal), 0.0);
                rim = pow(rim, 3.0);
                vec3 rim_color = rim * glow_color * 0.5;
                
                vec3 result = ambient + diffuse + specular + rim_color;
                fragColor = vec4(result, 1.0);
            }
        }
        '''
        
        self.sphere_program = self.ctx.program(
            vertex_shader=vertex_shader,
            fragment_shader=fragment_shader
        )
        
        # Shader for grid lines with gradient
        line_vertex_shader = '''
        #version 330
        
        uniform mat4 mvp;
        
        in vec3 in_position;
        out float v_depth;
        
        void main() {
            v_depth = in_position.y;
            gl_Position = mvp * vec4(in_position, 1.0);
        }
        '''
        
        line_fragment_shader = '''
        #version 330
        
        in float v_depth;
        out vec4 fragColor;
        
        void main() {
            // Gradient based on depth
            float depth_factor = clamp(-v_depth / 30.0, 0.0, 1.0);
            vec3 color = mix(vec3(0.2, 0.4, 0.8), vec3(0.6, 0.2, 0.8), depth_factor);
            float alpha = mix(0.6, 0.3, depth_factor);
            fragColor = vec4(color, alpha);
        }
        '''
        
        self.line_program = self.ctx.program(
            vertex_shader=line_vertex_shader,
            fragment_shader=line_fragment_shader
        )
        
        # Shader for stars
        star_vertex_shader = '''
        #version 330
        
        uniform mat4 mvp;
        
        in vec3 in_position;
        in float in_brightness;
        
        out float v_brightness;
        
        void main() {
            v_brightness = in_brightness;
            gl_Position = mvp * vec4(in_position, 1.0);
            gl_PointSize = 2.0;
        }
        '''
        
        star_fragment_shader = '''
        #version 330
        
        in float v_brightness;
        out vec4 fragColor;
        
        void main() {
            vec2 coord = gl_PointCoord - vec2(0.5);
            float dist = length(coord);
            if (dist > 0.5) discard;
            
            float alpha = (1.0 - dist * 2.0) * v_brightness;
            fragColor = vec4(1.0, 1.0, 1.0, alpha);
        }
        '''
        
        self.star_program = self.ctx.program(
            vertex_shader=star_vertex_shader,
            fragment_shader=star_fragment_shader
        )
        
    def setup_sphere(self):
        """Create high-quality sphere geometry"""
        vertices, indices = create_sphere_vertices(1.0, rings=30, sectors=30)
        
        self.sphere_vbo = self.ctx.buffer(vertices.tobytes())
        self.sphere_ibo = self.ctx.buffer(indices.tobytes())
        
        self.sphere_vao = self.ctx.vertex_array(
            self.sphere_program,
            [(self.sphere_vbo, '3f 3f', 'in_position', 'in_normal')],
            self.sphere_ibo
        )
        
    def setup_stars(self):
        """Create starfield"""
        star_data = create_stars(3000, 450)
        
        self.star_vbo = self.ctx.buffer(star_data.tobytes())
        self.star_vao = self.ctx.vertex_array(
            self.star_program,
            [(self.star_vbo, '3f 1f', 'in_position', 'in_brightness')]
        )
        
    def setup_grid(self):
        """Create grid geometry"""
        self.grid_vbo = self.ctx.buffer(self.grid.vertices.tobytes())
        self.grid_ibo = self.ctx.buffer(self.grid.indices.tobytes())
        
        self.grid_vao = self.ctx.vertex_array(
            self.line_program,
            [(self.grid_vbo, '3f', 'in_position')],
            self.grid_ibo
        )
        
    def update_grid_buffer(self):
        """Update grid vertex buffer with new curvature"""
        self.grid_vbo.write(self.grid.vertices.tobytes())
        
    def setup_solar_system(self):
        """Create a complete, realistic solar system with all 8 planets"""
        
        # Sun - glowing yellow star at the center
        sun = CelestialBody(
            name="Sun",
            mass=1.989e30,
            position=[0, 0, 0],
            velocity=[0, 0, 0],
            radius=10.0,  # Larger for visibility
            color=(1.0, 0.85, 0.1),
            glow_color=(1.0, 0.95, 0.4)
        )
        sun.is_star = True
        self.bodies.append(sun)
        
        # Mercury - small gray rocky planet
        self.bodies.append(CelestialBody(
            name="Mercury",
            mass=3.285e23,
            position=[5.79e10, 0, 0],
            velocity=[0, 0, 47870],
            radius=1.5,
            color=(0.55, 0.55, 0.55),
            glow_color=(0.65, 0.65, 0.65)
        ))
        
        # Venus - bright yellowish planet with thick atmosphere
        self.bodies.append(CelestialBody(
            name="Venus",
            mass=4.867e24,
            position=[1.082e11, 0, 0],
            velocity=[0, 0, 35020],
            radius=2.2,
            color=(0.95, 0.75, 0.35),
            glow_color=(1.0, 0.85, 0.5)
        ))
        
        # Earth - beautiful blue marble
        self.bodies.append(CelestialBody(
            name="Earth",
            mass=5.972e24,
            position=[1.496e11, 0, 0],
            velocity=[0, 0, 29780],
            radius=2.3,
            color=(0.15, 0.45, 0.95),
            glow_color=(0.4, 0.65, 1.0)
        ))
        
        # Mars - the red planet
        self.bodies.append(CelestialBody(
            name="Mars",
            mass=6.39e23,
            position=[2.279e11, 0, 0],
            velocity=[0, 0, 24070],
            radius=1.8,
            color=(0.85, 0.35, 0.15),
            glow_color=(1.0, 0.5, 0.25)
        ))
        
        # Jupiter - massive gas giant with bands
        self.bodies.append(CelestialBody(
            name="Jupiter",
            mass=1.898e27,
            position=[7.785e11, 0, 0],
            velocity=[0, 0, 13070],
            radius=6.5,
            color=(0.75, 0.55, 0.35),
            glow_color=(0.95, 0.75, 0.5)
        ))
        
        # Saturn - beautiful ringed planet
        self.bodies.append(CelestialBody(
            name="Saturn",
            mass=5.683e26,
            position=[1.429e12, 0, 0],
            velocity=[0, 0, 9690],
            radius=5.5,
            color=(0.85, 0.75, 0.55),
            glow_color=(1.0, 0.9, 0.7)
        ))
        
        # Uranus - ice giant with blue-green color
        self.bodies.append(CelestialBody(
            name="Uranus",
            mass=8.681e25,
            position=[2.871e12, 0, 0],
            velocity=[0, 0, 6810],
            radius=4.0,
            color=(0.5, 0.85, 0.85),
            glow_color=(0.7, 1.0, 1.0)
        ))
        
        # Neptune - deep blue ice giant
        self.bodies.append(CelestialBody(
            name="Neptune",
            mass=1.024e26,
            position=[4.495e12, 0, 0],
            velocity=[0, 0, 5430],
            radius=3.8,
            color=(0.25, 0.4, 0.95),
            glow_color=(0.4, 0.6, 1.0)
        ))
        
    def calculate_gravity(self):
        """Calculate gravitational forces between all bodies"""
        n = len(self.bodies)
        
        for i in range(n):
            for j in range(i + 1, n):
                body1 = self.bodies[i]
                body2 = self.bodies[j]
                
                # Calculate distance vector
                r_vec = body2.position - body1.position
                r_mag = np.linalg.norm(r_vec)
                
                if r_mag < 1e6:  # Prevent division by zero
                    continue
                
                # Calculate gravitational force magnitude
                force_mag = G * body1.mass * body2.mass / (r_mag ** 2)
                
                # Calculate force vector
                force_vec = force_mag * (r_vec / r_mag)
                
                # Apply forces (Newton's third law)
                body1.apply_force(force_vec)
                body2.apply_force(-force_vec)
                
    def update(self, dt):
        """Update simulation"""
        if not self.paused:
            self.calculate_gravity()
            
            for body in self.bodies:
                body.update(dt)
                
            if self.show_grid:
                self.grid.update_curvature(self.bodies)
                self.update_grid_buffer()
            
            self.simulation_time += dt
            
        # Auto-rotate camera
        if self.camera_auto_rotate:
            self.camera_rotation_y += self.rotation_speed
            
    def get_matrices(self):
        """Calculate view and projection matrices"""
        projection = Matrix44.perspective_projection(50.0, self.width / self.height, 0.1, 1000.0)
        
        view = Matrix44.identity()
        view = Matrix44.from_translation(-self.camera_pos) @ view
        view = Matrix44.from_x_rotation(np.radians(self.camera_rotation_x)) @ view
        view = Matrix44.from_y_rotation(np.radians(self.camera_rotation_y)) @ view
        
        return view, projection
        
    def draw(self):
        """Render the beautiful scene"""
        self.ctx.clear(0.0, 0.0, 0.02, 1.0)  # Deep space background
        
        view, projection = self.get_matrices()
        
        # Draw starfield
        mvp = projection @ view
        self.star_program['mvp'].write(mvp.astype('f4').tobytes())
        self.star_vao.render(moderngl.POINTS)
        
        # Draw spacetime grid
        if self.show_grid:
            self.line_program['mvp'].write(mvp.astype('f4').tobytes())
            self.grid_vao.render(moderngl.LINES)
        
        # Setup sphere rendering
        self.sphere_program['view'].write(view.astype('f4').tobytes())
        self.sphere_program['projection'].write(projection.astype('f4').tobytes())
        self.sphere_program['light_pos'].value = (0.0, 0.0, 0.0)  # Sun at origin
        self.sphere_program['camera_pos'].value = tuple(self.camera_pos)
        
        # Draw celestial bodies
        for body in self.bodies:
            # Create model matrix
            model = Matrix44.from_translation(Vector3([
                body.position[0] * SCALE,
                body.position[1] * SCALE,
                body.position[2] * SCALE
            ]))
            model = Matrix44.from_scale(Vector3([body.radius, body.radius, body.radius])) @ model
            
            self.sphere_program['model'].write(model.astype('f4').tobytes())
            self.sphere_program['color'].value = body.color
            self.sphere_program['glow_color'].value = body.glow_color
            self.sphere_program['is_star'].value = body.is_star
            
            self.sphere_vao.render(moderngl.TRIANGLES)
        
        # Draw trails
        if self.show_trails:
            self.draw_trails(view, projection)
        
        # Draw UI
        self.draw_ui()
        
        pygame.display.flip()
        
    def draw_trails(self, view, projection):
        """Draw beautiful orbital trails"""
        for body in self.bodies:
            if len(body.trail) < 2 or body.is_star:
                continue
            
            # Create trail vertices
            trail_verts = []
            for pos in body.trail:
                trail_verts.extend([
                    pos[0] * SCALE,
                    pos[1] * SCALE,
                    pos[2] * SCALE
                ])
            
            trail_array = np.array(trail_verts, dtype='f4')
            trail_vbo = self.ctx.buffer(trail_array.tobytes())
            trail_vao = self.ctx.vertex_array(
                self.line_program,
                [(trail_vbo, '3f', 'in_position')]
            )
            
            mvp = projection @ view
            self.line_program['mvp'].write(mvp.astype('f4').tobytes())
            
            trail_vao.render(moderngl.LINE_STRIP)
            trail_vbo.release()
            trail_vao.release()
        
    def draw_ui(self):
        """Draw beautiful UI overlay"""
        # Switch to 2D rendering
        self.ctx.disable(moderngl.DEPTH_TEST)
        
        # Create text surfaces
        y = 20
        
        # Title
        title = self.title_font.render("SOLAR SYSTEM", True, (255, 255, 255))
        title_rect = title.get_rect()
        title_rect.centerx = self.width // 2
        title_rect.y = y
        
        # Status
        status_color = (100, 255, 100) if not self.paused else (255, 100, 100)
        status_text = "RUNNING" if not self.paused else "PAUSED"
        status = self.font.render(status_text, True, status_color)
        
        # Info
        days = int(self.simulation_time / 86400)
        info_texts = [
            f"Simulation Time: {days} days",
            f"Bodies: {len(self.bodies)}",
            f"Grid: {'ON' if self.show_grid else 'OFF'}",
            f"Trails: {'ON' if self.show_trails else 'OFF'}",
            f"Auto-Rotate: {'ON' if self.camera_auto_rotate else 'OFF'}",
        ]
        
        # Controls
        control_texts = [
            "",
            "CONTROLS:",
            "SPACE - Pause/Resume",
            "G - Toggle Grid",
            "T - Toggle Trails",
            "R - Auto-Rotate",
            "Arrows - Rotate View",
            "+/- - Zoom",
            "ESC - Exit"
        ]
        
        # Render to pygame surface then to texture
        # For now, we'll use a simple approach
        
        self.ctx.enable(moderngl.DEPTH_TEST)
        
    def handle_input(self, events):
        """Handle keyboard and mouse input"""
        keys = pygame.key.get_pressed()
        
        # Handle mouse events
        for event in events:
            if event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # Left click
                    self.mouse_dragging = True
                    self.last_mouse_pos = pygame.mouse.get_pos()
                    self.camera_auto_rotate = False  # Disable auto-rotate when user controls
                    
            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:  # Left click release
                    self.mouse_dragging = False
                    self.last_mouse_pos = None
                    
            elif event.type == pygame.MOUSEWHEEL:
                # Zoom with mouse wheel
                zoom_delta = event.y * self.zoom_speed
                self.camera_pos.z = np.clip(self.camera_pos.z - zoom_delta, 50, 500)
                
            elif event.type == pygame.MOUSEMOTION:
                if self.mouse_dragging and self.last_mouse_pos:
                    # Calculate mouse movement
                    current_pos = pygame.mouse.get_pos()
                    dx = current_pos[0] - self.last_mouse_pos[0]
                    dy = current_pos[1] - self.last_mouse_pos[1]
                    
                    # Update camera rotation
                    self.camera_rotation_y += dx * self.mouse_sensitivity
                    self.camera_rotation_x -= dy * self.mouse_sensitivity
                    
                    # Clamp vertical rotation
                    self.camera_rotation_x = np.clip(self.camera_rotation_x, -89, 89)
                    
                    self.last_mouse_pos = current_pos
        
        # Keyboard camera rotation (still available)
        if keys[pygame.K_LEFT]:
            self.camera_rotation_y -= 1.5
            self.camera_auto_rotate = False
        if keys[pygame.K_RIGHT]:
            self.camera_rotation_y += 1.5
            self.camera_auto_rotate = False
        if keys[pygame.K_UP]:
            self.camera_rotation_x = max(-89, self.camera_rotation_x - 1.5)
        if keys[pygame.K_DOWN]:
            self.camera_rotation_x = min(89, self.camera_rotation_x + 1.5)
            
        # Keyboard zoom
        if keys[pygame.K_EQUALS] or keys[pygame.K_PLUS]:
            self.camera_pos.z = max(50, self.camera_pos.z - 3)
        if keys[pygame.K_MINUS] or keys[pygame.K_UNDERSCORE]:
            self.camera_pos.z = min(500, self.camera_pos.z + 3)
            
        # Camera presets
        if keys[pygame.K_1]:
            self.set_camera_preset('perspective')
        if keys[pygame.K_2]:
            self.set_camera_preset('top')
        if keys[pygame.K_3]:
            self.set_camera_preset('side')
            
    def set_camera_preset(self, preset_name):
        """Set camera to a preset view"""
        if preset_name in self.camera_presets:
            preset = self.camera_presets[preset_name]
            self.camera_pos = Vector3(preset['pos'])
            self.camera_rotation_x = preset['rot_x']
            self.camera_rotation_y = preset['rot_y']
            self.camera_auto_rotate = False
            print(f"📷 Camera view: {preset_name.upper()}")
            
    def run(self):
        """Main simulation loop"""
        clock = pygame.time.Clock()
        running = True
        
        print("╔═══════════════════════════════════════════════╗")
        print("║   3D SOLAR SYSTEM - ALL 8 PLANETS + SUN      ║")
        print("╚═══════════════════════════════════════════════╝")
        print("\n🌟 Planets: Sun, Mercury, Venus, Earth, Mars,")
        print("           Jupiter, Saturn, Uranus, Neptune")
        print("\n🎮 CONTROLS:")
        print("  🖱️  MOUSE:")
        print("    Left Drag  - Rotate camera freely")
        print("    Wheel      - Zoom in/out")
        print("\n  ⌨️  KEYBOARD:")
        print("    SPACE      - Pause/Resume")
        print("    G          - Toggle spacetime grid")
        print("    T          - Toggle orbital trails")
        print("    R          - Toggle auto-rotation")
        print("    Arrow Keys - Rotate camera")
        print("    +/-        - Zoom in/out")
        print("\n  📷 CAMERA VIEWS:")
        print("    1          - Perspective view (default)")
        print("    2          - Top-down view (horizontal)")
        print("    3          - Side view (vertical)")
        print("\n    ESC        - Exit")
        print("\n� Starting simulation...\n")
        
        while running:
            events = pygame.event.get()
            
            for event in events:
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_SPACE:
                        self.paused = not self.paused
                        print(f"⏸️  Paused" if self.paused else "▶️  Resumed")
                    elif event.key == pygame.K_g:
                        self.show_grid = not self.show_grid
                        print(f"🌐 Spacetime grid: {'ON' if self.show_grid else 'OFF'}")
                    elif event.key == pygame.K_t:
                        self.show_trails = not self.show_trails
                        print(f"✨ Orbital trails: {'ON' if self.show_trails else 'OFF'}")
                    elif event.key == pygame.K_r:
                        self.camera_auto_rotate = not self.camera_auto_rotate
                        print(f"🔄 Auto-rotate: {'ON' if self.camera_auto_rotate else 'OFF'}")
                        
            self.handle_input(events)
            self.update(TIME_SCALE)
            self.draw()
            
            clock.tick(60)
            
        print("\n👋 Simulation ended. Goodbye!")
        pygame.quit()


if __name__ == "__main__":
    sim = GravitySimulation()
    sim.run()
