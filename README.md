# Gravity Simulation Suite

A collection of beautiful Python gravity simulations using **ModernGL** for high-performance 3D rendering.

## Project Structure

```
gravity_sim/
├── main.py              # Full 3D Solar System with spacetime curvature
├── simple_gravity.py    # 2D bouncing ball demonstration
├── requirements.txt     # Python dependencies
├── README.md           # This file
├── CONTROLS.md         # Camera controls guide
└── PLANETS_GUIDE.md    # Complete planet reference
```

## Features

### 3D Solar System (main.py)

A stunning visualization of our complete solar system with:

**Beautiful Graphics**
- 3000+ star background for deep space atmosphere
- 9 celestial bodies (Sun + 8 planets: Mercury, Venus, Earth, Mars, Jupiter, Saturn, Uranus, Neptune)
- Enhanced lighting with rim effects and specular highlights
- Gradient spacetime grid showing gravitational wells
- Glowing sun with proper star rendering
- Smooth orbital trails

**Real Physics**
- Newtonian gravity (F = G × m₁ × m₂ / r²)
- N-body simulation with all gravitational interactions
- Verlet integration for stable orbits
- Spacetime curvature visualization (Flamm paraboloid)
- Realistic masses, distances, and velocities

**Interactive Controls**
- Full 3D camera rotation with mouse
- Mouse wheel zoom
- Camera preset views (perspective, top-down, side)
- Auto-rotation mode
- Pause/resume
- Toggle grid and trails

### Simple Gravity (simple_gravity.py)

A 2D demonstration showing:

- Gravity as acceleration (9.81 m/s²)
- Bouncing balls with energy damping
- Click to spawn new balls
- Velocity vectors visualization
- Perfect for understanding basic physics

## Quick Start

### Installation

```bash
# Navigate to the project
cd gravity_sim

# Create virtual environment
python3 -m venv venv

# Activate it
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
```

### Running the Simulations

**3D Solar System:**
```bash
./venv/bin/python main.py
```

**2D Bouncing Ball:**
```bash
./venv/bin/python simple_gravity.py
```

## Controls

### 3D Solar System

| Key | Action |
|-----|--------|
| **SPACE** | Pause/Resume simulation |
| **G** | Toggle spacetime grid |
| **T** | Toggle orbital trails |
| **R** | Toggle auto-rotation |
| **1** | Perspective view (default) |
| **2** | Top-down view (horizontal) |
| **3** | Side view (vertical) |
| **Left/Right Arrow** | Rotate camera left/right |
| **Up/Down Arrow** | Rotate camera up/down |
| **+ / -** | Zoom in/out |
| **Left Mouse Drag** | Free camera rotation |
| **Mouse Wheel** | Smooth zoom |
| **ESC** | Exit |

### 2D Bouncing Ball

| Key | Action |
|-----|--------|
| **SPACE** | Pause/Resume |
| **CLICK** | Add new ball at cursor |
| **R** | Reset simulation |
| **ESC** | Exit |

## Physics Explained

### Newtonian Gravity

Every object with mass attracts every other object:

```
F = G × m₁ × m₂ / r²
```

Where:
- `F` = Gravitational force (Newtons)
- `G` = 6.674 × 10⁻¹¹ (gravitational constant)
- `m₁, m₂` = Masses of the two objects (kg)
- `r` = Distance between centers (meters)

### From Force to Motion

1. **Force → Acceleration**: `a = F / m` (Newton's 2nd law)
2. **Acceleration → Velocity**: `v = v + a × dt`
3. **Velocity → Position**: `p = p + v × dt`

### Spacetime Curvature

The grid shows Einstein's general relativity concept:

- **Newtonian view**: Gravity is a force pulling objects together
- **Einsteinian view**: Mass curves spacetime, objects follow the curves

The Flamm paraboloid formula approximates this curvature:

```
y = -k × mass / distance
```

Massive objects create deeper "wells" in spacetime!

## Visual Enhancements

### Lighting System

- **Ambient**: Base illumination (15%)
- **Diffuse**: Surface angle to light
- **Specular**: Shiny highlights
- **Rim Lighting**: Edge glow for depth perception

### Shader Pipeline

1. **Vertex Shader**: Transforms 3D positions
2. **Fragment Shader**: Calculates per-pixel colors
3. **ModernGL**: Efficient GPU rendering

### Color Palette

| Body | Color | Glow |
|------|-------|------|
| Sun | Golden Yellow | Bright Yellow |
| Mercury | Gray | Light Gray |
| Venus | Yellowish | Warm Yellow |
| Earth | Blue | Cyan |
| Mars | Red | Bright Red |
| Jupiter | Brown/Tan | Warm Tan |
| Saturn | Pale Yellow | Light Yellow |
| Uranus | Cyan | Bright Cyan |
| Neptune | Deep Blue | Bright Blue |

## Customization

### Add Your Own Planet

```python
self.bodies.append(CelestialBody(
    name="MyPlanet",
    mass=5.972e24,              # Earth mass
    position=[2.0e11, 0, 0],    # 2 AU from sun
    velocity=[0, 0, 25000],     # Orbital velocity
    radius=2.0,                 # Visual size
    color=(0.5, 0.8, 0.3),     # RGB (0-1)
    glow_color=(0.7, 1.0, 0.5) # Rim light color
))
```

### Adjust Simulation Speed

```python
TIME_SCALE = 86400 * 2  # 2 days per frame (default)
TIME_SCALE = 86400      # 1 day per frame (slower)
TIME_SCALE = 86400 * 7  # 1 week per frame (faster)
```

### Modify Grid Detail

```python
self.grid = SpacetimeGrid(
    size=400,        # Grid size
    resolution=70    # Grid lines (higher = more detailed)
)
```

### Change Camera Settings

```python
self.camera_pos = Vector3([0.0, 120.0, 300.0])  # Position
self.camera_rotation_x = 35                      # Pitch
self.camera_rotation_y = 0                       # Yaw
self.rotation_speed = 0.15                       # Auto-rotate speed
```

## Performance

- **Target**: 60 FPS
- **Resolution**: 1600×900 (adjustable)
- **Anti-aliasing**: 4x MSAA
- **Bodies**: 9 (expandable to 20+ with good performance)
- **Stars**: 3000 background points
- **Grid**: 70×70 resolution

### Optimization Tips

1. **Reduce grid resolution** for better FPS
2. **Disable trails** if rendering is slow
3. **Lower star count** in `create_stars()`
4. **Reduce sphere detail** in `create_sphere_vertices()`

## Highlights

### What Makes This Special

1. **Modern OpenGL**: Uses shader-based rendering (not legacy OpenGL)
2. **Real Physics**: Actual gravitational equations, not approximations
3. **Beautiful Visuals**: Professional-grade graphics with lighting
4. **Educational**: Shows both Newtonian and Einsteinian gravity
5. **Interactive**: Full camera control and real-time adjustments
6. **Extensible**: Easy to add new bodies or features
7. **Complete Solar System**: All 8 planets plus the Sun

### Technical Achievements

- N-body gravitational simulation
- Verlet integration for stability
- Modern GLSL 330 shaders
- Procedural sphere generation
- Dynamic spacetime curvature
- Efficient GPU rendering
- Smooth 60 FPS performance
- Camera preset system
- Mouse-based free camera

## Learning Resources

### Physics Concepts

- **Gravity**: Newton's law of universal gravitation
- **Orbits**: Kepler's laws of planetary motion
- **Relativity**: Einstein's spacetime curvature
- **Numerical Integration**: Verlet method

### Graphics Concepts

- **3D Rendering**: Model-View-Projection matrices
- **Shaders**: Vertex and fragment programs
- **Lighting**: Phong reflection model
- **Procedural Geometry**: Sphere tessellation

## Camera Preset Views

### Perspective View (Press 1)
- Default 3D view at 35° angle
- Best for overall solar system visualization
- Shows depth and orbital relationships

### Top-Down View (Press 2)
- Looking straight down from above
- Perfect for seeing orbital mechanics
- Shows spacetime curvature clearly
- Horizontal plane view

### Side View (Press 3)
- Horizontal viewing angle
- Shows orbital planes
- Good for seeing planet alignment
- Vertical plane view

## Future Ideas

- Black hole with gravitational lensing
- Add moons to planets
- Collision detection and merging
- Relativistic effects (time dilation)
- Screenshot/video recording
- Save/load simulation states
- Click to select and follow bodies
- Graphs of orbital parameters
- Multiple star systems
- Texture mapping for planets
- Planet name labels in 3D space

## Troubleshooting

### "ModernGL context creation failed"
- Ensure you have OpenGL 3.3+ support
- Update graphics drivers

### "Simulation is slow"
- Reduce grid resolution
- Disable trails
- Lower star count

### "Bodies fly away"
- Check initial velocities
- Reduce TIME_SCALE
- Verify masses are correct

## License

MIT License - Free to use and modify!

## Credits

Inspired by the beauty of orbital mechanics and the elegance of physics. This project demonstrates both classical Newtonian gravity and Einstein's general relativity in an interactive, visually stunning way.

**Built with:**
- Python 3.11+
- ModernGL (modern OpenGL wrapper)
- Pygame (window management)
- NumPy (numerical computing)
- PyRR (3D math)

---

**Enjoy exploring the cosmos!**
