"""
Blender Battery Charging Animation Script

This script creates an animated battery charging visualization in Blender,
showing a battery filling from 5% to 100% with cool blinking bar effects.

Usage:
    1. Open Blender
    2. Go to Scripting workspace
    3. Open this script
    4. Click "Run Script"

Or run from command line:
    blender --python blender_battery_animation.py

Requirements:
    - Blender 2.80 or higher
"""

import bpy
import math
from mathutils import Vector


def clear_scene():
    """Remove all objects from the scene."""
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)
    
    # Clear all materials
    for material in bpy.data.materials:
        bpy.data.materials.remove(material)


def create_emission_material(name, base_color, emission_strength=2.0):
    """Create an emission material for glowing effect."""
    material = bpy.data.materials.new(name=name)
    material.use_nodes = True
    nodes = material.node_tree.nodes
    links = material.node_tree.links
    
    # Clear default nodes
    nodes.clear()
    
    # Create nodes
    output = nodes.new(type='ShaderNodeOutputMaterial')
    output.location = (300, 0)
    
    mix_shader = nodes.new(type='ShaderNodeMixShader')
    mix_shader.location = (100, 0)
    
    emission = nodes.new(type='ShaderNodeEmission')
    emission.location = (-100, 50)
    emission.inputs['Color'].default_value = (*base_color, 1.0)
    emission.inputs['Strength'].default_value = emission_strength
    
    principled = nodes.new(type='ShaderNodeBsdfPrincipled')
    principled.location = (-100, -100)
    principled.inputs['Base Color'].default_value = (*base_color, 1.0)
    principled.inputs['Metallic'].default_value = 0.3
    principled.inputs['Roughness'].default_value = 0.4
    
    # Create mix factor driver node for animation
    value_node = nodes.new(type='ShaderNodeValue')
    value_node.location = (-300, 100)
    value_node.name = "BlinkFactor"
    value_node.outputs[0].default_value = 0.5
    
    # Connect nodes
    links.new(value_node.outputs[0], mix_shader.inputs['Fac'])
    links.new(emission.outputs['Emission'], mix_shader.inputs[1])
    links.new(principled.outputs['BSDF'], mix_shader.inputs[2])
    links.new(mix_shader.outputs['Shader'], output.inputs['Surface'])
    
    return material


def create_battery_frame_material():
    """Create a metallic material for the battery frame."""
    material = bpy.data.materials.new(name="BatteryFrame")
    material.use_nodes = True
    nodes = material.node_tree.nodes
    
    principled = nodes.get('Principled BSDF')
    principled.inputs['Base Color'].default_value = (0.15, 0.15, 0.18, 1.0)
    principled.inputs['Metallic'].default_value = 0.9
    principled.inputs['Roughness'].default_value = 0.3
    
    return material


def create_battery_frame():
    """Create the battery outer frame/casing."""
    # Main body dimensions
    width = 4.0
    height = 2.0
    depth = 0.6
    wall_thickness = 0.1
    
    # Create outer shell
    bpy.ops.mesh.primitive_cube_add(size=1, location=(0, 0, 0))
    outer = bpy.context.active_object
    outer.name = "BatteryOuter"
    outer.scale = (width/2, height/2, depth/2)
    
    # Apply scale
    bpy.ops.object.transform_apply(scale=True)
    
    # Add bevel modifier for rounded edges
    bevel_mod = outer.modifiers.new(name="Bevel", type='BEVEL')
    bevel_mod.width = 0.1
    bevel_mod.segments = 3
    
    # Create inner cutout (to be subtracted)
    bpy.ops.mesh.primitive_cube_add(size=1, location=(0, 0, 0.05))
    inner = bpy.context.active_object
    inner.name = "BatteryInner"
    inner.scale = ((width - wall_thickness*2)/2, (height - wall_thickness*2)/2, depth/2)
    bpy.ops.object.transform_apply(scale=True)
    
    # Boolean difference
    bool_mod = outer.modifiers.new(name="Boolean", type='BOOLEAN')
    bool_mod.operation = 'DIFFERENCE'
    bool_mod.object = inner
    
    # Apply boolean modifier
    bpy.context.view_layer.objects.active = outer
    bpy.ops.object.modifier_apply(modifier="Boolean")
    
    # Delete inner object
    bpy.data.objects.remove(inner, do_unlink=True)
    
    # Create battery terminal (positive end)
    bpy.ops.mesh.primitive_cube_add(size=1, location=(width/2 + 0.15, 0, 0))
    terminal = bpy.context.active_object
    terminal.name = "BatteryTerminal"
    terminal.scale = (0.15, 0.4, 0.25)
    
    # Add bevel to terminal
    bevel_mod = terminal.modifiers.new(name="Bevel", type='BEVEL')
    bevel_mod.width = 0.03
    bevel_mod.segments = 2
    
    # Apply material
    frame_material = create_battery_frame_material()
    outer.data.materials.append(frame_material)
    terminal.data.materials.append(frame_material)
    
    return outer, terminal


def create_battery_bars(num_bars=10):
    """Create the battery charge level bars."""
    bars = []
    
    # Bar dimensions and positioning
    bar_width = 0.3
    bar_height = 1.6
    bar_depth = 0.4
    bar_spacing = 0.35
    start_x = -1.6
    
    # Color gradient from red (empty) to green (full)
    def get_bar_color(index, total):
        """Get color for bar based on position (red -> yellow -> green)."""
        t = index / (total - 1)
        if t < 0.5:
            # Red to yellow
            r = 1.0
            g = t * 2
            b = 0.0
        else:
            # Yellow to green
            r = 1.0 - (t - 0.5) * 2
            g = 1.0
            b = 0.0
        return (r, g, b)
    
    for i in range(num_bars):
        x_pos = start_x + i * bar_spacing
        color = get_bar_color(i, num_bars)
        
        # Create bar mesh
        bpy.ops.mesh.primitive_cube_add(size=1, location=(x_pos, 0, 0))
        bar = bpy.context.active_object
        bar.name = f"Bar_{i:02d}"
        bar.scale = (bar_width/2, bar_height/2, bar_depth/2)
        
        # Apply scale
        bpy.ops.object.transform_apply(scale=True)
        
        # Add bevel for slight rounding
        bevel_mod = bar.modifiers.new(name="Bevel", type='BEVEL')
        bevel_mod.width = 0.02
        bevel_mod.segments = 2
        
        # Create and apply material
        material = create_emission_material(f"BarMaterial_{i:02d}", color, emission_strength=3.0)
        bar.data.materials.append(material)
        
        bars.append(bar)
    
    return bars


def animate_charging(bars, start_percent=5, end_percent=100, 
                    total_frames=250, blink_speed=4):
    """
    Animate the battery charging from start_percent to end_percent.
    
    Args:
        bars: List of bar objects
        start_percent: Starting charge percentage (default 5%)
        end_percent: Ending charge percentage (default 100%)
        total_frames: Total animation duration in frames
        blink_speed: Speed of blinking animation (higher = faster)
    """
    scene = bpy.context.scene
    scene.frame_start = 1
    scene.frame_end = total_frames
    
    num_bars = len(bars)
    start_bars = int(start_percent / 100 * num_bars)
    end_bars = num_bars  # All bars for 100%
    
    # Calculate how many frames per bar activation
    bars_to_fill = end_bars - start_bars
    frames_per_bar = total_frames / bars_to_fill if bars_to_fill > 0 else total_frames
    
    for i, bar in enumerate(bars):
        material = bar.data.materials[0]
        nodes = material.node_tree.nodes
        
        # Find the value node for blink factor
        blink_node = None
        for node in nodes:
            if node.name == "BlinkFactor":
                blink_node = node
                break
        
        if not blink_node:
            continue
        
        # Find emission node
        emission_node = None
        for node in nodes:
            if node.type == 'EMISSION':
                emission_node = node
                break
        
        # Determine when this bar should start filling
        if i < start_bars:
            # Bar is already filled at start
            activation_frame = 1
        else:
            # Calculate when this bar activates
            bar_index = i - start_bars
            activation_frame = int(bar_index * frames_per_bar) + 1
        
        # Create animation for visibility (scale Y from 0 to 1)
        # Before activation: bar is invisible (scale 0)
        if i >= start_bars:
            # Set initial scale to 0
            bar.scale[1] = 0.001
            bar.keyframe_insert(data_path="scale", index=1, frame=1)
            bar.keyframe_insert(data_path="scale", index=1, frame=activation_frame - 1)
            
            # Animate bar appearing
            bar.scale[1] = 1.0
            bar.keyframe_insert(data_path="scale", index=1, frame=activation_frame + 5)
        
        # Create blinking animation for active/charging bar
        # The currently charging bar blinks, filled bars glow steadily
        
        if emission_node:
            emission_strength = emission_node.inputs['Strength']
            
            if i >= start_bars:
                # Initially dim
                emission_strength.default_value = 0.5
                emission_strength.keyframe_insert("default_value", frame=1)
                
                # Stay dim until activation
                emission_strength.keyframe_insert("default_value", frame=activation_frame - 1)
            
            # Determine the frame range where this bar is "charging" (blinking)
            if i < start_bars:
                # Already filled - steady glow
                emission_strength.default_value = 3.0
                emission_strength.keyframe_insert("default_value", frame=1)
            else:
                bar_index = i - start_bars
                start_blink = activation_frame
                
                # Calculate end of blinking (when next bar starts)
                if i < end_bars - 1:
                    end_blink = int((bar_index + 1) * frames_per_bar) + 1
                else:
                    end_blink = total_frames
                
                # Create blinking animation during charging
                blink_frames = int((end_blink - start_blink) / blink_speed)
                
                for j in range(max(1, blink_frames)):
                    frame = start_blink + j * blink_speed
                    if frame > total_frames:
                        break
                    
                    # Alternate between bright and dim
                    if j % 2 == 0:
                        emission_strength.default_value = 5.0  # Bright
                    else:
                        emission_strength.default_value = 1.5  # Dim
                    
                    emission_strength.keyframe_insert("default_value", frame=frame)
                
                # After charging complete, steady bright glow
                emission_strength.default_value = 3.5
                emission_strength.keyframe_insert("default_value", frame=end_blink)
                emission_strength.keyframe_insert("default_value", frame=total_frames)
    
    # Add a final flash effect at 100%
    for i, bar in enumerate(bars):
        material = bar.data.materials[0]
        for node in material.node_tree.nodes:
            if node.type == 'EMISSION':
                emission = node.inputs['Strength']
                
                # Big flash at the end
                emission.default_value = 8.0
                emission.keyframe_insert("default_value", frame=total_frames - 10)
                
                emission.default_value = 3.0
                emission.keyframe_insert("default_value", frame=total_frames)


def setup_lighting():
    """Set up lighting for the scene."""
    # Main key light
    bpy.ops.object.light_add(type='AREA', location=(3, -3, 4))
    key_light = bpy.context.active_object
    key_light.name = "KeyLight"
    key_light.data.energy = 200
    key_light.data.size = 4
    key_light.rotation_euler = (math.radians(45), 0, math.radians(45))
    
    # Fill light
    bpy.ops.object.light_add(type='AREA', location=(-3, -2, 3))
    fill_light = bpy.context.active_object
    fill_light.name = "FillLight"
    fill_light.data.energy = 100
    fill_light.data.size = 3
    fill_light.rotation_euler = (math.radians(45), 0, math.radians(-45))
    
    # Rim light from behind
    bpy.ops.object.light_add(type='AREA', location=(0, 4, 2))
    rim_light = bpy.context.active_object
    rim_light.name = "RimLight"
    rim_light.data.energy = 150
    rim_light.data.size = 5
    rim_light.rotation_euler = (math.radians(-60), 0, 0)


def setup_camera():
    """Set up camera for the scene."""
    bpy.ops.object.camera_add(location=(0, -6, 2))
    camera = bpy.context.active_object
    camera.name = "MainCamera"
    camera.rotation_euler = (math.radians(75), 0, 0)
    
    # Set as active camera
    bpy.context.scene.camera = camera
    
    # Add slight camera movement for dynamism
    camera.keyframe_insert(data_path="location", frame=1)
    camera.location = (0.3, -5.8, 2.1)
    camera.keyframe_insert(data_path="location", frame=250)
    
    return camera


def setup_world():
    """Set up world/environment settings."""
    world = bpy.context.scene.world
    if not world:
        world = bpy.data.worlds.new("World")
        bpy.context.scene.world = world
    
    world.use_nodes = True
    nodes = world.node_tree.nodes
    
    # Set background to dark gradient
    bg_node = nodes.get('Background')
    if bg_node:
        bg_node.inputs['Color'].default_value = (0.02, 0.02, 0.05, 1.0)
        bg_node.inputs['Strength'].default_value = 0.5


def setup_render_settings():
    """Configure render settings for animation."""
    scene = bpy.context.scene
    
    # Render settings
    scene.render.engine = 'CYCLES'
    scene.cycles.samples = 64  # Adjust for quality vs speed
    scene.cycles.use_denoising = True
    
    # Output settings
    scene.render.resolution_x = 1920
    scene.render.resolution_y = 1080
    scene.render.fps = 30
    
    # File output
    scene.render.image_settings.file_format = 'FFMPEG'
    scene.render.ffmpeg.format = 'MPEG4'
    scene.render.ffmpeg.codec = 'H264'
    scene.render.filepath = "//battery_charging_animation"
    
    # Enable bloom for glowing effect
    scene.view_settings.view_transform = 'Filmic'
    scene.view_settings.look = 'High Contrast'
    
    # Enable compositing for bloom
    scene.use_nodes = True
    tree = scene.node_tree
    
    # Clear existing nodes
    for node in tree.nodes:
        tree.nodes.remove(node)
    
    # Create compositing nodes
    render_layers = tree.nodes.new('CompositorNodeRLayers')
    render_layers.location = (0, 0)
    
    glare = tree.nodes.new('CompositorNodeGlare')
    glare.location = (200, 0)
    glare.glare_type = 'FOG_GLOW'
    glare.threshold = 0.5
    glare.size = 7
    glare.mix = 0.5
    
    composite = tree.nodes.new('CompositorNodeComposite')
    composite.location = (400, 0)
    
    # Connect nodes
    tree.links.new(render_layers.outputs['Image'], glare.inputs['Image'])
    tree.links.new(glare.outputs['Image'], composite.inputs['Image'])


def add_percentage_text():
    """Add animated percentage text display."""
    # Create text object
    bpy.ops.object.text_add(location=(0, -1.5, 0))
    text_obj = bpy.context.active_object
    text_obj.name = "PercentageText"
    text_obj.data.body = "5%"
    text_obj.data.align_x = 'CENTER'
    text_obj.data.align_y = 'CENTER'
    text_obj.data.size = 0.4
    text_obj.rotation_euler = (math.radians(90), 0, 0)
    
    # Create text material
    text_material = bpy.data.materials.new(name="TextMaterial")
    text_material.use_nodes = True
    nodes = text_material.node_tree.nodes
    principled = nodes.get('Principled BSDF')
    principled.inputs['Base Color'].default_value = (0.9, 0.9, 0.9, 1.0)
    principled.inputs['Emission Color'].default_value = (1.0, 1.0, 1.0, 1.0)
    principled.inputs['Emission Strength'].default_value = 1.0
    
    text_obj.data.materials.append(text_material)
    
    # Add driver for animated text (using frame change handler)
    # Note: In Blender, animated text requires a script handler
    return text_obj


def create_percentage_updater():
    """Create a frame change handler to update percentage text."""
    def update_percentage(scene):
        text_obj = bpy.data.objects.get("PercentageText")
        if not text_obj:
            return
        
        frame = scene.frame_current
        total_frames = scene.frame_end
        
        # Calculate current percentage (5% to 100%)
        progress = frame / total_frames
        percentage = int(5 + progress * 95)
        percentage = min(100, max(5, percentage))
        
        text_obj.data.body = f"{percentage}%"
    
    # Register the handler
    bpy.app.handlers.frame_change_post.clear()
    bpy.app.handlers.frame_change_post.append(update_percentage)


def main():
    """Main function to create the battery charging animation."""
    print("=" * 50)
    print("Creating Battery Charging Animation")
    print("=" * 50)
    
    # Clear existing scene
    print("Clearing scene...")
    clear_scene()
    
    # Create battery components
    print("Creating battery frame...")
    frame, terminal = create_battery_frame()
    
    print("Creating battery bars...")
    bars = create_battery_bars(num_bars=10)
    
    # Set up animation
    print("Setting up charging animation...")
    animate_charging(bars, start_percent=5, end_percent=100, 
                    total_frames=250, blink_speed=4)
    
    # Add percentage display
    print("Adding percentage text...")
    add_percentage_text()
    create_percentage_updater()
    
    # Set up scene
    print("Setting up lighting...")
    setup_lighting()
    
    print("Setting up camera...")
    setup_camera()
    
    print("Setting up world environment...")
    setup_world()
    
    print("Configuring render settings...")
    setup_render_settings()
    
    # Set viewport shading to rendered for preview
    for area in bpy.context.screen.areas:
        if area.type == 'VIEW_3D':
            for space in area.spaces:
                if space.type == 'VIEW_3D':
                    space.shading.type = 'MATERIAL'
    
    print("=" * 50)
    print("Battery charging animation created successfully!")
    print("=" * 50)
    print("\nTo render the animation:")
    print("1. Press Ctrl+F12 to render animation")
    print("2. Or go to Render > Render Animation")
    print("\nTo preview:")
    print("1. Press Space to play animation in viewport")
    print("2. Or use the timeline to scrub through frames")
    print("=" * 50)


# Run main function when script is executed
if __name__ == "__main__":
    main()
