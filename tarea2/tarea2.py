import pyglet
from OpenGL import GL
import numpy as np
import trimesh as tm
import networkx as nx
import os
import sys
from pathlib import Path
# No es necesaria la siguiente línea si el archivo está en el root del repositorio
sys.path.append(os.path.dirname(os.path.dirname((os.path.abspath(__file__)))))
import grafica.transformations as tr
import utils.shapes as shapes

import auxiliares.utils.shapes as shapes
from auxiliares.utils.camera import FreeCamera
from auxiliares.utils.scene_graph import SceneGraph
from auxiliares.utils.drawables import Model, Texture, DirectionalLight, Material
from auxiliares.utils.colliders import CollisionManager, AABB, Sphere
from auxiliares.utils.helpers import init_axis, init_pipeline, mesh_from_file, get_path


WIDTH = 640
HEIGHT = 640

class Controller(pyglet.window.Window):
    def __init__(self, title, *args, **kargs):
        super().__init__(*args, **kargs)
        self.set_minimum_size(240, 240) # Evita error cuando se redimensiona a 0
        self.set_caption(title)
        self.key_handler = pyglet.window.key.KeyStateHandler()
        self.push_handlers(self.key_handler)
        self.program_state = { "total_time": 0.0 }
        self.init()

    def init(self):
        GL.glClearColor(1, 1, 1, 1.0)
        GL.glEnable(GL.GL_DEPTH_TEST)
        GL.glEnable(GL.GL_CULL_FACE)
        GL.glCullFace(GL.GL_BACK)
        GL.glFrontFace(GL.GL_CCW)

    def is_key_pressed(self, key):
        return self.key_handler[key]

class Model():
    def __init__(self, position_data, index_data=None):
        self.position_data = position_data

        self.index_data = index_data
        if index_data is not None:
            self.index_data = np.array(index_data, dtype=np.uint32)

        self.gpu_data = None

    def init_gpu_data(self, pipeline):
        self.pipeline = pipeline
        if self.index_data is not None:
            self.gpu_data = pipeline.vertex_list_indexed(len(self.position_data) // 3, GL.GL_TRIANGLES, self.index_data)
        else:
            self.gpu_data = pipeline.vertex_list(len(self.position_data) // 3, GL.GL_TRIANGLES)
        
        self.gpu_data.position[:] = self.position_data

    def draw(self, mode = GL.GL_TRIANGLES):
        self.gpu_data.draw(mode)

class Mesh(Model):
    def __init__(self, asset_path):
        mesh_data = tm.load(asset_path)
        mesh_scale = tr.uniformScale(2.0 / mesh_data.scale)
        mesh_translate = tr.translate(*-mesh_data.centroid)
        mesh_data.apply_transform( mesh_scale @ mesh_translate)
        vertex_data = tm.rendering.mesh_to_vertexlist(mesh_data)
        indices = vertex_data[3]
        positions = vertex_data[4][1]

        super().__init__(positions, indices)

class Camera():
    def __init__(self, camera_type = "perspective"):
        self.position = np.array([1, 0, 0], dtype=np.float32)
        self.focus = np.array([0, 0, 0], dtype=np.float32)
        self.type = camera_type
        self.width = WIDTH
        self.height = HEIGHT

    def update(self):
        pass

    def get_view(self):
        lookAt_matrix = tr.lookAt(self.position, self.focus, np.array([0, 1, 0], dtype=np.float32))
        return np.reshape(lookAt_matrix, (16, 1), order="F")

    def get_projection(self):
        if self.type == "perspective":
            perspective_matrix = tr.perspective(90, self.width / self.height, 0.01, 100)
        elif self.type == "orthographic":
            depth = self.position - self.focus
            depth = np.linalg.norm(depth)
            perspective_matrix = tr.ortho(-(self.width/self.height) * depth, (self.width/self.height) * depth, -1 * depth, 1 * depth, 0.01, 100)
        return np.reshape(perspective_matrix, (16, 1), order="F")
    
    def resize(self, width, height):
        self.width = width
        self.height = height

class OrbitCamera(Camera):
    def __init__(self, distance, camera_type = "perspective"):
        super().__init__(camera_type)
        self.distance = distance
        self.phi = 0
        self.theta = np.pi / 2
        self.update()

    def update(self):
        if self.theta > np.pi:
            self.theta = np.pi
        elif self.theta < 0:
            self.theta = 0.0001

        self.position[0] = self.distance * np.sin(self.theta) * np.sin(self.phi)
        self.position[1] = self.distance * np.cos(self.theta)
        self.position[2] = self.distance * np.sin(self.theta) * np.cos(self.phi)

class SceneGraph():
    def __init__(self, camera=None):
        self.graph = nx.DiGraph(root="root")
        self.add_node("root")
        self.camera = camera

    def add_node(self,
                 name,
                 attach_to=None,
                 mesh=None,
                 color=[1, 1, 1],
                 transform=tr.identity(),
                 position=[0, 0, 0],
                 rotation=[0, 0, 0],
                 scale=[1, 1, 1],
                 mode=GL.GL_TRIANGLES):
        self.graph.add_node(
            name, 
            mesh=mesh, 
            color=color,
            transform=transform,
            position=np.array(position, dtype=np.float32),
            rotation=np.array(rotation, dtype=np.float32),
            scale=np.array(scale, dtype=np.float32),
            mode=mode)
        if attach_to is None:
            attach_to = "root"
        
        self.graph.add_edge(attach_to, name)

    def __getitem__(self, name):
        if name not in self.graph.nodes:
            raise KeyError(f"Node {name} not in graph")

        return self.graph.nodes[name]
    
    def __setitem__(self, name, value):
        if name not in self.graph.nodes:
            raise KeyError(f"Node {name} not in graph")

        self.graph.nodes[name] = value
    
    def get_transform(self, node):
        node = self.graph.nodes[node]
        transform = node["transform"]
        translation_matrix = tr.translate(node["position"][0], node["position"][1], node["position"][2])
        rotation_matrix = tr.rotationX(node["rotation"][0]) @ tr.rotationY(node["rotation"][1]) @ tr.rotationZ(node["rotation"][2])
        scale_matrix = tr.scale(node["scale"][0], node["scale"][1], node["scale"][2])
        return transform @ translation_matrix @ rotation_matrix @ scale_matrix

    def draw(self):
        root_key = self.graph.graph["root"]
        edges = list(nx.edge_dfs(self.graph, source=root_key))
        transformations = {root_key: self.get_transform(root_key)}

        for src, dst in edges:
            current_node = self.graph.nodes[dst]

            if not dst in transformations:
                transformations[dst] = transformations[src] @ self.get_transform(dst)

            if current_node["mesh"] is not None:
                current_pipeline = current_node["mesh"].pipeline
                current_pipeline.use()

                if self.camera is not None:
                    if "u_view" in current_pipeline.uniforms:
                        current_pipeline["u_view"] = self.camera.get_view()

                    if "u_projection" in current_pipeline.uniforms:
                        current_pipeline["u_projection"] = self.camera.get_projection()

                current_pipeline["u_model"] = np.reshape(transformations[dst], (16, 1), order="F")

                if "u_color" in current_pipeline.uniforms:
                    current_pipeline["u_color"] = np.array(current_node["color"], dtype=np.float32)
                current_node["mesh"].draw(current_node["mode"])

class Car():
    def __init__(self, camera):
        
        floor = Model(shapes.Cube["position"], index_data=shapes.Cube["indices"])
        floor.init_gpu_data(mesh_pipeline)

        chasis = Model(shapes.Cube["position"], index_data=shapes.Cube["indices"])
        chasis.init_gpu_data(mesh_pipeline)

        wheel = Mesh("./assets/llanta2.off")
        wheel.init_gpu_data(mesh_pipeline)

        garage = Mesh("./assets/1to64Parkingdiorama.stl")
        garage.init_gpu_data(mesh_pipeline)
        
        color_mesh_pipeline = init_pipeline(
        get_path("auxiliares/shaders/color_mesh.vert"),
        get_path("auxiliares/shaders/color_mesh.frag"))
    
        textured_mesh_pipeline = init_pipeline(
        get_path("auxiliares/shaders/textured_mesh.vert"),
        get_path("auxiliares/shaders/textured_mesh.frag"))
    
        color_mesh_lit_pipeline = init_pipeline(
        get_path("auxiliares/shaders/color_mesh_lit.vert"),
        get_path("auxiliares/shaders/color_mesh_lit.frag"))
    
        textured_mesh_lit_pipeline = init_pipeline(
        get_path("auxiliares/shaders/textured_mesh_lit.vert"),
        get_path("auxiliares/shaders/textured_mesh_lit.frag"))


        self.graph = SceneGraph(camera)

        self.graph.add_node("car", scale=[2.5, 2.5, 2.5])

        self.graph.add_node("sun",
                    pipeline=[color_mesh_lit_pipeline, textured_mesh_lit_pipeline],
                    light=DirectionalLight(),
                    rotation=[-np.pi/4, 0, 0],
                   )

        self.graph.add_node("chasis", attach_to="car", position=[0, 0.4, 0], pipeline=color_mesh_lit_pipeline, material = Material())

        self.graph.add_node("bottom_chasis", attach_to="chasis", mesh=chasis, color=shapes.RED, scale=[1.3, 0.3, 1.8])
        self.graph.add_node("top_chasis", attach_to="chasis", mesh=chasis, color=shapes.LIGHT_BLUE, scale=[1, 0.3, 1], position=[0, 0.3, -0.1])
        self.graph.add_node("aileron", attach_to="chasis", mesh=chasis, color=shapes.RED, scale=[1.3, 0.1, 0.025], position=[0, 0.17, -0.92], rotation=[-0.8, 0, 0])
        self.graph.add_node("left_light", attach_to="chasis", mesh=chasis, color=shapes.YELLOW, scale=[0.15, 0.15, 0.025], position=[0.5, 0, 0.92])
        self.graph.add_node("right_light", attach_to="chasis", mesh=chasis, color=shapes.YELLOW, scale=[0.15, 0.15, 0.025], position=[-0.5, 0, 0.92])


        self.graph.add_node("first_wheel", attach_to="car", mesh=wheel, color=shapes.BLACK, position=[0.7, 0.2, 0.4], scale=[0.35, 0.35, 0.35])
        self.graph.add_node("second_wheel", attach_to="car", mesh=wheel, color=shapes.BLACK, position=[-0.7, 0.2, 0.4], scale=[0.35, 0.35, 0.35])
        self.graph.add_node("third_wheel", attach_to="car", mesh=wheel, color=shapes.BLACK, position=[0.7, 0.2, -0.4], scale=[0.35, 0.35, 0.35])
        self.graph.add_node("fourth_wheel", attach_to="car", mesh=wheel, color=shapes.BLACK, position=[-0.7, 0.2, -0.4], scale=[0.35, 0.35, 0.35])
        
        self.graph.add_node("garage", mesh=garage, color=shapes.GRAY, rotation=[-(np.pi)/2, 0, 0], scale=[13, 10, 8], position=[0, 1.25, 0])

    def draw(self):
        self.graph.draw()

    def wheel_rotate(self, dt):
        period = dt*3
        self.graph["first_wheel"]["transform"] = tr.translate(0.7, 0.2, 0.4) @ tr.rotationX(period) @ tr.translate(-0.7, -0.2, -0.4)
        self.graph["second_wheel"]["transform"] = tr.translate(-0.7, 0.2, 0.4) @ tr.rotationX(period) @ tr.translate(0.7, -0.2, -0.4)
        self.graph["third_wheel"]["transform"] = tr.translate(0.7, 0.2, -0.4) @ tr.rotationX(period) @ tr.translate(-0.7, -0.2, 0.4)
        self.graph["fourth_wheel"]["transform"] = tr.translate(-0.7, 0.2, -0.4) @ tr.rotationX(period) @ tr.translate(0.7, -0.2, 0.4)  


if __name__ == "__main__":
    # Instancia del controller
    controller = Controller("Tarea 1", width=WIDTH, height=HEIGHT, resizable=True)

    with open(Path(os.path.dirname(__file__)) / "shaders/transform.vert") as f:
        color_vertex_source_code = f.read()

    with open(Path(os.path.dirname(__file__)) / "shaders/color.frag") as f:
        color_fragment_source_code = f.read()

    color_pipeline = pyglet.graphics.shader.ShaderProgram(
        pyglet.graphics.shader.Shader(color_vertex_source_code, "vertex"),
        pyglet.graphics.shader.Shader(color_fragment_source_code, "fragment")
    )

    camera = OrbitCamera(5, "perspective")
    camera.phi = np.pi / 4
    camera.theta = np.pi / 4

    axes = Model(shapes.Axes["position"])
    axes.init_gpu_data(color_pipeline)
    axes.gpu_data.color[:] = shapes.Axes["color"]


    with open(Path(os.path.dirname(__file__)) / "shaders/color_mesh.vert") as f:
        vertex_source_code = f.read()

    with open(Path(os.path.dirname(__file__)) / "shaders/color_mesh.frag") as f:
        fragment_source_code = f.read()

    mesh_pipeline = pyglet.graphics.shader.ShaderProgram(
        pyglet.graphics.shader.Shader(vertex_source_code, "vertex"),
        pyglet.graphics.shader.Shader(fragment_source_code, "fragment")
    )

    axis_scene = SceneGraph(camera)
    axis_scene.add_node("axes", attach_to="root", mesh=axes, mode=GL.GL_LINES)

    car = Car(camera)

    print("Controles Cámara:\n\tWASD: Rotar\n\t Q/E: Acercar/Alejar\n\t1/2: Cambiar tipo")
    def update(dt):
        controller.program_state["total_time"] += dt

        if controller.is_key_pressed(pyglet.window.key.A):
            camera.phi -= dt
        if controller.is_key_pressed(pyglet.window.key.D):
            camera.phi += dt
        if controller.is_key_pressed(pyglet.window.key.W):
            camera.theta -= dt
        if controller.is_key_pressed(pyglet.window.key.S):
            camera.theta += dt
        if controller.is_key_pressed(pyglet.window.key.Q):
            camera.distance += dt
        if controller.is_key_pressed(pyglet.window.key.E):
            camera.distance -= dt
        if controller.is_key_pressed(pyglet.window.key._1):
            camera.type = "perspective"
        if controller.is_key_pressed(pyglet.window.key._2):
            camera.type = "orthographic"

        car.wheel_rotate(controller.program_state["total_time"])
        
        camera.update()

    @controller.event
    def on_resize(width, height):
        camera.resize(width, height)

    # draw loop
    @controller.event
    def on_draw():
        controller.clear()
        axis_scene.draw()
        car.draw()

    pyglet.clock.schedule_interval(update, 1/60)
    pyglet.app.run()
