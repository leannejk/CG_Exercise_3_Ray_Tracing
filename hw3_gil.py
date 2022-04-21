from helper_classes import *
import matplotlib.pyplot as plt

def render_scene(camera, ambient, lights, objects, screen_size, max_depth):
    width, height = screen_size
    ratio = float(width) / height
    screen = (-1, 1 / ratio, 1, -1 / ratio)  # left, top, right, bottom

    image = np.zeros((height, width, 3))

    for i, y in enumerate(np.linspace(screen[1], screen[3], height)):
        for j, x in enumerate(np.linspace(screen[0], screen[2], width)):
            pixel = np.array([x, y, 0])
            color = np.zeros(3)

            # This is the main loop where each pixel color is computed.
            direction = normalize(pixel - camera)
            ray = Ray(camera, direction)
            nearest = ray.nearest_intersected_object(objects)
            if nearest is None:
                continue
            nearest_object, min_distance = nearest
            hit_p = camera + min_distance * direction
            surface_normal = np.zeros(3)
            if isinstance(nearest_object, Sphere):
                surface_normal = normalize(hit_p - nearest_object.center)
            elif isinstance(nearest_object, Triangle) or isinstance(nearest_object, Mesh) or isinstance(nearest_object, Plane):
                surface_normal = normalize(nearest_object.normal)
            close_hit_p = hit_p + 1e-5 * surface_normal
            color = get_color(camera, ambient, lights, nearest_object, objects, surface_normal, close_hit_p, min_distance, ray, max_depth, 1)           

            # We clip the values between 0 and 1 so all pixel values will make sense.
            image[i, j] = np.clip(color,0,1)

    return image


# Write your own objects and lights
def your_own_scene(): # the space blueberry
    camera = np.array([0,0,1])
    lights = []
    objects = []

    sphere_a = Sphere([0.2, -0.5, -1],0.5)
    sphere_a.set_material([0, 0, 0], [0, 0, 0], [0.3, 0.3, 0.3], 100, 0.2)

    v_list = np.array([[-1,1.2,2],[0.2,-1,1.7],[0.1,-1.2,-1.3],[0.3,1.2,-1.1]])
    f_list = np.array([[1,2,0],[0,2,3],[1,2,3],[1,2,2]])

    mesh = Mesh(v_list, f_list)
    mesh.set_material([1, 0, 0.3], [1, 0.4, 0], [0, 0.5, 0], 50, 0.7)
    mesh.apply_materials_to_triangles()

    background = Plane([0,1.2,1], [0,-1.2,-1])
    background.set_material([0.7, 0.7, 0.7], [0.8, 0.8, 0.8], [0.9, 0.9, 0.9], 850, 0.5)

    objects = [sphere_a, mesh, background]

    light_a = SpotLight(intensity= np.array([0, 0, 1]),position=np.array([0.2,0.2,0]), direction=([0,0,1]),
                        kc=0.1,kl=0.1,kq=0.1)
    
    light_b = PointLight(intensity= np.array([0.5, 0.5, 0.5]),position=np.array([1,1,1]),kc=0.4,kl=0.2,kq=0.3)

    lights = [light_a,light_b]

    ambient = np.array([0,0,0])

    return camera, lights, objects

