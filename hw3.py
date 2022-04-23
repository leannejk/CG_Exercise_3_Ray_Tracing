from helper_classes import *
import matplotlib.pyplot as plt

def render_scene(camera, ambient, lights, objects, screen_size, max_depth=3):
    width, height = screen_size
    ratio = float(width) / height
    screen = (-1, 1 / ratio, 1, -1 / ratio)  # left, top, right, bottom

    image = np.zeros((height, width, 3))

    for i, y in enumerate(np.linspace(screen[1], screen[3], height)):
        for j, x in enumerate(np.linspace(screen[0], screen[2], width)):
            pixel = np.array([x, y, 0])
            color = np.zeros(3)

            # This is the main loop where each pixel color is computed.
            # TODO
            
            direction = normalize(pixel - camera)
            ray = Ray(camera, direction)
            nearest_object, min_distance = ray.nearest_intersected_object(objects)
            if nearest_object is not None:
                hit_point = camera + min_distance * direction
                hit_point += 1e-5 * get_hit_normal(nearest_object, ray, hit_point) #TODO: CHANGE !!!!!
                color = get_color(camera, lights, ambient,objects, max_depth, 1, ray, hit_point, nearest_object)
                
                # We clip the values between 0 and 1 so all pixel values will make sense.
                image[i, j] = np.clip(color,0,1)

    return image


# Write your own objects and lights
# TODO
def your_own_scene():
    camera = np.array([0,0,1])

    light_b = SpotLight(intensity=np.array([1, 0, 1]), position=np.array([-0.5, 0.5, 0]), direction=([0, 0, 1]),
                        kc=0.1, kl=0.1, kq=0.1)
    light_a = DirectionalLight(intensity=np.array([1, 1, 1]), direction=np.array([0.5, 0.5, 0.5]))
    lights = [light_a, light_b]

    plane = Plane([0, 1, 0], [0, -0.3, 0])
    plane.set_material([0.2, 0.2, 0.2], [0.2, 0.2, 0.2], [1, 1, 1], 1000, 0.5)
    triangle = Triangle([1, 1, 1], [0, 2, -1], [0, -1, -1])
    triangle.set_material([0.8, 0, 0.4], [0.8, 0, 0.4], [1, 1, 1], 100, 0.5)
    sphere = Sphere([-0.4, 0.2, -1], 0.5)
    sphere.set_material([0, 0.5, 0.6], [0, 0.5, 0.6], [0.1, 0.2, 0.3], 1000, 0.5)
    objects = [plane, triangle, sphere]
    return camera, lights, objects

