from helper_classes import *
import matplotlib.pyplot as plt
import numpy as np


def render_scene(camera, ambient, lights, objects, screen_size, max_depth, refractive_index=1):
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

            color = np.zeros((3))
            reflection = 1
            ray = Ray(camera, normalize(pixel - camera))
            obj, min_d = ray.nearest_intersected_object(objects)
            if obj is not None:
                inter_point = ray.origin + min_d * ray.direction
                inter_point += 0.0001 * get_normal(obj, ray, inter_point)

                if i == 120 and j == 210:
                    color = get_color(camera, lights, ambient, ray, inter_point, obj, objects, 1, max_depth,
                                      refractive_index)
                    print("color " + str(color))
                else:
                    color = get_color(camera, lights, ambient, ray, inter_point, obj, objects, 1, max_depth,
                                      refractive_index)
                # print(color)

            # We clip the values between 0 and 1 so all pixel values will make sense.
            image[i, j] = np.clip(color, 0, 1)

    return image


# Write your own objects and lights
# TODO
def your_own_scene():
    sphere_a = Sphere([0, 0, -1], 0.5)
    sphere_a.set_material([0.4, 0.4, 0.4], [0, 0, 0], [0.4, 0.4, 0.4], 50, 0.1, 1.333)

    background = Plane([0, 0, 1], [0, 0, -8])
    background.set_material([1, 1, 1], [1, 1, 1], [1, 1, 1], 1000, 1)
    triangle = Triangle([1, -1, -3], [0, 1, -3], [0, -1, -3])
    triangle.set_material([1, 0, 0], [1, 0, 0], [0, 0, 0], 1, 0.1)

    light = PointLight(intensity=np.array([1, 1, 1]), position=np.array([1, 1, 1]), kc=0.1, kl=0.1, kq=0.1)
    ambient = np.array([0.2, 0.2, 0.2])
    camera = np.array([0, 0, 1])
    lights = [light]
    objects = [sphere_a, background, triangle]
    return camera, ambient, lights, objects


def obj_to_mesh(file_path):
    file = open(file_path, "r")
    f = []
    v = []
    i = 0
    for line in file.readlines():
        if line[0] == "v":
            i += 1
            v.append([float(x) for x in line[1:].split()])

        if line[0] == "f":
            f.append([int(x) - 1 for x in line[1:].split()])

    return Mesh(v, f)


def calc_ambient_color(obj, ambient):
    return np.array(obj.ambient * ambient, dtype="float64")


def get_normal(obj, ray, inter_point):
    normal = None
    if isinstance(obj, Sphere):
        normal = normalize(inter_point - obj.center)
    elif isinstance(obj, Mesh):
        _, obj = obj.intersect(ray)
        normal = normalize(obj.normal)
    else:
        normal = normalize(obj.normal)

    return normal


def calc_diffuse_color(light, inter_point, ray, obj, is_print=False):
    normal = get_normal(obj, ray, inter_point)  # maybe light ray
    # if 0 > any(obj.diffuse * light.get_intensity(inter_point) * np.dot(normal, light.get_light_ray(inter_point).direction)) > 1:
    #     print("calc_diffuse_color " + str(obj.diffuse * light.get_intensity(inter_point) * np.dot(normal, light.get_light_ray(inter_point).direction)))
    if is_print:
        print("calc_diffuse_color " + str(obj.diffuse * light.get_intensity(inter_point) * np.dot(normal,
                                                                                                  light.get_light_ray(
                                                                                                      inter_point).direction)))
        print(light.get_intensity(inter_point))
        print(light.get_light_ray(inter_point).direction)
        print(normal)

    return obj.diffuse * light.get_intensity(inter_point) * np.dot(normal, light.get_light_ray(inter_point).direction)


def calc_specular_color(light, inter_point, ray, obj, camera, is_print=False):
    Ks = obj.specular
    V = normalize(inter_point - camera)
    N = get_normal(obj, ray, inter_point)
    L = normalize(light.get_light_ray(inter_point).direction)
    # if isinstance(light, DirectionalLight):
    #     L *= -1
    Li = light.get_intensity(inter_point)
    shininess = obj.shininess
    L_gag = reflected(-1 * L, N)  # normalize(L - 2 * np.dot(L, N) * N)
    if is_print: print("calc_specular_color " + str(Ks * Li * (np.dot(V, L_gag) ** shininess)))
    # if 0 > any(Ks * Li * (np.dot(V, L_gag) ** shininess)) > 1:
    #     print("calc_specular_color " + str(Ks * Li * (np.dot(V, L_gag) ** shininess)))

    return Ks * Li * (np.dot(V, L_gag) ** shininess)


def cont_reflect_ray(ray, obj, inter_point):
    N = get_normal(obj, ray, inter_point)
    return Ray(inter_point, reflected(ray.direction, N))  # normalize(ray.direction - 2 * np.dot(ray.direction, N) * N))


def cont_refract_ray(ray, obj, inter_point, prev_n):
    norm_vec = get_normal(obj, ray, inter_point)
    dot = np.dot(ray.direction, norm_vec)
    n_r = obj.refraction

    if prev_n == n_r:
        # print("out , prev: " + str(prev_n) + " n: " + str(n_r))
        n_r, prev_n = prev_n, n_r
    # else: print("in , prev: " + str(prev_n) + " n: " + str(n_r))

    cos_I = np.dot(normalize(norm_vec), normalize(ray.direction))
    sin_I = np.sqrt(1 - np.power(cos_I, 2))
    sin_Ip = prev_n * sin_I / n_r
    cos_Ip = np.sqrt(1 - np.power(sin_Ip, 2))

    # Calculate the refractive vector
    r_vec = ray.direction
    T = np.array(((n_r / prev_n) * cos_I - cos_Ip) * norm_vec + (n_r / prev_n) * r_vec)
    return Ray(inter_point, T), n_r

#uv=ray.d n=normal new_ior=obj.refraction
def refract(inter_point, uv, n, new_ior, previous_ior=1):
    n = -n.copy()
    ni_over_nt = previous_ior / new_ior
    dt = np.dot(uv, n)
    discriminant = 1 - ni_over_nt * ni_over_nt * (1 - dt ** 2)
    refractDir = ni_over_nt * (uv - n * dt) - n * np.sqrt(abs(discriminant))

    return Ray(inter_point, refractDir)


def is_shadow(inter_point, light, objects, is_print=False):
    ray_to_light = light.get_light_ray(inter_point)
    _, min_d = ray_to_light.nearest_intersected_object(objects)
    # if is_print: print(min_d)
    if min_d is None or min_d >= light.get_distance_from_light(inter_point):
        return 1  # light.intensity

    return 0


def get_color(camera, lights, ambient, ray, inter_point, obj, objects, level, max_level, refractive_index):
    color = calc_ambient_color(obj, ambient)
    # print(color)

    for light in lights:
        color += (calc_diffuse_color(light, inter_point, ray, obj) + calc_specular_color(light, inter_point,
                                                                                         ray, obj, camera)) \
                 * is_shadow(inter_point, light, objects)

        # color += calc_specular_color(light, inter_point, ray, obj, camera)
        # check if shadow

    level += 1

    if level > max_level:
        return color

    r_ray = cont_reflect_ray(ray, obj, inter_point)
    r_obj, r_min_d = r_ray.nearest_intersected_object(objects)
    if r_min_d is not None:
        r_inter_point = r_ray.origin + r_min_d * r_ray.direction
        r_inter_point += 0.0001 * get_normal(r_obj, r_ray, r_inter_point)
        color += obj.reflection * get_color(camera, lights, ambient, r_ray, r_inter_point, r_obj, objects, level,
                                            max_level, refractive_index)

    n = get_normal(obj, ray, inter_point)
    if obj.refraction != 0:
        if obj.refraction == refractive_index:
            # refractDir_i = refract(ray_direction, plate.planeNormal, 1, previous_ior=plate.ior)
            t_ray = refract(inter_point, ray.direction, n, 1, previous_ior=obj.refraction)
        else:
            # refractDir_i = refract(ray_direction, plate.planeNormal, plate.ior, previous_ior=ior)
            t_ray = refract(inter_point, ray.direction, n, obj.refraction, previous_ior=refractive_index)
        # t_ray = refract(inter_point, ray, n, obj.refraction)#cont_refract_ray(ray, obj, inter_point, refractive_index)
        t_obj, t_min_d = t_ray.nearest_intersected_object(objects)

        if t_min_d is not None:
            t_inter_point = t_ray.origin + t_min_d * t_ray.direction
            t_inter_point += 0.0001 * get_normal(t_obj, t_ray, t_inter_point)
            color += 1 * get_color(camera, lights, ambient, t_ray, t_inter_point, t_obj, objects, level,
                                   max_level, refractive_index)

    return color
