import numpy as np


# This function gets a vector and returns its normalized form.
def normalize(vector):
    return vector / np.linalg.norm(vector)


# TODO:
# This function gets a vector and the normal of the surface it hit
# This function returns the vector that reflects from the surface
def reflected(vector, normal):
    v = vector - 2 * vector.dot(normal) * normal
    return v

## Lights


class LightSource:

    def __init__(self, intensity):
        self.intensity = intensity


class DirectionalLight(LightSource):

    def __init__(self, intensity, direction):
        super().__init__(intensity)
        self.direction = direction

    # This function returns the ray that goes from the light source to a point
    def get_light_ray(self,intersection_point):
        return Ray(intersection_point, normalize(self.direction))

    # This function returns the distance from a point to the light source
    def get_distance_from_light(self, intersection):
        return np.inf

    # This function returns the light intensity at a point
    def get_intensity(self, intersection):
        return self.intensity


class PointLight(LightSource):

    def __init__(self, intensity, position, kc, kl, kq):
        super().__init__(intensity)
        self.position = np.array(position)
        self.kc = kc
        self.kl = kl
        self.kq = kq

    # This function returns the ray that goes from the light source to a point
    def get_light_ray(self,intersection):
        return Ray(intersection, normalize(self.position - intersection))

    # This function returns the distance from a point to the light source
    def get_distance_from_light(self,intersection):
        return np.linalg.norm(intersection - self.position)

    # This function returns the light intensity at a point
    def get_intensity(self, intersection):
        d = self.get_distance_from_light(intersection)
        return self.intensity / (self.kc + self.kl*d + self.kq * (d**2))


class SpotLight(LightSource):


    def __init__(self, intensity, position, direction, kc, kl, kq):
        super().__init__(intensity)
        self.position = position
        self.direction = direction
        self.kc = kc
        self.kl = kl
        self.kq = kq

    # This function returns the ray that goes from the light source to a point
    def get_light_ray(self,intersection):
        return Ray(intersection, normalize(self.position - intersection))

    def get_distance_from_light(self,intersection):
        return np.linalg.norm(intersection - self.position)

    def get_intensity(self, intersection):
        d = self.get_distance_from_light(intersection)
        V = normalize(self.get_light_ray(intersection).direction)
        return (self.intensity * V.dot(normalize(self.direction))) / (self.kc + self.kl * d + self.kq * (d ** 2))



class Ray:
    def __init__(self, origin, direction):
        self.origin = origin
        self.direction = direction

    # The function is getting the collection of objects in the scene and looks for the one with minimum distance.
    # The function should return the nearest object and its distance (in two different arguments)
    def nearest_intersected_object(self, objects):
        nearest_object = None
        min_distance = np.inf
        for obj in objects:
            t, _ = obj.intersect(self)
            if t is not None and t < min_distance:
                nearest_object = obj
                min_distance = t
        
        if nearest_object is None:
            min_distance = None

        return nearest_object, min_distance


class Object3D:

    def set_material(self, ambient, diffuse, specular, shininess, reflection):
        self.ambient = ambient
        self.diffuse = diffuse
        self.specular = specular
        self.shininess = shininess
        self.reflection = reflection


class Plane(Object3D):
    def __init__(self, normal, point):
        self.normal = np.array(normal)
        self.point = np.array(point)

    def intersect(self, ray: Ray):
        v = self.point - ray.origin
        t = (np.dot(v, self.normal) / np.dot(self.normal, ray.direction))
        if t > 0:
            return t, self
        else:
            return (None, None)



class Triangle(Object3D):
    # Triangle gets 3 points as arguments
    def __init__(self, a, b, c):
        self.a = np.array(a)
        self.b = np.array(b)
        self.c = np.array(c)
        self.normal = self.compute_normal()

    def compute_normal(self):
        v = normalize(self.a - self.b)
        u = normalize(self.c - self.b)
        n = np.array(normalize(np.cross(u, v)))
        return n

    # Hint: First find the intersection on the plane
    # Later, find if the point is in the triangle using barycentric coordinates
    def intersect(self, ray: Ray):
        v = self.a - ray.origin
        t = (np.dot(v, self.normal) / np.dot(self.normal, ray.direction))
        if t > 0:
            p = ray.origin + t * ray.direction
            ab = self.b - self.a
            ap = p - self.a
            bc = self.c - self.b
            bp = p - self.b
            ca = self.a - self.c
            cp = p - self.c
            if np.dot(np.cross(ab, ap), self.normal) >= 0 and np.dot(np.cross(bc, bp), self.normal) >= 0 and np.dot(
                    np.cross(ca, cp), self.normal) >= 0:
                return t, self
        
        return (None, None)


class Sphere(Object3D):
    def __init__(self, center, radius: float):
        self.center = center
        self.radius = radius

    def intersect(self, ray: Ray):
        b = 2 * np.dot(ray.direction, ray.origin - self.center)
        c = np.linalg.norm(ray.origin - self.center) ** 2 - self.radius ** 2
        delta = b ** 2 - 4 * c
        if delta > 0:
            t1 = (-b + np.sqrt(delta)) / 2
            t2 = (-b - np.sqrt(delta)) / 2
            if t1 > 0 and t2 > 0:
                return min(t1, t2), self

        return (None, None)


class Mesh(Object3D):
    # Mesh are defined by a list of vertices, and a list of faces.
    # The faces are triplets of vertices by their index number.
    def __init__(self, v_list, f_list):
        self.v_list = v_list
        self.f_list = f_list
        self.triangle_list = self.create_triangle_list()

    def create_triangle_list(self):
        l = []
        for a, b, c in self.f_list:
            l.append(Triangle(self.v_list[a], self.v_list[b], self.v_list[c]))

        return l

    def apply_materials_to_triangles(self):
        for t in self.triangle_list:
            t.set_material(self.ambient,self.diffuse,self.specular,self.shininess,self.reflection)

    # Hint: Intersect returns both distance and nearest object.
    # Keep track of both.
    def intersect(self, ray: Ray):
        min_t = np.inf
        min_triangle = None
        for triangle in self.triangle_list:
            t, _ = triangle.intersect(ray)
            if t is not None and t < min_t:
                min_triangle = triangle
                min_t = t
        
        if min_triangle is None:
            min_t = None

        return min_t, min_triangle



# color functions
# TODO: CHANGE !!!!!
def get_normal(obj, ray, hit):
    normal = None
    if isinstance(obj, Sphere):
        normal = normalize(hit - obj.center)
    elif isinstance(obj, Mesh):
        _, obj = obj.intersect(ray)
        normal = normalize(obj.normal)
    else:
        normal = normalize(obj.normal)

    return normal

# TODO: CHANGE !!!!!
def get_ambient_color(ambient, obj):
    return np.array(obj.ambient * ambient, dtype="float64")

def get_diffuse_color(obj, hit, ray, light):
    normal = get_normal(obj, ray, hit)
    return obj.diffuse * light.get_intensity(hit) * normal.dot(light.get_light_ray(hit).direction)

def get_specular_color(obj, hit, ray, light, camera):
    reflected_ray = reflected(-1 * normalize(light.get_light_ray(hit).direction), get_normal(obj, ray, hit))
    return obj.specular * light.get_intensity(hit)  * (normalize(hit - camera).dot(reflected_ray)) ** obj.shininess

# TODO: CHANGE !!!!!
def is_shadow(inter_point, light, objects, is_print=False):
    ray_to_light = light.get_light_ray(inter_point)
    _, min_d = ray_to_light.nearest_intersected_object(objects)
    # if is_print: print(min_d)
    if min_d is None or min_d >= light.get_distance_from_light(inter_point):
        return 1  # light.intensity

    return 0

# TODO: CHANGE !!!!!
def get_reflective_ray(ray, obj, hit_point):
    N = get_normal(obj, ray, hit_point)
    return Ray(hit_point, reflected(ray.direction, N))

def get_color(camera, lights, ambient,objects, max_level, level, ray, hit_point, obj):
    # ambient and emission
    color = get_ambient_color(ambient, obj)

    for light in lights:
        if is_shadow(hit_point, light, objects) != 0:
            color += get_diffuse_color(obj, hit_point, ray, light)
            color += get_specular_color(obj, hit_point, ray, light, camera)
    
    level += 1
    if level > max_level:
        return color
    
    # reflection
    ref_ray = get_reflective_ray(ray, obj, hit_point)
    ref_nearest_object, ref_min_distance = ref_ray.nearest_intersected_object(objects)
    if ref_nearest_object is None:
        return color
    ref_hit_point = ref_ray.origin + ref_min_distance * ref_ray.direction
    ref_normal = get_normal(ref_nearest_object, ref_ray, ref_hit_point)
    ref_hit_point += 1e-5 * ref_normal
    color += obj.reflection * get_color(camera, lights, ambient, objects, max_level, level, ref_ray, ref_hit_point, ref_nearest_object)


    return color
