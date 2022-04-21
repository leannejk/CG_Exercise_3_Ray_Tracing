import numpy as np


# This function gets a vector and returns its normalized form.
def normalize(vector):
    return vector / np.linalg.norm(vector)


# This function gets a vector and the normal of the surface it hit
# This function returns the vector that reflects from the surface
def reflected(vector, normal):
    v = np.array(vector - (2 * (vector.dot(normal)) * normal))
    return v


## Lights


class LightSource:

    def __init__(self, intensity):
        self.intensity = intensity


class DirectionalLight(LightSource):

    def __init__(self, intensity, direction):
        super().__init__(intensity)
        self.direction = direction
        # TODO

    # This function returns the ray that goes from the light source to a point
    def get_light_ray(self, intersection_point):
        return Ray(intersection_point, self.direction)

    # This function returns the distance from a point to the light source
    def get_distance_from_light(self, intersection):
        # TODO ?????????
        return np.inf

    # This function returns the light intensity at a point
    def get_intensity(self, intersection):
        # TODO ?????????
        return self.intensity


class PointLight(LightSource):

    def __init__(self, intensity, position, kc, kl, kq):
        super().__init__(intensity)
        self.position = np.array(position)
        self.kc = kc
        self.kl = kl
        self.kq = kq

    # This function returns the ray that goes from the light source to a point
    def get_light_ray(self, intersection):
        return Ray(intersection, normalize(self.position - intersection))

    # This function returns the distance from a point to the light source
    def get_distance_from_light(self, intersection):
        return np.linalg.norm(intersection - self.position)

    # This function returns the light intensity at a point
    def get_intensity(self, intersection):
        d = self.get_distance_from_light(intersection)
        return self.intensity / (self.kc + self.kl * d + self.kq * (d ** 2))


class SpotLight(LightSource):

    def __init__(self, intensity, position, direction, kc, kl, kq):
        super().__init__(intensity)
        self.position = position  # maybe np array??
        self.direction = normalize(direction)
        self.kc = kc
        self.kl = kl
        self.kq = kq
        # TODO

    # This function returns the ray that goes from the light source to a point
    def get_light_ray(self, intersection):
        return Ray(intersection, normalize(self.position - intersection))

    def get_distance_from_light(self, intersection):
        return np.linalg.norm(intersection - self.position)

    def get_intensity(self, intersection):
        # TODO
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
                min_distance = t
                nearest_object = obj

        if nearest_object is None:
            min_distance = None

        return nearest_object, min_distance


class Object3D:

    def set_material(self, ambient, diffuse, specular, shininess, reflection, refraction=0):
        self.ambient = ambient
        self.diffuse = diffuse
        self.specular = specular
        self.shininess = shininess
        self.reflection = reflection
        self.refraction = refraction


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
            return None, None


class Triangle(Object3D):
    # Triangle gets 3 points as arguments
    def __init__(self, a, b, c):
        self.a = np.array(a)
        self.b = np.array(b)
        self.c = np.array(c)
        self.normal = self.compute_normal()

    def compute_normal(self):
        # TODO ???????
        v = normalize(self.a - self.b)
        u = normalize(self.c - self.b)
        n = np.array(normalize(np.cross(u, v)))
        return n

    # Hint: First find the intersection on the plane
    # Later, find if the point is in the triangle using barycentric coordinates
    def intersect(self, ray: Ray):
        eps = 0.00001
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

        return None, None
        # p = Plane(self.normal, self.a)
        # t, _ = p.intersect(ray)
        # if t is not None:
        #     AB = self.b - self.a
        #     AC = self.c - self.a
        #
        #     cross_AB_AC = np.sqrt(np.cross(AB, AC).dot(np.cross(AB, AC)))
        #     area_abc = cross_AB_AC / 2
        #     P = ray.origin + t * ray.direction
        #     PB = self.b - P
        #     PC = self.c - P
        #     PA = self.a - P
        #     cross_PB_PC = np.cross(PB, PC)
        #     cross_PC_PA = np.cross(PC, PA)
        #     alpha = np.sqrt(cross_PB_PC.dot(cross_PB_PC)) / (2 * area_abc)
        #     beta = np.sqrt(cross_PC_PA.dot(cross_PC_PA)) / (2 * area_abc)
        #     gamma = 1 - alpha - beta
        #     eps = 0.00001
        #
        #     if 0 - eps <= alpha <= 1 + eps and 0 - eps <= beta <= 1 + eps and 0 - eps <= gamma <= 1 + eps and 1 - eps <= alpha + beta + gamma <= 1 + eps:
        #         return t, self
        # return None, None


class Sphere(Object3D):
    def __init__(self, center, radius: float):
        self.center = center
        self.radius = radius

    def intersect(self, ray: Ray):
        b = 2 * np.dot(ray.direction, ray.origin - self.center)
        c = np.linalg.norm(ray.origin - self.center) ** 2 - self.radius ** 2  # if problem change
        delta = b ** 2 - 4 * c
        if delta > 0:
            t1 = (-b + np.sqrt(delta)) / 2
            t2 = (-b - np.sqrt(delta)) / 2
            if t1 > 0 and t2 > 0:
                return min(t1, t2), self

        return None, None


class Mesh(Object3D):
    # Mesh are defined by a list of vertices, and a list of faces.
    # The faces are triplets of vertices by their index number.
    def __init__(self, v_list, f_list):
        self.v_list = v_list
        self.f_list = f_list
        self.triangle_list = self.create_triangle_list()

    def create_triangle_list(self):
        l = []
        for (a, b, c) in self.f_list:
            l.append(Triangle(self.v_list[a], self.v_list[b], self.v_list[c]))

        return l

    def apply_materials_to_triangles(self):
        for t in self.triangle_list:
            t.set_material(self.ambient, self.diffuse, self.specular, self.shininess, self.reflection)

    # Hint: Intersect returns both distance and nearest object.
    # Keep track of both.
    def intersect(self, ray: Ray):
        # TODO
        min_d = np.inf
        closest_obj = None
        for triangle in self.triangle_list:
            t, _ = triangle.intersect(ray)
            if t is not None and t < min_d:
                min_d = t
                closest_obj = triangle

        if closest_obj is None:
            min_d = None
        #                                          ---- if problem change ----
        return min_d, closest_obj#ray.nearest_intersected_object(self.triangle_list)
