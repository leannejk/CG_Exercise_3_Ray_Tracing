import numpy as np

# This function gets a vector and returns its normalized form.
def normalize(vector):
    return vector / np.linalg.norm(vector)

# This function gets a vector and the normal of the surface it hit
# This function returns the vector that reflects from the surface
def reflected(vector, normal):
    v = vector - 2 * np.dot(vector, normal) * normal
    return v

def calc_ambient_color(ambient, object):
    return object.ambient * ambient

def calc_diffuse_color(object, light, surface_normal, hit_p):
    return object.diffuse * light.get_intensity(hit_p) * np.dot(surface_normal, light.get_light_ray(hit_p).direction)

def calc_specular_color(ray_direction, object, light, surface_normal, hit_p):
    return object.specular * light.get_intensity(hit_p) * (np.dot(normalize((-1)*ray_direction), normalize(reflected(-1*light.get_light_ray(hit_p).direction, surface_normal))) ** object.shininess)

def get_color(camera, ambient, lights, object, objects, surface_normal, hit_p, min_distance, ray, max_depth, depth):
    color = np.zeros(3)
    for light in lights:
        ray_to_light = light.get_light_ray(hit_p)
        if ray_to_light.nearest_intersected_object(objects) is None:
            min_distnace_to_light = np.inf
        else:
            _, min_distnace_to_light = ray_to_light.nearest_intersected_object(objects)
        if min_distnace_to_light >= light.get_distance_from_light(hit_p): # handling shadows
            color = color + calc_diffuse_color(object, light, surface_normal, hit_p) + calc_specular_color(ray.direction, object, light, surface_normal, hit_p)
    color = color + calc_ambient_color(ambient, object)
    
    depth += 1
    if depth > max_depth:
        return color

    # handling reflections
    r_ray = Ray(hit_p, reflected(ray.direction, surface_normal))
    r_nearest = r_ray.nearest_intersected_object(objects)
    if r_nearest is None:
        return color
    r_nearest_object, r_min_distance = r_nearest
    hit_r = r_ray.origin + r_min_distance * r_ray.direction
    r_surface_normal = np.zeros(3)
    if isinstance(r_nearest_object, Sphere):
        r_surface_normal = normalize(hit_r - r_nearest_object.center)
    elif isinstance(r_nearest_object, Triangle) or isinstance(r_nearest_object, Mesh) or isinstance(r_nearest_object, Plane):
        r_surface_normal = normalize(r_nearest_object.normal)
    close_hit_r = hit_r + 1e-5 * r_surface_normal
    color = color + object.reflection * get_color(camera, ambient, lights, r_nearest_object, objects, r_surface_normal, close_hit_r, r_min_distance, r_ray, max_depth, depth)
    return color


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
        return Ray(intersection,normalize(self.position - intersection))

    # This function returns the distance from a point to the light source
    def get_distance_from_light(self,intersection):
        return np.linalg.norm(intersection - self.position)

    # This function returns the light intensity at a point
    def get_intensity(self, intersection):
        d = self.get_distance_from_light(intersection)
        return self.intensity / (self.kc + self.kl*d + self.kq * (d**2))


class SpotLight(LightSource):


    def __init__(self, intensity, direction, position, kc, kl, kq):
        super().__init__(intensity)
        self.direction = normalize(direction)
        self.position = np.array(position)
        self.kc = kc
        self.kl = kl
        self.kq = kq

    # This function returns the ray that goes from the light source to a point
    def get_light_ray(self,intersection):
        return Ray(intersection,normalize(self.position - intersection))

    # This function returns the distance from a point to the light source
    def get_distance_from_light(self,intersection):
        return np.linalg.norm(intersection - self.position)

    # This function returns the light intensity at a point
    def get_intensity(self, intersection):
        d = self.get_distance_from_light(intersection)
        v = normalize(self.get_light_ray(intersection).direction)
        return (self.intensity * np.dot(v, self.direction)) / (self.kc + self.kl*d + self.kq * (d**2))


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
            ray_intersect = obj.intersect(self)
            if ray_intersect is None:
                continue
            distance, intersect_obj = ray_intersect
            if distance and distance < min_distance:
                nearest_object = intersect_obj
                min_distance = distance
        if nearest_object is None:
            return None
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
            return None


class Triangle(Object3D):
    # Triangle gets 3 points as arguments
    def __init__(self, a, b, c):
        self.a = np.array(a)
        self.b = np.array(b)
        self.c = np.array(c)
        self.normal = self.compute_normal()

    def compute_normal(self):
        v_1 = normalize(self.b - self.a)
        v_2 = normalize(self.c - self.a)
        orthogonal = np.cross(v_1, v_2)
        n = normalize(orthogonal)
        return n

    def compute_area(self):
        ab = self.b - self.a
        ac = self.c - self.a
        return np.linalg.norm(np.cross(ab, ac)) / 2

    # Hint: First find the intersection on the plane
    # Later, find if the point is in the triangle using barycentric coordinates
    def intersect(self, ray: Ray):
        plane = Plane(self.normal, self.a)
        plane_intersect = plane.intersect(ray)
        if plane_intersect is None:
            return None
        t, _ = plane_intersect
        p = ray.origin + t * ray.direction
        area = self.compute_area()
        pa = self.a - p
        pb = self.b - p
        pc = self.c - p
        alpha = np.linalg.norm(np.cross(pb,pc)) / (2 * area)
        beta = np.linalg.norm(np.cross(pc,pa)) / (2 * area)
        gamma = np.linalg.norm(np.cross(pa,pb)) / (2 * area)
        if (not (alpha >= 0 and alpha <= 1 and beta >= 0 and beta <= 1 and gamma >= 0 and gamma <= 1)) or (np.abs(alpha + beta + gamma - 1) >= 1e-5):
            return None
        return t, self


class Sphere(Object3D):
    def __init__(self, center, radius: float):
        self.center = center
        self.radius = radius

    def intersect(self, ray: Ray):
        a = np.linalg.norm(ray.direction) ** 2
        b = 2 * np.dot(ray.direction, ray.origin - self.center)
        c = np.linalg.norm(ray.origin - self.center) ** 2 - self.radius ** 2
        delta = b ** 2 - 4 * a * c
        if delta <= 0:
            return None
        t1 = (-b + np.sqrt(delta)) / (2 * a)
        t2 = (-b - np.sqrt(delta)) / (2 * a)
        if t1 > 0 and t2 > 0:
            return min(t1, t2), self


class Mesh(Object3D):
    # Mesh are defined by a list of vertices, and a list of faces.
    # The faces are triplets of vertices by their index number.
    def __init__(self, v_list, f_list):
        self.v_list = v_list
        self.f_list = f_list
        self.triangle_list = self.create_triangle_list()

    def create_triangle_list(self):
        l = []
        for f in self.f_list:
            triangle = Triangle(self.v_list[f[0]], self.v_list[f[1]], self.v_list[f[2]])
            l.append(triangle)
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
            triangle_intersect = triangle.intersect(ray)
            if triangle_intersect is None:
                continue
            t, candidate_triangle = triangle_intersect
            if t and t < min_t:
                min_triangle = candidate_triangle
                min_t = t
        if min_triangle is None:
            return None
        return min_t, min_triangle
