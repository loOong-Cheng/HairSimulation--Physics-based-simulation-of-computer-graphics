import taichi as ti
import taichi_glsl as ts
import numblend as nb
import numpy as np
import mathutils
import bpy

nb.init()
ti.init()

#----------------------------------------
# Configuration
#----------------------------------------

# user settings
frames = 250
steps = 10
dt = 4e-3
damping = 0.99
grav_acc = ti.Vector([0., 0., -9.8]) # gravity
# shape constraints
stiffness_global = 0.001
stiffness_local = 0.05
damping_ftl = 0.9
# collision
decimate_ratio = 0.3 # resolution of collision mesh
sdf_cell_size = 0.005
sdf_pad = 1.2
coll_margin = 0.0
# hair-hair interaction
vf_cell_size = 0.01
vf_pad = 2.
friction = 0.08
#repulsion = 0.1
# animation
def custom_translation(frame):
    return 0.1 * mathutils.Vector((0., ti.sin(frame*0.1), 0.))

# delete exsiting data blocks
for curve in bpy.data.curves:
    bpy.data.curves.remove(curve)
nb.delete_object('Mesh')
nb.delete_mesh('Mesh')
nb.delete_object('Collision')
nb.delete_mesh('Head.001')

# converte particle system to curves
head = bpy.data.objects['Head']
head.location = mathutils.Vector((0., 0., 0.))
ps = head.modifiers['ParticleSystem']
ps.show_viewport = True
ps_setting = bpy.data.particles['ParticleSettings']
bpy.context.view_layer.objects.active = head
bpy.ops.object.modifier_convert(modifier='ParticleSystem')
bpy.ops.object.modifier_add(type='SUBSURF') # converted curve has only 4 segments by default
bpy.context.object.modifiers['Subdivision'].levels = int(np.log2(ps_setting.hair_step / 4.))
bpy.ops.object.convert(target='CURVE')
hair = bpy.context.object
hair.data.bevel_depth = 0.005
hair.data.use_fill_caps = True
hair.active_material = bpy.data.materials['Hair']
curves = hair.data.splines

# hide particle system
ps.show_viewport = False
ps.show_render = False

# parameters
curves = hair.data.splines
M = len(curves)
N = len(curves[0].points)
length = ps_setting.hair_length / head.matrix_world.median_scale # in local coordinate system
mass = 1.
dx = length / N
w = 1. / mass
# initial locations
head_loc = head.location.copy()

# data
x = ti.Vector.field(3, dtype=ti.f32, shape=(M, N))
v = ti.Vector.field(3, dtype=ti.f32, shape=(M, N))
p = ti.Vector.field(3, dtype=ti.f32, shape=(M, N))
d = ti.Vector.field(3, dtype=ti.f32, shape=(M, N))
x_rest = ti.Vector.field(3, dtype=ti.f32, shape=(M, N))
v_world_old = ti.Vector.field(3, dtype=ti.f32, shape=())
v_world = ti.Vector.field(3, dtype=ti.f32, shape=())
f_ext = ti.Vector.field(3, dtype=ti.f32, shape=())

# generate collision mesh
coll_obj = bpy.data.objects.new('Collision', object_data=bpy.data.meshes['Head'].copy())
coll_obj.parent = head
bpy.context.view_layer.active_layer_collection.collection.objects.link(coll_obj)
bpy.context.view_layer.objects.active = coll_obj
bpy.ops.object.modifier_add(type='DECIMATE')
coll_obj.modifiers['Decimate'].ratio = decimate_ratio
bpy.ops.object.modifier_apply(modifier='Decimate')
coll_obj.hide_viewport = True
coll_obj.hide_render = True

def calculate_obb(obj):
    bbox = obj.bound_box
    obb_min = mathutils.Vector((1e10, 1e10, 1e10))
    obb_max = mathutils.Vector((-1e10, -1e10, -1e10))
    
    for i in range(8):
        pos = bbox[i] 
        for j in range(3):
            obb_min[j] = min(obb_min[j], pos[j])
            obb_max[j] = max(obb_max[j], pos[j])
            
    return obb_min, obb_max

# parameters for collision
sdf_h = 0.1 * sdf_cell_size
sdf_default_dist = 1e10
coll_obb_min, coll_obb_max = calculate_obb(coll_obj)
sdf_min = sdf_pad * coll_obb_min
sdf_max = sdf_pad * coll_obb_max
sdf_dim_x = int((sdf_max[0] - sdf_min[0]) / sdf_cell_size)
sdf_dim_y = int((sdf_max[1] - sdf_min[1]) / sdf_cell_size)
sdf_dim_z = int((sdf_max[2] - sdf_min[2]) / sdf_cell_size)
coll_obj.data.calc_loop_triangles()
triangles = coll_obj.data.loop_triangles
num_triangles = len(triangles)
num_vertices = len(coll_obj.data.vertices)
print('sdf_dim:', sdf_dim_x, sdf_dim_y, sdf_dim_z)

# data for collision
tri_idx = ti.Vector.field(3, dtype=ti.i32, shape=num_triangles)
tri_co = ti.Vector.field(3, dtype=ti.f32, shape=num_vertices)
sdf = ti.field(ti.f32, shape=(sdf_dim_x, sdf_dim_y, sdf_dim_z))
sdf_origin = ti.Vector.field(3, dtype=ti.f32, shape=())

# parameters for hair-hair interaction
vf_min = vf_pad * coll_obb_min
vf_max = vf_pad * coll_obb_max
vf_dim_x = int((vf_max[0] - vf_min[0]) / vf_cell_size)
vf_dim_y = int((vf_max[1] - vf_min[1]) / vf_cell_size)
vf_dim_z = int((vf_max[2] - vf_min[2]) / vf_cell_size)
print('vf_dim:', vf_dim_x, vf_dim_y, vf_dim_z)
vf_h = 0.1 * vf_cell_size

# data for hair-hair interaction
density = ti.field(ti.f32, shape=(vf_dim_x, vf_dim_y, vf_dim_z))
vf = ti.Vector.field(3, dtype=ti.f32, shape=(vf_dim_x, vf_dim_y, vf_dim_z))
vf_origin = ti.Vector.field(3, dtype=ti.f32, shape=())

#----------------------------------------
# Collision
#----------------------------------------

@ti.func
def get_sdf_coordinates(pos):
    return int((pos - sdf_origin[None]) // sdf_cell_size)

@ti.func
def get_sdf_cell_position(coord):
    return coord * sdf_cell_size + sdf_origin[None]

@ti.func
def distance_point_to_edge(p, x0, x1):
    x10 = x1 - x0
    t = ts.clamp((x1 - p).dot(x10) / x10.norm_sqr())
    point_on_edge = t * x0 + (1. - t) * x1
    a = p - point_on_edge
    d = a.norm()
    n = a / (d + 1e-30)
    return d, n

@ti.func
def signed_distance_point_to_triangle(p, x0, x1, x2):
    d = 0.
    x02 = x0 - x2
    l0 = x02.norm() + 1e-30
    x02 = x02 / l0
    x12 = x1 - x2
    l1 = x12.dot(x02)
    x12 = x12 - l1 * x02
    l2 = x12.norm() + 1e-30
    x12 = x12 / l2
    px2 = p - x2
    
    b = x12.dot(px2) / l2
    a = (x02.dot(px2) - l1 * b) / l0
    c = 1 - a - b
    
    nTri = (x1 - x0).cross(x2 - x0)
    n = ti.Vector([0., 0., 0.])
    tol = 1e-8
    
    if a >= -tol and b >= -tol and c >= -tol:
        n = p - (a * x0 + b * x1 + c * x2)
        d = n.norm()
        n1 = n / d
        n2 = nTri / (nTri.norm() + 1e-30)
        n = n1 if d > 0 else n2
    else:
        d, n = distance_point_to_edge(p, x0, x1)
        d12, n12 = distance_point_to_edge(p, x1, x2)
        d02, n02 = distance_point_to_edge(p, x0, x2)
        
        d = min(d, d12, d02)
        n = n12 if d == d12 else n
        n = n02 if d == d02 else n
        
    if (p - x0).dot(nTri) < 0.:
        d = -d
        
    return d 

@ti.kernel
def construct_signed_distance_field():
    for dummy in range(1): # no atomic_min
        for i in range(num_triangles):
            tri0 = tri_co[tri_idx[i][0]]
            tri1 = tri_co[tri_idx[i][1]]
            tri2 = tri_co[tri_idx[i][2]]
            obb_min = ti.Vector([
                        min(tri0.x, tri1.x, tri2.x),
                        min(tri0.y, tri1.y, tri2.y),
                        min(tri0.z, tri1.z, tri2.z)]) - sdf_cell_size
            obb_max = ti.Vector([
                        max(tri0.x, tri1.x, tri2.x),
                        max(tri0.y, tri1.y, tri2.y),
                        max(tri0.z, tri1.z, tri2.z)]) + sdf_cell_size
                        
            coord_min = get_sdf_coordinates(obb_min) - 1
            coord_max = get_sdf_coordinates(obb_max) + 1
            
            coord_min.x = max(0, min(coord_min.x, sdf_dim_x - 1))
            coord_min.y = max(0, min(coord_min.y, sdf_dim_y - 1))
            coord_min.z = max(0, min(coord_min.z, sdf_dim_z - 1))
            
            coord_max.x = max(0, min(coord_max.x, sdf_dim_x - 1))
            coord_max.y = max(0, min(coord_max.y, sdf_dim_y - 1))
            coord_max.z = max(0, min(coord_max.z, sdf_dim_z - 1))
            
            for a, b, c in ti.ndrange((coord_min.x, coord_max.x + 1),
                                      (coord_min.y, coord_max.y + 1),
                                      (coord_min.z, coord_max.z + 1)):
                cell_coord = ti.Vector([a, b, c])
                cell_pos = get_sdf_cell_position(cell_coord)
                dist = signed_distance_point_to_triangle(cell_pos, tri0, tri1, tri2)
                sdf[a, b, c] = min(sdf[a, b, c], dist)

@ti.func
def linear_interpolate(a, b, t):
    return a * (1. - t) + b * t

@ti.func
def bilinear_interpolate(a, b, c, d, p, q):
    return linear_interpolate(
                linear_interpolate(a, b, p),
                linear_interpolate(c, d, p),
                q)

@ti.func
def trilinear_interpolate(a, b, c, d, e, f, g, h, p, q, r):
    return linear_interpolate(
                bilinear_interpolate(a, b, c, d, p, q),
                bilinear_interpolate(e, f, g, h, p, q),
                r)
        
@ti.func
def get_signed_distance(pos):
    dist = 0.
    max_dist = -1e10
    coord = get_sdf_coordinates(pos)
    cx = coord.x
    cy = coord.y
    cz = coord.z
    
    if ((cx < 0 or cx >= sdf_dim_x - 1)
        or (cy < 0 or cy >= sdf_dim_y - 1)
        or (cz < 0 or cz >= sdf_dim_z - 1)):
        dist = sdf_default_dist
    else:
        dist_000 = sdf[cx, cy, cz]
        max_dist = max(max_dist, dist_000)
        dist_100 = sdf[cx + 1, cy, cz]
        max_dist = max(max_dist, dist_100)
        dist_010 = sdf[cx, cy + 1, cz]
        max_dist = max(max_dist, dist_010)
        dist_110 = sdf[cx + 1, cy + 1, cz]
        max_dist = max(max_dist, dist_110)
        dist_001 = sdf[cx, cy, cz + 1]
        max_dist = max(max_dist, dist_001)
        dist_101 = sdf[cx + 1, cy, cz + 1]
        max_dist = max(max_dist, dist_101)
        dist_011 = sdf[cx, cy + 1, cz + 1]
        max_dist = max(max_dist, dist_011)
        dist_111 = sdf[cx + 1, cy + 1, cz + 1]
        max_dist = max(max_dist, dist_111)
        
        if max_dist < sdf_default_dist:
            cell_pos = get_sdf_cell_position(coord)
            interp = (pos - cell_pos) / sdf_cell_size
            dist = trilinear_interpolate(
                        dist_000, dist_100, dist_010, dist_110,
                        dist_001, dist_101, dist_011, dist_111,
                        interp.x, interp.y, interp.z)
        else:
            dist = sdf_default_dist
            
    return dist
    
def init_sdf():
    sdf_origin[None] = sdf_min
    for i in range(num_triangles):
        tri_idx[i] = triangles[i].vertices
    for i in range(num_vertices):
        tri_co[i] = coll_obj.data.vertices[i].co
    for x, y, z in ti.ndrange(sdf_dim_x, sdf_dim_y, sdf_dim_z):
        sdf[x, y, z] = sdf_default_dist
    construct_signed_distance_field()

@ti.func
def solve_collision_constraints():
    offset_x = ti.Vector([sdf_h, 0., 0.])
    offset_y = ti.Vector([0., sdf_h, 0.])
    offset_z = ti.Vector([0., 0., sdf_h])
    for i, j in ti.ndrange(M, (1, N)):
        pos = p[i, j]
        dist = get_signed_distance(pos)
        if (dist < coll_margin):
            neighbor_dist = ti.Vector([
                get_signed_distance(pos + offset_x),
                get_signed_distance(pos + offset_y),
                get_signed_distance(pos + offset_z),
            ])
            # gradient of sdf
            grad = (neighbor_dist - dist).normalized()
            # project position
            p[i, j] += 0.01 * grad * (coll_margin - dist)
        
#----------------------------------------
# Shape Constraints
#----------------------------------------

@ti.func
def quat_normalize(q):
    # Input: 4d quaternian, Output: Normalized quaternion
    n = q.dot(q)
    if  n < 1e-10:
        q.w = 1.0
    else:
        q *= 1.0 / ti.sqrt(n)
    return q

@ti.func
def quat_from_two_unit_vector(u, v):
    # u, v have to be unit vector.
    r = 1.0 + u.dot(v)
    n = ti.Vector([0.0,0.0,0.0])
    if r < 1e-7:
        r = 0.0
        if ti.abs(u.x) > ti.abs(u.z):
            n = ti.Vector([-u[1], u[0], 0.0])
        else:
            n = ti.Vector([0.0, -u[2], u[1]])
    else:
        n = u.cross(v)
    q = ti.Vector([n[0], n[1], n[2], r])
    return quat_normalize(q)

@ti.func
def mul_quat_and_vector(q, v):
    qvec = ti.Vector([q[0], q[1], q[2]])
    uv = qvec.cross(v)
    uuv = qvec.cross(uv)
    uv *= (2.0 * q[3])
    uuv *= 2.0
    return v + uv + uuv

@ti.func
def solve_local_constraints():
    for i in range(M):
        for j in range(1, N-1):
            # add bone matrix for global coordinates rotation
            bind_pos = x_rest[i,j]
            bind_pos_before = x_rest[i,j-1]
            bind_pos_after = x_rest[i,j+1]
            
            vec_prc_bind = bind_pos - bind_pos_before
            last_vec = x_rest[i,j] - x_rest[i,j-1]
            rot_global = quat_from_two_unit_vector(vec_prc_bind.normalized(), last_vec.normalized())
            
            orgPos_i_plus_1_InGlobalFrame = mul_quat_and_vector(rot_global, vec_prc_bind) + p[i,j]
            dist = stiffness_local * (orgPos_i_plus_1_InGlobalFrame - p[i,j+1])
            p[i,j] -= dist
            p[i,j+1] += dist

@ti.func
def solve_global_constraints():
    for i,j in ti.ndrange(M, N):
        coord = ti.Vector([i,j])
        p[coord] += stiffness_global * ( ti.Vector([x_rest[coord].x, x_rest[coord].y, x_rest[coord].z]) - p[coord])


#----------------------------------------
# Length Constraints
#----------------------------------------

@ti.func
def solve_distance_constraints():
    for i in range(M):
        p[i, 0] = x[i, 0]
        for j in range(1, N):
            p1 = p[i, j - 1]
            p2 = p[i, j]
            delta = ((p1 - p2).norm() - dx) * (p1 - p2).normalized()
            d[i, j] = delta
            p[i, j] += delta

@ti.func
def correct_velocities_ftl():
    for i, j in ti.ndrange(M, (1, N - 1)):
        v[i, j] -= damping_ftl * d[i, j + 1] / dt

#----------------------------------------
# Hair-Hair Interaction
#----------------------------------------

def init_vf():
    vf_origin[None] = vf_min

@ti.func
def hair_interaction():
    for I in ti.grouped(vf):
        vf[I] = [0., 0., 0.]
        density[I] = 0.
    
    for I in ti.grouped(x):
        pos = x[I]
        xd = (pos.x - vf_origin[None].x) / vf_cell_size
        yd = (pos.y - vf_origin[None].y) / vf_cell_size
        zd = (pos.z - vf_origin[None].z) / vf_cell_size
        
        x_min = max(int(xd), 0)
        x_max = min(int(xd) + 1, vf_dim_x - 1)
        y_min = max(int(yd), 0)
        y_max = min(int(yd) + 1, vf_dim_y - 1)
        z_min = max(int(zd), 0)
        z_max = min(int(zd) + 1, vf_dim_z - 1)
        
        for a, b, c in ti.ndrange((x_min, x_max), (y_min, y_max), (z_min, z_max)):
            x_weight = ts.clamp(1. - ti.abs(xd - a))
            y_weight = ts.clamp(1. - ti.abs(yd - b))
            z_weight = ts.clamp(1. - ti.abs(zd - c))
            
            total_weight = x_weight * y_weight * z_weight
            
            ti.atomic_add(vf[a, b, c], total_weight * v[I])
            ti.atomic_add(density[a, b, c], total_weight)
            
    for I in ti.grouped(x):
        pos = x[I]
        xd = (pos.x - vf_origin[None].x) / vf_cell_size
        yd = (pos.y - vf_origin[None].y) / vf_cell_size
        zd = (pos.z - vf_origin[None].z) / vf_cell_size

        x_min = max(int(xd), 0)
        x_max = min(int(xd) + 1, vf_dim_x - 1)
        y_min = max(int(yd), 0)
        y_max = min(int(yd) + 1, vf_dim_y - 1)
        z_min = max(int(zd), 0)
        z_max = min(int(zd) + 1, vf_dim_z - 1)
        
        grid_velocity = ti.Vector([0., 0., 0.])
        density_p = 0.
        gradient_deinsity = ti.Vector([0., 0.,0. ])
        
        for a, b, c in ti.ndrange((x_min, x_max), (y_min, y_max), (z_min, z_max)):
            x_weight = ts.clamp(1. - ti.abs(xd - a))
            y_weight = ts.clamp(1. - ti.abs(yd - b))
            z_weight = ts.clamp(1. - ti.abs(zd - c))
            
            total_weight = x_weight * y_weight * z_weight
            
            grid_velocity += total_weight / density[a, b, c] * vf[a, b, c]
            
            density_p += total_weight * density[a, b, c]

           
        '''
        pos_x = pos+ti.Vector([vf_h, 0., 0.])
        
        xd = (pos_x.x - vf_origin[None].x) / vf_cell_size
        yd = (pos_x.y - vf_origin[None].y) / vf_cell_size
        zd = (pos_x.z - vf_origin[None].z) / vf_cell_size

        x_min = max(int(xd), 0)
        x_max = min(int(xd) + 1, vf_dim_x - 1)
        y_min = max(int(yd), 0)
        y_max = min(int(yd) + 1, vf_dim_y - 1)
        z_min = max(int(zd), 0)
        z_max = min(int(zd) + 1, vf_dim_z - 1)
        
        density_px = 0.
        
        for a, b, c in ti.ndrange((x_min, x_max), (y_min, y_max), (z_min, z_max)):
            x_weight = ts.clamp(1. - ti.abs(xd - a))
            y_weight = ts.clamp(1. - ti.abs(yd - b))
            z_weight = ts.clamp(1. - ti.abs(zd - c))
            
            total_weight = x_weight * y_weight * z_weight
            
            density_px += total_weight * density[a, b, c]
        
        # 
        pos_y = pos+ti.Vector([0, vf_h, 0.])
        xd = (pos_y.x - vf_origin[None].x) / vf_cell_size
        yd = (pos_y.y - vf_origin[None].y) / vf_cell_size
        zd = (pos_y.z - vf_origin[None].z) / vf_cell_size

        x_min = max(int(xd), 0)
        x_max = min(int(xd) + 1, vf_dim_x - 1)
        y_min = max(int(yd), 0)
        y_max = min(int(yd) + 1, vf_dim_y - 1)
        z_min = max(int(zd), 0)
        z_max = min(int(zd) + 1, vf_dim_z - 1)
        
        density_py = 0.
        
        for a, b, c in ti.ndrange((x_min, x_max), (y_min, y_max), (z_min, z_max)):
            x_weight = ts.clamp(1. - ti.abs(xd - a))
            y_weight = ts.clamp(1. - ti.abs(yd - b))
            z_weight = ts.clamp(1. - ti.abs(zd - c))
            
            total_weight = x_weight * y_weight * z_weight
            
            density_py += total_weight * density[a, b, c]
            
        pos_z = pos+ti.Vector([0., 0., vf_h])
        
        xd = (pos_z.x - vf_origin[None].x) / vf_cell_size
        yd = (pos_z.y - vf_origin[None].y) / vf_cell_size
        zd = (pos_z.z - vf_origin[None].z) / vf_cell_size

        x_min = max(int(xd), 0)
        x_max = min(int(xd) + 1, vf_dim_x - 1)
        y_min = max(int(yd), 0)
        y_max = min(int(yd) + 1, vf_dim_y - 1)
        z_min = max(int(zd), 0)
        z_max = min(int(zd) + 1, vf_dim_z - 1)
        
        density_pz = 0.
        
        for a, b, c in ti.ndrange((x_min, x_max), (y_min, y_max), (z_min, z_max)):
            x_weight = ts.clamp(1. - ti.abs(xd - a))
            y_weight = ts.clamp(1. - ti.abs(yd - b))
            z_weight = ts.clamp(1. - ti.abs(zd - c))
            
            total_weight = x_weight * y_weight * z_weight
            
            density_pz += total_weight * density[a, b, c]
        
        
        grad = ti.Vector([density_px-density_p,density_py-density_p, density_pz-density_p])
        '''
                
        v[I] = (1. - friction) * v[I] + friction * grid_velocity
        
        #v[I] = v[I]+ repulsion* grad.normalized() / dt 
        
        

#----------------------------------------
# PBD
#----------------------------------------

def local_to_world(pos_local, mat):
    pos_local_hom = mathutils.Vector((pos_local[0], pos_local[1], pos_local[2], 1.0))
    pos_world_hom = mat @ pos_local_hom
    pos_world = pos_world_hom.xyz / pos_world_hom.w
    return pos_world

def world_to_local(pos_world, mat):
    pos_world_hom = mathutils.Vector((pos_world[0], pos_world[1], pos_world[2], 1.0))
    pos_local_hom = mat.inverted() @ pos_world_hom
    pos_local = pos_local_hom.xyz / pos_local_hom.w
    return pos_local

# transform vectors
def world_to_local_v(vec_world, mat):
    vec_world_hom = mathutils.Vector((vec_world[0], vec_world[1], vec_world[2], 0.0))
    vec_local_hom = mat.inverted() @ vec_world_hom
    vec_local = vec_local_hom.xyz
    return vec_local

def init_data():
    mat = head.matrix_world
    for i, j in ti.ndrange(M, N):
        x[i, j] = world_to_local(curves[i].points[j].co.xyz, mat)
        x_rest[i, j] = world_to_local(curves[i].points[j].co.xyz, mat)
        v[i, j] = [0., 0., 0.]
    v_world_old[None] = [0., 0., 0.]

def update_data(frame):
    trans = custom_translation(frame)
    head.location = head_loc + trans
    bpy.context.view_layer.update()
    mat = head.matrix_world.copy()
    
    v_world[None] = world_to_local_v(trans / dt / steps, mat)
    f_ext[None] = mass * world_to_local_v(grav_acc, mat)
    for i, j in ti.ndrange(M, (1, N)):
        x[i, j] = world_to_local(curves[i].points[j].co.xyz, mat)
    
    return mat

@ti.kernel
def propose():
    for i, j in ti.ndrange(M, N):
        v[i, j] += v_world_old[None] - v_world[None]
    for i in range(M):
        p[i, 0] = x[i, 0]
    for i, j in ti.ndrange(M, (1, N)):
        p[i, j] = x[i, j] + dt * v[i, j] + dt * dt * w * f_ext[None]
    v_world_old[None] = v_world[None]

@ti.kernel
def project_constraints():
    solve_global_constraints()
    solve_collision_constraints()
    solve_local_constraints()
    solve_distance_constraints()
    
@ti.kernel
def update_velocities():
    for i, j in ti.ndrange(M, (1, N)):
        v[i, j] = damping * (p[i, j] - x[i, j]) / dt
        x[i, j] = p[i, j]
        
    correct_velocities_ftl()
    hair_interaction()

def update():
    propose()
    project_constraints()
    update_velocities()

#----------------------------------------
# Animation
#----------------------------------------

def curve_update(pos, mat):
    def callback():
        head.matrix_world = mat
        for i, j in ti.ndrange(M, N):
            curves[i].points[j].co.xyz = local_to_world(pos[i, j], mat)
    return nb.AnimUpdate(callback)

@nb.add_animation
def main():
    yield curve_update(x.to_numpy(), head.matrix_world)
    for frame in range(frames):
        mat = update_data(frame)
        for step in range(steps):
            update()
        yield curve_update(x.to_numpy(), mat)

init_data()
init_sdf()
init_vf()
bpy.context.scene.frame_current = 0