from skimage.filters import gaussian
from skimage.io import imread
from skimage.util import random_noise

from preprocess import make_figure

import numpy as np
from glumpy import app, gloo, gl
from glumpy.ext import png

def create_circular_mask(h, w, center=None, radius=None):
    if center is None:  # use the middle of the image
        center = (int(w / 2), int(h / 2))
    if radius is None:  # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w - center[0], h - center[1])

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2)

    mask = dist_from_center <= radius
    return mask



vertex = """
    uniform vec2 resolution;
    attribute vec3 center;
    attribute float radius;
    varying vec3 v_center;
    varying float v_radius;
    void main()
    {
        v_radius = radius;
        v_center = center;
        gl_PointSize = 2.0 + ceil(2.0*radius);
        gl_Position = vec4(2.0*center.xy/resolution-1.0, v_center.z, 1.0);
    } """

fragment = """
  varying vec3 v_center;
  varying float v_radius;
  void main()
  {
      vec2 p = (gl_FragCoord.xy - v_center.xy)/v_radius;
      float z = 1.0 - length(p);
      if (z < 0.0) discard;

      gl_FragDepth = 0.5*v_center.z + 0.5*(1.0 - z);


      vec3 color = vec3(1.0, 1.0, 1.0);
      vec3 normal = normalize(vec3(p.xy, z));
      vec3 direction = normalize(vec3(0.0, 0.0, 1.0));
      float diffuse = max(0.0, dot(direction, normal));
      float specular = pow(diffuse, 24.0);
      gl_FragColor = vec4(max(diffuse*color, specular*vec3(1.0)), 1.0);  
   } 
"""

window_size = 1024

np.random.seed(1)
V = np.zeros(100, [("center", np.float32, 3),
                   ("radius", np.float32, 1)])
V["center"] = np.random.uniform(50, window_size - 50, (len(V), 3))
V["center"][:, 2] = 0  # np.random.uniform(0, 1, len(V))
V["radius"] = np.random.uniform(5, 25, len(V))

window = app.Window(window_size, window_size)
points = gloo.Program(vertex, fragment)
points.bind(V.view(gloo.VertexBuffer))

framebuffer = np.zeros((window.height, 3 * window.width), dtype=np.uint8)


@window.event
def on_resize(width, height):
    points["resolution"] = width, height


@window.event
def on_draw(dt):
    window.clear()
    gl.glEnable(gl.GL_DEPTH_TEST)
    points.draw(gl.GL_POINTS)
    gl.glReadPixels(0, 0, window.width, window.height,
                    gl.GL_RGB, gl.GL_UNSIGNED_BYTE, framebuffer)
    png.from_array(framebuffer, 'RGB').save('screenshot.png')


app.run(framecount=1)

im = imread("screenshot.png", as_gray=True)
make_figure(im).show()
background = np.zeros_like(im)
noisy = random_noise(background, mode="gaussian")
noisy = random_noise(noisy, mode="poisson")

for (y, x, _z), r in V:
    mask = create_circular_mask(window_size, window_size, (y,x), r*.9)
    noisy[mask] = 0

res = gaussian(noisy+im, sigma=1)
make_figure(res).show()
# make_figure(noisy+im).savefig("screenshot.png")
