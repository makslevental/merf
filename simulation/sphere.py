# -----------------------------------------------------------------------------
# Python & OpenGL for Scientific Visualization
# www.labri.fr/perso/nrougier/python+opengl
# Copyright (c) 2018, Nicolas P. Rougier
# Distributed under the 2-Clause BSD License.
# -----------------------------------------------------------------------------
from math import sqrt

import numpy as np
from glumpy import app, gloo, gl


def mie(g, c):
    a = (1.0 - g ** 2) * (1.0 + c ** 2)
    b = 1.0 + g ** 2 - 2.0 * g * c
    b *= sqrt(b)
    b *= 2.0 + g ** 2
    return (3.0 / 8.0 / np.pi) * (a / b)


for c in np.linspace(-1, 1, 20):
    print(mie(.75, c))

vertex = """
    uniform vec2 resolution;
    attribute vec2 center;
    attribute float radius;
    varying vec2 v_center;
    varying float v_radius;
    void main()
    {
        v_radius = radius;
        v_center = center;
        gl_PointSize = 2.0 + ceil(2.0*radius);
        gl_Position = vec4(2.0*center/resolution-1.0, 0.0, 1.0);
    } """

fragment = """
  varying vec2 v_center;
  varying float v_radius;
  const float PI = 3.14159265359;
  float phase_mie(float g, float c) {
    float gg = g * g;
    float cc = c * c;
    
    float a = (1.0 - gg) * (1.0 + cc);

    float b = 1.0 + gg - 2.0 * g * c;
    b *= sqrt(b);
    b *= 2.0 + gg;

    return (3.0 / 8.0 / PI) * a / b;
  }
  void main()
  {
      vec2 p = (gl_FragCoord.xy - v_center)/v_radius;
      float z = 1.0 - length(p);
      if (z < 0.0) discard;

      vec3 color = vec3(1.0, 0.0, 0.0);
      vec3 normal = normalize(vec3(p.xy, z));
      vec3 direction = normalize(vec3(1,1,1));
      float diffuse = max(0.0, dot(direction, normal));
      float mie = phase_mie(-0.75, 0.0);
      float specular = pow(diffuse, 24.0);
      gl_FragColor = vec4(max(mie*diffuse*color, specular*vec3(1.0)), 1.0);
  } """

V = np.zeros(1, [("center", np.float32, 2), ("radius", np.float32, 1)])
V["center"] = 256, 256
V["radius"] = 250

window = app.Window(512, 512, color=(1, 1, 1, 1))
points = gloo.Program(vertex, fragment)
points.bind(V.view(gloo.VertexBuffer))


@window.event
def on_resize(width, height):
    points["resolution"] = width, height


@window.event
def on_draw(dt):
    window.clear()
    points.draw(gl.GL_POINTS)


app.run()
