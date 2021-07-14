# LumiRender
A GPU/CPU physically based renderer

## Features

### Render algorithm
| Feature                                                 | Progress  |
|---------------------------------------------------------|-----------|
| megakernel path tracing                                 | Done      |
| wavefront path tracing                                  | Planned   |
| bidirectional path tracing                              | Planned   |

### Backends
| Feature             | Progress                                            |
|---------------------|-----------------------------------------------------|
| GPU                 | Done (with CUDA + OptiX)                            |
| CPU                 | Planned (with embree)                               |

### Exporters
| Feature             | Progress                                            |
|---------------------|-----------------------------------------------------|
| Blender             | working                                             |
| 3DS Max             | Planned                                             |

## Gallery
megakernel pt + independent sampling + pinhole camera on GPU
![](gallery/cornell_box.png)