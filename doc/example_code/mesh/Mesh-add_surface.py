from abapy.mesh import RegularQuadMesh
mesh = RegularQuadMesh()
mesh.add_surface('topsurface', [ ('top', 1) ])
mesh.add_surface('topsurface', [ ('top', 2) ])
mesh.surfaces
