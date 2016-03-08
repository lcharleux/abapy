Mesh
====

Mesh processing tools.

Nodes
~~~~~

.. autoclass:: abapy.mesh.Nodes
  
Add/remove/get data
___________________

.. automethod:: abapy.mesh.Nodes.add_node
.. automethod:: abapy.mesh.Nodes.drop_node
.. automethod:: abapy.mesh.Nodes.add_set
.. automethod:: abapy.mesh.Nodes.add_set_by_func
.. automethod:: abapy.mesh.Nodes.drop_set

Modifications
_____________

.. automethod:: abapy.mesh.Nodes.translate
.. automethod:: abapy.mesh.Nodes.apply_displacement
.. automethod:: abapy.mesh.Nodes.closest_node
.. automethod:: abapy.mesh.Nodes.replace_node
.. automethod:: abapy.mesh.Mesh.apply_reflection

Export
______

.. automethod:: abapy.mesh.Nodes.dump2inp


Tools
_____

.. automethod:: abapy.mesh.Nodes.eval_function
.. automethod:: abapy.mesh.Nodes.eval_vectorFunction
.. automethod:: abapy.mesh.Nodes.eval_tensorFunction
.. automethod:: abapy.mesh.Nodes.boundingBox

Mesh
~~~~

.. autoclass:: abapy.mesh.Mesh
   
Add/remove/get data
___________________

.. automethod:: abapy.mesh.Mesh.add_element
.. automethod:: abapy.mesh.Mesh.drop_element
.. automethod:: abapy.mesh.Mesh.drop_node
.. automethod:: abapy.mesh.Mesh.add_set
.. automethod:: abapy.mesh.Mesh.drop_set
.. automethod:: abapy.mesh.Mesh.add_surface
.. automethod:: abapy.mesh.Mesh.node_set_to_surface
.. automethod:: abapy.mesh.Mesh.replace_node
.. automethod:: abapy.mesh.Mesh.simplify_nodes
.. automethod:: abapy.mesh.Mesh.add_field

Useful data
___________________

.. automethod:: abapy.mesh.Mesh.centroids
.. automethod:: abapy.mesh.Mesh.volume

Modifications
_____________

.. automethod:: abapy.mesh.Mesh.extrude
.. automethod:: abapy.mesh.Mesh.sweep
.. automethod:: abapy.mesh.Mesh.union
.. automethod:: abapy.mesh.Mesh.apply_reflection

Export
______

.. automethod:: abapy.mesh.Mesh.dump2inp
.. automethod:: abapy.mesh.Mesh.dump2vtk

Ploting tools
_____________

.. automethod:: abapy.mesh.Mesh.convert2tri3
.. automethod:: abapy.mesh.Mesh.dump2triplot
.. automethod:: abapy.mesh.Mesh.get_edges
.. automethod:: abapy.mesh.Mesh.get_border
.. automethod:: abapy.mesh.Mesh.dump2polygons
.. automethod:: abapy.mesh.Mesh.draw

Mesh generation
~~~~~~~~~~~~~~~

``RegularQuadMesh`` functions
_____________________________
.. autofunction:: abapy.mesh.RegularQuadMesh
.. autofunction:: abapy.mesh.RegularQuadMesh_like

Other meshes
__________________
.. autofunction:: abapy.mesh.TransitionMesh

.. note:: see also in ``abapy.indentation`` for indentation dedicated meshes.
  

