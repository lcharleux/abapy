Indentation
===============

Indentation simulation tools


Indentation meshes
~~~~~~~~~~~~~~~~~~

``ParamInfiniteMesh`` function
------------------------------

.. autofunction:: abapy.indentation.ParamInfiniteMesh

``IndentationMesh`` function
------------------------------
.. autofunction:: abapy.indentation.IndentationMesh






Indenters
~~~~~~~~~

``RigidCone2D`` class
----------------------

.. autoclass:: abapy.indentation.RigidCone2D
  :members:

``DeformableCone2D`` class
---------------------------

.. autoclass:: abapy.indentation.DeformableCone2D
  :members:

``DeformableCone3D`` class
---------------------------

.. autoclass:: abapy.indentation.DeformableCone3D
  :members:

Indenter miscellaneous
-------------------------
.. autofunction:: abapy.indentation.equivalent_half_angle


Simulation tools
~~~~~~~~~~~~~~~~~~~

Steps definition
----------------

.. autoclass:: abapy.indentation.Step
.. automethod:: abapy.indentation.Step.set_name
.. automethod:: abapy.indentation.Step.set_displacement
.. automethod:: abapy.indentation.Step.set_nframes
.. automethod:: abapy.indentation.Step.set_nlgeom
.. automethod:: abapy.indentation.Step.set_fieldOutputFreq
.. automethod:: abapy.indentation.Step.set_nodeFieldOutput
.. automethod:: abapy.indentation.Step.set_elemFieldOutput
.. automethod:: abapy.indentation.Step.dump2inp

Inp builder
--------------------

.. autofunction:: abapy.indentation.MakeInp


Simulation manager
------------------

.. autoclass:: abapy.indentation.Manager
  
Settings
________

.. automethod:: abapy.indentation.Manager.set_simname
.. automethod:: abapy.indentation.Manager.set_workdir
.. automethod:: abapy.indentation.Manager.set_abqlauncher
.. automethod:: abapy.indentation.Manager.set_samplemesh
.. automethod:: abapy.indentation.Manager.set_indenter
.. automethod:: abapy.indentation.Manager.set_samplemat
.. automethod:: abapy.indentation.Manager.set_steps
.. automethod:: abapy.indentation.Manager.set_files2delete
.. automethod:: abapy.indentation.Manager.set_abqpostproc
.. automethod:: abapy.indentation.Manager.set_pypostprocfunc

Launchers
__________
.. automethod:: abapy.indentation.Manager.erase_files
.. automethod:: abapy.indentation.Manager.make_inp
.. automethod:: abapy.indentation.Manager.run_sim
.. automethod:: abapy.indentation.Manager.run_abqpostproc
.. automethod:: abapy.indentation.Manager.run_pypostproc

Indentation post-processing
-------------------------------

Contact Data
____________
.. autoclass:: abapy.indentation.ContactData
  :members:

Get Contact Data
________________
.. autofunction:: abapy.indentation.Get_ContactData


Elasticity
~~~~~~~~~~

Hertz
--------
.. autoclass:: abapy.indentation.Hertz
  :members:

Hanson
----------
.. autoclass:: abapy.indentation.Hanson
  :members:

  
