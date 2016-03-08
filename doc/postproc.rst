Post Processing
===============

Finite Element Modeling post processing tools.

Field Outputs
~~~~~~~~~~~~~

Scalar fields
---------------------

.. autoclass:: abapy.postproc.FieldOutput

Add/remove/get data
___________________

.. automethod:: abapy.postproc.FieldOutput.add_data

.. automethod:: abapy.postproc.FieldOutput.get_data


VTK Export
__________

.. automethod:: abapy.postproc.FieldOutput.dump2vtk


Operations and invariants
_________________________
  

Vector fields
---------------------------

.. autoclass:: abapy.postproc.VectorFieldOutput

Add/remove/get data
___________________

.. automethod:: abapy.postproc.VectorFieldOutput.add_data

.. automethod:: abapy.postproc.VectorFieldOutput.get_data

.. automethod:: abapy.postproc.VectorFieldOutput.get_coord



VTK Export
__________

.. automethod:: abapy.postproc.VectorFieldOutput.dump2vtk


Operations
__________

.. automethod:: abapy.postproc.VectorFieldOutput.norm

.. automethod:: abapy.postproc.VectorFieldOutput.sum

.. automethod:: abapy.postproc.VectorFieldOutput.dot

.. automethod:: abapy.postproc.VectorFieldOutput.cross



Tensor fields
---------------------------

.. autoclass:: abapy.postproc.TensorFieldOutput

Add/remove/get data
___________________

.. automethod:: abapy.postproc.TensorFieldOutput.add_data

.. automethod:: abapy.postproc.TensorFieldOutput.get_data

.. automethod:: abapy.postproc.TensorFieldOutput.get_component

VTK Export
__________




.. automethod:: abapy.postproc.TensorFieldOutput.dump2vtk


Operations and invariants
_________________________


.. automethod:: abapy.postproc.TensorFieldOutput.sum

.. automethod:: abapy.postproc.TensorFieldOutput.trace

.. automethod:: abapy.postproc.TensorFieldOutput.deviatoric

.. automethod:: abapy.postproc.TensorFieldOutput.spheric

.. automethod:: abapy.postproc.TensorFieldOutput.i1

.. automethod:: abapy.postproc.TensorFieldOutput.i2

.. automethod:: abapy.postproc.TensorFieldOutput.i3

.. automethod:: abapy.postproc.TensorFieldOutput.j2

.. automethod:: abapy.postproc.TensorFieldOutput.j3

.. automethod:: abapy.postproc.TensorFieldOutput.eigen

.. automethod:: abapy.postproc.TensorFieldOutput.pressure

.. automethod:: abapy.postproc.TensorFieldOutput.vonmises

.. automethod:: abapy.postproc.TensorFieldOutput.tresca
  
Getting field outputs from an Abaqus ODB
----------------------------------------

Scalar fields
_____________

.. autofunction:: abapy.postproc.GetFieldOutput
.. autofunction:: abapy.postproc.MakeFieldOutputReport
.. autofunction:: abapy.postproc.ReadFieldOutputReport
.. autofunction:: abapy.postproc.GetFieldOutput_byRpt



Vector fields
_____________

.. autofunction:: abapy.postproc.GetVectorFieldOutput
.. autofunction:: abapy.postproc.GetVectorFieldOutput_byRpt

Tensor fields
_____________

.. autofunction:: abapy.postproc.GetTensorFieldOutput
.. autofunction:: abapy.postproc.GetTensorFieldOutput_byRpt


``ZeroFieldOutput_like``
------------------------

.. autofunction:: abapy.postproc.ZeroFieldOutput_like

``OneFieldOutput_like``
------------------------

.. autofunction:: abapy.postproc.OneFieldOutput_like

``Identity_like``
------------------------

.. autofunction:: abapy.postproc.Identity_like


History Outputs
~~~~~~~~~~~~~~~

``HistoryOutput`` class
-----------------------

.. autoclass:: abapy.postproc.HistoryOutput
  
Add/get data
____________

.. automethod:: abapy.postproc.HistoryOutput.add_step

.. automethod:: abapy.postproc.HistoryOutput.plotable

.. automethod:: abapy.postproc.HistoryOutput.toArray

Utilities
_________

.. automethod:: abapy.postproc.HistoryOutput.total
.. automethod:: abapy.postproc.HistoryOutput.integral
.. automethod:: abapy.postproc.HistoryOutput.average



.. automethod:: abapy.postproc.HistoryOutput.data_min
.. automethod:: abapy.postproc.HistoryOutput.data_max
.. automethod:: abapy.postproc.HistoryOutput.time_min
.. automethod:: abapy.postproc.HistoryOutput.time_max

.. automethod:: abapy.postproc.HistoryOutput.duration

``GetHistoryOutputByKey`` function
----------------------------------

.. autofunction:: abapy.postproc.GetHistoryOutputByKey


Mesh
~~~~

``GetMesh`` function
-----------------------

.. autofunction:: abapy.postproc.GetMesh


