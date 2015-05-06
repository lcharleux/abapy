'''
Abapy Documentation
=================================


Abaqus Python "**AbaPy**" contains tools to build, postprocess and plot automatic finite element simulations using Abaqus. It is divided into four parts:
  * *mesh*: contains finite element mesh utilities allowing building, modifying, ploting and exporting to various formats.
  * *postproc*: contains utilities to read in Abaqus odb files and represent history and field outputs using custom classes. These classes allow easy calculations, export, storage (using pickle/cPickle).
  * *materials*: contains functions to preprocess materials before finite element simulations.
  * *indentation*: contains all indentation dedicated tools.
  * *advanced_examples*: contains examples using abapy and other python packages to perform research higher level tasks.

.. plot:: example_code/logo/abapy_logo.py


.. codeauthor:: Ludovic Charleux <ludovic.charleux@univ-savoie.fr>
     

Installation can be performed in many ways, here a two:
  
* The right way:
  
.. code-block:: bash

   pip install git+https://github.com/lcharleux/abapy.git

* If you are contributing to the module, you can just clone the repository:
    
.. code-block:: bash

   git clone https://github.com/lcharleux/abapy.git   

And remember to add the abapy/abapy directory to your ``PYTHONPATH``. For example, the following code can be used under Linux (in ``.bashrc`` or ``.profile``):

.. code-block:: bash

  export PYTHONPATH=$PYTHONPATH:yourpath/abapy 


.. toctree::
   :maxdepth: 2
'''

import indentation, materials, mesh, misc, postproc
