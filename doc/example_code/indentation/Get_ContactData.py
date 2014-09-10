from abapy.indentation import Get_ContactData
from odbAccess import openOdb
from abapy.misc import dump

# Odb opening  
Berk_name = 'workdir/indentation_berko'
Axi_name = 'workdir/indentation_axi'
Berk_odb = openOdb(Berk_name + '.odb')
Axi_odb = openOdb(Axi_name + '.odb')
# Getting data
Berk_out = Get_ContactData(odb = Berk_odb, instance = 'I_SAMPLE', node_set = 'TOP_NODES')
Axi_out = Get_ContactData(odb = Axi_odb, instance = 'I_SAMPLE', node_set = 'TOP_NODES')
# Dumping data
dump(Axi_out, 'ContactData_axi.pckl')
dump(Berk_out, 'ContactData_berk.pckl')


