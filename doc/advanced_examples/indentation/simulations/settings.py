from classes import Simulation, Database_Manager

#----------------------------------------------------------------------------------------------------------------
# SETTINGS
work_dir      = 'workdir/'
plot_dir      = 'plots/'
database_dir  = 'database/'
database_name = 'database'
abqlauncher   = '/opt/Abaqus/6.9/Commands/abaqus'
cls           = Simulation
#----------------------------------------------------------------------------------------------------------------

#----------------------------------------------------------------------------------------------------------------
# Starting Database Manager
db_manager =  Database_Manager(
  work_dir = work_dir,
  database_dir = database_dir,
  database_name = database_name,
  abqlauncher = abqlauncher,
  cls = cls)
#----------------------------------------------------------------------------------------------------------------
