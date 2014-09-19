# DOC COMPILER
# Use only as a standalone script in order to compile the Sphinx doc


if __name__ == '__main__':
  import os
  files_in_dir = os.listdir('./')
  if 'workdir' not in files_in_dir: 
    os.system('mkdir workdir')
  os.system('sphinx-build -a . _build/html')  
  
