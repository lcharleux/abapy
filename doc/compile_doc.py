# DOC COMPILER
# Use only as a standalone script in order to compile the Sphinx doc


if __name__ == '__main__':
  import os, subprocess
  os.system('git checkout master')
  proc = subprocess.Popen(["git status"], stdout=subprocess.PIPE, shell=True)
  (out, err) = proc.communicate()
  out = out.split('\n')[0]
  if out == "# On branch master":
    os.system('sphinx-build -a . _build/html')  
    os.system('git checkout gh-pages')
    proc = subprocess.Popen(["git status"], stdout=subprocess.PIPE, shell=True)
    (out, err) = proc.communicate()
    out = out.split('\n')[0]
    if out == "# On branch gh-pages":
      os.system('cp -R _build/html/* ../')
    else: 
      print "Commit before compiling doc"
  else:
    print "Commit before compiling doc"
          
  
  
