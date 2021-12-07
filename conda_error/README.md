# Conda Verification Error

See [this GitHub issue](https://github.com/ContinuumIO/anaconda-issues/issues/12089O) for a summary of the problem. 

All the problems are related to ncurses, which is a package required by conda for linux-based machines (it deals with command line related stuff). In ncurses, the subfolder names in 

```
{PATH_TO_MINICONDA}/pkgs/ncurses-6.3-h7f8727e_2/share/terminfo/
``` 

have a mix of upper and lower cases. When a new conda environment is created, some checks are performed. The directories the checks looks for do not match what we have because our OS (in the backend, a Windows Server machine) is case insensitive. 

This creates 2 problems:
1. Python shell bug -> see bionet01 documentation for more details
2. New conda environments cannot be created.

**So we need to rename all these files (most are actually simlinks) to what the checks are looking for.** Once done correctly, the 2 problems mentioned above will be addressed (but for 1, only when you're in the environment you created).
The error message produced shows all the files it looks for, so we save them in a text file and process it with this script. 

# How to use

1. Try creating an environment, e.g. `conda create --name py38 python=3.8`
2. Copy the error message into a text file, save it as `errorconda.txt` (replace the existing one in this folder)
3. Run the Python script `conda_workaround.py`, it has additional features (like backup and rollback) as compared to this notebook.
4. If there's a bug, run the script again and somehow it works on the second try...
5. Remember to delete the envirnoment conda previously tried to create `rm {PATH_TO_MINICONDA}/envs/py38` before trying Step 1 again (replace miniconda with anaconda as needed)