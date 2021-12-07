"""
script to solve ncurses error
"""

import os
import enum
import stat
import shutil
import argparse


class PathType(enum.Enum):
    dir = 0     # directory
    chr = 1     # character special device file
    blk = 2     # block special device file
    reg = 3     # regular file
    fifo = 4    # FIFO (named pipe)
    lnk = 5     # symbolic link
    sock = 6    # socket
    door = 7    # door  (Py 3.4+)
    port = 8    # event port  (Py 3.4+)
    wht = 9     # whiteout (Py 3.4+)
    unknown = 10

    @classmethod
    def get(cls, path):
        if not isinstance(path, int):
            path = os.stat(path).st_mode
        for path_type in cls:
            method = getattr(stat, 'S_IS' + path_type.name.upper())
            if method and method(path):
                return path_type.name
        return cls.unknown.name


PathType.__new__ = (lambda cls, path: cls.get(path))


def read_errorconda(errorconda, verbose=False):
    all_paths = []
    all_directories = []

    with open(errorconda, 'r') as f:
        for l in f.readlines():
            if 'located' in l:
                directory = l.split(' ')[-1].strip()
                all_directories.append(directory)
            elif 'path' in l:
                path = l.split(' ')[-1][1:-2]
                all_paths.append(path)

    if not len(all_directories) == len(all_paths):
        raise Exception(
            'invalid errorconda.txt, number of directories found '
            'does not match the number of paths found'
        )
    if not all_directories:
        raise Exception(
            'invalid errorconda.txt, cannot find keywords "located" and "path"'
        )
    all_paths_complete = [os.path.join(d, p) for d, p in zip(all_directories, all_paths)]
    
    if verbose:
        print('\nWhat they want')
        for i, path in enumerate(all_paths_complete, start=1):
            print('{}.\t{}'.format(i, path))
    
    return all_directories, all_paths, all_paths_complete


def create_backup(all_directories, verbose=False):
    if verbose:
        print("\nCreating backup")
    for directory in set(all_directories):
        backup = os.path.join(directory, 'share/terminfo_copy')
        source = os.path.join(directory, 'share/terminfo')
        if not os.path.isdir(backup):
            shutil.copytree(source, backup, dirs_exist_ok=True)
            if verbose:
                print("source: {}\nbackup: {}".format(source, backup))


def solve(all_paths_complete, all_directories, verbose=False):
    if verbose:
        print("\nSolving error")

    all_installed = dict()
    for directory in set(all_directories):
        main_dir = os.path.join(directory, 'share/terminfo')
        for d in os.listdir(main_dir):
            d = os.path.join(main_dir, d)
            for p in os.listdir(d):
                installed = os.path.join(d, p)
                all_installed[installed.lower()] = installed
    
    try:
        for i, path in enumerate(all_paths_complete, start=1): 
            if path.lower() not in all_installed:
                raise FileNotFoundError("{} not found".format(path))
                
            wrong = all_installed[path.lower()]
            if path == wrong:
                if verbose:
                    print("{}.\t{} already done!".format(i, path))
                continue
            
            installed = os.path.basename(wrong)
            temp = os.path.join(directory, 'temp_{}'.format(installed))
            if os.path.islink(wrong):
                target = os.path.join(directory, os.readlink(wrong))
                os.remove(wrong)
                os.symlink(target, path)

            else:
                if not os.path.isfile(wrong):
                    raise FileNotFoundError("{} is not a file".format(wrong))
                shutil.copyfile(wrong, temp)
                os.remove(wrong)
                shutil.copyfile(temp, path)
                if not os.path.exists(path):
                    raise FileNotFoundError("{} failed to be renamed to {}".format(wrong, path))
                os.remove(temp)
            
            if verbose:
                print("{}.\t{} renamed to {}".format(i, wrong, path))

        # if os.path.isdir(directory + '/share/terminfo_copy/'):
        #     shutil.rmtree(directory + '/share/terminfo_copy/')
        #     print("Removed backup since bug fix succeeded and we don't need it anymore")

    except Exception as e:
        print("\nAn exception has occurred: {}".format(e))
        print("Not all files are replaced successfully!")
        print("Rolling back the changes")
        for directory in set(all_directories):
            backup = os.path.join(directory, 'share/terminfo_copy')
            source = os.path.join(directory, 'share/terminfo')
            shutil.rmtree(source)
            shutil.copytree(backup, source, dirs_exist_ok=True)        
        print("Rollback successful! Try to run the script again, somehow it works on the second try...")


def main(args):
    verbose = args.verbose
    errorconda = args.errorconda

    all_directories, all_paths, all_paths_complete = read_errorconda(errorconda, verbose)
    create_backup(all_directories, verbose)
    solve(all_paths_complete, all_directories, verbose)
    if verbose:
        print("\nCompleted!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--errorconda', type=str, default='./errorconda.txt',
                        help='the path to errorconda.txt')
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()
    main(args)