import os
from shutil import copyfile

# pattern =
curdir = os.getcwd()
while not curdir.endswith("NatCo-Project"):
    os.chdir(os.path.dirname(curdir))
    curdir = os.getcwd()


print(os.getcwd())
comp_10_iter_dir = "data/archives/Autorun_10iters_28-05-2021_10:42:26"
com_1_iter_dir = "data/archives/Autorun_1iter_28-05-2021_10:45:40"

c10_iter_folder = 9
c1_iter_folder = 0

comp_10_target_dir = 'data/turing_test_data/computer_g10'
comp_1_target_dir = 'data/turing_test_data/computer_g1'



l= [
    (comp_10_iter_dir,c10_iter_folder,comp_10_target_dir),
    (com_1_iter_dir,c1_iter_folder,comp_1_target_dir)
]
for srcdir,iterdir,trgtdir in l:
    i=0
    for run in os.listdir(srcdir):
        if run.startswith("run"):
            targetiter = os.path.join(curdir,srcdir,run,f'iter{iterdir}')
            for iterfile in os.listdir(targetiter):
                copyfile(os.path.join(curdir,targetiter,iterfile),os.path.join(curdir,trgtdir,f'{i}.png'))
                i+=1


human_10_iter_dirs = "david,ron,stijn".split(',')
human_10_iter_dirs = [os.path.join(curdir,'data','turing_test_data',i) for i in human_10_iter_dirs]
m=0
for i in human_10_iter_dirs:
    for j in os.listdir(i):
        if j.startswith('run'):
            for k in os.listdir(os.path.join(i,j,'iter10')):
                copyfile(os.path.join(i,j,'iter10',k),os.path.join(curdir,'data','turing_test_data','human_g10',f'{m}.png'))
                m+=1

