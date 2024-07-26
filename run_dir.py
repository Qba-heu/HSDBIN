import subprocess

command = 'python main_t2l.py --patch_size 19 ' \
          '--centroid_path ./Estimated_prototypes/10centers_192dim.pth ' \
          '--training_sample 0.005 ' \
          '--source_HSI PaviaU ' \
          '--disjoint True ' \
          '--epoch 100 ' \
          '--runs 5 ' \
          '--smoothing 0.2 ' \
          '--batch_size 32 -' \
          '-lambda1 0.1 ' \
          '--lambda2 0.1' \
          '--alpha 0.9'

subprocess.run(command,shell=True)

