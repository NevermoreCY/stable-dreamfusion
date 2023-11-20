import os
import json
import sys
import argparse
import shutil
from tqdm import tqdm

def doArgs(argList):
    parser = argparse.ArgumentParser()

    #parser.add_argument('-v', "--verbose", action="store_true", help="Enable verbose debugging", default=False)
    parser.add_argument('--job_num',type=int, help="Input file name", required=True)
    # parser.add_argument('--output', action="store", dest="outputFn", type=str, help="Output file name", required=True)

    return parser.parse_args(argList)


def main():
    args = doArgs(sys.argv[1:])
    count = 0
    job_num = args.job_num

    histo_count = {}
    folders = '/yuch_ws/zero123/objaverse-rendering/views_shape'
    good_path = '/yuch_ws/zero123/zero123/good_samples.json'

    with open(good_path,'r') as f:
        folder_list = json.load(f)

    for folder in folder_list:
        histo_count[folder] = 0

    start_idx = job_num* 10
    end_idx = (job_num+1)* 10
    print("\n\n\n\n\n\n\n\n********** start idx end idx are ", start_idx,end_idx )

    folder_list = folder_list[start_idx:end_idx]

    for folder in folder_list:

        img_path = folders + '/' + folder + '/' + 'canny.png'
        prompt_path = folders + '/' + folder + '/' + 'BLIP_best_text_v2.txt'

        with open(prompt_path, 'r') as f:
            prompt = f.readline()
        work_space = 'results/control3d_' + prompt + '_'+ folder + '_'+ str(histo_count[folder])
        iters = 10000



        # train

        train_cmd = ('python3 main.py -O --image '+ img_path + ' --workspace ' + work_space+ ' --iters ' + str(iters) +
               ' --control_text ' + prompt + ' --zero123_ckpt pretrained/zero123/control_3d.ckpt --save_guidance --save_guidance_interval 10')

        print('****** Train cmd is ', train_cmd)
        os.system(train_cmd)
        print("****** training done, start testing")
        # test

        test_cmd = 'python3 main.py -O --workspace ' + work_space + ' --test --save_mesh --zero123_ckpt pretrained/zero123/control_3d.ckpt --save_guidance --save_guidance_interval 10'
        print('****** Test cmd is ', test_cmd)
        os.system(test_cmd)

if __name__ == '__main__':
    #sys.argv = ["programName.py","--input","test.txt","--output","tmp/test.txt"]
    main()


