import os
import json
import sys
import argparse
import shutil
from tqdm import tqdm
import random

def doArgs(argList):
    parser = argparse.ArgumentParser()

    #parser.add_argument('-v', "--verbose", action="store_true", help="Enable verbose debugging", default=False)
    parser.add_argument('--job_num',type=int, help="Input file name", required=True)
    # parser.add_argument('--output', action="store", dest="outputFn", type=str, help="Output file name", required=True)
    parser.add_argument('--color', type=str, help="whether to add color string", default=False)
    parser.add_argument('--exp_name', type=str, help="the experiment name", default=False)

    parser.add_argument('--total_num', type=int, help="whether to add color string", required=True)

    return parser.parse_args(argList)


def main():
    args = doArgs(sys.argv[1:])
    job_num = args.job_num
    color = args.color
    exp_name = args.exp_name
    color_list = ['black', 'white', 'red', 'green', 'yellow', 'blue', 'brown', 'orange', 'pink', 'purple' ,'grey','silver','golden']
    histo_count = {}
    folders = '/yuch_ws/zero123/objaverse-rendering/views_shape'
    good_path = '/yuch_ws/zero123/zero123/good_samples.json'

    total_num = args.total_num

    with open(good_path,'r') as f:
        folder_list = json.load(f)

    # with open('histo')
    for folder in folder_list:
        histo_count[folder] = total_num
    #
    # with open('hist_count.json','r') as f :
    #     histo_count = json.load(f)
    #

    start_idx = job_num* 10
    end_idx = (job_num+1)* 10
    print("\n\n\n\n\n\n\n\n********** start idx end idx are ", start_idx,end_idx )
    random.seed(10)
    random.shuffle(folder_list)
    folder_list = folder_list[start_idx:end_idx]

    for folder in folder_list:

        img_path = folders + '/' + folder + '/' + 'canny.png'
        prompt_path = folders + '/' + folder + '/' + 'BLIP_best_text_v2.txt'

        with open(prompt_path, 'r') as f:
            prompt = f.readline()

        # s= 1
        iters = 20000

        if color:
            color_id = random.randint(0,len(color_list)-1)

            work_space = 'results/control3d_' + color_list[color_id] + '_' + prompt + '_' + folder + '_' + exp_name + str(histo_count[folder])
            # histo_count[folder] += 1

            prompt = color_list[color_id] + ' ' + prompt

            # train
            train_cmd = ('python3 main.py -O --image ' + img_path + ' --workspace ' + work_space + ' --iters ' + str(
                iters) +
                         ' --control_text ' +'\"'+ prompt + '\"' + ' --zero123_ckpt pretrained/zero123/control_3d.ckpt --save_guidance --save_guidance_interval 100 --eval_interval 5')

            print('****** Train cmd is ', train_cmd)
            os.system(train_cmd)
            print("****** training done, start testing")
            # test

            test_cmd = 'python3 main.py -O --workspace ' + work_space + ' --test --save_mesh --zero123_ckpt pretrained/zero123/control_3d.ckpt --save_guidance --save_guidance_interval 100 --eval_interval 5'
            print('****** Test cmd is ', test_cmd)
            os.system(test_cmd)
        else:

            work_space = 'results/control3d_' + prompt + '_'+ folder +  '_' + exp_name + str(histo_count[folder])
            # train
            train_cmd = ('python3 main.py -O --image '+ img_path + ' --workspace ' + work_space+ ' --iters ' + str(iters) +
                   ' --control_text ' + prompt + ' --zero123_ckpt pretrained/zero123/control_3d.ckpt --save_guidance --save_guidance_interval 100 --eval_interval 5')

            print('****** Train cmd is ', train_cmd)
            os.system(train_cmd)
            print("****** training done, start testing")
            # test

            test_cmd = 'python3 main.py -O --workspace ' + work_space + ' --test --save_mesh --zero123_ckpt pretrained/zero123/control_3d.ckpt --save_guidance --save_guidance_interval 100 --eval_interval 5'
            print('****** Test cmd is ', test_cmd)
            os.system(test_cmd)
    # with open('hist_count.json','w') as f:
    #     json.dump(histo_count,f)

if __name__ == '__main__':
    #sys.argv = ["programName.py","--input","test.txt","--output","tmp/test.txt"]
    main()


