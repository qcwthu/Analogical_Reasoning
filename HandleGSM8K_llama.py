import os
import sys
import argparse
import random
import numpy as np
from loguru import logger
import json
from openai import OpenAI


import time
import re

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument('--data_root', type=str, default='data')
    parser.add_argument("--task", type=str, default="gsm8k", choices=["gsm8k"], help="task used for experiment")
    parser.add_argument('--api_key', type=str, default='EMPTY')
    parser.add_argument("--subset", type=int, default=1, help="whether sample subset. 1: sample, other: not sample")
    parser.add_argument("--samplenum", type=int, default=500, help="sample number")
    parser.add_argument('--self_generate_type', type=str, default='relevant', choices=["relevant", "random_math", "random_any", "random_na", "random_any_nomath", "random_abcd", "strange_string", "random_any_nomath_nonumber", "zero"], help="which type of self-generated examples to use")
    parser.add_argument('--model', type=str, default='llama-2-70b', help="which model to use")
    parser.add_argument("--sleeptime", type=float, default=1.0, help="sleep time after each query")
    args = parser.parse_args()
    return args

def get_data(args):
    if args.task == 'gsm8k':
        data_root = args.data_root
        testdata_path = os.path.join(data_root, 'gsm8k', 'test.jsonl')
        testdata = []
        testlabel = []
        ###read json file line by line
        with open(testdata_path, 'r') as f:
            for line in f:
                onedata = json.loads(line)
                testdata.append(onedata["question"])
                testlabel.append(onedata["answer"].split("#### ")[-1])
        return testdata, testlabel
    else:
        raise ValueError('task not supported: {}'.format(args.task))

def get_response(alltestdata, alltestlabel, args):
    logger.info("get response")
    start = time.time()
    client = OpenAI(api_key=args.api_key, base_url="http://localhost:8000/v1")
    sleeptime = args.sleeptime
    answerpath = args.answerpath

    count = len(alltestdata)
    logger.info("number of data: {}", count)
    allflag = [0 for i in range(count)]  ####the number of all samples

    if not os.path.exists(answerpath):
        print("no response file! create now")
        f = open(answerpath, "w+")
        f.close()
    else:
        with open(answerpath, "r") as f:
            for oneline in f:
                onedata = oneline.strip().split("\t")
                if len(onedata) != 5:
                    continue
                thisindex = int(onedata[0])
                allflag[thisindex] = 1

    logger.info("sum(allflag) {}, len(allflag) {}", sum(allflag), len(allflag))
    if sum(allflag) == len(allflag):
        print("finished response!")
        return

    while True:
        fw = open(answerpath, "a+")
        for i in range(len(alltestdata)):
            # if i < 498:
            #     allflag[i] = 1
            #     continue
            if allflag[i] == 1:
                continue
            fullprompt = args.self_generate_prompt.replace("&&&&&&&&&&&&", alltestdata[i])
            # logger.info(fullprompt)
            # exit -1
            try:
                responsefromgpt = client.chat.completions.create(model=args.model,
                messages=[
                    {"role": "user", "content": fullprompt},
                ],
                temperature=0,
                stop=None)
            except Exception as exc:
                logger.info(exc)
                time.sleep(sleeptime)
                continue
            time.sleep(sleeptime)
            # print(responsefromgpt)
            oneresponse = responsefromgpt.choices[0].message.content.replace('\n', '&&&&&&')
            # print(oneresponse)
            # exit -1
            ###get the answer
            onepred = oneresponse.split("\\boxed{")[-1].split("}")[0]
            onepred = onepred.replace(",", "")
            oneanswerlist = [s for s in re.findall(r'-?\d+\.?\d*', onepred)]
            if len(oneanswerlist) > 0:
                oneanswer = oneanswerlist[0]
            else:
                oneanswer = '-1234560.789'
            allflag[i] = 1
            #print(i, alltestlabel[i], oneanswer, onepred, oneresponse)
            print(i, alltestlabel[i], oneanswer, onepred)
            fw.write(str(i) + "\t" + alltestlabel[i] + "\t" + oneanswer + "\t" + onepred + "\t" + oneresponse.replace("\t"," ") + "\n")
            fw.flush()
        fw.close()
        iffinish = True
        for oneflag in allflag:
            if oneflag == 0:
                iffinish = False
                break

        if iffinish:
            break
    end = time.time()
    print("all used time: ", end - start)

def getaccuracy_new(path, args):
    f = open(path, 'r')
    allnum = 0
    accnum = 0
    while True:
        oneline = f.readline().strip()
        if not oneline:
            break
        content = oneline.split('\t')
        if len(content) != 5:
            continue
        solution = content[4].replace('&&&&&&', '\n')
        if "Solve the Initial Problem" in solution:
            solution = solution.split("Solve the Initial Problem")[-1]
        if "\\boxed{" in solution:
            ####for python HandleGSM8K.py --self_generate_type random_any_nomath_nonumber --seed 1000
            if args.self_generate_type == 'random_any_nomath_nonumber' and args.seed == 1000:
                if "}" in solution:
                    solution = solution.split("\\boxed{")[-1].split("}")[0]
                else:
                    solution = solution.split("\\boxed{")[-1]
            else:
                assert "}" in solution
                solution = solution.split("\\boxed{")[-1].split("}")[0]
            
        solution = solution.replace(",", "")
        answerlist = [s for s in re.findall(r'-?\d+\.?\d*', solution)]
        if len(answerlist) > 0:
            #pred = answerlist[0]    ####0 or -1?
            pred = answerlist[-1]
        else:
            pred = '-1234560.789'
        
        allnum += 1
        gold = float(content[1].replace(",", ""))

        if pred[-1] == ".":
            pred = pred[:-1]
        if gold == float(pred):
            accnum += 1
        else:
            print(gold, float(pred))
    f.close()
    print(f"final {args.task} {args.model} {args.self_generate_type} {args.seed} {args.sample_num} result")
    print(accnum, allnum, float(accnum) / float(allnum))

if __name__ == '__main__':
    args = parse_args()
    logger.info(args)
    seed_everything(args.seed)

    ####get data
    alltestdata, alltestlabel = get_data(args)

    ####determine whether to sample test samples for experiments
    if args.subset == 1:
        sample_num = args.samplenum
        all_test_index = [i for i in range(len(alltestdata))]
        sampled_test_index = random.sample(all_test_index, min(sample_num, len(alltestdata)))
        print(len(sampled_test_index))
        sampled_test_data = [alltestdata[i] for i in sampled_test_index]    ####used for testing
        sampled_test_label = [alltestlabel[i] for i in sampled_test_index]
        args.sample_num = str(sample_num)
    else:
        all_test_index = [i for i in range(len(alltestdata))]
        sampled_test_index = random.sample(all_test_index, len(all_test_index))
        sampled_test_data = [alltestdata[i] for i in sampled_test_index]  ####used for testing
        sampled_test_label = [alltestlabel[i] for i in sampled_test_index]
        args.sample_num = "full"
    logger.info("len(sampled_test_data) {}", len(sampled_test_data))

    ####define prompt for generating in-context learning examples
    if args.task == 'gsm8k':
        if args.self_generate_type == 'relevant':
            args.self_generate_prompt = f'Your task is to tackle mathematical problems. When presented with a math problem, recall relevant problems as examples. Afterward, proceed to solve the initial problem.\n\n# Initial Problem:\n&&&&&&&&&&&&\n\n# Instructions:\nMake sure that your response follows the instructions below.\n\n## Analogous Problems:\nOffer five diverse examples of math problems that are relevant or analogous to the initial problem. For each problem, elaborate on the solution and conclude with the ultimate answer (enclosed in \\boxed{{}}). For each problem:\n- After "Q: ", describe the problem\n- After "A: ", explain the solution and enclose the ultimate answer in \\boxed{{}}.\n\n## Solve the Initial Problem:\nQ: Copy and paste the initial problem here.\nA: Explain the solution and enclose the ultimate answer in \\boxed{{}} here.\n'
        elif args.self_generate_type == 'zero':
            args.self_generate_prompt = f'Your task is to tackle mathematical problems.\n\n# Problem:\n&&&&&&&&&&&&\n\n# Instructions:\nMake sure that your response follows the instructions below.\n\nA: Explain the solution and enclose the ultimate answer in \\boxed{{}} here.\n'
        elif args.self_generate_type == 'random_na':
            args.self_generate_prompt = f'Your task is to tackle mathematical problems. When presented with a math problem, recall n/a problems as examples. Afterward, proceed to solve the initial problem.\n\n# Initial Problem:\n&&&&&&&&&&&&\n\n# Instructions:\nMake sure that your response follows the instructions below.\n\n## N/A Problems:\nOffer five diverse examples of math problems that are n/a to the initial problem. For each problem, elaborate on the solution and conclude with the ultimate answer (enclosed in \\boxed{{}}). For each problem:\n- After "Q: ", describe the problem\n- After "A: ", explain the solution and enclose the ultimate answer in \\boxed{{}}.\n\n## Solve the Initial Problem:\nQ: Copy and paste the initial problem here.\nA: Explain the solution and enclose the ultimate answer in \\boxed{{}} here.\n'
        elif args.self_generate_type == 'random_abcd':
            args.self_generate_prompt = f'Your task is to tackle mathematical problems. When presented with a math problem, recall abcd problems as examples. Afterward, proceed to solve the initial problem.\n\n# Initial Problem:\n&&&&&&&&&&&&\n\n# Instructions:\nMake sure that your response follows the instructions below.\n\n## ABCD Problems:\nOffer five diverse examples of math problems that are abcd to the initial problem. For each problem, elaborate on the solution and conclude with the ultimate answer (enclosed in \\boxed{{}}). For each problem:\n- After "Q: ", describe the problem\n- After "A: ", explain the solution and enclose the ultimate answer in \\boxed{{}}.\n\n## Solve the Initial Problem:\nQ: Copy and paste the initial problem here.\nA: Explain the solution and enclose the ultimate answer in \\boxed{{}} here.\n'
        elif args.self_generate_type == 'strange_string':
            args.self_generate_prompt = f'Your task is to tackle mathematical problems. When presented with a math problem, recall vhu%5sfjw9@ problems as examples. Afterward, proceed to solve the initial problem.\n\n# Initial Problem:\n&&&&&&&&&&&&\n\n# Instructions:\nMake sure that your response follows the instructions below.\n\n## vhu%5sfjw9@ Problems:\nOffer five diverse examples of math problems that are vhu%5sfjw9@ to the initial problem. For each problem, elaborate on the solution and conclude with the ultimate answer (enclosed in \\boxed{{}}). For each problem:\n- After "Q: ", describe the problem\n- After "A: ", explain the solution and enclose the ultimate answer in \\boxed{{}}.\n\n## Solve the Initial Problem:\nQ: Copy and paste the initial problem here.\nA: Explain the solution and enclose the ultimate answer in \\boxed{{}} here.\n'
        elif args.self_generate_type == 'random_math':
            args.self_generate_prompt = f'Your task is to tackle mathematical problems. When presented with a math problem, recall random problems as examples. Afterward, proceed to solve the initial problem.\n\n# Initial Problem:\n&&&&&&&&&&&&\n\n# Instructions:\nMake sure that your response follows the instructions below.\n\n## Random Problems:\nRandomly offer five diverse examples of math problems. For each problem, elaborate on the solution and conclude with the ultimate answer (enclosed in \\boxed{{}}). For each problem:\n- After "Q: ", describe the problem\n- After "A: ", explain the solution and enclose the ultimate answer in \\boxed{{}}.\n\n## Solve the Initial Problem:\nQ: Copy and paste the initial problem here.\nA: Explain the solution and enclose the ultimate answer in \\boxed{{}} here.\n'
        elif args.self_generate_type == 'random_any':
            args.self_generate_prompt = f'Your task is to tackle mathematical problems. When presented with a math problem, recall random problems as examples. Afterward, proceed to solve the initial problem.\n\n# Initial Problem:\n&&&&&&&&&&&&\n\n# Instructions:\nMake sure that your response follows the instructions below.\n\n## Random Problems:\nRandomly offer five diverse examples of any type of problems. For each problem, elaborate on the solution and conclude with the ultimate answer (enclosed in \\boxed{{}}). For each problem:\n- After "Q: ", describe the problem\n- After "A: ", explain the solution and enclose the ultimate answer in \\boxed{{}}.\n\n## Solve the Initial Problem:\nQ: Copy and paste the initial problem here.\nA: Explain the solution and enclose the ultimate answer in \\boxed{{}} here.\n'
        elif args.self_generate_type == 'random_any_nomath':
            args.self_generate_prompt = f'Your task is to tackle mathematical problems. When presented with a math problem, recall random problems (remember not to output math problems) as examples. Afterward, proceed to solve the initial problem.\n\n# Initial Problem:\n&&&&&&&&&&&&\n\n# Instructions:\nMake sure that your response follows the instructions below.\n\n## Random Problems:\nRandomly offer five diverse examples of any type, except math problems. For each problem, elaborate on the solution and conclude with the ultimate answer (enclosed in \\boxed{{}}). For each problem:\n- After "Q: ", describe the problem\n- After "A: ", explain the solution and enclose the ultimate answer in \\boxed{{}}.\n\n## Solve the Initial Problem:\nQ: Copy and paste the initial problem here.\nA: Explain the solution and enclose the ultimate answer in \\boxed{{}} here.\n'
        elif args.self_generate_type == 'random_any_nomath_nonumber':
            args.self_generate_prompt = f'Your task is to tackle mathematical problems. When presented with a math problem, recall random biological problems (remember not to output math problems) as examples. Afterward, proceed to solve the initial problem.\n\n# Initial Problem:\n&&&&&&&&&&&&\n\n# Instructions:\nMake sure that your response follows the instructions below.\n\n## Random Problems:\nRandomly offer five diverse examples of biological problems (remember not to output math problems). For each problem, elaborate on the solution and conclude with the ultimate answer (enclosed in \\boxed{{}}). For each problem:\n- After "Q: ", describe the problem\n- After "A: ", explain the solution and enclose the ultimate answer in \\boxed{{}}.\n\n## Solve the Initial Problem:\nQ: Copy and paste the initial problem here.\nA: Explain the solution and enclose the ultimate answer in \\boxed{{}} here.\n'
        else:
            raise ValueError('self_generate_type not supported: {}'.format(args.self_generate_type))
    else:
        raise ValueError('task not supported: {}'.format(args.task))
    print(args.self_generate_prompt)
    # exit -1

    ####set output path
    args.answerpath = args.data_root + "/" + args.task + "/" + args.model.replace("/", "-") + "_" + args.self_generate_type + "_response_" + str(args.seed) + "_" + args.sample_num
    logger.info(args.answerpath)

    ####get response from args.model for each test sample
    get_response(sampled_test_data, sampled_test_label, args)

    ####calculate accuracy
    getaccuracy_new(args.answerpath, args)