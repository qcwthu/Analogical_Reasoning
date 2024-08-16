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
    parser.add_argument("--task", type=str, default="bbh", choices=["bbh"], help="task used for experiment")
    parser.add_argument('--api_key', type=str, default='YOUR_OPENAI_API_KEY')
    parser.add_argument("--subset", type=int, default=0, help="whether sample subset. 1: sample, other: not sample")
    parser.add_argument("--samplenum", type=int, default=500, help="sample number")
    parser.add_argument('--self_generate_type', type=str, default='relevant', choices=["relevant", "random_reasoning", "random_na", "random_any_noreasoning", "random_abcd", "strange_string", "random_any_noreasoning_nonumber", "zero"], help="which type of self-generated examples to use")
    parser.add_argument('--model', type=str, default='gpt-3.5-turbo', help="which model to use")
    parser.add_argument("--sleeptime", type=float, default=1.0, help="sleep time after each query")
    args = parser.parse_args()
    return args

MULTIPLE_CHOICE_TASKS = [
        'temporal_sequences', 'disambiguation_qa', 'date_understanding', 'tracking_shuffled_objects_three_objects', 'penguins_in_a_table', 
        'geometric_shapes', 'snarks', 'ruin_names', 'tracking_shuffled_objects_seven_objects', 'tracking_shuffled_objects_five_objects', 
        'logical_deduction_three_objects', 'hyperbaton', 'logical_deduction_five_objects', 'logical_deduction_seven_objects', 'movie_recommendation', 
        'salient_translation_error_detection', 'reasoning_about_colored_objects', 
]
FREE_FORM_TASKS = [
        'multistep_arithmetic_two', 'navigate', 'dyck_languages', 'word_sorting', 'sports_understanding', 
        'boolean_expressions', 'object_counting', 'formal_fallacies', 'causal_judgement', 'web_of_lies', 
]
# TO_TEST_TASKS = [
#         'word_sorting', 'logical_deduction_five_objects', 'temporal_sequences', 'reasoning_about_colored_objects', 'formal_fallacies'
# ]
TO_TEST_TASKS = [
        'temporal_sequences',
        'logical_deduction_five_objects',
        'reasoning_about_colored_objects',
        'formal_fallacies',
        'word_sorting'
]

def get_data(args):
    if args.task == 'bbh':
        testdata = {}
        testlabel = {}
        data_root = args.data_root
        testdata_path = os.path.join(data_root, 'bbh')
        for task in TO_TEST_TASKS:
            testdata[task] = []
            testlabel[task] = []
            #print('loading %s ...' % task)
            all_data = json.load(open(f'{testdata_path}/{task}.json'))['examples']
            for one_data in all_data:
                testdata[task].append(one_data['input'])
                testlabel[task].append(one_data['target'])
        return testdata, testlabel
    else:
        raise ValueError('task not supported: {}'.format(args.task))

def extract_ans(ans, mode):
    onepred = ans.replace('&&&&&&', '\n')
    if "Solve the Initial Problem" in onepred:
        onepred = onepred.split("Solve the Initial Problem")[-1]
    if "\\boxed{" in onepred:
        onepred = onepred.split("\\boxed{")[-1]
        curindex = -100
        for idx in range(len(onepred)):
            onechar = onepred[idx]
            if onechar == "{":
                curindex = -1
            if onechar == "}":
                if curindex == -1:
                    curindex = idx
                else:
                    curindex = idx
                    break
        onepred = onepred[0:curindex]
    else:
        return onepred.replace("\n", " ")
    
    if mode == 'multiple_choice':
        options = ['(A)', '(B)', '(C)', '(D)', '(E)', '(F)', '(G)', '(H)', '(I)', '(J)', '(K)', '(L)', '(M)', '(N)', '(O)', '(P)', '(Q)', '(R)', '(S)', '(T)', '(U)', '(V)', '(W)', '(X)', '(Y)', '(Z)']
        for option in options:
            if option in onepred:
                onepred = option[1]
                break
        return onepred
    elif mode == 'free_form':
        if onepred[-1] == '.':
            onepred = onepred[:-1]
        return onepred

def get_response(alltestdata, alltestlabel, args):
    start = time.time()
    client = OpenAI(api_key=args.api_key)
    sleeptime = args.sleeptime
    for onetask in alltestdata.keys():
        logger.info("get response for task {}", onetask)
        answerpath = args.answerpath + "/" + onetask

        count = len(alltestdata[onetask])
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
            logger.info("finished response for task {}", onetask)
            continue

        while True:
            fw = open(answerpath, "a+")
            for i in range(len(alltestdata[onetask])):
                # if i > 1000:
                #     allflag[i] = 1
                #     continue
                if allflag[i] == 1:
                    continue
                fullprompt = args.self_generate_prompt.replace("&&&&&&&&&&&&", alltestdata[onetask][i])
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
                oneresponse = responsefromgpt.choices[0].message.content.replace('\n', '&&&&&&')
                ###get the answer
                if onetask in MULTIPLE_CHOICE_TASKS:
                    mode = 'multiple_choice'
                elif onetask in FREE_FORM_TASKS:
                    mode = 'free_form'
                onepred = extract_ans(oneresponse, mode)
                oneanswer = onepred
                if mode == 'multiple_choice':
                    gold = alltestlabel[onetask][i][1]
                elif mode == 'free_form':
                    gold = alltestlabel[onetask][i]
                allflag[i] = 1
                #print(i, gold, oneanswer, onepred, oneresponse)
                #exit -1
                print(i, gold, oneanswer, onepred)
                fw.write(str(i) + "\t" + gold + "\t" + oneanswer + "\t" + onepred + "\t" +  oneresponse.replace("\t"," ") + "\n")
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

num_to_word = {
    '0': 'zero', '1': 'one', '2': 'two', '3': 'three', '4': 'four', '5': 'five',
    '6': 'six', '7': 'seven', '8': 'eight', '9': 'nine', '10': 'ten',
    '11': 'eleven', '12': 'twelve', '13': 'thirteen', '14': 'fourteen', 
    '15': 'fifteen', '16': 'sixteen', '17': 'seventeen', '18': 'eighteen',
    '19': 'nineteen', '20': 'twenty'
}

def getaccuracy_new(path, sampled_test_data, args):
    for onetask in sampled_test_data.keys():
        if onetask in MULTIPLE_CHOICE_TASKS:
            mode = 'multiple_choice'
        elif onetask in FREE_FORM_TASKS:
            mode = 'free_form'
        taskpath = path + "/" + onetask
        f = open(taskpath, 'r')
        allnum = 0
        accnum = 0
        while True:
            oneline = f.readline().strip()
            if not oneline:
                break
            content = oneline.split('\t')
            if len(content) != 5:
                continue
            allnum += 1
            gold = content[1]
            if mode == 'multiple_choice':
                oneidx = int(content[0])
                oneinput = sampled_test_data[onetask][oneidx]
                options = oneinput.split('Options:\n')[-1].split('\n')
                allchoices = {}
                for oneoption in options:
                    allchoices[oneoption[1]] = oneoption[4:]
                gold_mc = allchoices[gold]
            pred = content[2]
            ####handle "\text{}" for logical_deduction_five_objects
            if onetask == 'logical_deduction_five_objects':
                if "\\text{" in pred:
                    pred = pred.replace("\\text{", "").replace("}", "")
                if gold == pred:
                    accnum += 1
                else:
                    if len(pred) > 1:
                        if '\\' in pred:
                            pred = pred.replace('\\', '')
                        if pred.lower() in gold_mc.lower():
                            #print(pred, gold_mc)
                            accnum += 1
                        else:
                            print(gold, pred, gold_mc, "  aaaa  ")
                    else:
                        print(gold, pred, gold_mc, "  bbbb  ")
            elif onetask == 'reasoning_about_colored_objects':
                if "\\text{" in pred:
                    pred = pred.replace("\\text{", "").replace("}", "")
                if '"' in pred:
                    pred = pred.replace('"', '')
                if gold == pred or gold_mc.lower() == pred.lower():
                    accnum += 1
                else:
                    if pred in num_to_word.keys():
                        if num_to_word[pred] == gold_mc.lower():
                            #print(pred, num_to_word[pred], gold_mc)
                            accnum += 1
                        else:
                            print(gold, pred, gold_mc, "  aaaa  ")
                    else:
                        print(gold, pred, gold_mc, "  bbbb  ")
            elif onetask == 'temporal_sequences':
                if "\\text{" in pred:
                    pred = pred.replace("\\text{", "").replace("}", "")
                if gold == pred or gold_mc.lower() == pred.lower():
                    accnum += 1
                else:
                    gold_mc_handle_space = re.sub(r'\s+', '', gold_mc)
                    pred_handle_space = re.sub(r'\s+', '', pred)
                    if '\\' in gold_mc_handle_space:
                        gold_mc_handle_space = gold_mc_handle_space.replace('\\', '')
                    if '\\' in pred_handle_space:
                        pred_handle_space = pred_handle_space.replace('\\', '')
                    if gold_mc_handle_space.lower() == pred_handle_space.lower():
                        accnum += 1
                    else:
                        if '-' in gold_mc_handle_space:
                            gold_mc_handle_space = gold_mc_handle_space.replace('-', 'to')
                        if '-' in pred_handle_space:
                            pred_handle_space = pred_handle_space.replace('-', 'to')
                        if gold_mc_handle_space.lower() == pred_handle_space.lower():
                            accnum += 1
                        else:
                            ####one more type error: B (9pm, 10pm) 9pm to 10pm
                            print(gold, pred, gold_mc)
            elif onetask == 'formal_fallacies':
                if "\\text{" in pred:
                    pred = pred.replace("\\text{", "").replace("}", "")
                if gold.lower() == pred.lower():
                    accnum += 1
                else:
                    print(gold, "  &&&&  ", pred)
            elif onetask == 'word_sorting':
                if "\\text{" in pred:
                    pred = pred.replace("\\text{", "").replace("}", "")
                if "," in pred:
                    pred = pred.replace(",", "")
                if '"' in pred:
                    pred = pred.replace('"', '')
                if gold.lower() == pred.lower():
                    accnum += 1
                else:
                    print(gold, "  &&&&  ", pred)
            else:
                raise ValueError('task not supported: {}'.format(onetask))
        f.close()
        print(f"final {args.task} {onetask} {args.model} {args.self_generate_type} {args.seed} {args.sample_num} result")
        print(onetask, accnum, allnum, float(accnum) / float(allnum))

if __name__ == '__main__':
    args = parse_args()
    logger.info(args)
    seed_everything(args.seed)

    ####get data
    alltestdata, alltestlabel = get_data(args)

    ####determine whether to sample test samples for experiments
    if args.subset == 1:
        sample_num = args.samplenum
        sampled_test_data = {}
        sampled_test_label = {}
        for onetask in alltestdata.keys():
            all_test_index = [i for i in range(len(alltestdata[onetask]))]
            sampled_test_index = random.sample(all_test_index, min(sample_num, len(alltestdata[onetask])))
            print(len(sampled_test_index))
            sampled_test_data[onetask] = [alltestdata[onetask][i] for i in sampled_test_index]    ####used for testing
            sampled_test_label[onetask] = [alltestlabel[onetask][i] for i in sampled_test_index]
            args.sample_num = str(sample_num)
    else:
        sampled_test_data = {}
        sampled_test_label = {}
        for onetask in alltestdata.keys():
            all_test_index = [i for i in range(len(alltestdata[onetask]))]
            sampled_test_index = random.sample(all_test_index, len(all_test_index))
            sampled_test_data[onetask] = [alltestdata[onetask][i] for i in sampled_test_index]  ####used for testing
            sampled_test_label[onetask] = [alltestlabel[onetask][i] for i in sampled_test_index]
            args.sample_num = "full"
    for onetask in sampled_test_data.keys():
        logger.info("len(sampled_test_data[{}]) {}", onetask, len(sampled_test_data[onetask]))
    
    ####define prompt for generating in-context learning examples
    if args.task == 'bbh':
        if args.self_generate_type == 'relevant':
            args.self_generate_prompt = f'Your task is to tackle reasoning problems. When presented with a problem, recall relevant problems as examples. Afterward, proceed to solve the initial problem.\n\n# Initial Problem:\n&&&&&&&&&&&&\n\n# Instructions:\nMake sure to include all of the following points:\n\n## Relevant Problems:\nRecall three examples of problems that are relevant to the initial problem. Note that your problems must be distinct from each other and from the initial problem. For each problem:\n- After "Q: ", describe the problem\n- After "A: ", explain the solution and enclose the ultimate answer in \\boxed{{}}.\n\n## Solve the Initial Problem:\n' + 'Say "Let\'s solve the following reasoning problem." Then formulate your response in the following format:' + '\nQ: Copy and paste the initial problem here.\nA: Explain the solution and enclose the ultimate answer in \\boxed{} here.\n'
        elif args.self_generate_type == 'zero':
            args.self_generate_prompt = f'Your task is to tackle reasoning problems.\n\n# Problem:\n&&&&&&&&&&&&\n\n# Instructions:\nMake sure to include all of the following points:\n\nA: Explain the solution and enclose the ultimate answer in \\boxed{{}} here.\n'
        elif args.self_generate_type == 'random_na':
            args.self_generate_prompt = f'Your task is to tackle reasoning problems. When presented with a problem, recall n/a problems as examples. Afterward, proceed to solve the initial problem.\n\n# Initial Problem:\n&&&&&&&&&&&&\n\n# Instructions:\nMake sure to include all of the following points:\n\n## N/A Problems:\nRecall three examples of problems that are n/a to the initial problem. Note that your problems must be distinct from each other and from the initial problem. For each problem:\n- After "Q: ", describe the problem\n- After "A: ", explain the solution and enclose the ultimate answer in \\boxed{{}}.\n\n## Solve the Initial Problem:\n' + 'Say "Let\'s solve the following reasoning problem." Then formulate your response in the following format:' + '\nQ: Copy and paste the initial problem here.\nA: Explain the solution and enclose the ultimate answer in \\boxed{} here.\n'
        elif args.self_generate_type == 'random_abcd':
            args.self_generate_prompt = f'Your task is to tackle reasoning problems. When presented with a problem, recall abcd problems as examples. Afterward, proceed to solve the initial problem.\n\n# Initial Problem:\n&&&&&&&&&&&&\n\n# Instructions:\nMake sure to include all of the following points:\n\n## ABCD Problems:\nRecall three examples of problems that are abcd to the initial problem. Note that your problems must be distinct from each other and from the initial problem. For each problem:\n- After "Q: ", describe the problem\n- After "A: ", explain the solution and enclose the ultimate answer in \\boxed{{}}.\n\n## Solve the Initial Problem:\n' + 'Say "Let\'s solve the following reasoning problem." Then formulate your response in the following format:' + '\nQ: Copy and paste the initial problem here.\nA: Explain the solution and enclose the ultimate answer in \\boxed{} here.\n'
        elif args.self_generate_type == 'strange_string':
            args.self_generate_prompt = f'Your task is to tackle reasoning problems. When presented with a problem, recall vhu%5sfjw9@ problems as examples. Afterward, proceed to solve the initial problem.\n\n# Initial Problem:\n&&&&&&&&&&&&\n\n# Instructions:\nMake sure to include all of the following points:\n\n## vhu%5sfjw9@ Problems:\nRecall three examples of problems that are vhu%5sfjw9@ to the initial problem. Note that your problems must be distinct from each other and from the initial problem. For each problem:\n- After "Q: ", describe the problem\n- After "A: ", explain the solution and enclose the ultimate answer in \\boxed{{}}.\n\n## Solve the Initial Problem:\n' + 'Say "Let\'s solve the following reasoning problem." Then formulate your response in the following format:' + '\nQ: Copy and paste the initial problem here.\nA: Explain the solution and enclose the ultimate answer in \\boxed{} here.\n'
        elif args.self_generate_type == 'random_reasoning':
            args.self_generate_prompt = f'Your task is to tackle reasoning problems. When presented with a problem, recall random problems as examples. Afterward, proceed to solve the initial problem.\n\n# Initial Problem:\n&&&&&&&&&&&&\n\n# Instructions:\nMake sure to include all of the following points:\n\n## Random Problems:\nRandomly recall three examples of reasoning problems. Note that your problems must be distinct from each other and from the initial problem. For each problem:\n- After "Q: ", describe the problem\n- After "A: ", explain the solution and enclose the ultimate answer in \\boxed{{}}.\n\n## Solve the Initial Problem:\n' + 'Say "Let\'s solve the following reasoning problem." Then formulate your response in the following format:' + '\nQ: Copy and paste the initial problem here.\nA: Explain the solution and enclose the ultimate answer in \\boxed{} here.\n'
        elif args.self_generate_type == 'random_any_noreasoning':
            args.self_generate_prompt = f'Your task is to tackle reasoning problems. When presented with a problem, recall random problems (remember not to output reasoning problems) as examples. Afterward, proceed to solve the initial problem.\n\n# Initial Problem:\n&&&&&&&&&&&&\n\n# Instructions:\nMake sure to include all of the following points:\n\n## Random Problems:\nRandomly recall three examples of any type, except reasoning problems. Note that your problems must be distinct from each other and from the initial problem. For each problem:\n- After "Q: ", describe the problem\n- After "A: ", explain the solution and enclose the ultimate answer in \\boxed{{}}.\n\n## Solve the Initial Problem:\n' + 'Say "Let\'s solve the following reasoning problem." Then formulate your response in the following format:' + '\nQ: Copy and paste the initial problem here.\nA: Explain the solution and enclose the ultimate answer in \\boxed{} here.\n'
        elif args.self_generate_type == 'random_any_noreasoning_nonumber':
            args.self_generate_prompt = f'Your task is to tackle reasoning problems. When presented with a problem, recall random biological problems (remember not to output reasoning problems) as examples. Afterward, proceed to solve the initial problem.\n\n# Initial Problem:\n&&&&&&&&&&&&\n\n# Instructions:\nMake sure to include all of the following points:\n\n## Random Problems:\nRandomly recall three examples of biological problems (remember not to output reasoning problems). Note that your problems must be distinct from each other and from the initial problem. For each problem:\n- After "Q: ", describe the problem\n- After "A: ", explain the solution and enclose the ultimate answer in \\boxed{{}}.\n\n## Solve the Initial Problem:\n' + 'Say "Let\'s solve the following reasoning problem." Then formulate your response in the following format:' + '\nQ: Copy and paste the initial problem here.\nA: Explain the solution and enclose the ultimate answer in \\boxed{} here.\n'
        else:
            raise ValueError('self_generate_type not supported: {}'.format(args.self_generate_type))
    else:
        raise ValueError('task not supported: {}'.format(args.task))
    logger.info(args.self_generate_prompt)

    ####set output path
    args.answerpath = args.data_root + "/" + args.task + "/" + args.model + "_" + args.self_generate_type + "_response_" + str(args.seed) + "_" + args.sample_num
    logger.info(args.answerpath)
    if not os.path.exists(args.answerpath):
        os.makedirs(args.answerpath)

    ####get response from args.model for each test sample
    get_response(sampled_test_data, sampled_test_label, args)

    ####calculate accuracy
    getaccuracy_new(args.answerpath, sampled_test_data, args)
