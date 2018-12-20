#!/usr/bin/env python

# Copyright 2017 Pegah Ghahremani
#
# Apache 2.0.

""" This script generates examples for multilingual training of neural network.
    This scripts produces 3 sets of files --
    egs.*.scp, egs.output.*.ark, egs.weight.*.ark

---------------------------------------
    egs.*.scp are the SCP files of the training examples.

    egs.weight.*.ark map from the key of the example to the language-specific
    weight of that example.

    egs.output.*.ark map from the key of the example to the name of the output-node 
    in the neural net for that specific language, e.g.
    'output-2'.
---------------------------------------



    # 1 egs.ranges.*.txt
    This script additionally produces temporary files -- egs.ranges.*.txt,
    which are consumed by this script itself.

    Each line in egs.ranges.*.txt corresponds to ranges of examples
    There is one egs.ranges.*.txt file for each of the egs.*.scp files.

    selected from one of the input languages's scp files as:
    <lang> <local-scp-line> <num-examples>

    可以被解释为 选择 <num-example> 个样本, 
    从egs_scp_list 中的lang/egs.scp 中选择 num-example(minibatch_size) 个example.
    0 (lang1-id)  0 (start eg index) 256 (num-example)
    这样就在一次minibatch中就会只有一个任务， 而不同的minibatch就会是不同的训练任务，进而实现多语种同时训练。

    That can be interpreted as selecting <num-example> examples starting from
    <local-scp-line> line from {lang}_th 'egs' file in "egs_scp_list".
    (note that <local-scp-line> is the zero-based line number.)

    Example lines might look like:
    0 0 256
    2 1024 256

    根据egs.ranges.*.txt的描述 生成egs.*.scp, 
    从某个lang的scp-file中 从<local-scp-line> 中选择连续的<num-examples>个样本cp-> egs.*.scp
    貌似并没有使用egs.ranges.*.txt， 而是根据all_egs生成的.
    从 egs_scp_list 中选择<lang-id> egs.scp 文件做源文件
    然后egs.ranges.*.scp 就描述了<num-examples>个egs -> egs.*.scp
    egs.*.scp is generated using egs.ranges.*.txt as following:
    "<num-examples>" consecutive examples starting from line "<local-scp-line>"
    from {lang}_th input scp-file is copied to egs.*.scp.


    --egs-prefix option can be used to generate train and diagnostics egs files.
    If --egs-prefix=train_diagnostics. is passed, then the files produced by the
    script will be named with the prefix as "train_diagnostics."
    instead of "egs."
    i.e. the files produced are -- train_diagnostics.*.scp,
    train_diagnostics.output.*.ark, train_diagnostics.weight.*.ark and
    train_diagnostics.ranges.*.txt.
    The other egs-prefix options used in the recipes are "valid_diagnositics."
    for validation examples and "combine." for examples used for model
    combination.


    # Usage:
    You can call this script as (e.g.):
    allocate_multilingual_examples.py [opts] example-scp-lists
        multilingual-egs-dir

    allocate_multilingual_examples.py --minibatch-size 128
        --lang2weight  "0.2,0.8" exp/lang1/egs.scp exp/lang2/egs.scp  exp/multi/egs

    To avoid loading whole scp files from all languages in memory,
    input egs.scp files are processed line by line using readline() for input
    languages. 


    为了实现更多的随机交叉多个语种, 需要生成临时的 scp.<job>.<archive_index> 文件, 
    然后合并<jobs> 最终实现多个egs.<archives>.scp
    To have more randomization across different archives,
    "num-jobs * num-archives" temporary scp.<job>.<archive_index> files are created
    in egs/temp dir and all "num_jobs" scp.*.<archive_index> combined into
    egs.<archive_index>.scp.
"""

from __future__ import print_function
import os, argparse, sys, random
import logging
import traceback

sys.path.insert(0, 'steps')
import libs.common as common_lib

logger = logging.getLogger('libs')
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s [%(pathname)s:%(lineno)s - "
                              "%(funcName)s - %(levelname)s ] %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.info('Start generating multilingual examples')


def get_args():

    parser = argparse.ArgumentParser(
        description=""" This script generates examples for multilingual training
        of neural network by producing 3 sets of primary files
        as egs.*.scp, egs.output.*.ark, egs.weight.*.ark.
        egs.*.scp are the SCP files of the training examples.
        egs.weight.*.ark map from the key of the example to the language-specific
        weight of that example.
        egs.output.*.ark map from the key of the example to the name of
        the output-node in the neural net for that specific language, e.g.
        'output-2'.""",
        epilog="Called by steps/nnet3/multilingual/combine_egs.sh")

    parser.add_argument("--samples-per-iter", type=int, default=40000,
                        help="The target number of egs in each archive of egs, "
                        "(prior to merging egs). ")
    parser.add_argument("--num-jobs", type=int, default=20,
                        help="This can be used for better randomization in distributing "
                        "examples for different languages across egs.*.scp files, "
                        "where egs.<job>.*.scp are generated "
                        "randomly and combined across all jobs in egs.*.scp files.")
    parser.add_argument("--random-lang", type=str, action=common_lib.StrToBoolAction,
                        help="If true, egs.ranges.*.txt are generated "
                        "randomly w.r.t distribution of remaining examples in "
                        "each language, otherwise it is generated sequentially.",
                        default=True, choices = ["false", "true"])
    parser.add_argument("--max-archives", type=int, default=1000,
                        help="max number of archives used to generate egs.*.scp")
    parser.add_argument("--seed", type=int, default=1,
                        help="Seed for random number generator")
    parser.add_argument("--minibatch-size", type=int, default=512,
                        help="It is the number of consecutive egs that is taken "
                        "from each input scp source, and it only affects locality "
                        "of disk access. This does not have to be actual minibatch size.")
    parser.add_argument("--egs-prefix", type=str, default="egs.",
                        help="option can be used to generated example scp, weight "
                        "and output files for training and diagnostics."
                        "If --egs-prefix=combine. , then files produced "
                        "by the sript will be named with this prefix as "
                        "combine.output.*.ark, combine.weight.*.ark, combine.*.scp, "
                        "combine.ranges.*.ark.")
    parser.add_argument("--lang2weight", type=str,
                        help="comma-separated list of weights, one per language."
                        "The language order is as egs_scp_lists.")
# now the positional arguments
    parser.add_argument("egs_scp_lists", nargs='+',
                        help="list of egs.scp files per input language."
                           "e.g. exp/lang1/egs/egs.scp exp/lang2/egs/egs.scp")
    parser.add_argument("egs_dir",
                        help="Name of egs directory e.g. exp/tdnn_multilingual_sp/egs")


    print(sys.argv, file=sys.stderr)
    args = parser.parse_args()

    return args


def select_random_lang(lang_len, tot_egs, random_selection):
    """ Returns a random language index w.r.t
        amount of examples in each language.
        It works based on sampling from a
        discrete distribution, where it returns i
        with prob(i) = (num_egs in lang(i)/ tot_egs).
        tot_egs is sum of lang_len.
    """
    assert(tot_egs > 0)
    rand_int = random.randint(0, tot_egs - 1)
    count = 0
    # random select a right lang. and return the random lang.
    for l in range(len(lang_len)):
        if random_selection:
            if  rand_int <= (count + lang_len[l]):
                return l
            else:
                count += lang_len[l]
        else:
            if (lang_len[l] > 0):
                return l
    return -1


def process_multilingual_egs(args):
    args = get_args()
    random.seed(args.seed)
    rand_select = args.random_lang

    # read input languages egs.scp
    scp_lists = args.egs_scp_lists
    num_langs = len(scp_lists)
    scp_files = [open(scp_lists[lang], 'r') for lang in range(num_langs)]

    # each lang's egsCnt
    lang2len = [0] * num_langs
    for lang in range(num_langs):
        lang2len[lang] = sum(1 for line in open(scp_lists[lang]))
        logger.info("Number of examples for language {0} "
                    "is {1}.".format(lang, lang2len[lang]))

    # If weights are not provided, the weights are 1.0.
    if args.lang2weight is None:
        lang2weight = [ 1.0 ] * num_langs
    else:
        lang2weight = args.lang2weight.split(",")
        assert(len(lang2weight) == num_langs)

    # create tempdir
    if not os.path.exists("{0}/temp".format(args.egs_dir)):
        os.makedirs("{0}/temp".format(args.egs_dir))
    # 实际上就是多任务训练方法， 并没有特别特殊的. num_tasks
    num_lang_file = open("{0}/info/{1}num_tasks".format(args.egs_dir, args.egs_prefix), "w")
    print("{0}".format(num_langs), file=num_lang_file)

    # Each element of all_egs is an array of 3-tuples
    # (lang-id, local-start-egs-line, num-egs)
    all_egs = []
    # each lang's egs cnt, 
    # the lang2len never change, lang2len 
    #     就是保存每个语种的egs数量
    # the lang_len will be consumed, when the combine going
    #     lang_len 是在描述，剩余未进行组合的每个语种的egs数量
    lang_len = lang2len[:]

    #  ----------- archive cnt, egs cnt
    #              归档文件总数， egs 总数
    # total num of egs in all languages
    tot_num_egs = sum(lang2len[i] for i in range(len(lang2len)))
    # tot_num_egs = samples_per_iter * num_archives
    num_archives = max(1, min(args.max_archives, tot_num_egs / args.samples_per_iter))
    num_arch_file = open("{0}/info/{1}num_archives".format(
                            args.egs_dir,
                            args.egs_prefix),
                         "w")
    print("{0}".format(num_archives), file=num_arch_file)
    num_arch_file.close()
    
    # ------ 每个archive里每个job的egs总数
    # tot_num_egs = samples_per_iter * num_archives
    # tot_num_egs = num_jobs * this_num_egs_per_archive * num_archives
    # ==> samples_per_iter = num_jobs * this_num_egs_per_archives, 
    # 一个 archive， 是一次迭代消耗的egs 归档， 而一次迭代，是多个job 并行。
    # 一个archives中有 this_num_egs_per_archives * num_jobs
    # 这就是上面说的，为了增加随机性， 使用一个临时无用的job 概念，对archive 进行再划分.
    this_num_egs_per_archive = tot_num_egs / (num_archives * args.num_jobs)

    logger.info("Generating {0}scp.<job>.<archive_index> temporary files used to "
                "generate {0}<archive_index>.scp.".format(args.egs_prefix))

    for job in range(args.num_jobs):
        for archive_index in range(num_archives):

            # egsscp.job-id.archive-id
            # one archive file --- egscp.job.archive.
            # at then end , will combine the jobs egsscp.job.archive => egsscp.archive.
            archfile = open("{0}/temp/{1}scp.{2}.{3}"
                            "".format(args.egs_dir, args.egs_prefix,
                                      job + 1, archive_index + 1),
                            "w")
            # this_egs 保存的是minibatch的描述。
            # 一个archive 
            #     n x job  (job== this_num_egs_per_archive)
            #         n x minibatch
            this_egs = [] # this will be array of 3-tuples (lang-id start-frame num-frames)

            num_egs = 0
            # 累计存入的egs数量， 每个job 在每个archive上应该保存的egs数量
            while num_egs <= this_num_egs_per_archive:
                # all egs.
                num_left_egs = sum(num_left_egs_per_lang for
                                   num_left_egs_per_lang in lang_len)
                if num_left_egs > 0:
                    # random select a lang.
                    # 随机选择一个语种
                    lang_id = select_random_lang(lang_len, num_left_egs, rand_select)
                    # 从选择语种中，取得该语种剩余的egs(lang_len)开始点
                    # from the selected lang , select start_eg_index.
                    start_egs = lang2len[lang_id] - lang_len[lang_id]
                    
                    # this_egs    
                    #        -- job 内minibatch的描述。 minibatch_per_job* minibatch_size 个egs
                    # 内部
                    # ---    minibatch-0     -- lang_id  start_pos  minibatch_size (examples)
                    # ---    minibatch-1
                    # ---    minibatch-2
                    #           
                    this_egs.append((lang_id, start_egs, args.minibatch_size))
                    
                    # 从选择语种的scp_file 中逐行取得<eg-name save-postion> 
                    # 写入archfile(临时 egsscp.job.archive文件)
                    # from the scp_files[lang_id] read minibatch write to archfile.
                    for scpline in range(args.minibatch_size):
                        scp_key = scp_files[lang_id].readline().splitlines()[0]
                        print("{0} {1}".format(scp_key, lang_id),
                              file=archfile)
                    
                    # 更新 lang_len, 该语种剩余的数量逐渐减少， 下一次选择egs的开始点向后。
                    lang_len[lang_id] = lang_len[lang_id] - args.minibatch_size
                    num_egs = num_egs + args.minibatch_size

                    # 如果该语种剩余的数量 小于minibatch， 直接不要了。
                    # If num of remaining egs in each lang is less than minibatch_size,
                    # they are discarded.
                    if lang_len[lang_id] < args.minibatch_size:
                        lang_len[lang_id] = 0
                        logger.info("Done processing data for language {0}".format(
                            lang_id))
                else:
                    logger.info("Done processing data for all languages.")
                    break

            # all_egs 保存 this_egs
            # 是个 二级list
            # 
            # 即 
            # 保存了 num_archives * job 个 对job内的描述。
            # [
            # ---  archive-0 job-0  [this_egs]
            # ---  archive-0 job-1  [this_egs]
            # ---  archive-0 job-2  [this_egs]
            # ---  archive-1 job-0  [this_egs]
            # ---  archive-2 job-1  [this_egs]
            # ---  archive-3 job-2  [this_egs]
            # ]
            # .....
            all_egs.append(this_egs)
            archfile.close()

    # 下面 组合为了增加随机性 而引入的job域, 最终保持为 num_archives个egs.*.scp
    logger.info("combining egs.<job>.*.scp across all jobs into egs.*.scp file.")
    for archive in range(num_archives):
        logger.info("Combine {0}job.{1}.scp across all jobs into "
                    "{0}{1}.scp.".format(args.egs_prefix, archive))
        this_ranges = []
        # ranges
        f = open("{0}/temp/{1}ranges.{2}.txt".format(
                    args.egs_dir, args.egs_prefix, archive + 1),
                 'w')
        # output, <eg-name  output-node>
        o = open("{0}/{1}output.{2}.ark".format(
                    args.egs_dir, args.egs_prefix, archive + 1),
                 'w')
        # weight
        w = open("{0}/{1}weight.{2}.ark".format(
                    args.egs_dir, args.egs_prefix, archive + 1),
                 'w')
        # egs.*.scp  <eg-name save-postion>
        scp_per_archive_file = open("{0}/{1}{2}.scp"
                                    "".format(args.egs_dir,
                                              args.egs_prefix, archive + 1),
                                    'w')

        # check files before writing.
        if f is None:
            raise Exception("Error opening file {0}".format(f))
        if o is None:
            raise Exception("Error opening file {0}".format(o))
        if w is None:
            raise Exception("Error opening file {0}".format(w))
        if scp_per_archive_file is None:
            raise Exception("Error opening file {0}".format(scp_per_archive_file))

        # 当前archive下的所有job （<job>.<archive>) 进行合并
        for job in range(args.num_jobs):
            scp = ("{0}/temp/{1}scp.{2}.{3}".format(args.egs_dir, args.egs_prefix,
                                                    job + 1, archive + 1))
            with open(scp, "r") as scpfile:
                # 逐个eg
                for line in scpfile:
                    scp_line = line.splitlines()[0].split()
                    print("{0} {1}".format(scp_line[0], scp_line[1]),
                          file=scp_per_archive_file)

                    print("{0} output-{1}".format(scp_line[0], scp_line[2]),
                          file=o)

                    print("{0} {1}".format(
                            scp_line[0],
                            lang2weight[int(scp_line[2])]),
                          file=w)
            os.remove(scp)

            # 保存了 num_archives * job 个job内的minibatch的描述.
            # ranges ???
            for (lang_id, start_eg_line, num_egs) in all_egs[num_archives * job + archive]:
                this_ranges.append((lang_id, start_eg_line, num_egs))
            # write egs.ranges.*.txt
            for (lang_id, start_eg_line, num_egs) in this_ranges:
                print("{0} {1} {2}".format(lang_id, start_eg_line, num_egs), file=f)

        f.close()
        o.close()
        w.close()
        scp_per_archive_file.close()
    logger.info("finished generating {0}*.scp, {0}output.*.ark "
                "and {0}weight.*.ark files.".format(args.egs_prefix))


def main():
    try:
        args = get_args()
        process_multilingual_egs(args)
    except Exception as e:
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
  main()
