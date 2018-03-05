
#!/usr/bin/env python

# Copyright 2016    Vijayaditya Peddinti.
#           2016    Vimal Manohar
#           2017 Johns Hopkins University (author: Daniel Povey)
# Apache 2.0.






# stage=0
# train_stage=-10
# affix=
# common_egs_dir=

# # training options
# initial_effective_lrate=0.0015
# final_effective_lrate=0.00015
# num_epochs=4
# num_jobs_initial=2
# num_jobs_final=12
# remove_egs=true

# # feature options
# use_ivectors=true

# # End configuration section.


# dir=exp/nnet3/tdnn_sp${affix:+_$affix}

# # exp/tri5a 最新模型结果
# gmm_dir=exp/tri5a

# train_set=train_sp

# # 目标对齐输出目录
# ali_dir=${gmm_dir}_sp_ali

# # exp/tri5a/graph 最新模型构建的图 HCLG.fst
# graph_dir=$gmm_dir/graph


# usefor:
# and 
#   提取ivector 特征 == >  ivectordir=exp/nnet3/ivectors_train_sp (copy-feats ark:feats.ark ark,t:-|head)
#   generate sp data
#   generate sp and hires feature
#   generate some ali results.
# out:
#   data/train_sp           speed perturbed data's feats.scp
#   mfcc_perturbed          sp data's mfcc-perturbed-feature.ark (mfcc_perturbed_features cmvn.)
#   exp/tri5a_sp_ali        use the final.mdl in tri5a to generate the ali-result of data/train_sp.

#   data/train_sp_hires    
#   mfcc_perturbed_hires    high soluation mfcc-features extra from the data has been perturbed.

#   exp/nnet3/ivectors_train_sp ivectors-features dim =100.

# local/nnet3/run_ivector_common.sh --stage $stage || exit ;1




# steps/nnet3/train_dnn.py
#   --stage=$train_stage \
#   --cmd="$decode_cmd" \
#   --feat.online-ivector-dir exp/nnet3/ivectors_${train_set} \    (dim = 100)
#   --feat.cmvn-opts="--norm-means=false --norm-vars=false" \
#   --trainer.num-epochs $num_epochs \
#   --trainer.optimization.num-jobs-initial $num_jobs_initial \
#   --trainer.optimization.num-jobs-final $num_jobs_final \
#   --trainer.optimization.initial-effective-lrate $initial_effective_lrate \
#   --trainer.optimization.final-effective-lrate $final_effective_lrate \

#   --egs.dir "$common_egs_dir" \                          # ""
#   --cleanup.remove-egs $remove_egs \                     # true
#   --cleanup.preserve-model-interval 500 \
#   --use-gpu true \

#     =============== feat-dir ali-dir dir =========== 
#   --feat-dir=data/${train_set}_hires \                    # data/train_sp_hires (dim = 40)
#   --ali-dir $ali_dir \                                    # exp/tri5a_sp_ali
#   --dir=$dir \                                            # dir=exp/nnet3/tdnn_sp

#   --lang data/lang \
#   --reporting.email="$reporting_email" || exit 1;



""" This script is based on steps/nnet3/tdnn/train.sh
"""

from __future__ import print_function
import argparse
import logging
import os
import pprint
import shutil
import sys
import traceback

sys.path.insert(0, 'steps')
import libs.nnet3.train.common as common_train_lib
import libs.common as common_lib
import libs.nnet3.train.frame_level_objf as train_lib
import libs.nnet3.report.log_parse as nnet3_log_parse

def train(args, run_opts):
    """ The main function for training.

    Args:
        args: a Namespace object with the required parameters
            obtained from the function process_args()

        run_opts: RunOpts object obtained from the process_args()
    """

    arg_string = pprint.pformat(vars(args))
    logger.info("Arguments for the experiment\n{0}".format(arg_string))

    # Set some variables.
    # num_leaves = common_lib.get_number_of_leaves_from_tree(args.ali_dir)
    num_jobs = common_lib.get_number_of_jobs(args.ali_dir)
    feat_dim = common_lib.get_feat_dim(args.feat_dir)
    ivector_dim = common_lib.get_ivector_dim(args.online_ivector_dir)
    ivector_id = common_lib.get_ivector_extractor_id(args.online_ivector_dir)

    # split the training data into parts for individual jobs
    # we will use the same number of jobs as that used for alignment
    # 将特征数据 按照job数量进行划分, 会使用相同数量的job 进行对齐.
    # feat_dir -- data/train_set_hires 划分为jos 个子集.
    common_lib.execute_command("utils/split_data.sh {0} {1}".format(
        args.feat_dir, num_jobs))
    # copy tri5a_sp_ali/tree(最后训练好的模型tree) --->  exp/nneet3/tdnn_sp
    shutil.copy('{0}/tree'.format(args.ali_dir), args.dir)

    
    # 将ali_dir(exp/tri5a_sp_ali/num_jobs) 读取写入 目标目录 exp/nnet3/tdnn_sp/num_jos
    with open('{0}/num_jobs'.format(args.dir), 'w') as f:
        f.write(str(num_jobs))

    #  目标目录中 存入 nnet3结构/ 以及nnet3的 left right context配置信息
    config_dir = '{0}/configs'.format(args.dir)
    var_file = '{0}/vars'.format(config_dir)

    variables = common_train_lib.parse_generic_config_vars_file(var_file)
    # liujunnan@innovem:configs$ cat vars 
    # model_left_context=16
    # model_right_context=12
    # liujunnan@innovem:configs$
    

    # Set some variables.
    model_left_context = variables['model_left_context']
    model_right_context = variables['model_right_context']

    left_context = model_left_context
    right_context = model_right_context


    # 初始化 原始nnet结构, 在训练 LDA预处理矩阵之前
    # 第一个配置 只做了哪些初始化拼接操作
    # 这样做,是因为这样很方变能够获得LDA变换矩阵的统计信息.
    if (args.stage <= -5) and os.path.exists(args.dir+"/configs/init.config"):
        logger.info("Initializing a basic network for estimating preconditioning matrix")
        
        common_lib.execute_command(
            """{command} {dir}/log/nnet_init.log \
                    nnet3-init --srand=-2 {dir}/configs/init.config \
                    {dir}/init.raw""".format(command=run_opts.command,
                                             dir=args.dir))

    # ================== 生成egs 的过程
    # egs 目录为 目标目录下的 exp/nnet3/tdnn_sp/egs/
    default_egs_dir = '{0}/egs'.format(args.dir)
    if (args.stage <= -4) and args.egs_dir is None:
        logger.info("Generating egs")

        train_lib.acoustic_model.generate_egs(
            data=args.feat_dir,                          # data/train_sp_hires 
            alidir=args.ali_dir,                         # exp/tri5a_sp_ali
            egs_dir=default_egs_dir,                     # exp/nnet3/tdnn_sp/egs
            
            left_context=left_context,
            right_context=right_context,
            
            run_opts=run_opts,
            stage=args.egs_stage,
            
            online_ivector_dir=args.online_ivector_dir,          # exp/nnet3/ivectors_train_sp
            samples_per_iter=args.samples_per_iter,
            frames_per_eg_str=str(args.frames_per_eg),
            
            srand=args.srand,
            
            egs_opts=args.egs_opts,
            cmvn_opts=args.cmvn_opts,

            transform_dir=args.transform_dir)

        
        egs_dir = default_egs_dir

    [egs_left_context, egs_right_context,
     frames_per_eg_str, num_archives] = (
         common_train_lib.verify_egs_dir(egs_dir, feat_dim,
                                         ivector_dim, ivector_id,
                                         left_context, right_context))
    assert str(args.frames_per_eg) == frames_per_eg_str





    

    if args.num_jobs_final > num_archives:
        raise Exception('num_jobs_final cannot exceed the number of archives '
                        'in the egs directory')

    # copy the properties of the egs to dir for
    # use during decoding
    common_train_lib.copy_egs_properties_to_exp_dir(egs_dir, args.dir)

    if args.stage <= -3 and os.path.exists(args.dir+"/configs/init.config"):
        logger.info('Computing the preconditioning matrix for input features')

        train_lib.common.compute_preconditioning_matrix(
            args.dir, egs_dir, num_archives, run_opts,
            max_lda_jobs=args.max_lda_jobs,
            rand_prune=args.rand_prune)

    if args.stage <= -2:
        logger.info("Computing initial vector for FixedScaleComponent before"
                    " softmax, using priors^{prior_scale} and rescaling to"
                    " average 1".format(
                        prior_scale=args.presoftmax_prior_scale_power))

        common_train_lib.compute_presoftmax_prior_scale(
            args.dir, args.ali_dir, num_jobs, run_opts,
            presoftmax_prior_scale_power=args.presoftmax_prior_scale_power)

    if args.stage <= -1:
        logger.info("Preparing the initial acoustic model.")
        train_lib.acoustic_model.prepare_initial_acoustic_model(
            args.dir, args.ali_dir, run_opts)

    # set num_iters so that as close as possible, we process the data
    # $num_epochs times, i.e. $num_iters*$avg_num_jobs) ==
    # $num_epochs*$num_archives, where
    # avg_num_jobs=(num_jobs_initial+num_jobs_final)/2.
    num_archives_expanded = num_archives * args.frames_per_eg
    num_archives_to_process = int(args.num_epochs * num_archives_expanded)
    num_archives_processed = 0
    num_iters = ((num_archives_to_process * 2)
                 / (args.num_jobs_initial + args.num_jobs_final))

    # If do_final_combination is True, compute the set of models_to_combine.
    # Otherwise, models_to_combine will be none.
    if args.do_final_combination:
        models_to_combine = common_train_lib.get_model_combine_iters(
            num_iters, args.num_epochs,
            num_archives_expanded, args.max_models_combine,
            args.num_jobs_final)
    else:
        models_to_combine = None

    logger.info("Training will run for {0} epochs = "
                "{1} iterations".format(args.num_epochs, num_iters))

    for iter in range(num_iters):
        if (args.exit_stage is not None) and (iter == args.exit_stage):
            logger.info("Exiting early due to --exit-stage {0}".format(iter))
            return
        current_num_jobs = int(0.5 + args.num_jobs_initial
                               + (args.num_jobs_final - args.num_jobs_initial)
                               * float(iter) / num_iters)

        if args.stage <= iter:
            lrate = common_train_lib.get_learning_rate(iter, current_num_jobs,
                                                       num_iters,
                                                       num_archives_processed,
                                                       num_archives_to_process,
                                                       args.initial_effective_lrate,
                                                       args.final_effective_lrate)
            shrinkage_value = 1.0 - (args.proportional_shrink * lrate)
            if shrinkage_value <= 0.5:
                raise Exception("proportional-shrink={0} is too large, it gives "
                                "shrink-value={1}".format(args.proportional_shrink,
                                                          shrinkage_value))

            train_lib.common.train_one_iteration(
                dir=args.dir,
                iter=iter,
                srand=args.srand,
                egs_dir=egs_dir,
                num_jobs=current_num_jobs,
                num_archives_processed=num_archives_processed,
                num_archives=num_archives,
                learning_rate=lrate,
                dropout_edit_string=common_train_lib.get_dropout_edit_string(
                    args.dropout_schedule,
                    float(num_archives_processed) / num_archives_to_process,
                    iter),
                minibatch_size_str=args.minibatch_size,
                frames_per_eg=args.frames_per_eg,
                momentum=args.momentum,
                max_param_change=args.max_param_change,
                shrinkage_value=shrinkage_value,
                shuffle_buffer_size=args.shuffle_buffer_size,
                run_opts=run_opts)

            if args.cleanup:
                # do a clean up everythin but the last 2 models, under certain
                # conditions
                common_train_lib.remove_model(
                    args.dir, iter-2, num_iters, models_to_combine,
                    args.preserve_model_interval)

            if args.email is not None:
                reporting_iter_interval = num_iters * args.reporting_interval
                if iter % reporting_iter_interval == 0:
                    # lets do some reporting
                    [report, times, data] = (
                        nnet3_log_parse.generate_acc_logprob_report(args.dir))
                    message = report
                    subject = ("Update : Expt {dir} : "
                               "Iter {iter}".format(dir=args.dir, iter=iter))
                    common_lib.send_mail(message, subject, args.email)

        num_archives_processed = num_archives_processed + current_num_jobs

    if args.stage <= num_iters:
        if args.do_final_combination:
            logger.info("Doing final combination to produce final.mdl")
            train_lib.common.combine_models(
                dir=args.dir, num_iters=num_iters,
                models_to_combine=models_to_combine,
                egs_dir=egs_dir,
                minibatch_size_str=args.minibatch_size, run_opts=run_opts,
                sum_to_one_penalty=args.combine_sum_to_one_penalty)
    
    if args.stage <= num_iters + 1:
        logger.info("Getting average posterior for purposes of "
                    "adjusting the priors.")
        
        # If args.do_final_combination is true, we will use the combined model.
        # Otherwise, we will use the last_numbered model.
        real_iter = 'combined' if args.do_final_combination else num_iters
        avg_post_vec_file = train_lib.common.compute_average_posterior(
            dir=args.dir, iter=real_iter, 
            egs_dir=egs_dir, num_archives=num_archives,
            prior_subset_size=args.prior_subset_size, run_opts=run_opts)

        logger.info("Re-adjusting priors based on computed posteriors")
        combined_or_last_numbered_model = "{dir}/{iter}.mdl".format(dir=args.dir,
                iter=real_iter)
        final_model = "{dir}/final.mdl".format(dir=args.dir)
        train_lib.common.adjust_am_priors(args.dir, combined_or_last_numbered_model,
                avg_post_vec_file, final_model, run_opts)


    if args.cleanup:
        logger.info("Cleaning up the experiment directory "
                    "{0}".format(args.dir))
        remove_egs = args.remove_egs
        if args.egs_dir is not None:
            # this egs_dir was not created by this experiment so we will not
            # delete it
            remove_egs = False

        common_train_lib.clean_nnet_dir(
            nnet_dir=args.dir, num_iters=num_iters, egs_dir=egs_dir,
            preserve_model_interval=args.preserve_model_interval,
            remove_egs=remove_egs)

    # do some reporting
    [report, times, data] = nnet3_log_parse.generate_acc_logprob_report(args.dir)
    if args.email is not None:
        common_lib.send_mail(report, "Update : Expt {0} : "
                                     "complete".format(args.dir), args.email)

    with open("{dir}/accuracy.report".format(dir=args.dir), "w") as f:
        f.write(report)

    common_lib.execute_command("steps/info/nnet3_dir_info.pl "
                               "{0}".format(args.dir))




    
def main():
    [args, run_opts] = get_args()

   
    try:
        # 开始训练.
        train(args, run_opts)
        common_lib.wait_for_background_commands()
    except BaseException as e:
        # look for BaseException so we catch KeyboardInterrupt, which is
        # what we get when a background thread dies.
        if args.email is not None:
            message = ("Training session for experiment {dir} "
                       "died due to an error.".format(dir=args.dir))
            common_lib.send_mail(message, message, args.email)
        if not isinstance(e, KeyboardInterrupt):
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
