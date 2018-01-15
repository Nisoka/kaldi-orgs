// #DNN training, both xent-交叉熵 and MPE
// . ./path.sh ## Source the tools/utils (import the queue.pl)

// local/nnet/run_dnn.sh --stage 0 --nj $n  exp/tri4b exp/tri4b_ali exp/tri4b_ali_cv || exit 1;

// exp 中 trinb 都是进行训练过程中的各种信息以及训练得到 mdl结果等信息, 其中包含的对齐结果都是临时对齐,并不包含最终得到的mdl的对齐.
// exp 中 trinb_ali 是使用刚刚训练完成的 mdl 进行对齐的对齐结果, 所以一般下次训练是用上次的 trinb_ali作为输入.

// gmmdir=$1      exp/tri4b   上次训练的输出模型
// alidir=$2      exp/tri4b_ali 上次训练的对齐结果
// alidir_cv=$3   exp/tri4b_ali_cv  cv（交叉验证集）对齐结果


void generra_fbanks(){
  // ---------------- #generate fbanks-------------
  //   echo "DNN training: stage 0: feature generation"
  //   rm -rf data/fbank && mkdir -p data/fbank &&  cp -R data/{train,dev,test,test_phone} data/fbank || exit 1;
  //   for x in train dev test; do
  //     echo "producing fbank for $x"
  //     #fbank generation
  //     steps/make_fbank.sh --nj $nj --cmd "$train_cmd" data/fbank/$x exp/make_fbank/$x fbank/$x || exit 1
  //     #ompute cmvn
  //     steps/compute_cmvn_stats.sh data/fbank/$x exp/fbank_cmvn/$x fbank/$x || exit 1
  //   done

  //   echo "producing test_fbank_phone"
  //   cp data/fbank/test/feats.scp data/fbank/test_phone && cp data/fbank/test/cmvn.scp data/fbank/test_phone || exit 1;
}


void xEnt_training(){
  // ---------------- #xEnt training ---------------
  
  //   outdir=exp/tri4b_dnn  -- 说明是以tri4b的模型为基础 再进行训练的结果.
  
  //   #NN training
  
  //   (tail --pid=$$ -F $outdir/log/train_nnet.log 2>/dev/null)& # forward log
  
  //     steps/nnet/train.sh --copy_feats false --cmvn-opts "--norm-means=true --norm-vars=false" --hid-layers 4 --hid-dim 1024 \
  //     --learn-rate 0.008
  //             data/fbank/train     data/fbank/dev     data/lang     $alidir    $alidir_cv $outdir
  
  //   #Decode (reuse HCLG graph in gmmdir)
  //   (
  //     steps/nnet/decode.sh --nj $nj --cmd "$decode_cmd" --srcdir $outdir --config conf/decode_dnn.config --acwt 0.1 \
  //       $gmmdir/graph_word data/fbank/test $outdir/decode_test_word || exit 1;
  //   )&
  //   (
  //    steps/nnet/decode.sh --nj $nj --cmd "$decode_cmd" --srcdir $outdir --config conf/decode_dnn.config --acwt 0.1 \
  //      $gmmdir/graph_phone data/fbank/test_phone $outdir/decode_test_phone || exit 1;
  //   )&

}


void training_sh(){
  // view in nnet-train.cpp
  
}

































      
#MPE training

srcdir=exp/tri4b_dnn
acwt=0.1

if [ $stage -le 2 ]; then
  # generate lattices and alignments
  steps/nnet/align.sh --nj $nj --cmd "$train_cmd" \
    data/fbank/train data/lang $srcdir ${srcdir}_ali || exit 1;
  steps/nnet/make_denlats.sh --nj $nj --cmd "$decode_cmd" --config conf/decode_dnn.config --acwt $acwt \
    data/fbank/train data/lang $srcdir ${srcdir}_denlats || exit 1;
fi

if [ $stage -le 3 ]; then
  outdir=exp/tri4b_dnn_mpe
  #Re-train the DNN by 3 iteration of MPE
  steps/nnet/train_mpe.sh --cmd "$cuda_cmd" --num-iters 3 --acwt $acwt --do-smbr false \
    data/fbank/train data/lang $srcdir ${srcdir}_ali ${srcdir}_denlats $outdir || exit 1
  #Decode (reuse HCLG graph)
  for ITER in 3 2 1; do
   (
    steps/nnet/decode.sh --nj $nj --cmd "$decode_cmd" --nnet $outdir/${ITER}.nnet --config conf/decode_dnn.config --acwt $acwt \
      $gmmdir/graph_word data/fbank/test $outdir/decode_test_word_it${ITER} || exit 1;
   )&
   (
   steps/nnet/decode.sh --nj $nj --cmd "$decode_cmd" --nnet $outdir/${ITER}.nnet --config conf/decode_dnn.config --acwt $acwt \
     $gmmdir/graph_phone data/fbank/test_phone $outdir/decode_test_phone_it${ITER} || exit 1;
   )&
  done
fi

