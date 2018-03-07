// #!/bin/bash
// # Copyright 2010-2012 Microsoft Corporation
// #           2012-2013 Johns Hopkins University (Author: Daniel Povey)
// # Apache 2.0

// # This script creates a fully expanded decoding graph (HCLG) that represents
// # all the language-model, pronunciation dictionary (lexicon), context-dependency,
// # and HMM structure in our model.  The output is a Finite State Transducer
// # that has word-ids on the output, and pdf-ids on the input (these are indexes
// # that resolve to Gaussian Mixture Models).
// # See
// #  http://kaldi-asr.org/doc/graph_recipe_test.html
// # (this is compiled from this repository using Doxygen,
// # the source for this part is in src/doc/graph_recipe_test.dox)


// utils/mkgraph.sh $opt data/graph/lang $srcdir $srcdir/graph_word

// lang=$1                     //data/graph/lang
// tree=$2/tree                //exp/mono/tree    ctx
// model=$2/final.mdl          //exp/mono/final.mdl
// dir=$3                      //exp/mono/graph_word/ ...

// 从决策树中获得 context-depent 属性.
// N=$(tree-info $tree | grep "context-width" | cut -d' ' -f2)
// P=$(tree-info $tree | grep "central-position" | cut -d' ' -f2)


// [[]] 与 [] 相似, 但是[[]]提供了更多的操作符号 eg || 替代 -o.
// 判断是否存在required的所有文件, 存在了 就进行compose determinize 等操作。

// if [[ ! -s $lang/tmp/LG.fst || $lang/tmp/LG.fst -ot $lang/G.fst || \
//       $lang/tmp/LG.fst -ot $lang/L_disambig.fst ]]; then

//   fsttablecompose $lang/L_disambig.fst $lang/G.fst | fstdeterminizestar --use-log=true | \
//     fstminimizeencoded | fstpushspecial | \
//     fstarcsort --sort_type=ilabel > $lang/tmp/LG.fst.$$ || exit 1;
//   mv $lang/tmp/LG.fst.$$ $lang/tmp/LG.fst
//   fstisstochastic $lang/tmp/LG.fst || echo "[info]: LG not stochastic."

// fi




// clg=$lang/tmp/CLG_${N}_${P}.fst
// clg_tmp=$clg.$$
// ilabels=$lang/tmp/ilabels_${N}_${P}
// ilabels_tmp=$ilabels.$$
// trap "rm -f $clg_tmp $ilabels_tmp" EXIT HUP INT PIPE TERM
// if [[ ! -s $clg || $clg -ot $lang/tmp/LG.fst \
//     || ! -s $ilabels || $ilabels -ot $lang/tmp/LG.fst ]]; then
//   fstcomposecontext --context-size=$N --central-position=$P \
//    --read-disambig-syms=$lang/phones/disambig.int \
//    --write-disambig-syms=$lang/tmp/disambig_ilabels_${N}_${P}.int \
//     $ilabels_tmp < $lang/tmp/LG.fst |\
//     fstarcsort --sort_type=ilabel > $clg_tmp
//   mv $clg_tmp $clg
//   mv $ilabels_tmp $ilabels
//   fstisstochastic $clg || echo "[info]: CLG not stochastic."
// fi

// trap "rm -f $dir/Ha.fst.$$" EXIT HUP INT PIPE TERM
// if [[ ! -s $dir/Ha.fst || $dir/Ha.fst -ot $model  \
//     || $dir/Ha.fst -ot $lang/tmp/ilabels_${N}_${P} ]]; then
//   make-h-transducer --disambig-syms-out=$dir/disambig_tid.int \
//     --transition-scale=$tscale $lang/tmp/ilabels_${N}_${P} $tree $model \
//      > $dir/Ha.fst.$$  || exit 1;
//   mv $dir/Ha.fst.$$ $dir/Ha.fst
// fi

// trap "rm -f $dir/HCLGa.fst.$$" EXIT HUP INT PIPE TERM
// if [[ ! -s $dir/HCLGa.fst || $dir/HCLGa.fst -ot $dir/Ha.fst || \
//       $dir/HCLGa.fst -ot $clg ]]; then
//   if $remove_oov; then
//     [ ! -f $lang/oov.int ] && \
//       echo "$0: --remove-oov option: no file $lang/oov.int" && exit 1;
//     clg="fstrmsymbols --remove-arcs=true --apply-to-output=true $lang/oov.int $clg|"
//   fi
//   fsttablecompose $dir/Ha.fst "$clg" | fstdeterminizestar --use-log=true \
//     | fstrmsymbols $dir/disambig_tid.int | fstrmepslocal | \
//      fstminimizeencoded > $dir/HCLGa.fst.$$ || exit 1;
//   mv $dir/HCLGa.fst.$$ $dir/HCLGa.fst
//   fstisstochastic $dir/HCLGa.fst || echo "HCLGa is not stochastic"
// fi

// trap "rm -f $dir/HCLG.fst.$$" EXIT HUP INT PIPE TERM
// if [[ ! -s $dir/HCLG.fst || $dir/HCLG.fst -ot $dir/HCLGa.fst ]]; then
//   add-self-loops --self-loop-scale=$loopscale --reorder=true \
//     $model < $dir/HCLGa.fst | fstconvert --fst_type=const > $dir/HCLG.fst.$$ || exit 1;
//   mv $dir/HCLG.fst.$$ $dir/HCLG.fst
//   if [ $tscale == 1.0 -a $loopscale == 1.0 ]; then
//     # No point doing this test if transition-scale not 1, as it is bound to fail.
//     fstisstochastic $dir/HCLG.fst || echo "[info]: final HCLG is not stochastic."
//   fi
// fi

// # note: the empty FST has 66 bytes.  this check is for whether the final FST
// # is the empty file or is the empty FST.
// if ! [ $(head -c 67 $dir/HCLG.fst | wc -c) -eq 67 ]; then
//   echo "$0: it looks like the result in $dir/HCLG.fst is empty"
//   exit 1
// fi

// # save space.
// rm $dir/HCLGa.fst $dir/Ha.fst 2>/dev/null || true

// # keep a copy of the lexicon and a list of silence phones with HCLG...
// # this means we can decode without reference to the $lang directory.


// cp $lang/words.txt $dir/ || exit 1;
// mkdir -p $dir/phones
// cp $lang/phones/word_boundary.* $dir/phones/ 2>/dev/null # might be needed for ctm scoring,
// cp $lang/phones/align_lexicon.* $dir/phones/ 2>/dev/null # might be needed for ctm scoring,
// cp $lang/phones/optional_silence.* $dir/phones/ 2>/dev/null # might be needed for analyzing alignments.
//   # but ignore the error if it's not there.

// cp $lang/phones/disambig.{txt,int} $dir/phones/ 2> /dev/null
// cp $lang/phones/silence.csl $dir/phones/ || exit 1;
// cp $lang/phones.txt $dir/ 2> /dev/null # ignore the error if it's not there.

// am-info --print-args=false $model | grep pdfs | awk '{print $NF}' > $dir/num_pdfs
