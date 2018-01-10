

// $decoder --cmd "$decode_cmd" --nj $nj $srcdir/graph_word $datadir/test $srcdir/decode_test_word || exit 1



// graphdir=$1      图路径 -- exp/mono/graph_word  刚刚mkgraph 构建得到的HCLG.FST
// data=$2          数据  --  data/mfcc/test
// dir=$3           输出结果/log --  exp/mono/decode_test_word.

// srcdir=`dirname $dir`;   
// sdata=$data/split$nj;

//  一般结果是 final.mdl 上面得到的转移模型
// if [ -z "$model" ]; then # if --model <mdl> was not specified on the command line...
//   if [ -z $iter ]; then
//      model=$srcdir/final.mdl;
//   else
//      model=$srcdir/$iter.mdl;
//   fi
// fi


// 特征 、 cmvn（应用说话人特征均值倒谱系数）   model？现在无定义  当前图HCLG.FST.  
// for f in $sdata/1/feats.scp $sdata/1/cmvn.scp $model $graphdir/HCLG.fst; do
//   [ ! -f $f ] && echo "decode.sh: no such file $f" && exit 1;
// done

// srcdir  exp/mono/
// splice_opts=`cat $srcdir/splice_opts 2>/dev/null` # frame-splicing options.
// cmvn_opts=`cat $srcdir/cmvn_opts 2>/dev/null`
// delta_opts=`cat $srcdir/delta_opts 2>/dev/null`



// if [ $stage -le 0 ]; then

//     gmm-latgen-faster$thread_string --max-active=$max_active --beam=$beam --lattice-beam=$lattice_beam \
//     --acoustic-scale=$acwt --allow-partial=true --word-symbol-table=$graphdir/words.txt $decode_extra_opts \
//     $model $graphdir/HCLG.fst "$feats" "ark:|gzip -c > $dir/lat.JOB.gz" || exit 1;
// fi


