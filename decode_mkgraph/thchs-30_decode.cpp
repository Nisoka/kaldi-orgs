
// # local/thchs-30_decode.sh --mono true --nj $n "steps/decode.sh" exp/mono data/mfcc & 

// nj=8
// mono=false

// decoder=$1  // steps/decode.sh 
// srcdir=$2   // exp/mono   对齐结果 ali.JOB.gz
// datadir=$3  // data/mfcc


// if [ $mono = true ];then
//   echo  "using monophone to generate graph"
//   opt="--mono"
// fi



// #decode word
// 根据G.fst L.fst   exp/mono/graph_word
// utils/mkgraph.sh $opt data/graph/lang $srcdir $srcdir/graph_word
// $decoder --cmd "$decode_cmd" --nj $nj $srcdir/graph_word $datadir/test $srcdir/decode_test_word || exit 1

// #decode phone
// utils/mkgraph.sh $opt data/graph_phone/lang $srcdir $srcdir/graph_phone  || exit 1;
// $decoder --cmd "$decode_cmd" --nj $nj $srcdir/graph_phone $datadir/test_phone $srcdir/decode_test_phone || exit 1


