#!/bin/bash
#

if [ -f path.sh ]; then . ./path.sh; fi

mkdir -p data/lang_test

# 上一步生成的语言模型 lm_unpruned.gz
arpa_lm=data/local/lm/3gram-mincount/lm_unpruned.gz
[ ! -f $arpa_lm ] && echo No such file $arpa_lm && exit 1;


mkdir -p data/lang_test
cp -r data/lang/* data/lang_test

# 转换生成 G.fst
gunzip -c "$arpa_lm" | \
  arpa2fst --disambig-symbol=#0 \
           --read-symbol-table=data/lang_test/words.txt - data/lang_test/G.fst

# 检测 G.fst的随机化
echo  "Checking how stochastic G is (the first of these numbers should be small):"
fstisstochastic data/lang_test/G.fst


# 生成L.fst
## Check lexicon.
## just have a look and make sure it seems sane.
echo "First few lines of lexicon FST:"
fstprint   --isymbols=data/lang/phones.txt --osymbols=data/lang/words.txt data/lang/L.fst  | head


echo Performing further checks

# Checking that G.fst is determinizable.
fstdeterminize data/lang_test/G.fst /dev/null || echo Error determinizing G.

# Checking that L_disambig.fst is determinizable.
fstdeterminize data/lang_test/L_disambig.fst /dev/null || echo Error determinizing L.


# 检测 L.fst G.fst compose  之后是确定化的
# Checking that disambiguated lexicon times G is determinizable
# Note: we do this with fstdeterminizestar not fstdeterminize, as
# fstdeterminize was taking forever (presumbaly relates to a bug
# in this version of OpenFst that makes determinization slow for
# some case).
fsttablecompose data/lang_test/L_disambig.fst data/lang_test/G.fst | \
   fstdeterminizestar >/dev/null || echo Error

# Checking that LG is stochastic:
fsttablecompose data/lang/L_disambig.fst data/lang_test/G.fst | \
   fstisstochastic || echo "[log:] LG is not stochastic"


echo "$0 succeeded"
