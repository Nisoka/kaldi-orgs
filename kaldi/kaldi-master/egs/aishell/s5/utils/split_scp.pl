#!/usr/bin/env perl
use warnings; #sed replacement for -w perl parameter
# Copyright 2010-2011 Microsoft Corporation

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#  http://www.apache.org/licenses/LICENSE-2.0
#
# THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
# WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
# MERCHANTABLITY OR NON-INFRINGEMENT.
# See the Apache 2 License for the specific language governing permissions and
# limitations under the License.


# 这个脚本用来 split .scp 文件,
# This program splits up any kind of .scp or archive-type file.
# 如果没指定　utt2spk 选项，　会工作在任何ｔｅｘｔ文件，　并等价划分为　numsplit
# If there is no utt2spk option it will work on any text  file and
# will split it up with an approximately(大约) equal number of lines in
# each but.
#　使用--utt2spk 选项　会工作在 具有utt-id 作为开始字段的文件
# With the --utt2spk option it will work on anything that has the
# utterance-id as the first entry on each line; the utt2spk file is
# of the form "utterance speaker" (on each line).
# 会将文件划分成等大小的块, 如果你使用utt2spk选项, 他会确保这些块 与 spker 边界相符合.
# 这种情况下, 如果块比spker多, 一些chunk会为empty, 并且打印错误信息, 并exit退出.
# (这就是错误情况, 如果块数量 > spker 数量
# It splits it into equal size chunks as far as it can.  If you use the utt2spk
# option it will make sure these chunks coincide with speaker boundaries.  In
# this case, if there are more chunks than speakers (and in some other
# circumstances), some of the resulting chunks will be empty and it will print
# an error message and exit with nonzero status.
#
# You will normally call this like:
# split_scp.pl scp scp.1 scp.2 scp.3 ...
# or
# split_scp.pl --utt2spk=utt2spk scp scp.1 scp.2 scp.3 ...
# Note that you can use this script to split the utt2spk file itself,
# e.g. split_scp.pl --utt2spk=utt2spk utt2spk utt2spk.1 utt2spk.2 ...

# You can also call the scripts like:
# split_scp.pl -j 3 0 scp scp.0
# [note: with this option, it assumes zero-based indexing of the split parts,
# i.e. the second number must be 0 <= n < num-jobs.]

$num_jobs = 0;
$job_id = 0;
$utt2spk_file = "";

for ($x = 1; $x <= 2 && @ARGV > 0; $x++) {
    if ($ARGV[0] eq "-j") {
        shift @ARGV;
        $num_jobs = shift @ARGV;
        $job_id = shift @ARGV;
        if ($num_jobs <= 0 || $job_id < 0 || $job_id >= $num_jobs) {
            die "Invalid num-jobs and job-id: $num_jobs and $job_id";
        }
    }
    # 第一个参数是　－－utt2spk=utt2spk  然后去除utt2spk 赋值给
    # utt2spk_file = utt2spk
    if ($ARGV[0] =~ "--utt2spk=(.+)") {
        $utt2spk_file=$1;
        shift;
    }
}

if(($num_jobs == 0 && @ARGV < 2) || ($num_jobs > 0 && (@ARGV < 1 || @ARGV > 2))) {
    die "Usage: split_scp.pl [--utt2spk=<utt2spk_file>] in.scp out1.scp out2.scp ... \n" .
        " or: split_scp.pl -j num-jobs job-id [--utt2spk=<utt2spk_file>] in.scp [out.scp]\n" .
        " ... where 0 <= job-id < num-jobs.";
}



# split_scp.pl [--utt2spk=<utt2spk_file>] in.scp out1.scp out2.scp ...
# 使用方式 此时 @ARGV = in.scp out1.scp out2.scp
# shift @ARGV 取出 第一个 in.scp
# $inscp = in.scp
# @OUTPUTS = out1.scp out2.scp ...
$error = 0;
$inscp = shift @ARGV;
# 如果没有多 jobs 则进行单线程划分,
if ($num_jobs == 0) { # without -j option
    @OUTPUTS = @ARGV;
} else {
    for ($j = 0; $j < $num_jobs; $j++) {
        if ($j == $job_id) {
            if (@ARGV > 0) { push @OUTPUTS, $ARGV[0]; }
            else { push @OUTPUTS, "-"; }
        } else {
            push @OUTPUTS, "/dev/null";
        }
    }
}




if ($utt2spk_file ne "") {  # We have the --utt2spk option...
    # 将文件打开 写入到 U中
    open(U, "<$utt2spk_file") || die "Failed to open utt2spk file $utt2spk_file";
    # while(<U>) 逐行读取
    while(<U>) {
        # 划分行为 utt spker
        @A = split;
        @A == 2 || die "Bad line $_ in utt2spk file $utt2spk_file";
        ($u,$s) = @A;
        # utt2spker[utt] = spker
        $utt2spk{$u} = $s;
    }

    # inscp
    #     utt  some.
    open(I, "<$inscp") || die "Opening input scp file $inscp";
    # 声明数组
    @spkrs = ();
    while(<I>) {
        @A = split;
        if(@A == 0) { die "Empty or space-only line in scp file $inscp"; }
        $u = $A[0];
        $s = $utt2spk{$u};
        if(!defined $s) { die "No such utterance $u in utt2spk file $utt2spk_file"; }
        # 1 将spker 加入到 @spkrs
        # 2 spk_count{spker} = 0
          # spk_count  保存 inscp 有多少某个spker的utt
        # 3 spk_data{spker} = []
          # spk_data  保存 inscp 有多少某个spker的utt
        if(!defined $spk_count{$s}) {
            push @spkrs, $s;
            $spk_count{$s} = 0;
            $spk_data{$s} = [];  # ref to new empty array.
        }

        # spker 保存的utt总数
        $spk_count{$s}++;
        # spker 包含的 utt some
        # "$_" 默认的输入, 当前输入就是 <I>
        push @{$spk_data{$s}}, $_;
    }


    # Now split as equally as possible ..
    # First allocate spks to files by allocating an approximately
    # equal number of speakers.
    $numspks = @spkrs;  # number of speakers.
    $numscps = @OUTPUTS; # number of output files.

    # 如果划分太多了 > numspker . 那么直接出错.
    if ($numspks < $numscps) {
      die "Refusing to split data because number of speakers $numspks is less " .
          "than the number of output .scp files $numscps";
    }
    # 构造每个划分的数组
    for($scpidx = 0; $scpidx < $numscps; $scpidx++) {
        $scparray[$scpidx] = []; # [] is array reference.
    }

    # 每个spker 计算应该被划分到的 scpidx
    for ($spkidx = 0; $spkidx < $numspks; $spkidx++) {
        $scpidx = int(($spkidx*$numscps) / $numspks);
        # 该spker
        $spk = $spkrs[$spkidx];
        # scpidx 中 保存对应被划分到其中的 spker
        push @{$scparray[$scpidx]}, $spk;
        # scpidx 保存的utt总数 就应该是 所有属于其的 spker 的utt总数 和 .
        $scpcount[$scpidx] += $spk_count{$spk};
    }

    # Now will try to reassign beginning + ending speakers
    # to different scp's and see if it gets more balanced.
    # 拟合参数 找到最好的划分方法
    # Suppose objf we're minimizing is sum_i (num utts in scp[i] - average)^2.
    # We can show that if considering changing just 2 scp's, we minimize
    # this by minimizing the squared difference in sizes.  This is
    # equivalent to minimizing the absolute difference in sizes.  This
    # shows this method is bound to converge.

    $changed = 1;
    while($changed) {
        $changed = 0;
        for($scpidx = 0; $scpidx < $numscps; $scpidx++) {
            # First try to reassign ending spk of this scp.
            if($scpidx < $numscps-1) {
                $sz = @{$scparray[$scpidx]};
                if($sz > 0) {
                    $spk = $scparray[$scpidx]->[$sz-1];
                    $count = $spk_count{$spk};
                    $nutt1 = $scpcount[$scpidx];
                    $nutt2 = $scpcount[$scpidx+1];
                    if( abs( ($nutt2+$count) - ($nutt1-$count))
                        < abs($nutt2 - $nutt1))  { # Would decrease
                        # size-diff by reassigning spk...
                        $scpcount[$scpidx+1] += $count;
                        $scpcount[$scpidx] -= $count;
                        pop @{$scparray[$scpidx]};
                        unshift @{$scparray[$scpidx+1]}, $spk;
                        $changed = 1;
                    }
                }
            }
            if($scpidx > 0 && @{$scparray[$scpidx]} > 0) {
                $spk = $scparray[$scpidx]->[0];
                $count = $spk_count{$spk};
                $nutt1 = $scpcount[$scpidx-1];
                $nutt2 = $scpcount[$scpidx];
                if( abs( ($nutt2-$count) - ($nutt1+$count))
                    < abs($nutt2 - $nutt1))  { # Would decrease
                    # size-diff by reassigning spk...
                    $scpcount[$scpidx-1] += $count;
                    $scpcount[$scpidx] -= $count;
                    shift @{$scparray[$scpidx]};
                    push @{$scparray[$scpidx-1]}, $spk;
                    $changed = 1;
                }
            }
        }
    }
    # Now print out the files...
    for($scpidx = 0; $scpidx < $numscps; $scpidx++) {
        $scpfn = $OUTPUTS[$scpidx];
        open(F, ">$scpfn") || die "Could not open scp file $scpfn for writing.";
        $count = 0;
        if(@{$scparray[$scpidx]} == 0) {
            print STDERR "Error: split_scp.pl producing empty .scp file $scpfn (too many splits and too few speakers?)\n";
            $error = 1;
        } else {
            foreach $spk ( @{$scparray[$scpidx]} ) {
                print F @{$spk_data{$spk}};
                $count += $spk_count{$spk};
            }
            if($count != $scpcount[$scpidx]) { die "Count mismatch [code error]"; }
        }
        close(F);
    }
} else {
   # This block is the "normal" case where there is no --utt2spk
   # option and we just break into equal size chunks.

    open(I, "<$inscp") || die "Opening input scp file $inscp";

    $numscps = @OUTPUTS;  # size of array.
    @F = ();
    while(<I>) {
        push @F, $_;
    }
    $numlines = @F;
    if($numlines == 0) {
        print STDERR "split_scp.pl: error: empty input scp file $inscp , ";
        $error = 1;
    }
    $linesperscp = int( $numlines / $numscps); # the "whole part"..
    $linesperscp >= 1 || die "You are splitting into too many pieces! [reduce \$nj]";
    $remainder = $numlines - ($linesperscp * $numscps);
    ($remainder >= 0 && $remainder < $numlines) || die "bad remainder $remainder";
    # [just doing int() rounds down].
    $n = 0;
    for($scpidx = 0; $scpidx < @OUTPUTS; $scpidx++) {
        $scpfile = $OUTPUTS[$scpidx];
        open(O, ">$scpfile") || die "Opening output scp file $scpfile";
        for($k = 0; $k < $linesperscp + ($scpidx < $remainder ? 1 : 0); $k++) {
            print O $F[$n++];
        }
        close(O) || die "Closing scp file $scpfile";
    }
    $n == $numlines || die "split_scp.pl: code error., $n != $numlines";
}

exit ($error ? 1 : 0);
