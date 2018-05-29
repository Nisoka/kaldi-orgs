#!/usr/bin/perl
#
# Copyright 2015   David Snyder
# Apache 2.0.
# Usage: make_sre.pl <path-to-data> <name-of-source> <sre-ref> <output-dir>

# 生成信息文件 dir: $out_dir
# 1 wav.scp      utt-id    wav-path
# 2 utt2spk      utt-id    spker-gender
# 3 spk2gender   spker     gender


if (@ARGV != 4) {
  print STDERR "Usage: $0 <path-to-data> <name-of-source> <sre-ref> <output-dir>\n";
  print STDERR "e.g. $0 /export/corpora5/LDC/LDC2006S44 sre2004 sre_ref data/sre2004\n";
  exit(1);
}

($db_base, $sre_name, $sre_ref_filename, $out_dir) = @ARGV;

# utt-id utt-path 字典
%utt2sph = ();
# 说话人性别 字典(包含所有说话人 key:spker+uttid, value: gender)
%spk2gender = ();

$tmp_dir = "$out_dir/tmp";
if (system("mkdir -p $tmp_dir") != 0) {
  die "Error making directory $tmp_dir";
}

# 获得.sph 语音数据 list  > data/sre_nn/tmp/sph.list
if (system("find $db_base -name '*.sph' > $tmp_dir/sph.list") != 0) {
  die "Error getting list of sph files";
}

# 读取 uttlist
open(WAVLIST, "<", "$tmp_dir/sph.list") or die "cannot open wav list";

while(<WAVLIST>) {
    chomp;
    # wav-path
    $sph = $_;
    
    @A1 = split("/",$sph);
    @A2 = split("[./]",$A1[$#A1]);
    
    $uttId=$A2[0];

    # 构造uttid - utt-path 字典  key:uttid   value:wavpath
    $utt2sph{$uttId} = $sph;
}

open(GNDR,">", "$out_dir/spk2gender") or die "Could not open the output file $out_dir/spk2gender";
open(SPKR,">", "$out_dir/utt2spk") or die "Could not open the output file $out_dir/utt2spk";
open(WAV,">", "$out_dir/wav.scp") or die "Could not open the output file $out_dir/wav.scp";

# uttinfos : spker gender other_sre_name utt-id channel
open(SRE_REF, "<", $sre_ref_filename) or die "Cannot open SRE reference.";




while (<SRE_REF>) {

  chomp;
  ($speaker, $gender, $other_sre_name, $utt_id, $channel) = split(" ", $_);
  $channel_num = "1";
  if ($channel eq "A") {
    $channel_num = "1";
  } else {
    $channel_num = "2";
  }

  # 1 wav.scp    utt-id  -- wav.path
  # 2 utt2spk    utt-id  -- spker-gender
  if (($other_sre_name eq $sre_name) and (exists $utt2sph{$utt_id})) {
      # 完整utt-id = spker - gender - name - old-utt-id - channel
      $full_utt_id = "$speaker-$gender-$sre_name-$utt_id-$channel";
      # spker2gender
      $spk2gender{"$speaker-$gender"} = $gender;
      print WAV "$full_utt_id"," sph2pipe -f wav -p -c $channel_num $utt2sph{$utt_id} |\n";
      print SPKR "$full_utt_id $speaker-$gender","\n";
  }
}

# spk2gender   spker -- gender
# 生成spker - gender 文件 spk2gender
foreach $speaker (keys %spk2gender) {
  print GNDR "$speaker $spk2gender{$speaker}\n";
}


close(GNDR) || die;
close(SPKR) || die;
close(WAV) || die;
close(SRE_REF) || die;
