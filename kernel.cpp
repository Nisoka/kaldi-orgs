void GMM(){
  /*
    1 Gaussiable 统计量的结构, 各自代表EM算法中的什么部分.
    2 怎么进行的EM算法的参数更新
    3 
   */
}



void FST(){
  // H C L G
 
  /*
    1 compose
  */
  
  /*
    2 各种优化操作之后 得到的fst图的结构情况

     例如fst图最后经过facotrization 之后会将多个HMM-state合并成一个WFST状态, 然后一个WFST状态代表是一个子word。
     然后不同的WFST通过 epsilon转移连接, 这样解码时候 就是循环判断是否是epsilon, 然后不同解码方式循环的过程.

     但是每个epsilon 转移 在什么时候增加的, 怎么讲多个HMM-state 合并成一个WFST的操作 一点不了解.

  */
}









