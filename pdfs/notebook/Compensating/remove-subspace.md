# vector remove direction
direction 是一个方向向量
vector想要移除direction方向，需要
1 vector映射到 direction方向的投影向量
$$ projectVec = a . b / |b|  * b / |b| $$
2 vector 减去映射投影向量
$$ resVec = vector - projectVec $$

# vector remove subspace
方法与 vector remove direction 相同
需要减去所有subspace中的direction即可。
