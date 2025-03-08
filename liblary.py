
# 素因数分解のリストを返す
def prime_factorize(n):
    a = []
    while n % 2 == 0:
        a.append(2)
        n //= 2
    f = 3
    while f * f <= n:
        if n % f == 0:
            a.append(f)
            n //= f
        else:
            f += 2
    if n != 1:
        a.append(n)
    return a


# 約数列挙
def make_divisors(n):
    lower_divisors , upper_divisors = [], []
    i = 1
    while i*i <= n:
        if n % i == 0:
            lower_divisors.append(i)
            if i != n // i:
                upper_divisors.append(n//i)
        i += 1
    return lower_divisors + upper_divisors[::-1]


# 素数判定リスト作成
import math
def sieve_of_eratosthenes(n):
    prime = [True for i in range(n+1)]
    prime[0] = False
    prime[1] = False
    sqrt_n = math.ceil(math.sqrt(n))
    for i in range(2, sqrt_n):
        if prime[i]:
            for j in range(2*i, n+1, i):
                prime[j] = False

    # return [i for i in range(n+1) if prime[i]] #素数列挙
    return prime


# n進数から10進数へ
def Base_n_to_10(X,n):
    out = 0
    for i in range(1,len(str(X))+1):                                    
        out += int(X[-i])*(n**(i-1))
    return out


# 10進数からn進数へ
def Base_10_to_n(X, n):
    if (int(X//n)):
        return Base_10_to_n(int(X//n), n)+str(X%n)
    return str(X%n)


# 重複組み合わせの総数
import math
def combr(n, r):
    return math.comb(n + r - 1, r)


# 二項係数をmod pで
def cmb(n, r, mod):
    if (r < 0) or (n < r):
        return 0
    r = min(r, n - r)
    return fact[n] * factinv[r] * factinv[n-r] % mod
mod=998244353
N = 10 ** 6  # N は必要分だけ用意する
fact = [1, 1]  # fact[n] = (n! mod p)
factinv = [1, 1]  # factinv[n] = ((n!)^(-1) mod p)
inv = [0, 1]  # factinv 計算用
for i in range(2, N + 1):
    fact.append((fact[-1] * i) % mod)
    inv.append((-inv[mod % i] * (mod // i)) % mod)
    factinv.append((factinv[-1] * inv[-1]) % mod)

# メモ化
from functools import lru_cache
@lru_cache(None)
def calc(x):
    pass


# https://tjkendev.github.io/procon-library/python/geometry/polygon_area.html
# 多角形の面積
def polygon_area(N, P):
    return abs(sum(P[i][0]*P[i-1][1] - P[i][1]*P[i-1][0] for i in range(N))) / 2.


# 最大フロー用の辺の構造体
class maxflow_edge:
	def __init__(self, to, cap, rev):
		self.to = to
		self.cap = cap
		self.rev = rev
                
def dfs(pos, goal, F, G, used):
	if pos == goal:
		return F # ゴールに到着：フローを流せる！
	# 探索する
	used[pos] = True
	for e in G[pos]:
		# 容量が 1 以上でかつ、まだ訪問していない頂点にのみ行く
		if e.cap > 0 and not used[e.to]:
			flow = dfs(e.to, goal, min(F, e.cap), G, used)
			# フローを流せる場合、残余グラフの容量を flow だけ増減させる
			if flow >= 1:
				e.cap -= flow
				G[e.to][e.rev].cap += flow
				return flow
	# すべての辺を探索しても見つからなかった…
	return 0

#  頂点 s から頂点 t までの最大フローの総流量を返す（頂点数 N、辺のリスト edges）
def maxflow(N, s, t, edges):
	# 初期状態の残余グラフを構築
	# （ここは書籍とは少し異なる実装をしているため、8 行目は G[a] に追加された後なので len(G[a]) - 1 となっていることに注意）
	G = [ list() for i in range(N + 1) ]
	for a, b, c in edges:
		G[a].append(maxflow_edge(b, c, len(G[b])))
		G[b].append(maxflow_edge(a, 0, len(G[a]) - 1))
	INF = 10 ** 10
	total_flow = 0
	while True:
		used = [ False ] * (N + 1)
		F = dfs(s, t, INF, G, used)
		if F > 0:
			total_flow += F
		else:
			break # フローを流せなくなったら、操作終了
	return total_flow


# ローリングハッシュ
class RollingHash:
    def __init__(self, string, base=29, mod=10 ** 9 + 7):
        self.base = base
        self.mod = mod
        self.hash = [0] * (len(string) + 1)
        self.base_mod = [1] * (len(string) + 1)

        for i in range(len(string)):
            self.hash[i + 1] = (base * self.hash[i] + ord(string[i]) - ord('a') + 1) % mod
            self.base_mod[i + 1] = self.base_mod[i] * base % mod

    def get(self, left, right):
        """[l,r)のハッシュ値を取得"""
        return (self.hash[right] - self.hash[left] * self.base_mod[right - left]) % self.mod

# フェニック木
class Fenwick_Tree:
    def __init__(self, n):
        self._n = n
        self.data = [0] * n
 
    def add(self, p, x):
        assert 0 <= p < self._n
        p += 1
        while p <= self._n:
            self.data[p - 1] += x
            p += p & -p
 
    def sum(self, l, r):
        assert 0 <= l <= r <= self._n
        return self._sum(r) - self._sum(l)
 
    def _sum(self, r):
        s = 0
        while r > 0:
            s += self.data[r - 1]
            r -= r & -r
        return s

# 区間加算・区間和取得するBIP
class RangeBIT:    
    def __init__(self, n):
        class BIT:
            def __init__(self, n):
                self.size = n
                self.arr = [0] * (n + 1)

            def __getitem__(self, i):
                return self.sum(i + 1) - self.sum(i)

            def sum(self, i):
                s = 0
                tmp = i
                while tmp:
                    s += self.arr[tmp]
                    tmp -= tmp & -tmp
                return s

            def add(self, i, x):
                tmp = i + 1
                while tmp <= self.size:
                    self.arr[tmp] += x
                    tmp += tmp & -tmp
        self.bit0 = BIT(n)
        self.bit1 = BIT(n)

    def add(self, i, j, x):
        self.bit0.add(i, -x * i)
        self.bit0.add(j, x * j)
        self.bit1.add(i, x)
        self.bit1.add(j, -x)

    def sum(self, i, j):
        si = self.bit0.sum(i) + self.bit1.sum(i) * i
        sj = self.bit0.sum(j) + self.bit1.sum(j) * j
        return sj - si

# 遅延セグ木(区間更新用)
class LazySegTree_RUQ:
    def __init__(self,init_val,segfunc,ide_ele):
        n = len(init_val)
        self.segfunc = segfunc
        self.ide_ele = ide_ele
        self.num = 1<<(n-1).bit_length()
        self.tree = [ide_ele]*2*self.num
        self.lazy = [None]*2*self.num
        for i in range(n):
            self.tree[self.num+i] = init_val[i]
        for i in range(self.num-1,0,-1):
            self.tree[i] = self.segfunc(self.tree[2*i],self.tree[2*i+1])
    def gindex(self,l,r):
        l += self.num
        r += self.num
        lm = l>>(l&-l).bit_length()
        rm = r>>(r&-r).bit_length()
        while r>l:
            if l<=lm:
                yield l
            if r<=rm:
                yield r
            r >>= 1
            l >>= 1
        while l:
            yield l
            l >>= 1
    def propagates(self,*ids):
        for i in reversed(ids):
            v = self.lazy[i]
            if v is None:
                continue
            self.lazy[i] = None
            self.lazy[2*i] = v
            self.lazy[2*i+1] = v
            self.tree[2*i] = v
            self.tree[2*i+1] = v
    def update(self,l,r,x):
        """
        [l, r)の区間をxに更新
        l: index(0-index)
        r: index(0-index)
        """
        ids = self.gindex(l,r)
        self.propagates(*self.gindex(l,r))
        l += self.num
        r += self.num
        while l<r:
            if l&1:
                self.lazy[l] = x
                self.tree[l] = x
                l += 1
            if r&1:
                self.lazy[r-1] = x
                self.tree[r-1] = x
            r >>= 1
            l >>= 1
        for i in ids:
            self.tree[i] = self.segfunc(self.tree[2*i], self.tree[2*i+1])
    def query(self,l,r):
        """
        [l, r)のsegfuncしたものを得る
        l: index(0-index)
        r: index(0-index)
        """
        ids = self.gindex(l,r)
        self.propagates(*self.gindex(l,r))
        res = self.ide_ele
        l += self.num
        r += self.num
        while l<r:
            if l&1:
                res = self.segfunc(res,self.tree[l])
                l += 1
            if r&1:
                res = self.segfunc(res,self.tree[r-1])
            l >>= 1
            r >>= 1
        return res

# 最小:min(x,y) 最大:max(x,y)
def segfunc(x, y):
    pass
    # return 
# 最小:INF 最大:-INF
ide_ele = 0

# 遅延セグ木(区間更新・区間加算用)
class LazySegTree_RSQ_RUQ:
    def __init__(self, init_list, func, ide_ele):
        n = len(init_list)
        self.func = func
        self.ide_ele = ide_ele
        self.tree_height = (n - 1).bit_length()
        self.num = 1 << self.tree_height
        self.tree = [self.ide_ele] * 2 * self.num
        self.lazy = [None] * 2 * self.num

        for i in range(n):
            self.tree[self.num + i] = init_list[i]
        for i in range(self.num - 1, 0, -1):
            self.tree[i] = self.func(self.tree[2 * i], self.tree[2 * i + 1])

    def _get_index(self, left, right):
        """
        [left, right) で伝播するインデックスを収集
        """
        # 最下層から上に調べていく
        # 最下層の伝播範囲 [i_left, i_right)
        i_left = left + self.num
        i_right = right + self.num

        # bit列で右側から見て連続してる 0 の数 (Number of Training Zero)
        l_ntz = (i_left & -i_left).bit_length() - 1
        r_ntz = (i_right & -i_right).bit_length() - 1

        indexes = []
        for i in range(self.tree_height):
            # 下から i 番目の伝播範囲 [i_left, i_right)
            i_left >>= 1
            i_right >>= 1
            if r_ntz <= i:
                indexes.append(i_right)
            if i_left < i_right and l_ntz <= i:
                indexes.append(i_left)
        return indexes

    def _propagate(self, indexes):
        # 上から伝播していくので reversed
        for i in reversed(indexes):
            v = self.lazy[i]
            if v is None:
                continue
            v >>= 1
            self.lazy[2 * i] = v
            self.tree[2 * i] = v
            self.lazy[2 * i + 1] = v
            self.tree[2 * i + 1] = v
            self.lazy[i] = None

    def update(self, left, right, x):
        """区間 [left, right) を x で更新"""
        indexes = self._get_index(left, right)
        self._propagate(indexes)

        # 木の最下層から update
        left += self.num
        right += self.num
        while left < right:
            if right & 1:
                self.lazy[right - 1] = x
                self.tree[right - 1] = x
            if left & 1:
                self.lazy[left] = x
                self.tree[left] = x
                left += 1
            left >>= 1
            right >>= 1
            x <<= 1
        for i in indexes:
            self.tree[i] = self.func(self.tree[2 * i], self.tree[2 * i + 1])

    def query(self, left, right):
        """区間 [left, right) で query"""
        indexes = self._get_index(left, right)
        self._propagate(indexes)

        # 木の最下層から query
        left += self.num
        right += self.num
        res = self.ide_ele
        while left < right:
            if right & 1:
                res = self.func(res, self.tree[right - 1])
            if left & 1:
                res = self.func(res, self.tree[left])
                left += 1
            left >>= 1
            right >>= 1
        return res

def segfunc(x, y):
    return x+y
ide_ele = 0

# https://output-zakki.com/topological_sort/
# トポロジカルソート
from collections import deque
def topological_sort(G, into_num):
    #入ってくる有向辺を持たないノードを列挙
    q = deque()
    #V: 頂点数
    for i in range(V):
        if into_num[i]==0:
            q.append(i)
    
    #以下、幅優先探索
    ans = []
    while q:
        v = q.popleft()
        ans.append(v)
        for adj in G[v]:
            into_num[adj] -= 1 #入次数を減らす
            if into_num[adj]==0:
                q.append(adj) #入次数が0になったら、キューに入れる
    
    return ans


# ポテンシャル付きUnion-Find
from typing import Callable, Generic, TypeVar, List
T = TypeVar('T')
# 初期化関数
def init() -> int:
    return 0  # 整数の場合、単位元は 0 とします

# 2項間加算関数
def add(a: int, b: int) -> int:
    return a + b

# 逆関数
def sub(a: int, b: int) -> int:
    return a - b
 
class UnionFindWithPotential(Generic[T]):
 
    def __init__(self,
                 n: int,
                 init: Callable[[], T],
                 func: Callable[[T, T], T],
                 rev_func: Callable[[T, T], T]):
        """
        :param n:
        :param init: 単位元の生成関数
        :param func: 2項間加算関数（add）
        :param rev_func: 逆関数（sub）
        """
        self.table: List[int] = [-1] * n
        self.values: List[T] = [init() for _ in range(n)]
        self.init: Callable[[], T] = init
        self.func: Callable[[T, T], T] = func
        self.rev_func: Callable[[T, T], T] = rev_func
 
    def root(self, x: int) -> int:
        stack = []
        tbl = self.table
        vals = self.values
 
        while tbl[x] >= 0:
            stack.append(x)
            x = tbl[x]
        if stack:
            val = self.init()
            while stack:
                y = stack.pop()
                val = self.func(val, vals[y])
                vals[y] = val
                tbl[y] = x
        return x
 
    def is_same(self, x: int, y: int) -> bool:
        return self.root(x) == self.root(y)
 
    def diff(self, x: int, y: int) -> T:
        """
        x と y の差（y - x）を取得。同じグループに属さない場合は None。
        """
        if not self.is_same(x, y):
            return None
        vx = self.values[x]
        vy = self.values[y]
        return self.rev_func(vy, vx)
 
    def unite(self, x: int, y: int, d: T) -> bool:
        """
        x と y のグループを、y - x = d となるように統合。
        既に x と y が同グループで、矛盾する場合は AssertionError。矛盾しない場合はFalse。
        同グループで無く、新たな統合が発生した場合はTrue。
        """
        rx = self.root(x)
        ry = self.root(y)
        vx = self.values[x]
        vy = self.values[y]
        if rx == ry:
            assert self.rev_func(vy, vx) == d
            return False
 
        rd = self.rev_func(self.func(vx, d), vy)
        self.table[rx] += self.table[ry]
        self.table[ry] = rx
        self.values[ry] = rd
        return True
 
    def get_size(self, x: int) -> int:
        return -self.table[self.root(x)]


# 行列累乗
class Matrix():

    def __init__(self, MOD=-1):
        """剰余計算する場合はMODを指定"""
        self.MOD = MOD

    def mul(self, a, b):
        L, M, N = len(a), len(b), len(b[0])
        assert len(a[0]) == M
        c = [[0] * N for _ in range(L)]
        for i in range(L):
            for j in range(N):
                for k in range(M):
                    c[i][j] += a[i][k] * b[k][j]
                    if self.MOD != -1:
                        c[i][j] %= self.MOD
        return c
    
    def pow(self, x, n):
        y = [[0] * len(x) for _ in range(len(x))]
        for i in range(len(x)):
            y[i][i] = 1
        while n > 0:
            if n & 1:
                y = self.mul(x, y)
            x = self.mul(x, x)
            n >>= 1
        return y


# 強連結成分分解
def scc(N,edges):
    M=len(edges)
    start=[0]*(N+1)
    elist=[0]*M
    for e in edges:
        start[e[0]+1]+=1
    for i in range(1,N+1):
        start[i]+=start[i-1]
    counter=start[:]
    for e in edges:
        elist[counter[e[0]]]=e[1]
        counter[e[0]]+=1
    visited=[]
    low=[0]*N
    Ord=[-1]*N
    ids=[0]*N
    NG=[0,0]
    def dfs(v):
        stack=[(v,-1,0),(v,-1,1)]
        while stack:
            v,bef,t=stack.pop()
            if t:
                if bef!=-1 and Ord[v]!=-1:
                    low[bef]=min(low[bef],Ord[v])
                    stack.pop()
                    continue
                low[v]=NG[0]
                Ord[v]=NG[0]
                NG[0]+=1
                visited.append(v)
                for i in range(start[v],start[v+1]):
                    to=elist[i]
                    if Ord[to]==-1:
                        stack.append((to,v,0))
                        stack.append((to,v,1))
                    else:
                        low[v]=min(low[v],Ord[to])
            else:
                if low[v]==Ord[v]:
                    while(True):
                        u=visited.pop()
                        Ord[u]=N
                        ids[u]=NG[1]
                        if u==v:
                            break
                    NG[1]+=1
                low[bef]=min(low[bef],low[v])
    for i in range(N):
        if Ord[i]==-1:
            dfs(i)
    for i in range(N):
        ids[i]=NG[1]-1-ids[i]
    group_num=NG[1]
    counts=[0]*group_num
    for x in ids:
        counts[x]+=1
    groups=[[] for i in range(group_num)]
    for i in range(N):
        groups[ids[i]].append(i)
    return groups

# 最小共通祖先(LCA)

# N: 頂点数
# G[v]: 頂点vの子頂点 (親頂点は含まない)
N = n
G = [[] for i in range(N)]

# Euler Tour の構築
S = []
F = [0]*N
depth = [0]*N
def dfs(v, d):
    F[v] = len(S)
    depth[v] = d
    S.append(v)
    for w in G[v]:
        dfs(w, d+1)
        S.append(v)
dfs(0, 0)

# 存在しない範囲は深さが他よりも大きくなるようにする
INF = (N, None)

# LCAを計算するクエリの前計算
M = 2*N
M0 = 2**(M-1).bit_length()
data = [INF]*(2*M0)
for i, v in enumerate(S):
    data[M0-1+i] = (depth[v], i)
for i in range(M0-2, -1, -1):
    data[i] = min(data[2*i+1], data[2*i+2])

# LCAの計算 (generatorで最小値を求める)
def _query(a, b):
    yield INF
    a += M0; b += M0
    while a < b:
        if b & 1:
            b -= 1
            yield data[b-1]
        if a & 1:
            yield data[a-1]
            a += 1
        a >>= 1; b >>= 1

# LCAの計算 (外から呼び出す関数)
def query(u, v):
    fu = F[u]; fv = F[v]
    if fu > fv:
        fu, fv = fv, fu
    return S[min(_query(fu, fv+1))[1]]

# 繰り返し二乗法を用いたpow
def pow(A, B, p):
    res = 1  # Initialize result
    A = A % p  # Update A if it is more , than or equal to p
    if (A == 0):
        return 0
    while (B > 0):
        if ((B & 1) == 1):  # If B is odd, multiply, A with result
            res = (res * A) % p
 
        B = B >> 1  # B = B/2
        A = (A * A) % p
    return res

# LIS(最長増加部分列)
import bisect
seq = []
LIS = [seq[0]]
for i in range(len(seq)):
    if seq[i] > LIS[-1]:
        LIS.append(seq[i])
    else:
        LIS[bisect.bisect_left(LIS, seq[i])] = seq[i]

# LCS(最長共通部分列)
def lcs(S, T):
    L1 = len(S)
    L2 = len(T)
    dp = [[0]*(L2+1) for i in range(L1+1)]

    for i in range(L1-1, -1, -1):
        for j in range(L2-1, -1, -1):
            r = max(dp[i+1][j], dp[i][j+1])
            if S[i] == T[j]:
                r = max(r, dp[i+1][j+1] + 1)
            dp[i][j] = r

    # dp[0][0] が長さの解

    # ここからは復元処理
    res = []
    i = 0; j = 0
    while i < L1 and j < L2:
        if S[i] == T[j]:
            res.append(S[i])
            i += 1; j += 1
        elif dp[i][j] == dp[i+1][j]:
            i += 1
        elif dp[i][j] == dp[i][j+1]:
            j += 1
    return "".join(res)

# Mo's Algorithm
# https://qiita.com/hyouchun/items/55877339a682e018bf1b
import math
from operator import itemgetter

class MoStatus():
    def __init__(self, max_element):
        self.cnt = [0] * (max_element + 1)
        self.val = 0

    def add(self, element):
        self.cnt[element] += 1
        # TODO:valの更新

    def discard(self, element):
        self.cnt[element] -= 1
        # TODO:valの更新

class Mo():
    def __init__(self, lis, init_queries):
        self.N = len(lis)
        self.lis = lis

        self.Q = len(init_queries)
        self.max_r = -1
        self.init_queries = []
        for qi, query in enumerate(init_queries):
            l, r = query
            self.init_queries.append((l, r, qi))
            if self.max_r < r:
                self.max_r = r

        self.status = MoStatus(max_element=max(self.lis))

        self.section_width = None
        self.separate_cnt = None
        self.separated_queries = None
        self.separated_queries_generator()

        self.ans = [0] * self.Q
        self.solve()

    def separated_queries_generator(self):
        self.section_width = int(
            math.sqrt(3) * self.max_r / math.sqrt(2 * self.Q)) + 1
        self.separate_cnt = (
                                    self.max_r + self.section_width - 1) // self.section_width
        self.separated_queries = [[] for _ in range(self.separate_cnt + 1)]
        for query in self.init_queries:
            l, r, qi = query
            idx = l // self.section_width
            self.separated_queries[idx].append(query)
        for i in range(self.separate_cnt):
            self.separated_queries[i].sort(key=itemgetter(1), reverse=i % 2)

    def solve(self):
        prev_l, prev_r = 0, -1
        for queries_list in self.separated_queries:
            for query in queries_list:
                nl, nr, qi = query
                if nl < prev_l:
                    for i in range(nl, prev_l):
                        element = self.lis[i]
                        self.status.add(element)
                else:
                    for i in range(prev_l, nl):
                        element = self.lis[i]
                        self.status.discard(element)
                if prev_r < nr:
                    for i in range(nr, prev_r, -1):
                        element = self.lis[i]
                        self.status.add(element)
                else:
                    for i in range(prev_r, nr, -1):
                        element = self.lis[i]
                        self.status.discard(element)
                prev_l, prev_r = nl, nr
                self.ans[qi] = self.status.val


# 全方位木DP (ReRooting)
from collections.abc import Callable
DataType = int
CostType = int
class RerootingTreeDP:
    __slots__ = ('N', 'G')

    def __init__(self, vertex_num: int) -> None:
        self.N = vertex_num
        self.G: list[list[tuple[int, CostType]]] = [[] for _ in range(vertex_num)]

    def add_directional_edge(self, u: int, v: int, cost: CostType = 1) -> None:
        G = self.G
        G[u].append((v, cost))

    def solve(self,
              merge: Callable[[DataType, DataType], DataType],
              e: Callable[[], DataType],
              leaf: Callable[[], DataType],
              apply: Callable[[DataType, int, int, CostType], DataType]
              ) -> list[DataType]:
        '''
        Args:
          merge: (child-data-1, child-data-2) -> merged-data
          e: () -> zero-data
          leaf: () -> leaf-data
          apply: (child-data, child-node, parent-node, cost-of-child-to-parent) -> parent-data-after-one-child-applied
        '''
        N, G = self.N, self.G
        P, O, Pc = self._dfs()

        to_leaf = [leaf() for _ in range(N)]
        for v in reversed(O):
            p = P[v]
            if p != v:
                to_leaf[p] = merge(to_leaf[p], apply(to_leaf[v], v, p, Pc[v]))

        to_root = [e() for _ in range(N)]
        ans = [e() for _ in range(N)]
        for v in O:
            p = P[v]

            to_nv: list[DataType] = []
            push = to_nv.append
            for nv, c in G[v]:
                if nv == p:
                    push(apply(to_root[v], nv, v, c))
                else:
                    push(apply(to_leaf[nv], nv, v, c))

            L = [e()]
            push = L.append
            for dp in to_nv:
                push(merge(L[-1], dp))
            R = [e()]
            push = R.append
            for dp in reversed(to_nv):
                push(merge(R[-1], dp))
            R.reverse()

            ans[v] = L[-1]

            for i, (nv, c) in enumerate(G[v]):
                if nv == p: continue
                to_root[nv] = merge(L[i], R[i + 1])
        return ans

    def _dfs(self, root=0) -> tuple[list[int], list[int], list[int]]:
        N, G = self.N, self.G
        parent = [-1] * N
        parent[root] = root
        order: list[int] = []
        append_order = order.append
        parent_cost = [0] * N
        stk = [root]
        push, pop = stk.append, stk.pop
        while stk:
            v = pop()
            append_order(v)
            for nv, c in G[v]:
                if parent[nv] >= 0: continue
                parent[nv] = v
                parent_cost[nv] = c
                push(nv)
        return (parent, order, parent_cost)

# Merge Sort Tree
from bisect import bisect
from heapq import merge

class MergesortTree:
    def __init__(self, arr):
        self.N0 = 2 ** (len(arr) - 1).bit_length()
        self.data = [None] * (2 * self.N0)
        self.data_acc = [[0] for _ in range(2 * self.N0)]
        for i, a in enumerate(arr):
            self.data[self.N0 - 1 + i] = [a]
            self.data_acc[self.N0 - 1 + i].append(a)
        for i in range(len(arr), self.N0):
            self.data[self.N0 - 1 + i] = []
        for i in range(self.N0 - 2, -1, -1):
            (*self.data[i],) = merge(self.data[2 * i + 1], self.data[2 * i + 2])
            for tmp in self.data[i]:
                self.data_acc[i].append(self.data_acc[i][-1] + tmp)

    # Return count of A_i where A_i < k in [l, r) and their sum.
    def query1(self, l, r, k):
        L = l + self.N0
        R = r + self.N0
        cnt = 0
        tot = 0
        while L < R:
            if R & 1:
                R -= 1
                idx = bisect(self.data[R - 1], k - 1)
                cnt += idx
                tot += self.data_acc[R - 1][idx]
            if L & 1:
                idx = bisect(self.data[L - 1], k - 1)
                cnt += idx
                tot += self.data_acc[L - 1][idx]
                L += 1
            L >>= 1
            R >>= 1
        return cnt, tot

    # Return count and sum elements A_i in [l, r) where a <= A_i < b.
    def query(self, l, r, a, b):
        L = l + self.N0
        R = r + self.N0
        cnt = 0
        tot = 0
        while L < R:
            if R & 1:
                R -= 1
                b_idx = bisect(self.data[R - 1], b - 1)
                a_idx = bisect(self.data[R - 1], a - 1)
                cnt += b_idx - a_idx
                tot += self.data_acc[R - 1][b_idx] - self.data_acc[R - 1][a_idx]
            if L & 1:
                b_idx = bisect(self.data[L - 1], b - 1)
                a_idx = bisect(self.data[L - 1], a - 1)
                cnt += b_idx - a_idx
                tot += self.data_acc[L - 1][b_idx] - self.data_acc[L - 1][a_idx]
                L += 1
            L >>= 1
            R >>= 1
        return cnt, tot

# 拡張ユークリッド互除法
def extended_gcd(a, b):
    """
    2つの整数aとbの最大公約数（GCD）を拡張ユークリッド互除法で求め、
    さらに、GCDをaとbの線形結合として表す係数xとyを求める。

    Returns:
    tuple: (gcd, x, y) 
    gcdはaとbの最大公約数、xとyはa*x + b*y = gcd(a, b)を満たす係数。

    例:
    >>> gcd, x, y = extended_gcd(30, 20)
    >>> gcd
    10
    >>> x
    -1
    >>> y
    2
    >>> 30 * x + 20 * y
    10
    """
    if a == 0:
        return b, 0, 1
    else:
        gcd, x1, y1 = extended_gcd(b % a, a)
        x = y1 - (b // a) * x1
        y = x1
        return gcd, x, y

# 中国剰余定理
def crt(r1 :int, m1 :int, r2 :int, m2 :int):
    # x = r1 (mod m1),  x = r2 (mod m2)
    # <-> x = r3 (mod m3)
 
    g,p,q = extended_gcd(m1, m2)
 
    if (r2 - r1) % g != 0:
        return 0, -1
    
    m3 = m1 * m2 // g  # lcm of m1 and m2
    r3 = r1 + m1 * ((r2 - r1) // g * p)
    r3 %= m3
 
    return (r3, m3)

# Z algorithm
def z_algorithm(s):
    '''
    入力された文字列またはリストのZ配列を返す
    '''

    if isinstance(s, str):
        s = [ord(c) for c in s]

    n = len(s)
    if n == 0:
        return []

    z = [0] * n
    j = 0
    for i in range(1, n):
        z[i] = 0 if j + z[j] <= i else min(j + z[j] - i, z[i - j])
        while i + z[i] < n and s[z[i]] == s[i + z[i]]:
            z[i] += 1
        if j + z[j] < i + z[i]:
            j = i
    z[0] = n

    return z

import functools
import typing
# Suffix Array
class SuffixArray:
    def __init__(self, s: typing.Union[str, typing.List[int]], upper: typing.Optional[int] = None):
        '''
        サフィックス配列とLCP配列を構築するためのクラス
        
        引数:
        s: 処理する文字列または整数のリスト。
        upper: sが整数リストの場合の最大値（デフォルトはNone）。
        '''
        self.s = s
        self.upper = upper
        self.sa = self._suffix_array()
        self.lcp = self._lcp_array()

    def _sa_naive(self, s: typing.List[int]) -> typing.List[int]:
        sa = list(range(len(s)))
        return sorted(sa, key=lambda i: s[i:])

    def _sa_doubling(self, s: typing.List[int]) -> typing.List[int]:
        n = len(s)
        sa = list(range(n))
        rnk = s.copy()
        tmp = [0] * n
        k = 1
        while k < n:
            def cmp(x: int, y: int) -> int:
                if rnk[x] != rnk[y]:
                    return rnk[x] - rnk[y]
                rx = rnk[x + k] if x + k < n else -1
                ry = rnk[y + k] if y + k < n else -1
                return rx - ry
            sa.sort(key=functools.cmp_to_key(cmp))
            tmp[sa[0]] = 0
            for i in range(1, n):
                tmp[sa[i]] = tmp[sa[i - 1]] + (1 if cmp(sa[i - 1], sa[i]) else 0)
            tmp, rnk = rnk, tmp
            k *= 2
        return sa

    def _sa_is(self, s: typing.List[int], upper: int) -> typing.List[int]:
        threshold_naive = 10
        threshold_doubling = 40

        n = len(s)

        if n == 0:
            return []
        if n == 1:
            return [0]
        if n == 2:
            if s[0] < s[1]:
                return [0, 1]
            else:
                return [1, 0]

        if n < threshold_naive:
            return self._sa_naive(s)
        if n < threshold_doubling:
            return self._sa_doubling(s)

        sa = [0] * n
        ls = [False] * n
        for i in range(n - 2, -1, -1):
            if s[i] == s[i + 1]:
                ls[i] = ls[i + 1]
            else:
                ls[i] = s[i] < s[i + 1]

        sum_l = [0] * (upper + 1)
        sum_s = [0] * (upper + 1)
        for i in range(n):
            if not ls[i]:
                sum_s[s[i]] += 1
            else:
                sum_l[s[i] + 1] += 1
        for i in range(upper + 1):
            sum_s[i] += sum_l[i]
            if i < upper:
                sum_l[i + 1] += sum_s[i]

        def induce(lms: typing.List[int]) -> None:
            nonlocal sa
            sa = [-1] * n

            buf = sum_s.copy()
            for d in lms:
                if d == n:
                    continue
                sa[buf[s[d]]] = d
                buf[s[d]] += 1

            buf = sum_l.copy()
            sa[buf[s[n - 1]]] = n - 1
            buf[s[n - 1]] += 1
            for i in range(n):
                v = sa[i]
                if v >= 1 and not ls[v - 1]:
                    sa[buf[s[v - 1]]] = v - 1
                    buf[s[v - 1]] += 1

            buf = sum_l.copy()
            for i in range(n - 1, -1, -1):
                v = sa[i]
                if v >= 1 and ls[v - 1]:
                    buf[s[v - 1] + 1] -= 1
                    sa[buf[s[v - 1] + 1]] = v - 1

        lms_map = [-1] * (n + 1)
        m = 0
        for i in range(1, n):
            if not ls[i - 1] and ls[i]:
                lms_map[i] = m
                m += 1
        lms = []
        for i in range(1, n):
            if not ls[i - 1] and ls[i]:
                lms.append(i)

        induce(lms)

        if m:
            sorted_lms = []
            for v in sa:
                if lms_map[v] != -1:
                    sorted_lms.append(v)
            rec_s = [0] * m
            rec_upper = 0
            rec_s[lms_map[sorted_lms[0]]] = 0
            for i in range(1, m):
                left = sorted_lms[i - 1]
                right = sorted_lms[i]
                if lms_map[left] + 1 < m:
                    end_l = lms[lms_map[left] + 1]
                else:
                    end_l = n
                if lms_map[right] + 1 < m:
                    end_r = lms[lms_map[right] + 1]
                else:
                    end_r = n

                same = True
                if end_l - left != end_r - right:
                    same = False
                else:
                    while left < end_l:
                        if s[left] != s[right]:
                            break
                        left += 1
                        right += 1
                    if left == n or s[left] != s[right]:
                        same = False

                if not same:
                    rec_upper += 1
                rec_s[lms_map[sorted_lms[i]]] = rec_upper

            rec_sa = self._sa_is(rec_s, rec_upper)

            for i in range(m):
                sorted_lms[i] = lms[rec_sa[i]]
            induce(sorted_lms)

        return sa

    def _suffix_array(self) -> typing.List[int]:
        '''
        SA-IS, リニアタイムサフィックス配列構築
        参考:
        G. Nong, S. Zhang, and W. H. Chan,
        Two Efficient Algorithms for Linear Time Suffix Array Construction
        '''
        s = self.s
        upper = self.upper

        if isinstance(s, str):
            return self._sa_is([ord(c) for c in s], 255)
        elif upper is None:
            n = len(s)
            idx = list(range(n))

            def cmp(left: int, right: int) -> int:
                return typing.cast(int, s[left]) - typing.cast(int, s[right])

            idx.sort(key=functools.cmp_to_key(cmp))
            s2 = [0] * n
            now = 0
            for i in range(n):
                if i and s[idx[i - 1]] != s[idx[i]]:
                    now += 1
                s2[idx[i]] = now
            return self._sa_is(s2, now)
        else:
            assert 0 <= upper
            for d in s:
                assert 0 <= d <= upper

            return self._sa_is(s, upper)

    def _lcp_array(self) -> typing.List[int]:
        '''
        最長共通接頭辞(LCP)配列を計算します。
        参考:
        T. Kasai, G. Lee, H. Arimura, S. Arikawa, and K. Park,
        Linear-Time Longest-Common-Prefix Computation in Suffix Arrays and Its
        Applications

        返り値:
        LCP配列 (リスト): 入力された文字列または整数リストのLCP配列。
        '''
        s = self.s
        sa = self.sa

        if isinstance(s, str):
            s = [ord(c) for c in s]

        n = len(s)
        assert n >= 1

        rnk = [0] * n
        for i in range(n):
            rnk[sa[i]] = i

        lcp = [0] * (n - 1)
        h = 0
        for i in range(n):
            if h > 0:
                h -= 1
            if rnk[i] == 0:
                continue
            j = sa[rnk[i] - 1]
            while j + h < n and i + h < n:
                if s[j + h] != s[i + h]:
                    break
                h += 1
            lcp[rnk[i] - 1] = h

        return lcp

# LowLink
def lowLink(edges):
    """
    edges[from] = [to1, to2, ...]
    """
    n = len(edges)
    ord = [-1] * n
    low = [-1] * n
    isartic = [False] * n
    bridge = []

    def dfs(root, k):
        x = root * (n + 1) + n
        stack = [~x, x]
        cnt = 0
        while stack:
            tmp = stack.pop()
            if tmp >= 0:
                pos = tmp // (n + 1)
                bpos = tmp - (n + 1) * pos

                if bpos != n and ord[pos] != -1:
                    low[bpos] = min(low[bpos], ord[pos])
                    stack.pop()
                    continue

                low[pos] = ord[pos] = k
                k += 1
                for npos in edges[pos]:
                    if npos == bpos:
                        continue

                    if ord[npos] == -1:
                        if bpos == n:
                            cnt += 1
                        x = npos * (n + 1) + pos
                        stack.append(~x)
                        stack.append(x)
                    else:
                        low[pos] = min(low[pos], ord[npos])
                        if npos == root:
                            cnt -= 1
            else:
                tmp = ~tmp
                pos = tmp // (n + 1)
                bpos = tmp - (n + 1) * pos

                if bpos != n and ord[bpos] < low[pos]:
                    bridge.append((min(pos, bpos), max(pos, bpos)))

                if bpos != n and bpos != root and low[pos] >= ord[bpos]:
                    isartic[bpos] = True

                if bpos != n:
                    low[bpos] = min(low[bpos], low[pos])

        if cnt >= 2:
            isartic[root] = True

        return k

    k = 0
    for i in range(n):
        if ord[i] == -1:
            k = dfs(i, k)

    return isartic, bridge

# convolution
class FFT():
    def primitive_root_constexpr(self,m):
        if m==2:return 1
        if m==167772161:return 3
        if m==469762049:return 3
        if m==754974721:return 11
        if m==998244353:return 3
        divs=[0]*20
        divs[0]=2
        cnt=1
        x=(m-1)//2
        while(x%2==0):x//=2
        i=3
        while(i*i<=x):
            if (x%i==0):
                divs[cnt]=i
                cnt+=1
                while(x%i==0):
                    x//=i
            i+=2
        if x>1:
            divs[cnt]=x
            cnt+=1
        g=2
        while(1):
            ok=True
            for i in range(cnt):
                if pow(g,(m-1)//divs[i],m)==1:
                    ok=False
                    break
            if ok:
                return g
            g+=1
    def bsf(self,x):
        res=0
        while(x%2==0):
            res+=1
            x//=2
        return res
    rank2=0
    root=[]
    iroot=[]
    rate2=[]
    irate2=[]
    rate3=[]
    irate3=[]
    
    def __init__(self,MOD):
        self.mod=MOD
        self.g=self.primitive_root_constexpr(self.mod)
        self.rank2=self.bsf(self.mod-1)
        self.root=[0 for i in range(self.rank2+1)]
        self.iroot=[0 for i in range(self.rank2+1)]
        self.rate2=[0 for i in range(self.rank2)]
        self.irate2=[0 for i in range(self.rank2)]
        self.rate3=[0 for i in range(self.rank2-1)]
        self.irate3=[0 for i in range(self.rank2-1)]
        self.root[self.rank2]=pow(self.g,(self.mod-1)>>self.rank2,self.mod)
        self.iroot[self.rank2]=pow(self.root[self.rank2],self.mod-2,self.mod)
        for i in range(self.rank2-1,-1,-1):
            self.root[i]=(self.root[i+1]**2)%self.mod
            self.iroot[i]=(self.iroot[i+1]**2)%self.mod
        prod=1;iprod=1
        for i in range(self.rank2-1):
            self.rate2[i]=(self.root[i+2]*prod)%self.mod
            self.irate2[i]=(self.iroot[i+2]*iprod)%self.mod
            prod=(prod*self.iroot[i+2])%self.mod
            iprod=(iprod*self.root[i+2])%self.mod
        prod=1;iprod=1
        for i in range(self.rank2-2):
            self.rate3[i]=(self.root[i+3]*prod)%self.mod
            self.irate3[i]=(self.iroot[i+3]*iprod)%self.mod
            prod=(prod*self.iroot[i+3])%self.mod
            iprod=(iprod*self.root[i+3])%self.mod
    def butterfly(self,a):
        n=len(a)
        h=(n-1).bit_length()
        
        LEN=0
        while(LEN<h):
            if (h-LEN==1):
                p=1<<(h-LEN-1)
                rot=1
                for s in range(1<<LEN):
                    offset=s<<(h-LEN)
                    for i in range(p):
                        l=a[i+offset]
                        r=a[i+offset+p]*rot
                        a[i+offset]=(l+r)%self.mod
                        a[i+offset+p]=(l-r)%self.mod
                    rot*=self.rate2[(~s & -~s).bit_length()-1]
                    rot%=self.mod
                LEN+=1
            else:
                p=1<<(h-LEN-2)
                rot=1
                imag=self.root[2]
                for s in range(1<<LEN):
                    rot2=(rot*rot)%self.mod
                    rot3=(rot2*rot)%self.mod
                    offset=s<<(h-LEN)
                    for i in range(p):
                        a0=a[i+offset]
                        a1=a[i+offset+p]*rot
                        a2=a[i+offset+2*p]*rot2
                        a3=a[i+offset+3*p]*rot3
                        a1na3imag=(a1-a3)%self.mod*imag
                        a[i+offset]=(a0+a2+a1+a3)%self.mod
                        a[i+offset+p]=(a0+a2-a1-a3)%self.mod
                        a[i+offset+2*p]=(a0-a2+a1na3imag)%self.mod
                        a[i+offset+3*p]=(a0-a2-a1na3imag)%self.mod
                    rot*=self.rate3[(~s & -~s).bit_length()-1]
                    rot%=self.mod
                LEN+=2
                
    def butterfly_inv(self,a):
        n=len(a)
        h=(n-1).bit_length()
        LEN=h
        while(LEN):
            if (LEN==1):
                p=1<<(h-LEN)
                irot=1
                for s in range(1<<(LEN-1)):
                    offset=s<<(h-LEN+1)
                    for i in range(p):
                        l=a[i+offset]
                        r=a[i+offset+p]
                        a[i+offset]=(l+r)%self.mod
                        a[i+offset+p]=(l-r)*irot%self.mod
                    irot*=self.irate2[(~s & -~s).bit_length()-1]
                    irot%=self.mod
                LEN-=1
            else:
                p=1<<(h-LEN)
                irot=1
                iimag=self.iroot[2]
                for s in range(1<<(LEN-2)):
                    irot2=(irot*irot)%self.mod
                    irot3=(irot*irot2)%self.mod
                    offset=s<<(h-LEN+2)
                    for i in range(p):
                        a0=a[i+offset]
                        a1=a[i+offset+p]
                        a2=a[i+offset+2*p]
                        a3=a[i+offset+3*p]
                        a2na3iimag=(a2-a3)*iimag%self.mod
                        a[i+offset]=(a0+a1+a2+a3)%self.mod
                        a[i+offset+p]=(a0-a1+a2na3iimag)*irot%self.mod
                        a[i+offset+2*p]=(a0+a1-a2-a3)*irot2%self.mod
                        a[i+offset+3*p]=(a0-a1-a2na3iimag)*irot3%self.mod
                    irot*=self.irate3[(~s & -~s).bit_length()-1]
                    irot%=self.mod
                LEN-=2
    def convolution(self,a,b):
        n=len(a);m=len(b)
        if not(a) or not(b):
            return []
        if min(n,m)<=40:
            res=[0]*(n+m-1)
            for i in range(n):
                for j in range(m):
                    res[i+j]+=a[i]*b[j]
                    res[i+j]%=self.mod
            return res
        z=1<<((n+m-2).bit_length())
        a=a+[0]*(z-n)
        b=b+[0]*(z-m)
        self.butterfly(a)
        self.butterfly(b)
        c=[(a[i]*b[i])%self.mod for i in range(z)]
        self.butterfly_inv(c)
        iz=pow(z,self.mod-2,self.mod)
        for i in range(n+m-1):
            c[i]=(c[i]*iz)%self.mod
        return c[:n+m-1]
