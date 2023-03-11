
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

# 素数判定
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

    return prime

# n進数から10進数へ
def Base_n_to_10(X,n):
    out = 0
    for i in range(1,len(str(X))+1):                                    
        out += int(X[-i])*(n**(i-1))
    return out

# 10進数からn進数へ
def Base_10_to_n(X, n):
    if (int(X/n)):
        return Base_10_to_n(int(X/n), n)+str(X%n)
    return str(X%n)

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

# lcm
from math import gcd
def lcm(a,b):
    return a//gcd(a,b)*b

# メモ化
from functools import lru_cache
@lru_cache
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
        return (self.hash[right] - self.hash[left] * self.base_mod[right - left]) % self.mod

# https://qiita.com/takayg1/items/c811bd07c21923d7ec69
# セグメント木
class SegTree:
    """
    init(init_val, ide_ele): 配列init_valで初期化 O(N)
    update(k, x): k番目の値をxに更新 O(logN)
    query(l, r): 区間[l, r)をsegfuncしたものを返す O(logN)
    """
    def __init__(self, init_val, segfunc, ide_ele):
        """
        init_val: 配列の初期値
        segfunc: 区間にしたい操作
        ide_ele: 単位元
        n: 要素数
        num: n以上の最小の2のべき乗
        tree: セグメント木(1-index)
        """
        n = len(init_val)
        self.segfunc = segfunc
        self.ide_ele = ide_ele
        self.num = 1 << (n - 1).bit_length()
        self.tree = [ide_ele] * 2 * self.num
        # 配列の値を葉にセット
        for i in range(n):
            self.tree[self.num + i] = init_val[i]
        # 構築していく
        for i in range(self.num - 1, 0, -1):
            self.tree[i] = self.segfunc(self.tree[2 * i], self.tree[2 * i + 1])

    def update(self, k, x):
        """
        k番目の値をxに更新
        k: index(0-index)
        x: update value
        """
        k += self.num
        self.tree[k] = x
        while k > 1:
            self.tree[k >> 1] = self.segfunc(self.tree[k], self.tree[k ^ 1])
            k >>= 1

    def query(self, l, r):
        """
        [l, r)のsegfuncしたものを得る
        l: index(0-index)
        r: index(0-index)
        """
        res = self.ide_ele

        l += self.num
        r += self.num
        while l < r:
            if l & 1:
                res = self.segfunc(res, self.tree[l])
                l += 1
            if r & 1:
                res = self.segfunc(res, self.tree[r - 1])
            l >>= 1
            r >>= 1
        return res
# 最小:min(x,y) 最大:max(x,y) 区間和:x+y 区間積:x*y 最大公約数 math.gcd(x,y)
def segfunc(x, y):
    return 
INF=float('inf')
# 最小:INF 最大:-INF 区間和:0 区間積:1 最大公約数 0
ide_ele = 

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


