
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
mod = 10 ** 9 + 7
N = 10 ** 6  # N は必要分だけ用意する
fact = [1, 1]  # fact[n] = (n! mod p)
factinv = [1, 1]  # factinv[n] = ((n!)^(-1) mod p)
inv = [0, 1]  # factinv 計算用
for i in range(2, N + 1):
    fact.append((fact[-1] * i) % mod)
    inv.append((-inv[mod % i] * (mod // i)) % mod)
    factinv.append((factinv[-1] * inv[-1]) % mod)


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
        """[l,r)のハッシュ値を取得"""
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
    # return 
INF=float('inf')
# 最小:INF 最大:-INF 区間和:0 区間積:1 最大公約数 0
ide_ele = 

# 遅延セグ木(区間加算用）
def segfunc(x,y):
    return x+y
class LazySegTree_RAQ:
    def __init__(self,init_val,segfunc,ide_ele):
        n = len(init_val)
        self.segfunc = segfunc
        self.ide_ele = ide_ele
        self.num = 1<<(n-1).bit_length()
        self.tree = [ide_ele]*2*self.num
        self.lazy = [0]*2*self.num
        for i in range(n):
            self.tree[self.num+i] = init_val[i]
        for i in range(self.num-1,0,-1):
            self.tree[i] = self.segfunc(self.tree[2*i], self.tree[2*i+1])
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
            if v==0:
                continue
            self.lazy[i] = 0
            self.lazy[2*i] += v
            self.lazy[2*i+1] += v
            self.tree[2*i] += v
            self.tree[2*i+1] += v
    def add(self,l,r,x):
        ids = self.gindex(l,r)
        l += self.num
        r += self.num
        while l<r:
            if l&1:
                self.lazy[l] += x
                self.tree[l] += x
                l += 1
            if r&1:
                self.lazy[r-1] += x
                self.tree[r-1] += x
            r >>= 1
            l >>= 1
        for i in ids:
            self.tree[i] = self.segfunc(self.tree[2*i], self.tree[2*i+1]) + self.lazy[i]
    def query(self,l,r):
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

# 遅延セグ木(区間更新用)
def segfunc(x,y):
    return min(x,y)
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


# 行列
import math
def Extended_Euclid(n,m):
    stack=[]
    while m:
        stack.append((n,m))
        n,m=m,n%m
    if n>=0:
        x,y=1,0
    else:
        x,y=-1,0
    for i in range(len(stack)-1,-1,-1):
        n,m=stack[i]
        x,y=y,x-(n//m)*y
    return x,y 
class MOD:
    def __init__(self,p,e=None):
        self.p=p
        self.e=e
        if self.e==None:
            self.mod=self.p
        else:
            self.mod=self.p**self.e
 
    def Pow(self,a,n):
        a%=self.mod
        if n>=0:
            return pow(a,n,self.mod)
        else:
            assert math.gcd(a,self.mod)==1
            x=Extended_Euclid(a,self.mod)[0]
            return pow(x,-n,self.mod)
 
    def Build_Fact(self,N):
        assert N>=0
        self.factorial=[1]
        if self.e==None:
            for i in range(1,N+1):
                self.factorial.append(self.factorial[-1]*i%self.mod)
        else:
            self.cnt=[0]*(N+1)
            for i in range(1,N+1):
                self.cnt[i]=self.cnt[i-1]
                ii=i
                while ii%self.p==0:
                    ii//=self.p
                    self.cnt[i]+=1
                self.factorial.append(self.factorial[-1]*ii%self.mod)
        self.factorial_inve=[None]*(N+1)
        self.factorial_inve[-1]=self.Pow(self.factorial[-1],-1)
        for i in range(N-1,-1,-1):
            ii=i+1
            while ii%self.p==0:
                ii//=self.p
            self.factorial_inve[i]=(self.factorial_inve[i+1]*ii)%self.mod
 
    def Fact(self,N):
        if N<0:
            return 0
        retu=self.factorial[N]
        if self.e!=None and self.cnt[N]:
            retu*=pow(self.p,self.cnt[N],self.mod)%self.mod
            retu%=self.mod
        return retu
 
    def Fact_Inve(self,N):
        if self.e!=None and self.cnt[N]:
            return None
        return self.factorial_inve[N]
 
    def Comb(self,N,K,divisible_count=False):
        if K<0 or K>N:
            return 0
        retu=self.factorial[N]*self.factorial_inve[K]%self.mod*self.factorial_inve[N-K]%self.mod
        if self.e!=None:
            cnt=self.cnt[N]-self.cnt[N-K]-self.cnt[K]
            if divisible_count:
                return retu,cnt
            else:
                retu*=pow(self.p,cnt,self.mod)
                retu%=self.mod
        return retu
class Matrix:
    def __init__(self,H=0,W=0,matrix=False,eps=0,mod=0,identity=0):
        if identity:
            if H:
                self.H=H
                self.W=H
            else:
                self.H=W
                self.W=W
            self.matrix=[[0]*self.W for i in range(self.H)]
            for i in range(self.H):
                self.matrix[i][i]=identity
        elif matrix:
            self.matrix=matrix
            self.H=len(self.matrix)
            self.W=len(self.matrix[0]) if self.matrix else 0
        else:
            self.H=H
            self.W=W
            self.matrix=[[0]*self.W for i in range(self.H)]
        self.mod=mod
        self.eps=eps
 
    def __eq__(self,other):
        if type(other)!=Matrix:
            return False
        if self.H!=other.H:
            return False
        if self.mod:
            for i in range(self.H):
                for j in range(self.W):
                    if self.matrix[i][j]%self.mod!=other.matrix[i][j]%self.mod:
                        return False
        else:
            for i in range(self.H):
                for j in range(self.W):
                    if self.eps<abs(self.matrix[i][j]-other.matrix[i][j]):
                        return False
        return True
 
    def __ne__(self,other):
        if type(other)!=Matrix:
            return True
        if self.H!=other.H:
            return True
        if self.mod:
            for i in range(self.H):
                for j in range(self.W):
                    if self.matrix[i][j]%self.mod!=other.matrix[i][j]%self.mod:
                        return True
        else:
            for i in range(self.H):
                for j in range(self.W):
                    if self.eps<abs(self.matrix[i][j]-other.matrix[i][j]):
                        return True
        return False
 
    def __add__(self,other):
        if type(other)==Matrix:
            assert self.H==other.H
            assert self.W==other.W
            if self.mod:
                summ=Matrix(matrix=[[(self.matrix[i][j]+other.matrix[i][j])%self.mod for j in range(self.W)] for i in range(self.H)],eps=self.eps,mod=self.mod)
            else:
                summ=Matrix(matrix=[[self.matrix[i][j]+other.matrix[i][j] for j in range(self.W)] for i in range(self.H)],eps=self.eps,mod=self.mod)
        else:
            if self.mod:
                summ=Matrix(matrix=[[(self.matrix[i][j]+other)%self.mod for j in range(self.W)] for i in range(self.H)],eps=self.eps,mod=self.mod)
            else:
                summ=Matrix(matrix=[[self.matrix[i][j]+other for j in range(self.W)] for i in range(self.H)],eps=self.eps,mod=self.mod)
        return summ
 
    def __sub__(self,other):
        if type(other)==Matrix:
            assert self.H==other.H
            assert self.W==other.W
            if self.mod:
                diff=Matrix(matrix=[[(self.matrix[i][j]-other.matrix[i][j])%self.mod for j in range(self.W)] for i in range(self.H)],eps=self.eps,mod=self.mod)
            else:
                diff=Matrix(matrix=[[self.matrix[i][j]-other.matrix[i][j] for j in range(self.W)] for i in range(self.H)],eps=self.eps,mod=self.mod)
        else:
            if self.mod:
                diff=Matrix(matrix=[[(self.matrix[i][j]-other)%self.mod for j in range(self.W)] for i in range(self.H)],eps=self.eps,mod=self.mod)
            else:
                diff=Matrix(matrix=[[self.matrix[i][j]-other for j in range(self.W)] for i in range(self.H)],eps=self.eps,mod=self.mod)
        return diff
 
    def __mul__(self,other):
        if type(other)==Matrix:
            assert self.H==other.H
            assert self.W==other.W
            if self.mod:
                prod=Matrix(matrix=[[(self.matrix[i][j]*other.matrix[i][j])%self.mod for j in range(self.W)] for i in range(self.H)],eps=self.eps,mod=self.mod)
            else:
                prod=Matrix(matrix=[[self.matrix[i][j]*other.matrix[i][j] for j in range(self.W)] for i in range(self.H)],eps=self.eps,mod=self.mod)
        else:
            if self.mod:
                prod=Matrix(matrix=[[(self.matrix[i][j]*other)%self.mod for j in range(self.W)] for i in range(self.H)],eps=self.eps,mod=self.mod)
            else:
                prod=Matrix(matrix=[[self.matrix[i][j]*other for j in range(self.W)] for i in range(self.H)],eps=self.eps,mod=self.mod)
        return prod
 
    def __matmul__(self,other):
        if type(other)==Matrix:
            assert self.W==other.H
            prod=Matrix(H=self.H,W=other.W,eps=self.eps,mod=self.mod)
            for i in range(self.H):
                for j in range(other.W):
                    for k in range(self.W):
                        prod.matrix[i][j]+=self.matrix[i][k]*other.matrix[k][j]
                        if self.mod:
                            prod.matrix[i][j]%=self.mod
        elif type(other)==int:
            assert self.H==self.W
            if other==0:
                prod=Matrix(H=self.H,eps=self.eps,mod=self.mod,identity=1)
            elif other==1:
                prod=Matrix(matrix=[[self.matrix[i][j] for j in range(self.W)] for i in range(self.H)],eps=self.eps,mod=self.mod)
            else:
                prod=Matrix(H=self.H,eps=self.eps,mod=self.mod,identity=1)
                doub=Matrix(matrix=[[self.matrix[i][j] for j in range(self.W)] for i in range(self.H)],eps=self.eps,mod=self.mod)
                while other>=2:
                    if other&1:
                        prod@=doub
                    doub@=doub
                    other>>=1
                prod@=doub
        return prod
 
    def __truediv__(self,other):
        if type(other)==Matrix:
            assert self.H==other.H
            assert self.W==other.W
            if self.mod:
                quot=Matrix(matrix=[[(self.matrix[i][j]*MOD(self.mod).Pow(other.matrix[i][j],-1))%self.mod for j in range(self.W)] for i in range(self.H)],eps=self.eps,mod=self.mod)
            else:
                quot=Matrix(matrix=[[self.matrix[i][j]/other.matrix[i][j] for j in range(self.W)] for i in range(self.H)],eps=self.eps,mod=self.mod)
        else:
            if self.mod:
                inve=MOD(self.mod).Pow(other,-1)
                quot=Matrix(matrix=[[(self.matrix[i][j]*inve)%self.mod for j in range(self.W)] for i in range(self.H)],eps=self.eps,mod=self.mod)
            else:
                quot=Matrix(matrix=[[self.matrix[i][j]/other for j in range(self.W)] for i in range(self.H)],eps=self.eps,mod=self.mod)
        return quot
 
    def __floordiv__(self,other):
        if type(other)==Matrix:
            assert self.H==other.H
            assert self.W==other.W
            quot=Matrix(matrix=[[self.matrix[i][j]//other.matrix[i][j] for j in range(self.W)] for i in range(self.H)],eps=self.eps,mod=self.mod)
        else:
            quot=Matrix(matrix=[[self.matrix[i][j]//other for j in range(self.W)] for i in range(self.H)],eps=self.eps,mod=self.mod)
        return quot
 
    def __mod__(self,other):
        if type(other)==Matrix:
            assert self.H==other.H
            assert self.W==other.W
            rema=Matrix(matrix=[[self.matrix[i][j]%other.matrix[i][j] for j in range(self.W)] for i in range(self.H)],eps=self.eps,mod=self.mod)
        else:
            rema=Matrix(matrix=[[self.matrix[i][j]%other for j in range(self.W)] for i in range(self.H)],eps=self.eps,mod=self.mod)
        return rema
 
    def __pow__(self,other):
        if type(other)==Matrix:
            assert self.H==other.H
            assert self.W==other.W
            if self.mod:
                powe=Matrix(matrix=[[pow(self.matrix[i][j],other.matrix[i][j],self.mod) for j in range(self.W)] for i in range(self.H)],eps=self.eps,mod=self.mod)
            else:
                powe=Matrix(matrix=[[pow(self.matrix[i][j],other.matrix[i][j]) for j in range(self.W)] for i in range(self.H)],eps=self.eps,mod=self.mod)
        else:
            if self.mod:
                powe=Matrix(matrix=[[pow(self.matrix[i][j],other,self.mod) for j in range(self.W)] for i in range(self.H)],eps=self.eps,mod=self.mod)
            else:
                powe=Matrix(matrix=[[pow(self.matrix[i][j],other) for j in range(self.W)] for i in range(self.H)],eps=self.eps,mod=self.mod)
        return powe
 
    def __lshift__(self,other):
        if type(other)==Matrix:
            assert self.H==other.H
            assert self.W==other.W
            lshi=Matrix(matrix=[[self.matrix[i][j]<<other.matrix[i][j] for j in range(self.W)] for i in range(self.H)],eps=self.eps,mod=self.mod)
        else:
            lshi=Matrix(matrix=[[self.matrix[i][j]<<other for j in range(self.W)] for i in range(self.H)],eps=self.eps,mod=self.mod)
        return lshi
 
    def __rshift__(self,other):
        if type(other)==Matrix:
            assert self.H==other.H
            assert self.W==other.W
            rshi=Matrix(matrix=[[self.matrix[i][j]>>other.matrix[i][j] for j in range(self.W)] for i in range(self.H)],eps=self.eps,mod=self.mod)
        else:
            rshi=Matrix(matrix=[[self.matrix[i][j]>>other for j in range(self.W)] for i in range(self.H)],eps=self.eps,mod=self.mod)
        return rshi
 
    def __and__(self,other):
        if type(other)==Matrix:
            assert self.H==other.H
            assert self.W==other.W
            conj=Matrix(matrix=[[self.matrix[i][j]&other.matrix[i][j] for j in range(self.W)] for i in range(self.H)],eps=self.eps,mod=self.mod)
        else:
            conj=Matrix(matrix=[[self.matrix[i][j]&other for j in range(self.W)] for i in range(self.H)],eps=self.eps,mod=self.mod)
        return conj
 
    def __or__(self,other):
        if type(other)==Matrix:
            assert self.H==other.H
            assert self.W==other.W
            disj=Matrix(matrix=[[self.matrix[i][j]|other.matrix[i][j] for j in range(self.W)] for i in range(self.H)],eps=self.eps,mod=self.mod)
        else:
            disj=Matrix(matrix=[[self.matrix[i][j]|other for j in range(self.W)] for i in range(self.H)],eps=self.eps,mod=self.mod)
        return disj
 
    def __xor__(self,other):
        if type(other)==Matrix:
            assert self.H==other.H
            assert self.W==other.W
            excl=Matrix(matrix=[[self.matrix[i][j]^other.matrix[i][j] for j in range(self.W)] for i in range(self.H)],eps=self.eps,mod=self.mod)
        else:
            excl=Matrix(matrix=[[self.matrix[i][j]^other for j in range(self.W)] for i in range(self.H)],eps=self.eps,mod=self.mod)
        return excl
 
    def __iadd__(self,other):
        if type(other)==Matrix:
            assert self.H==other.H
            assert self.W==other.W
            for i in range(self.H):
                for j in range(self.W):
                    self.matrix[i][j]+=other.matrix[i][j]
                    if self.mod:
                        self.matrix[i][j]%=self.mod
        else:
            for i in range(self.H):
                for j in range(self.W):
                    self.matrix[i][j]+=other
                    if self.mod:
                        self.matrix[i][j]%=self.mod
        return self
 
    def __isub__(self,other):
        if type(other)==Matrix:
            assert self.H==other.H
            assert self.W==other.W
            for i in range(self.H):
                for j in range(self.W):
                    self.matrix[i][j]-=other.matrix[i][j]
                    if self.mod:
                        self.matrix[i][j]%=self.mod
        else:
            for i in range(self.H):
                for j in range(self.W):
                    self.matrix[i][j]-=other
                    if self.mod:
                        self.matrix[i][j]%=self.mod
        return self
 
    def __imul__(self,other):
        if type(other)==Matrix:
            assert self.H==other.H
            assert self.W==other.W
            for i in range(self.H):
                for j in range(self.W):
                    self.matrix[i][j]*=other.matrix[i][j]
                    if self.mod:
                        self.matrix[i][j]%=self.mod
        else:
            for i in range(self.H):
                for j in range(self.W):
                    self.matrix[i][j]*=other
                    if self.mod:
                        self.matrix[i][j]%=self.mod
        return self
 
    def __imatmul__(self,other):
        if type(other)==Matrix:
            assert self.W==other.H
            prod=Matrix(H=self.H,W=other.W,eps=self.eps,mod=self.mod)
            for i in range(self.H):
                for j in range(other.W):
                    for k in range(self.W):
                        prod.matrix[i][j]+=self.matrix[i][k]*other.matrix[k][j]
                        if self.mod:
                            prod.matrix[i][j]%=self.mod
        elif type(other)==int:
            assert self.H==self.W
            if other==0:
                return Matrix(H=self.H,eps=self.eps,mod=self.mod,identity=1)
            elif other==1:
                prod=Matrix(matrix=[[self.matrix[i][j] for j in range(self.W)] for i in range(self.H)],eps=self.eps,mod=self.mod)
            else:
                prod=Matrix(H=self.H,eps=self.eps,mod=self.mod,identity=1)
                doub=self
                while other>=2:
                    if other&1:
                        prod@=doub
                    doub@=doub
                    other>>=1
                prod@=doub
        return prod
 
    def __itruediv__(self,other):
        if type(other)==Matrix:
            assert self.H==other.H
            assert self.W==other.W
            for i in range(self.H):
                for j in range(self.W):
                    if self.mod:
                        self.matrix[i][j]=self.matrix[i][j]*MOD(self.mod).Pow(other.matrix[i][j],-1)%self.mod
                    else:
                        self.matrix[i][j]/=other.matrix[i][j]
        else:
            if self.mod:
                inve=MOD(self.mod).Pow(other,-1)
            for i in range(self.H):
                for j in range(self.W):
                    if self.mod:
                        self.matrix[i][j]=self.matrix[i][j]*inve%self.mod
                    else:
                        self.matrix[i][j]/=other
        return self
 
    def __ifloordiv__(self,other):
        if type(other)==Matrix:
            assert self.H==other.H
            assert self.W==other.W
            for i in range(self.H):
                for j in range(self.W):
                    self.matrix[i][j]//=other.matrix[i][j]
        else:
            for i in range(self.H):
                for j in range(self.W):
                    self.matrix[i][j]//=other
        return self
 
    def __imod__(self,other):
        if type(other)==Matrix:
            assert self.H==other.H
            assert self.W==other.W
            for i in range(self.H):
                for j in range(self.W):
                    self.matrix[i][j]%=other.matrix[i][j]
        else:
            for i in range(self.H):
                for j in range(self.W):
                    self.matrix[i][j]%=other
        return self
 
    def __ipow__(self,other):
        if type(other)==Matrix:
            assert self.H==other.H
            assert self.W==other.W
            for i in range(self.H):
                for j in range(self.W):
                    if self.mod:
                        self.matrix[i][j]=pow(self.matrix[i][j],other.matrix[i][j],self.mod)
                    else:
                        self.matrix[i][j]=pow(self.matrix[i][j],other.matrix[i][j])
        else:
            for i in range(self.H):
                for j in range(self.W):
                    if self.mod:
                        self.matrix[i][j]=pow(self.matrix[i][j],other,self.mod)
                    else:
                        self.matrix[i][j]=pow(self.matrix[i][j],other)
        return self
 
    def __ilshift__(self,other):
        if type(other)==Matrix:
            assert self.H==other.H
            assert self.W==other.W
            for i in range(self.H):
                for j in range(self.W):
                    self.matrix[i][j]<<=other.matrix[i][j]
        else:
            for i in range(self.H):
                for j in range(self.W):
                    self.matrix[i][j]<<=other
        return self
 
    def __irshift__(self,other):
        if type(other)==Matrix:
            assert self.H==other.H
            assert self.W==other.W
            for i in range(self.H):
                for j in range(self.W):
                    self.matrix[i][j]>>=other.matrix[i][j]
        else:
            for i in range(self.H):
                for j in range(self.W):
                    self.matrix[i][j]>>=other
        return self
 
    def __iand__(self,other):
        if type(other)==Matrix:
            assert self.H==other.H
            assert self.W==other.W
            for i in range(self.H):
                for j in range(self.W):
                    self.matrix[i][j]&=other.matrix[i][j]
        else:
            for i in range(self.H):
                for j in range(self.W):
                    self.matrix[i][j]&=other
        return self
 
    def __ior__(self,other):
        if type(other)==Matrix:
            assert self.H==other.H
            assert self.W==other.W
            for i in range(self.H):
                for j in range(self.W):
                    self.matrix[i][j]|=other.matrix[i][j]
        else:
            for i in range(self.H):
                for j in range(self.W):
                    self.matrix[i][j]|=other
        return self
 
    def __ixor__(self,other):
        if type(other)==Matrix:
            assert self.H==other.H
            assert self.W==other.W
            for i in range(self.H):
                for j in range(self.W):
                    self.matrix[i][j]^=other.matrix[i][j]
        else:
            for i in range(self.H):
                for j in range(self.W):
                    self.matrix[i][j]^=other
        return self
 
    def __neg__(self):
        if self.mod:
            nega=Matrix(matrix=[[(-self.matrix[i][j])%self.mod for j in range(self.W)] for i in range(self.H)],eps=self.eps,mod=self.mod)
        else:
            nega=Matrix(matrix=[[-self.matrix[i][j] for j in range(self.W)] for i in range(self.H)],eps=self.eps,mod=self.mod)
        return nega
 
    def __pos__(self):
        posi=Matrix(matrix=[[self.matrix[i][j] for j in range(self.W)] for i in range(self.H)],eps=self.eps,mod=self.mod)
        return posi
 
    def __invert__(self):
        inve=Matrix(matrix=[[~self.matrix[i][j] for j in range(self.W)] for i in range(self.H)],eps=self.eps,mod=self.mod)
        return inve
 
    def __abs__(self):
        abso=Matrix(matrix=[[abs(self.matrix[i][j]) for j in range(self.W)] for i in range(self.H)],eps=self.eps,mod=self.mod)
        return abso
 
    def __getitem__(self,i):
        if type(i)==int:
            return self.matrix[i]
        elif type(i)==tuple:
            i,j=i
            if type(i)==int:
                i=slice(i,i+1)
            if type(j)==int:
                j=slice(j,j+1)
            return Matrix(matrix=[lst[j] for lst in self.matrix[i]],eps=self.eps,mod=self.mod)
 
    def __contains__(self,x):
        for i in range(self.H):
            if x in self.matrix[i]:
                return True
        return False
 
    def __str__(self):
        digit=[max(len(str(self.matrix[i][j])) for i in range(self.H)) for j in range(self.W)]
        return "\n".join([(" [" if i else "[[")+", ".join([str(self.matrix[i][j]).rjust(digit[j]," ") for j in range(self.W)])+"]" for i in range(self.H)])+"]"
 
    def __bool__(self):
        return True
 
    def Transpose(self):
        return Matrix(matrix=[[self.matrix[i][j] for i in range(self.H)] for j in range(self.W)])
 
    def Trace(self):
        assert self.H==self.W
        trace=sum(self.matrix[i][i] for i in range(self.H))
        if self.mod:
            trace%=self.mod
        return trace
 
    def Elem_Raw_Operate_1(self,i0,i1):
        self.matrix[i0],self.matrix[i1]=self.matrix[i1],self.matrix[i0]
 
    def Elem_Raw_Operate_2(self,i,c):
        if self.mod:
            self.matrix[i]=[self.matrix[i][j]*c%self.mod for j in range(self.W)]
        else:
            self.matrix[i]=[self.matrix[i][j]*c for j in range(self.W)]
 
    def Elem_Raw_Operate_3(self,i0,i1,c):
        if self.mod:
            self.matrix[i0]=[(self.matrix[i0][j]+c*self.matrix[i1][j])%self.mod for j in range(self.W)]
        else:
            self.matrix[i0]=[self.matrix[i0][j]+c*self.matrix[i1][j] for j in range(self.W)]
 
    def Elimination(self,determinant=False,inverse_matrix=False,linear_equation=False,rank=False,upper_triangular=False):
        h=0
        ut=Matrix(matrix=[[self.matrix[i][j] for j in range(self.W)] for i in range(self.H)],eps=self.eps,mod=self.mod)
        if determinant or inverse_matrix:
            assert self.H==self.W
            det=1
        if inverse_matrix:
            assert self.H==self.W
            im=Matrix(H=self.H,eps=self.eps,mod=self.mod,identity=1)
        if linear_equation:
            assert self.H==linear_equation.H
            le=Matrix(matrix=[[linear_equation.matrix[i][j] for j in range(linear_equation.W)] for i in range(linear_equation.H)],eps=self.eps,mod=self.mod)
        for j in range(ut.W):
            for i in range(h,ut.H):
                if abs(ut.matrix[i][j])>ut.eps:
                    if determinant or inverse_matrix:
                        det*=ut.matrix[i][j]
                        if self.mod:
                            det%=self.mod
                    if self.mod:
                        inve=MOD(self.mod).Pow(ut.matrix[i][j],-1)
                    else:
                        inve=1/ut.matrix[i][j]
 
                    ut.Elem_Raw_Operate_1(i,h)
                    if determinant and i!=h and self.mod:
                        det=(-det)%self.mod
                    if inverse_matrix:
                        im.Elem_Raw_Operate_1(i,h)
                    if linear_equation:
                        le.Elem_Raw_Operate_1(i,h)
 
                    ut.Elem_Raw_Operate_2(h,inve)
                    if inverse_matrix:
                        im.Elem_Raw_Operate_2(h,inve)
                    if linear_equation:
                        le.Elem_Raw_Operate_2(h,inve)
 
                    for ii in range(ut.H):
                        if ii==h:
                            continue
                        x=-ut.matrix[ii][j]
                        ut.Elem_Raw_Operate_3(ii,h,x)
                        if inverse_matrix:
                            im.Elem_Raw_Operate_3(ii,h,x)
                        if linear_equation:
                            le.Elem_Raw_Operate_3(ii,h,x)
                    h+=1
                    break
            else:
                det=0
        if any(le[i][0] for i in range(h,self.H)):
            le=None
        tpl=()
        if determinant:
            tpl+=(det,)
        if inverse_matrix:
            if det==0:
                im=None
            tpl+=(im,)
        if linear_equation:
            tpl+=(le,)
        if rank:
            tpl+=(h,)
        if upper_triangular:
            tpl+=(ut,)
        if len(tpl)==1:
            tpl=tpl[0]
        return tpl
    
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
N = ...
G = [[...] for i in range(N)]

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

