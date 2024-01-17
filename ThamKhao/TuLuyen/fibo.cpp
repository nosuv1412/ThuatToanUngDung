#include<bits/stdc++.h>
using namespace std;

map <int,long long> a; 

long fibo(long long n){
	if(n==1) return 0;
//	auto p=a.find(n);
//	if(p!=end(a)) return p->second;

	map <int, long long>::iterator p = a.find(n);
	if(p != a.end()) return p->second;
	
	long long kq=0;
	if(n%2==0) kq= fibo(n/2)+1;
	else kq= fibo(3*n +1)+1;
	a[n]=kq;
	return kq;
	
}

int main(){
	int n;cin>>n;
	cout<<fibo(n);
}

//F(N) = 0 neu N = 1
//F(N) = 1 + F(N/2) neu N chan > 1
//F(N) = 1 + F(3N + 1) neu N le > 1
