#include<bits/stdc++.h>
using namespace std;

const unsigned long long M = 1000000007;
map<int, long long> a;
long long sinh(int n){
	if(n==0) return 1;
	if(n<0) return 0;
	map <int, long long>::iterator p = a.find(n);
	if(p != a.end()) return p->second;
	a[n]= sinh(n-2)%M+sinh(n-3)%M;
	return a[n];
}

long long f2(int n){
	a[0]=1;
	a[1]=0;
	a[2]=1;
	a[3]=1;
	for(int i=3;i<=n;i++) a[i]=a[i-2]%M+a[i-3]%M;
	return a[n];
}

int main(){
	int n;cin>>n;
	cout<<sinh(n)%M<<endl;
//	cout<<f2(n)%M;
}

