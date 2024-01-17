#include <bits/stdc++.h>

using namespace std; 

map <int, long long> a;
const unsigned long long M = 1000000007;

long long f(int n){
	if(n==0) return 1;
	if(n<0) return 0;
	map <int, long long>::iterator p = a.find(n);
	if(p != a.end()) return p->second;
	a[n]=f(n-2)%M+f(n-3)%M+f(n-4)%M;
	return a[n];
}

long long f2(int n){
	a[0]=1;
	a[1]=0;
	a[2]=1;
	a[3]=1;
	for(int i=4;i<=n;i++) a[i]=a[i-2]%M+a[i-3]%M+a[i-4]%M;
	return a[n];
}
int main(){
	int n;
	cin>>n;
	cout<<f2(n)%M;
}
