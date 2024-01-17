#include<bits/stdc++.h>
using namespace std;

map<pair<int,int>,long long> a;
long long C(int k,int n){
	if(k==0||k==n) return 1;
	if(k>n) return 0;
//	auto p=a.find({k,n});
//	if(p!=a.end()) return p->second;

	map<pair<int,int>,long long>::iterator p = a.find({k,n});
	if(p != a.end()) return p->second;
	
	a[{k,n}]= C(k,n-1)+C(k-1,n-1); 
	return a[{k,n}];
}

int main(){
	int n,k;
	cout<<"nhap n: ";cin>>n;
	cout<<"nhap k: ";cin>>k;
	cout<<C(k,n);
}
