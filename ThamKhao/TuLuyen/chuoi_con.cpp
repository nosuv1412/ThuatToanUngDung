#include<bits/stdc++.h>
using namespace std;

const int MAX=1000;

long long f[MAX]={1},t=0;

int main(){
	string s;
	getline(cin,s);
	for(int p=1;p<=s.length();p++){
		f[p]=1;
		for(int k=p-1;k>0;k--){
			f[p]+=f[k];
			if(s[p-1]==s[k-1]){
				f[p]=f[p]-1;
				break;
			}	
		}
		t+=f[p];
	}
	cout<<t;
}



