#include<iostream>
#include<string>
using namespace std;
const long long MAX=999999;

long long n,m,a[MAX];
string Z, W;

int init(){
	getline(cin,Z);
	W="";
	n=Z.length();
	
	for(int i=n-1;i>=0;i--)
		W+=Z[i];
	cin>>m;
	for(int i=1;i<=m;i++)
		cin>>a[i];
}

int gen(int t,int k){
	for(int i=1;i<=m;i++)
	{
		t=a[i]%n;
		k=a[i]/n;
		if(k%2==1) cout<<W[t]<<endl;
		else if(k%2 == 0) cout<<Z[t]<<endl;
	}
}

int main()
{
	init();
	gen(0,0);
	return 0;
}
