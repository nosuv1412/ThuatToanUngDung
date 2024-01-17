
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;


ll MOD(string num,int mod)
{
	ll res=0;
	
	for(int i=0;i<num.length();i++)
	res=(res*10+num[i]-'0')%mod;
	
	return res;
}

ll ModExponent(ll a,ll b,ll m)
{
	ll result;
	if(a==0)
	return 0;
	else if(b==0)
	return 1;
	else if(b&1)
	{
		result=a%m;
		result=result*ModExponent(a,b-1,m);
	}
	else{
		result=ModExponent(a,b/2,m);
		result=((result%m)*(result%m))%m;
	}
	return (result%m+m)%m;
}

int main()
{
//	string a = "11";
//	string b = "22";
	string a,b;
	cout<<"A = ";
	getline(cin,a);
	cout<<"B = ";
	getline(cin,b);
	ll m;
	cout<<"M = ";cin>>m;
	ll remainderA = MOD(a,m);
	ll remainderB = MOD(b,m);

	cout << "G = "<<ModExponent(remainderA, remainderB, m);
	
	return 0;
}
