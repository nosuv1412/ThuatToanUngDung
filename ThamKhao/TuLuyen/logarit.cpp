#include<bits/stdc++.h>
using namespace std;

int main(){
	int a,b,c,d;
	cin>>a;
	cin>>b;
	cin>>c;
	cin>>d;
	float kq=log(pow(a,b))-log(pow(c,d));
	if(kq>0) cout<<"lon hon";
	if(kq==0) cout<<"bang nhau";
	if(kq<0) cout<<"nho hon";
	
}
