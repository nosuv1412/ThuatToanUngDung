#include <iostream>

using namespace std;
/*
	g1 =1
	g3 =3
	g2n = gn
	g4n+1 = 2g(2n+1) - gn
	g4n+3 = 3g(2n+1) - 2gn
	
	g1 = 1
	g2 = 1
	g3 = 3
	g4 = 1
	g5 = 2 * g3 - g1 = 5
	g6 = 3
	g7 = 3 * g3 - 2 = 7
	g8 = 1
	g9 = 2 * g5 - g2 = 9
	g10 = g5 = 5
*/
long long fn ( long n){
	long long temp;
	if(n==1) return 1;
	if(n==3) return 3;
	if(n%2==0)
	{
		temp = n/2;
		return fn(temp);
	 } 
	if(n%4==1){
		temp = (n-1)/4;
		return 2*fn(2*temp+1) - fn(temp);
	}
	if(n%4==3){
		temp = (n-3)/4;
		return 3*fn(2*temp+1) - 2*fn(temp);
	}
}
int main()
{
	long long n ;
	cout<<"Nhap n = "; cin>>n;
	while(n<=0){
		cout<<"Nhap n = "; cin>>n;
	}
	cout<<"g("<<n<<") = "<<fn(n);
return 0;
}

